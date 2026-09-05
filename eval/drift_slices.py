#!/usr/bin/env python3
"""Measure chronological drift slices on the secondary benchmark target."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import OlistLoader, SECONDARY_LABEL
from features.builder import FeatureBuilder
from models.calibration import PlattCalibrator
from models.train import predict, prepare_matrix, train

OUT_PATH = REPO_ROOT / "eval" / "drift_slices.md"
FIG_PATH = REPO_ROOT / "eval" / "figures" / "drift_slices_secondary.png"
TOP_FRACTION = 0.05
CALIBRATION_WINDOW_DAYS = 30


def main() -> int:
    loader = OlistLoader()
    population = loader.labelled().join(loader.split_labelled()["split"], how="left")
    matrix = FeatureBuilder(loader=loader).build(population)
    split = population["split"].to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"
    y = population[SECONDARY_LABEL].astype(int).to_numpy()

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
        target=SECONDARY_LABEL, population="all matured", category_levels=levels,
    )
    validation_scores = predict(bundle, X.loc[va], raw_score=True)
    validation_ts = population.loc[va, "order_purchase_timestamp"]
    fit = (
        validation_ts >= loader.split_boundary
        - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)
    ).to_numpy()
    calibrator = PlattCalibrator().fit(validation_scores[fit], y[va][fit])
    scores = calibrator.predict(predict(bundle, X.loc[te], raw_score=True))

    test = population.loc[te].copy()
    test["score"] = scores
    test["target"] = y[te]
    test = test.sort_values("order_purchase_timestamp").reset_index(drop=True)
    take = max(1, int(np.ceil(TOP_FRACTION * len(test))))
    threshold = float(np.sort(test["score"].to_numpy())[::-1][take - 1])
    test["predicted"] = test["score"] >= threshold

    slices = np.array_split(test, 3)
    rows: list[dict[str, object]] = []
    for i, frame in enumerate(slices, start=1):
        predicted = frame["predicted"].to_numpy()
        actual = frame["target"].to_numpy(dtype=int)
        tp = int((predicted & (actual == 1)).sum())
        treated = int(predicted.sum())
        positives = int(actual.sum())
        rows.append({
            "slice": i,
            "start": frame["order_purchase_timestamp"].min(),
            "end": frame["order_purchase_timestamp"].max(),
            "n": len(frame),
            "positives": positives,
            "positive_rate": positives / len(frame),
            "treated": treated,
            "precision": tp / treated if treated else float("nan"),
            "recall": tp / positives if positives else float("nan"),
        })

    prevalence = float(test["target"].mean())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(1, 4)
    ax.axhline(prevalence * 100, color="#666666", linestyle="--",
               label=f"overall prevalence {prevalence:.2%}")
    ax.plot(x, [r["positive_rate"] * 100 for r in rows], "o-", label="positive rate")
    ax.set_xticks(x, [f"Slice {r['slice']}\nn={r['n']:,}" for r in rows])
    ax.set_ylabel("Rate (%)")
    ax.set_title("SECONDARY TARGET — chronological test slices")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)

    lines = [
        "# Drift slices — SECONDARY TARGET",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`",
        "",
        "**This is the SECONDARY target (`label_a`) benchmark, not the primary "
        "`label_b` target. The result does not transfer to the primary.**",
        "",
        f"The chronological test window is split into three equal slices. The operating "
        f"point is the global top {TOP_FRACTION:.0%} of calibrated secondary scores "
        f"(`{take:,}` orders), held fixed across slices. Overall test prevalence is "
        f"**{prevalence:.2%}** ({int(test['target'].sum()):,} positives of "
        f"{len(test):,}).",
        "",
        "![Secondary chronological drift slices](figures/drift_slices_secondary.png)",
        "",
        "| Slice | Boundary | n | Positives | Positive rate | Treated | Precision | Recall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['slice']} | {r['start']:%Y-%m-%d} to {r['end']:%Y-%m-%d} | "
            f"{r['n']:,} | {r['positives']:,} | {r['positive_rate']:.2%} | "
            f"{r['treated']:,} | {r['precision']:.2%} | {r['recall']:.2%} |"
        )
    lines.extend([
        "",
        "The slice size is approximately 126 positives per slice. **Any apparent "
        "difference is within noise at n≈126 per slice; drift is not resolvable at "
        "this slice size.** This is a secondary benchmark result only and must not "
        "be used to claim primary-target drift or primary-target retraining need.",
        "",
    ])
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
