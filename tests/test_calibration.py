"""
Step 4 calibration assertions.

The four the task names - calibrator never sees test, fit population equals the
early-stopping population, calibrated output monotone in raw score, determinism - plus
unit coverage of the Platt fit and the metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import PRIMARY_LABEL, VALIDATION_DAYS, OlistLoader
from features.builder import FeatureBuilder
from models.calibration import (
    PlattCalibrator,
    brier_score,
    reliability,
    smece,
    uniform_mass_bins,
)
from models.train import predict, prepare_matrix, train

# Committed in eval/calibration.md.
CHOSEN_WINDOW_DAYS = 30
CEILING = 0.06856          # max calibrated P on test, 5 dp
TOP_DECILE_RATIO = 1.31


# ---------------------------------------------------------------------------------
# Platt unit behaviour
# ---------------------------------------------------------------------------------

def test_platt_recovers_known_parameters():
    rng = np.random.default_rng(0)
    f = rng.normal(size=4000)
    y = (rng.random(4000) < 1.0 / (1.0 + np.exp(1.5 * f + 1.0))).astype(int)
    c = PlattCalibrator().fit(f, y)
    assert c.a_ == pytest.approx(1.5, abs=0.15)
    assert c.b_ == pytest.approx(1.0, abs=0.15)


def test_platt_is_monotone_in_the_raw_score():
    rng = np.random.default_rng(1)
    f = rng.normal(size=2000)
    y = (rng.random(2000) < 1.0 / (1.0 + np.exp(2.0 * f))).astype(int)
    c = PlattCalibrator().fit(f, y)
    order = np.argsort(f)
    p = c.predict(f[order])
    d = np.diff(p)
    assert np.all(d >= -1e-12) or np.all(d <= 1e-12)


def test_platt_stays_finite_on_a_separable_score():
    """
    Target smoothing is the reason. With hard 0/1 labels a perfectly separated score
    drives the slope to infinity and every probability to 0 or 1.
    """
    f = np.concatenate([np.full(200, -5.0), np.full(8, 5.0)])
    y = np.concatenate([np.zeros(200, int), np.ones(8, int)])
    c = PlattCalibrator().fit(f, y)
    assert np.isfinite([c.a_, c.b_]).all()
    p = c.predict(f)
    assert p.min() > 0.0 and p.max() < 1.0


def test_platt_needs_both_classes():
    with pytest.raises(ValueError):
        PlattCalibrator().fit(np.arange(10.0), np.zeros(10, dtype=int))


def test_platt_fit_is_deterministic():
    rng = np.random.default_rng(2)
    f = rng.normal(size=1500)
    y = (rng.random(1500) < 0.02).astype(int)
    a = PlattCalibrator().fit(f, y)
    b = PlattCalibrator().fit(f, y)
    assert (a.a_, a.b_) == (b.a_, b.b_)


def test_calibration_improves_brier_on_a_miscalibrated_score():
    rng = np.random.default_rng(3)
    f = rng.normal(size=3000)
    y = (rng.random(3000) < 1.0 / (1.0 + np.exp(1.2 * f))).astype(int)
    overconfident = 1.0 / (1.0 + np.exp(4.0 * f))
    c = PlattCalibrator().fit(f, y)
    assert brier_score(y, c.predict(f)) < brier_score(y, overconfident)


# ---------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------

def test_uniform_mass_bins_are_roughly_equal_count():
    rng = np.random.default_rng(4)
    p = rng.beta(1, 60, size=10000)
    b = uniform_mass_bins(p, 10)
    counts = np.bincount(b)
    assert len(counts) == 10
    assert counts.max() / counts.min() < 1.5


def test_uniform_mass_bins_survive_heavy_ties():
    p = np.concatenate([np.full(900, 0.01), np.linspace(0.02, 0.2, 100)])
    b = uniform_mass_bins(p, 10)
    assert len(np.unique(b)) >= 2      # degenerate but must not crash or collapse


def test_smece_is_zero_for_a_perfectly_calibrated_scorer():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, size=40000)
    y = (rng.random(40000) < p).astype(int)
    assert smece(y, p, 0.05) < 0.02


def test_smece_grows_with_miscalibration():
    rng = np.random.default_rng(6)
    p = rng.uniform(0.05, 0.95, size=20000)
    y = (rng.random(20000) < p).astype(int)
    good = smece(y, p, 0.05)
    bad = smece(y, np.clip(p * 2.0, 0, 1), 0.05)
    assert bad > good


def test_reliability_bins_sum_to_the_sample():
    rng = np.random.default_rng(7)
    p = rng.beta(1, 80, size=5000)
    y = (rng.random(5000) < p).astype(int)
    rep = reliability(y, p, n_bins=10, n_boot=200)
    assert rep.counts.sum() == 5000
    assert bool(np.all(rep.ci_low <= rep.observed))
    assert bool(np.all(rep.observed <= rep.ci_high))


# ---------------------------------------------------------------------------------
# The four named assertions, on the real pipeline
# ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(loader: OlistLoader):
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    matrix = FeatureBuilder().build(risk)
    split = risk["split"].to_numpy()
    y = risk[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
        target=PRIMARY_LABEL, population="risk_set", category_levels=levels,
    )
    s_va = predict(bundle, X.loc[va], raw_score=True)
    s_te = predict(bundle, X.loc[te], raw_score=True)

    boundary = loader.split_boundary
    ts_va = risk.loc[va, "order_purchase_timestamp"]
    fit_mask = (ts_va >= boundary - pd.Timedelta(days=CHOSEN_WINDOW_DAYS)).to_numpy()
    cal = PlattCalibrator().fit(s_va[fit_mask], y[va][fit_mask])
    return {
        "risk": risk, "loader": loader, "cal": cal, "s_va": s_va, "s_te": s_te,
        "y_va": y[va], "y_te": y[te], "fit_mask": fit_mask, "ts_va": ts_va,
        "boundary": boundary,
    }


def test_calibrator_never_sees_test(pipeline):
    """
    The fit window is drawn from the validation split only, and validation and test are
    disjoint in both identity and time.
    """
    risk, loader = pipeline["risk"], pipeline["loader"]
    va = risk[risk["split"] == "validation"]
    te = risk[risk["split"] == "test"]
    assert set(va["order_id"]).isdisjoint(set(te["order_id"]))
    assert va["order_purchase_timestamp"].max() < te["order_purchase_timestamp"].min()

    fit_ts = pipeline["ts_va"][pipeline["fit_mask"]]
    assert bool((fit_ts < loader.split_boundary).all())


def test_fit_population_is_the_early_stopping_population(pipeline):
    """
    The calibrator's rows are a subset of the window the model early-stopped on - the
    same validation split, not a different slice of data.
    """
    risk = pipeline["risk"]
    va_ids = set(risk.loc[risk["split"] == "validation", "order_id"])
    fit_ids = set(
        risk.loc[risk["split"] == "validation", "order_id"].to_numpy()[
            pipeline["fit_mask"]
        ]
    )
    assert fit_ids <= va_ids
    assert len(fit_ids) > 0


def test_validation_window_length_matches_the_declared_constant(pipeline):
    span = (
        pipeline["ts_va"].max() - (pipeline["boundary"] - pd.Timedelta(days=VALIDATION_DAYS))
    ).days
    assert 0 <= span <= VALIDATION_DAYS


def test_calibrated_output_is_monotone_in_raw_score(pipeline):
    """Platt is monotone by construction; a failure here is a wiring error."""
    s_te = pipeline["s_te"]
    p = pipeline["cal"].predict(s_te)
    order = np.argsort(s_te, kind="stable")
    d = np.diff(p[order])
    assert np.all(d >= -1e-12) or np.all(d <= 1e-12)


def test_calibrated_predictions_are_deterministic(pipeline):
    cal2 = PlattCalibrator().fit(
        pipeline["s_va"][pipeline["fit_mask"]],
        pipeline["y_va"][pipeline["fit_mask"]],
    )
    assert np.array_equal(cal2.predict(pipeline["s_te"]),
                          pipeline["cal"].predict(pipeline["s_te"]))


# ---------------------------------------------------------------------------------
# Committed results
# ---------------------------------------------------------------------------------

def test_probability_ceiling_is_pinned(pipeline):
    """
    Step 5's per-order thresholds must live inside this range. If the ceiling moves, the
    policy layer's feasible threshold band moves with it.
    """
    p = pipeline["cal"].predict(pipeline["s_te"])
    assert p.max() == pytest.approx(CEILING, abs=5e-4)
    assert p.min() > 0.0


def test_top_decile_calibration_is_pinned(pipeline):
    p = pipeline["cal"].predict(pipeline["s_te"])
    y = pipeline["y_te"]
    dec = uniform_mass_bins(p, 10)
    top = dec == dec.max()
    ratio = float(p[top].mean()) / float(y[top].mean())
    assert ratio == pytest.approx(TOP_DECILE_RATIO, abs=0.05)


def test_calibration_beats_the_uncalibrated_probability_on_brier(pipeline):
    y, s_te = pipeline["y_te"], pipeline["s_te"]
    raw_prob = 1.0 / (1.0 + np.exp(-s_te))
    p = pipeline["cal"].predict(s_te)
    assert brier_score(y, p) < brier_score(y, raw_prob)
