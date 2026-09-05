"""Generate semi-synthetic intervention worlds from real Olist test covariates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import OlistLoader, PRIMARY_LABEL
from features.builder import FeatureBuilder
from policy.effectiveness import EFFECTIVENESS, REASONS

TARGET_TREATED_FRACTION = 0.20
DEFAULT_XI = (0.0, 1.0, 2.0)
SEED = 20260905


@dataclass(frozen=True)
class SimulationWorld:
    xi: float
    features: pd.DataFrame
    y_obs: np.ndarray
    score: np.ndarray
    treatment_probability: np.ndarray
    treatment: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    y: np.ndarray
    reason: np.ndarray
    tier: str
    tau: np.ndarray

    @property
    def treated_fraction(self) -> float:
        return float(self.treatment.mean())

    @property
    def expected_treated_fraction(self) -> float:
        return float(self.treatment_probability.mean())

    def oracle_prevented_failures(self) -> int:
        return int((self.y0 - self.y1).sum())

    def oracle_policy_value(self) -> float:
        """Observed failure cost remaining after applying the planted response."""
        return float(self.y1.sum())


def _real_test_window() -> tuple[pd.DataFrame, np.ndarray]:
    loader = OlistLoader()
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    test = risk[risk["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    features = FeatureBuilder(loader=loader).build(test).reset_index(drop=True)
    return features, test[PRIMARY_LABEL].astype(int).to_numpy()


def held_out_index(features: pd.DataFrame) -> np.ndarray:
    """Standardized observable covariate index; deliberately not the shipped score."""
    columns = ("order_value", "order_freight", "n_items", "n_missing_features")
    parts = []
    for column in columns:
        if column not in features:
            continue
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(float)
        center = np.nanmedian(values)
        scale = np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
        parts.append(np.nan_to_num((values - center) / (scale or 1.0)))
    if not parts:
        raise ValueError("held-out index has no admissible covariates")
    return np.mean(parts, axis=0)


def _assignment_probability(score: np.ndarray, xi: float) -> np.ndarray:
    target = TARGET_TREATED_FRACTION
    lo, hi = -20.0, 20.0
    for _ in range(80):
        alpha = (lo + hi) / 2.0
        probability = 1.0 / (1.0 + np.exp(-(alpha + xi * score)))
        if probability.mean() < target:
            lo = alpha
        else:
            hi = alpha
    alpha = (lo + hi) / 2.0
    return 1.0 / (1.0 + np.exp(-(alpha + xi * score)))


def _reason(features: pd.DataFrame) -> np.ndarray:
    values = {
        "order_composition": np.nan_to_num(features["order_value"].to_numpy(float)),
        "pincode": np.nan_to_num(features["pincode_failure_rate_smoothed"].to_numpy(float)),
        "customer_history": np.nan_to_num(features["cust_prior_failure_rate"].to_numpy(float)),
        "availability": np.nan_to_num(features["n_missing_features"].to_numpy(float)),
    }
    matrix = np.column_stack([
        (value - np.median(value)) / (np.percentile(value, 75) - np.percentile(value, 25) or 1.0)
        for value in values.values()
    ])
    return np.asarray(REASONS)[np.argmax(matrix, axis=1)]


def simulate(xi: float, *, seed: int = SEED, tier: str = "confirm") -> SimulationWorld:
    features, y_obs = _real_test_window()
    score = held_out_index(features)
    probability = _assignment_probability(score, xi)
    rng = np.random.default_rng(seed + int(xi * 1000))
    treatment = rng.binomial(1, probability).astype(bool)
    reason = _reason(features)
    tau = np.asarray([EFFECTIVENESS[item][tier] for item in reason], dtype=float)
    y0 = y_obs.copy()
    y1 = y0 * rng.binomial(1, 1.0 - tau)
    observed = np.where(treatment, y1, y0)
    return SimulationWorld(
        xi, features, y_obs, score, probability, treatment, y0, y1,
        observed, reason, tier, tau,
    )


def generate(xis: tuple[float, ...] = DEFAULT_XI) -> tuple[SimulationWorld, ...]:
    return tuple(simulate(xi) for xi in xis)


def main() -> int:
    # Tier-2 output is intentionally not written under eval/; callers must quarantine it.
    generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
