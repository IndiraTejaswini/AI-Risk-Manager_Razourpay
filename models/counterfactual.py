"""Safe score counterfactuals for the shipped monotone features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.constraints import MONOTONE_CONSTRAINTS
from models.train import predict

__all__ = ["CounterfactualError", "counterfactual_score", "top_modifiable_feature"]


class CounterfactualError(ValueError):
    """Raised when a counterfactual is requested outside the shipped guarantees."""


def counterfactual_score(
    bundle,
    calibrator,
    features: pd.DataFrame,
    feature: str,
    improved_value: float,
) -> float:
    """Score one modified row; only shipped monotone features are permitted."""
    if feature not in MONOTONE_CONSTRAINTS:
        raise CounterfactualError(
            f"{feature!r} is not a shipped monotone feature; refusing counterfactual"
        )
    if feature not in features.columns:
        raise CounterfactualError(f"{feature!r} is absent from the scored feature frame")
    modified = features.copy()
    modified.loc[modified.index[0], feature] = improved_value
    margin = predict(bundle, modified, raw_score=True)
    return float(calibrator.predict(margin)[0])


def top_modifiable_feature(explanation, current: pd.DataFrame) -> tuple[str, float]:
    """Choose the strongest positive constrained attribution and a plausible improvement."""
    candidates = [
        (name, float(explanation.values[0, i]))
        for i, name in enumerate(explanation.feature_names)
        if name in MONOTONE_CONSTRAINTS
    ]
    if not candidates:
        raise CounterfactualError("no constrained feature contributes positive risk")
    feature, _ = max(candidates, key=lambda item: abs(item[1]))
    value = float(current.iloc[0][feature])
    if not np.isfinite(value):
        value = 0.0
    direction = MONOTONE_CONSTRAINTS[feature]
    improved = value - max(abs(value) * 0.25, 1e-6) if direction > 0 else value + max(abs(value) * 0.25, 1e-6)
    return feature, improved
