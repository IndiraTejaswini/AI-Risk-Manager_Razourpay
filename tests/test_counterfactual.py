"""Counterfactuals may only use guarantees carried by the shipped model."""

from __future__ import annotations

import pytest

from api.service import ScoringService
from models.constraints import MONOTONE_CONSTRAINTS, UNSHIPPED_CONSTRAINTS
from models.counterfactual import CounterfactualError, counterfactual_score
from models.train import prepare_matrix, predict


@pytest.fixture(scope="module")
def svc():
    return ScoringService()


def _features(svc):
    row = svc.history.iloc[[0]]
    matrix = svc.builder.build(row, store=svc.store)
    X, _ = prepare_matrix(matrix, category_levels=svc.category_levels)
    return X


def test_counterfactual_only_on_constrained_features(svc):
    features = _features(svc)
    for feature in ("order_value", next(iter(UNSHIPPED_CONSTRAINTS))):
        with pytest.raises(CounterfactualError):
            counterfactual_score(
                svc.bundle, svc.calibrator, features, feature, 0.0
            )


@pytest.mark.parametrize("feature", sorted(MONOTONE_CONSTRAINTS))
def test_counterfactual_direction_matches_constraint(svc, feature):
    features = _features(svc)
    current = float(features.iloc[0][feature])
    step = max(abs(current) * 0.25, 1e-6)
    improved = current - step if MONOTONE_CONSTRAINTS[feature] > 0 else current + step
    original = float(svc.calibrator.predict(predict(svc.bundle, features, raw_score=True))[0])
    changed = counterfactual_score(
        svc.bundle, svc.calibrator, features, feature, improved
    )
    assert changed <= original + 1e-12


def test_explain_endpoint_not_in_score_path():
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as test_client:
        response = test_client.get("/openapi.json")
        assert "/score" in response.json()["paths"]
        assert "/explain/{order_id}" in response.json()["paths"]
