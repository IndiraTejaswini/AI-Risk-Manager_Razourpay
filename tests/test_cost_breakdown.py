"""The response cost explanation must reconcile with the policy threshold."""

from __future__ import annotations

import pytest

from api.contract import ScoreRequest
from api.service import ScoringService
from models.train import predict, prepare_matrix
from tests.test_api import _payload_for


@pytest.fixture(scope="module")
def svc():
    return ScoringService()


@pytest.fixture(scope="module")
def batch(svc):
    risk = svc.loader.risk_set().join(svc.loader.split_labelled()["split"], how="left")
    matrix = svc.builder.build(risk)
    X, _ = prepare_matrix(matrix, category_levels=svc.category_levels)
    test_rows = risk["split"] == "test"
    margin = predict(svc.bundle, X.loc[test_rows], raw_score=True)
    return risk.loc[test_rows].reset_index(drop=True), svc.calibrator.predict(margin)


def test_cost_breakdown_matches_policy(svc, batch):
    rows, _ = batch
    for i in range(3):
        payload = _payload_for(svc, rows.iloc[i])
        response, _ = svc.score(ScoreRequest.model_validate(payload), payload)
        costs = response.cost_breakdown

        expected_threshold = (
            costs.impression_cost + costs.expected_triggered_cost
        ) / (costs.expected_rto_loss + costs.expected_triggered_cost)

        assert costs.currency == "BRL"
        assert costs.basis == "assumed_cost_constants"
        assert response.threshold_used == pytest.approx(
            expected_threshold, abs=1e-12
        )
