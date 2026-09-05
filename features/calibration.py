"""
Calibration window candidates.

ARCHITECTURE.md 6 fits a 2-parameter Platt map on the temporal validation window and
selects the window length from {30, 60, 90} days by Brier score.  This module reports
whether each candidate can support that fit at all, before any fitting happens.

Admissibility is events-per-parameter.  Peduzzi's floor for logistic regression is 10
events per parameter; Platt has two, so 20 positives.  A candidate below that is
**dropped with a stated reason** rather than fitted badly and reported as a number.

Sample size is not the only failure mode, and on this panel it is not the binding one.
The candidates differ in prevalence, and a calibration map fit at a base rate far from
the one it will be applied at is miscalibrated by construction - so the prevalence gap
against the test window is reported alongside the counts and is the figure that actually
discriminates between the candidates here.

No model is fit in this module.  It reports what a fit would have to work with.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.loader import PRIMARY_LABEL, OlistLoader

__all__ = [
    "CANDIDATE_DAYS",
    "MIN_EVENTS_PER_PARAM",
    "PLATT_N_PARAMS",
    "CalibrationCandidate",
    "calibration_window_candidates",
]

CANDIDATE_DAYS = (30, 60, 90)

#: Platt scaling fits P = 1 / (1 + exp(A*f + B)) - a slope and an intercept.
PLATT_N_PARAMS = 2

#: Peduzzi et al.'s events-per-variable floor for a stable logistic fit.
MIN_EVENTS_PER_PARAM = 10

TS = "order_purchase_timestamp"


@dataclass(frozen=True)
class CalibrationCandidate:
    days: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_orders: int
    n_positives: int
    prevalence: float
    test_prevalence: float
    admissible: bool
    exclusion_reason: str | None

    @property
    def events_per_param(self) -> float:
        return self.n_positives / PLATT_N_PARAMS

    @property
    def prevalence_ratio(self) -> float:
        """Candidate prevalence relative to the window the map will be applied to."""
        return self.prevalence / self.test_prevalence


def calibration_window_candidates(
    loader: OlistLoader | None = None,
    label: str = PRIMARY_LABEL,
) -> list[CalibrationCandidate]:
    """
    Evaluate each candidate window against the primary target's population.

    Windows are trailing: ``[boundary - N days, boundary)``.  They are nested, and all
    of them sit inside validation by construction, because VALIDATION_DAYS is set to the
    longest candidate.
    """
    loader = loader or OlistLoader()
    risk = loader.risk_set()
    boundary = loader.split_boundary

    test = risk[risk["is_test"]]
    test_prevalence = float(test[label].mean())

    out: list[CalibrationCandidate] = []
    floor = MIN_EVENTS_PER_PARAM * PLATT_N_PARAMS

    for days in CANDIDATE_DAYS:
        start = boundary - pd.Timedelta(days=days)
        window = risk[(risk[TS] >= start) & (risk[TS] < boundary)]
        n_orders = len(window)
        n_pos = int(window[label].sum())
        admissible = n_pos >= floor
        reason = (
            None
            if admissible
            else (
                f"{n_pos} positives is below the {floor}-positive floor "
                f"({MIN_EVENTS_PER_PARAM} events per parameter x {PLATT_N_PARAMS} Platt "
                "parameters); the fit would not be stable, so the candidate is dropped "
                "rather than fitted badly"
            )
        )
        out.append(
            CalibrationCandidate(
                days=days,
                start=start,
                end=boundary,
                n_orders=n_orders,
                n_positives=n_pos,
                prevalence=float(window[label].mean()) if n_orders else 0.0,
                test_prevalence=test_prevalence,
                admissible=admissible,
                exclusion_reason=reason,
            )
        )
    return out
