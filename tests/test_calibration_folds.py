"""
The cross-validation fold generator in ``scripts/07_calibrate.py`` must tile its index
range.

WHY THIS FILE EXISTS
--------------------------------------------------------------------------------------
It did not tile, for as long as the file existed, and nothing caught it.

``_time_folds`` read ``np.arange(n)[i::1][j * n // k:(j + 1) * n // k]`` with ``i == j``.
Re-slicing a progressively shorter array before applying fold ``j``'s slice shifted every
fold after the first by its own index and dropped the boundary element, leaving exactly
``k - 1`` indices unassigned.  ``_crossfit_brier`` allocated its output with ``np.empty``
and never wrote those slots, so a model-selection metric averaged uninitialised memory.

The garbage was usually small enough to look plausible.  On one run a slot held ~1e73 and
the Brier score came back as 1e+143, which flipped the calibration-window selection from
30 days to 60.  Three consecutive runs of unchanged code produced three different
selections.

None of that was caught by a test, because no test asserted the folds tile.  Covering the
index set is not the only property that matters either - these are *time-ordered* folds,
so contiguity and ordering are asserted here rather than inferred from the function name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_calibrate_module():
    """Import ``scripts/07_calibrate.py``, whose name is not a valid identifier."""
    path = REPO_ROOT / "scripts" / "07_calibrate.py"
    spec = importlib.util.spec_from_file_location("calibrate_step", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calibrate = _load_calibrate_module()
_time_folds = calibrate._time_folds
_crossfit_brier = calibrate._crossfit_brier


# The three real window populations on this panel, plus small non-divisible sizes where
# an off-by-one in the boundary arithmetic has nowhere to hide.
SIZES = [7380, 14148, 21329, 100, 101, 103, 7]
FOLD_COUNTS = [3, 5]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("k", FOLD_COUNTS)
def test_folds_tile_the_index_range_exactly(n: int, k: int) -> None:
    """
    The concatenation of the folds is a permutation of ``range(n)``.

    This is the assertion the old implementation failed.  At n=7380, k=5 it left
    [1476, 2953, 4430, 5907] unassigned.
    """
    folds = _time_folds(n, k)
    combined = np.sort(np.concatenate(folds))
    assert np.array_equal(combined, np.arange(n)), (
        f"folds do not tile range({n}) with k={k}: "
        f"missing {np.setdiff1d(np.arange(n), combined).tolist()}, "
        f"{len(combined) - len(np.unique(combined))} duplicated"
    )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("k", FOLD_COUNTS)
def test_folds_are_disjoint(n: int, k: int) -> None:
    """No index appears in two folds - a row cannot be held out twice."""
    folds = _time_folds(n, k)
    combined = np.concatenate(folds)
    assert len(combined) == len(np.unique(combined)), (
        f"folds overlap at n={n}, k={k}: {len(combined)} entries but "
        f"{len(np.unique(combined))} distinct"
    )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("k", FOLD_COUNTS)
def test_folds_are_contiguous_and_time_ordered(n: int, k: int) -> None:
    """
    Each fold is an ascending run of consecutive indices, and the folds advance.

    The caller sorts by timestamp before indexing, so "contiguous" here means
    contiguous *in time*: a fold that covered a scattered index set would silently stop
    being a time-blocked fold while still tiling.
    """
    folds = _time_folds(n, k)
    assert len(folds) == k, f"expected {k} folds, got {len(folds)}"

    previous_end = -1
    for j, fold in enumerate(folds):
        assert fold.size > 0, f"fold {j} is empty at n={n}, k={k}"
        assert np.array_equal(fold, np.arange(fold[0], fold[-1] + 1)), (
            f"fold {j} is not a contiguous ascending run at n={n}, k={k}: {fold}"
        )
        assert int(fold[0]) == previous_end + 1, (
            f"fold {j} starts at {fold[0]}, expected {previous_end + 1} "
            f"(n={n}, k={k}) - folds must advance without gaps"
        )
        previous_end = int(fold[-1])

    assert previous_end == n - 1, (
        f"folds end at {previous_end}, expected {n - 1} (n={n}, k={k})"
    )


@pytest.mark.parametrize("k", [0, 1, -1])
def test_fewer_than_two_folds_is_rejected(k: int) -> None:
    """k < 2 is not cross-fitting; it must fail loudly rather than return one block."""
    with pytest.raises(ValueError, match="at least 2 folds"):
        _time_folds(100, k)


@pytest.mark.parametrize("n,k", [(3, 5), (4, 5), (1, 3)])
def test_more_folds_than_rows_is_rejected(n: int, k: int) -> None:
    """
    k > n cannot be split without empty folds.

    An empty fold is not harmless here: it would hand ``_crossfit_brier`` an empty
    held-out set, and the surrounding code would quietly do nothing for it.
    """
    with pytest.raises(ValueError, match="without empty folds"):
        _time_folds(n, k)


def test_crossfit_leaves_no_row_unpredicted() -> None:
    """
    ``_crossfit_brier`` returns a prediction for every validation row.

    The end-to-end property: a NaN surviving to the return means some row was never
    written, which is what put uninitialised memory into the Brier mean.  Built on
    synthetic data so it runs in milliseconds and needs no Olist download.
    """
    import pandas as pd

    rng = np.random.default_rng(20260904)
    n = 900
    boundary = pd.Timestamp("2018-01-01")

    # Rows spread either side of every candidate window, so both the fold path and the
    # out-of-window path are exercised.
    ts = pd.Series(boundary - pd.to_timedelta(rng.integers(1, 120, size=n), unit="D"))
    scores = rng.normal(size=n)
    y = (rng.random(n) < 0.05).astype(int)

    for days in (30, 60, 90):
        p, n_in, n_pos = _crossfit_brier(scores, y, ts, boundary, days)
        assert len(p) == n
        assert not np.isnan(p).any(), (
            f"{int(np.isnan(p).sum())} of {n} rows unpredicted at days={days}"
        )
        assert np.isfinite(p).all(), f"non-finite prediction at days={days}"
        assert ((p >= 0.0) & (p <= 1.0)).all(), (
            f"cross-fit produced values outside [0, 1] at days={days}: "
            f"min {p.min()}, max {p.max()} - these are probabilities and a Brier score "
            "computed from them would be meaningless"
        )
        assert 0 < n_in <= n
        assert 0 <= n_pos <= n_in


def test_brier_from_crossfit_is_a_probability_scale_number() -> None:
    """
    The regression in one number.

    A Brier score is a mean of squared errors between probabilities and 0/1 labels, so
    it lies in [0, 1] by construction.  The defect produced 1e+143.  Asserting the range
    catches any recurrence regardless of which slot the garbage lands in.
    """
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 600
    boundary = pd.Timestamp("2018-01-01")
    ts = pd.Series(boundary - pd.to_timedelta(rng.integers(1, 120, size=n), unit="D"))
    scores = rng.normal(size=n)
    y = (rng.random(n) < 0.08).astype(int)

    p, _, _ = _crossfit_brier(scores, y, ts, boundary, 60)
    brier = float(((p - y) ** 2).mean())
    assert 0.0 <= brier <= 1.0, f"Brier score {brier} is not on the probability scale"
