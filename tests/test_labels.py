"""
Label construction, and the primary/secondary designation.

label_b (PRIMARY)   entered fulfillment and never delivered.  On the risk set this is
                    P(fails | ships).
label_a (SECONDARY) any non-delivered order.  Unconditional P(fails).

The counts asserted here are the ones committed in eval/label_composition.md.  They are
characterisation tests: if a change moves them, that is either a bug or a decision that
has to be reflected in the artifacts, and either way it should not pass silently.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from data.loader import PRIMARY_LABEL, SECONDARY_LABEL, OlistLoader

# Committed in eval/label_composition.md and eval/positive_counts.md.
MATURE_ORDERS = 99_433
RISK_SET_ORDERS = 97_658
LABEL_A_TOTAL, LABEL_A_TRAIN, LABEL_A_TEST = 2_955, 2_576, 379
LABEL_B_TOTAL, LABEL_B_TRAIN, LABEL_B_TEST = 1_182, 1_028, 154
CANCELLED_IN_TRANSIT = 75


def test_primary_is_label_b():
    assert PRIMARY_LABEL == "label_b"
    assert SECONDARY_LABEL == "label_a"


def test_population_sizes(labelled, risk_set):
    assert len(labelled) == MATURE_ORDERS
    assert len(risk_set) == RISK_SET_ORDERS


def test_label_a_counts(labelled):
    assert int(labelled["label_a"].sum()) == LABEL_A_TOTAL
    assert int((labelled["label_a"] & ~labelled["is_test"]).sum()) == LABEL_A_TRAIN
    assert int((labelled["label_a"] & labelled["is_test"]).sum()) == LABEL_A_TEST


def test_label_b_counts(labelled):
    assert int(labelled["label_b"].sum()) == LABEL_B_TOTAL
    assert int((labelled["label_b"] & ~labelled["is_test"]).sum()) == LABEL_B_TRAIN
    assert int((labelled["label_b"] & labelled["is_test"]).sum()) == LABEL_B_TEST


def test_secondary_buys_positives_at_the_cost_of_specificity(labelled):
    """The trade-off the two labels exist to document, asserted as a direction."""
    a_test = int((labelled["label_a"] & labelled["is_test"]).sum())
    b_test = int((labelled["label_b"] & labelled["is_test"]).sum())
    assert a_test > b_test
    # Every label_b positive is a label_a positive; the reverse does not hold.
    assert bool((labelled["label_b"] & ~labelled["label_a"]).sum() == 0)
    assert int((labelled["label_a"] & ~labelled["label_b"]).sum()) == (
        LABEL_A_TOTAL - LABEL_B_TOTAL
    )


def test_label_b_is_fully_contained_in_the_risk_set(labelled, risk_set):
    """No label_b positive may sit outside the population it is defined on."""
    assert int(labelled.loc[~labelled["entered_fulfillment"], "label_b"].sum()) == 0
    assert int(risk_set["label_b"].sum()) == LABEL_B_TOTAL


def test_risk_set_carries_no_delivered_only_negatives_that_never_shipped(risk_set):
    assert bool(risk_set["entered_fulfillment"].all())


def test_conditional_prevalence_differs_from_unconditional(labelled, risk_set):
    """
    P(fails | ships) is computed on a smaller denominator than P(fails would be).
    The gap is small on this panel but must not be zero, or the conditioning is inert.
    """
    conditional = risk_set["label_b"].mean()
    unconditional = labelled["label_b"].mean()
    assert conditional > unconditional


def test_fulfillment_flag_beats_status_alone(labelled):
    """
    75 `canceled` orders carry a carrier date - they shipped and were cancelled in
    transit.  A status-only flag would drop them from the risk set.
    """
    cancelled = labelled["order_status"] == "canceled"
    assert int((cancelled & labelled["entered_fulfillment"]).sum()) == CANCELLED_IN_TRANSIT
    shipped_status_only = int((labelled["order_status"] == "shipped").sum())
    assert int(labelled["label_b"].sum()) == shipped_status_only + CANCELLED_IN_TRANSIT


def test_statuses_that_cannot_ship_carry_no_carrier_date(labelled):
    never = {"unavailable", "invoiced", "processing", "created", "approved"}
    subset = labelled[labelled["order_status"].isin(never)]
    assert int(subset["entered_fulfillment"].sum()) == 0


def test_labels_are_boolean(labelled):
    for col in ("label_a", "label_b", "entered_fulfillment", "is_test"):
        assert labelled[col].dtype == bool, col


def test_load_is_deterministic(loader: OlistLoader):
    pd.testing.assert_frame_equal(loader.labelled(), loader.labelled())


# ---------------------------------------------------------------------------------
# Where the data comes from
# ---------------------------------------------------------------------------------

def test_dataset_path_is_resolved_once_per_process(loader: OlistLoader):
    """
    `kagglehub.dataset_download` reaches the network to check the dataset version even
    when every byte is cached, and the suite constructs dozens of loaders.  Resolving
    once per process is what stops a transient Kaggle outage failing a test that needs
    no network - which is exactly how it failed before the memo existed.
    """
    import data.loader as loader_module

    assert loader_module._DATASET_DIR is not None or "OLIST_DATA_DIR" in os.environ
    first = OlistLoader().data_dir
    assert all(OlistLoader().data_dir == first for _ in range(5))


def _offline(monkeypatch):
    """Stand in for kagglehub with a module whose download always fails."""
    import sys
    import types

    fake = types.ModuleType("kagglehub")

    def refuse(*args, **kwargs):
        raise ConnectionError("simulated RemoteDisconnected")

    fake.dataset_download = refuse
    monkeypatch.setitem(sys.modules, "kagglehub", fake)


def test_a_network_failure_falls_back_to_the_cache(monkeypatch, loader: OlistLoader):
    """A blip must not end a run whose data is already on disk."""
    import data.loader as loader_module

    if loader_module._cached_dataset_dir() is None:
        pytest.skip("no kagglehub cache on this machine; OLIST_DATA_DIR is in use")

    _offline(monkeypatch)
    monkeypatch.setattr(loader_module, "_DATASET_DIR", None)
    resolved = loader_module._resolve_dataset_dir()
    assert (resolved / "olist_orders_dataset.csv").is_file()


def test_no_cache_and_no_network_names_the_env_var(monkeypatch):
    """
    The one case where the run genuinely cannot proceed.  It has to say what to set -
    the same requirement the container image is held to.
    """
    import data.loader as loader_module

    _offline(monkeypatch)
    monkeypatch.setattr(loader_module, "_DATASET_DIR", None)
    monkeypatch.setattr(loader_module, "_cached_dataset_dir", lambda: None)

    with pytest.raises(RuntimeError, match="OLIST_DATA_DIR"):
        loader_module._resolve_dataset_dir()
