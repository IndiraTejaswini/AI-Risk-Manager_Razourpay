"""
Three-way chronological split: train / temporal validation / test.

ARCHITECTURE.md 9 requires train, a temporal validation window used for early stopping
(5.2) and the calibration fit (6), and test.  The assertions that matter are that the
three are exhaustive, disjoint, and strictly ordered in time - and specifically that the
validation window never intersects test, because validation is what selects the
calibration length and the early-stopping round.
"""

from __future__ import annotations

import pandas as pd

from data.loader import VALIDATION_DAYS, OlistLoader

TS = "order_purchase_timestamp"
SPLITS = ("train", "validation", "test")


def test_split_labels_cover_every_row(loader: OlistLoader):
    df = loader.split_labelled()
    assert set(df["split"].unique()) == set(SPLITS)
    assert int(df["split"].isna().sum()) == 0


def test_splits_are_disjoint(loader: OlistLoader):
    df = loader.split_labelled()
    counts = df["split"].value_counts()
    assert counts.sum() == len(df)


def test_splits_are_chronologically_ordered(loader: OlistLoader):
    df = loader.split_labelled()
    bounds = df.groupby("split")[TS].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["validation", "min"]
    assert bounds.loc["validation", "max"] < bounds.loc["test", "min"]


def test_validation_never_intersects_test(loader: OlistLoader):
    """The assertion this file exists for."""
    df = loader.split_labelled()
    val = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    assert not set(val["order_id"]) & set(test["order_id"])
    assert val[TS].max() < test[TS].min()
    assert bool((val[TS] < loader.split_boundary).all())
    assert bool((test[TS] >= loader.split_boundary).all())


def test_validation_window_is_the_declared_length(loader: OlistLoader):
    df = loader.split_labelled()
    val = df[df["split"] == "validation"]
    expected_start = loader.split_boundary - pd.Timedelta(days=VALIDATION_DAYS)
    assert bool((val[TS] >= expected_start).all())
    assert bool((df.loc[df["split"] == "train", TS] < expected_start).all())


def test_test_split_matches_the_committed_boundary(loader: OlistLoader):
    df = loader.split_labelled()
    assert int((df["split"] == "test").sum()) == int(loader.labelled()["is_test"].sum())


def test_every_split_contains_positives(loader: OlistLoader):
    """A validation window with no positives cannot select a calibration length."""
    df = loader.split_labelled()
    for split in SPLITS:
        assert int(df.loc[df["split"] == split, "label_b"].sum()) > 0, split


def test_split_labels_are_stable_across_calls(loader: OlistLoader):
    pd.testing.assert_frame_equal(loader.split_labelled(), loader.split_labelled())
