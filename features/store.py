"""
Point-in-time history store.

--------------------------------------------------------------------------------------
ONE DEFINITION, TWO EVALUATION STRATEGIES
--------------------------------------------------------------------------------------

The serving path was re-deriving every point-in-time aggregate from ~97k historical rows
on each request, which cost 3.2s against a 200ms budget (``eval/latency.md``, step 7).

This is the index that removes it.  It is deliberately **not** a second definition of any
feature.  The formulas - the smoothing, the ratios, the fallbacks - stay exactly where
they were, in ``features/builder.py``.  What changes is only *how the prior window is
evaluated*:

    row-scan   sort the population, walk it, take the value at the start of each
               tie block                                   (features/pit.py, batch)

    index      binary-search a per-group sorted timestamp array and read a cumulative
               sum precomputed by the same pandas kernel the scan uses, so the float
               accumulation is identical and not merely close   (this module, serving)

Both answer the identical question - "over rows in this group with ``ts`` strictly less
than *t*, what is the count / sum / last timestamp" - and ``searchsorted(..., "left")``
is exactly the strict inequality, so ties are excluded on both paths for the same reason.

That they agree is **asserted, not assumed**: ``tests/test_store.py`` checks the store's
answers against the row-scan for every group and every feature, and the end-to-end API
parity test still requires 1e-12 agreement with the batch matrix.

--------------------------------------------------------------------------------------
STRICTLY POINT-IN-TIME
--------------------------------------------------------------------------------------

The store holds the *whole* history and is queried **by timestamp**, so it cannot leak
the future: a query at time *t* reads only entries before *t*, whatever else the store
contains.  Storing a single "state as of the end of history" would have been cheaper and
would have broken point-in-time correctness the moment an order was scored whose
timestamp sits inside the history rather than after it - which is precisely what the
parity test does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["HistoryStore"]


class _GroupIndex:
    """Sorted timestamps plus cumulative sums for one grouping column."""

    __slots__ = ("offsets", "keys", "ts", "vals", "cums", "key_to_slot", "prefix")

    def __init__(
        self,
        group: pd.Series,
        ts: pd.Series,
        values: dict[str, np.ndarray],
        prefix: bool = False,
    ) -> None:
        order = np.lexsort((ts.to_numpy(), group.to_numpy().astype(str)))
        g = group.to_numpy().astype(str)[order]
        t = ts.to_numpy()[order]

        boundaries = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1])))
        self.keys = g[boundaries]
        self.offsets = np.append(boundaries, len(g))
        self.key_to_slot = {k: i for i, k in enumerate(self.keys)}
        self.ts = t
        self.prefix = prefix
        ordered = {n: np.nan_to_num(v[order], nan=0.0) for n, v in values.items()}

        # Two strategies, chosen for arithmetic exactness rather than convenience.
        #
        # A single prefix-sum across the whole index would accumulate across group
        # boundaries, so a small group's sum becomes the difference of two large numbers
        # and disagrees with the row-scan in the last bits - measured at 3.4e-10.
        #
        # For grouped indexes the answer is a per-group INCLUSIVE cumsum, computed here
        # with the same pandas kernel the row-scan uses (features/pit.py:
        # `vals.groupby(...).cumsum()`) over the same per-group element order.  Both
        # sorts put a group's rows in ascending timestamp with ties in original frame
        # order, so the two accumulate identical elements in an identical sequence and
        # the result is bit-identical by construction rather than by luck.
        #
        # An earlier version summed the slice at query time with `.sum()`.  That is
        # numpy's PAIRWISE summation against pandas' sequential accumulation, and the
        # two differ in the last bits once a group is large: it agreed on customers
        # (median one prior order) and disagreed by 1.5e-11 on sellers, which have
        # thousands.  api/service.py's startup self-check is what caught it.
        #
        # The global index is a single group, so its prefix IS the scan's cumsum order
        # and can be precomputed the same way - which matters, because that slice can be
        # 97k long.
        if prefix:
            self.cums = {
                n: np.concatenate(([0.0], np.cumsum(v))) for n, v in ordered.items()
            }
            self.vals = {}
        else:
            codes = np.repeat(np.arange(len(self.keys)), np.diff(self.offsets))
            self.cums = {}
            self.vals = {
                n: pd.Series(v).groupby(codes, sort=False).cumsum().to_numpy()
                for n, v in ordered.items()
            }

    def query_first(self, keys: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Earliest prior timestamp per query row; NaT where the group has no prior."""
        out = np.full(len(keys), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        for i, (k, t) in enumerate(zip(keys, ts)):
            slot = self.key_to_slot.get(str(k))
            if slot is None:
                continue
            lo, hi = self.offsets[slot], self.offsets[slot + 1]
            if int(np.searchsorted(self.ts[lo:hi], t, side="left")):
                out[i] = self.ts[lo]
        return out

    def query(self, keys: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
        """Prior count, prior sums, and the most recent prior timestamp per query row."""
        n = len(keys)
        counts = np.zeros(n, dtype=np.int64)
        names = self.cums if self.prefix else self.vals
        sums = {name: np.zeros(n, dtype=np.float64) for name in names}
        last = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

        for i, (k, t) in enumerate(zip(keys, ts)):
            slot = self.key_to_slot.get(str(k))
            if slot is None:
                continue
            lo, hi = self.offsets[slot], self.offsets[slot + 1]
            # "left" is the strict inequality: entries equal to t are excluded, which is
            # the same tie rule the row-scan applies via its tie-block collapse.
            j = int(np.searchsorted(self.ts[lo:hi], t, side="left"))
            counts[i] = j
            if j:
                last[i] = self.ts[lo + j - 1]
            if self.prefix:
                for name, c in self.cums.items():
                    sums[name][i] = c[lo + j] - c[lo]
            elif j:
                # Inclusive per-group cumsum: the value at the j-th prior row IS the sum
                # over the first j rows of the group.  O(1), and the identical float
                # accumulation the row-scan performs.
                for name, v in self.vals.items():
                    sums[name][i] = v[lo + j - 1]
        return counts, sums, last


class HistoryStore:
    """
    Prior-window index over a historical population.

    Parameters
    ----------
    history:
        Rows the aggregates are drawn from.  Must carry the grouping columns, the
        timestamp, and every value column that will be summed.
    group_cols:
        Grouping columns to index, e.g. ``customer_unique_id`` and
        ``customer_zip_code_prefix``.
    value_cols:
        Columns to maintain cumulative sums for.
    ts_col:
        Timestamp column defining the ordering.
    edges:
        Optional second history frame at a **different cardinality** - the order-seller
        edge frame, one row per (order, seller).  Seller history is aggregated with max
        across an order's sellers (features/builder.py, MULTI_SELLER_RULE), so its prior
        window is over edges and not over orders; indexing it from ``history`` would
        count a two-seller order once.  The indexes it produces are queried through the
        same :meth:`prior`, with the caller supplying an edge-shaped frame.
    edge_group_cols, edge_value_cols:
        Grouping and value columns to index from ``edges``.

    The global (ungrouped) index is always built from ``history``, never from ``edges``:
    the prior it serves is the global **order** failure rate, which is what the batch
    path smooths toward.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        group_cols: tuple[str, ...],
        value_cols: tuple[str, ...],
        ts_col: str = "order_purchase_timestamp",
        edges: pd.DataFrame | None = None,
        edge_group_cols: tuple[str, ...] = (),
        edge_value_cols: tuple[str, ...] = (),
    ) -> None:
        self.ts_col = ts_col
        self.group_cols = tuple(group_cols)
        self.value_cols = tuple(value_cols)
        self.edge_group_cols = tuple(edge_group_cols)

        values = {
            c: history[c].astype("float64").to_numpy() for c in value_cols
        }
        self._index = {
            g: _GroupIndex(history[g], history[ts_col], values) for g in group_cols
        }

        if edge_group_cols:
            if edges is None:
                raise ValueError("edge_group_cols given without an edges frame")
            ev = {c: edges[c].astype("float64").to_numpy() for c in edge_value_cols}
            overlap = set(edge_group_cols) & set(group_cols)
            if overlap:
                raise ValueError(
                    "group column(s) indexed from both frames: "
                    + ", ".join(sorted(overlap))
                    + ". A query would silently get whichever won."
                )
            for g in edge_group_cols:
                self._index[g] = _GroupIndex(edges[g], edges[ts_col], ev)

        # Ungrouped index for the global expanding mean, held as one synthetic group so
        # the same query path serves it.
        self._global = _GroupIndex(
            pd.Series(["__all__"] * len(history), index=history.index),
            history[ts_col],
            values,
            prefix=True,
        )

    # -- queries -----------------------------------------------------------------------

    def prior(
        self, frame: pd.DataFrame, group_col: str
    ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        idx = self._index[group_col]
        return idx.query(
            frame[group_col].to_numpy(), frame[self.ts_col].to_numpy()
        )

    def prior_first(self, frame: pd.DataFrame, group_col: str) -> np.ndarray:
        idx = self._index[group_col]
        return idx.query_first(
            frame[group_col].to_numpy(), frame[self.ts_col].to_numpy()
        )

    def prior_global(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        counts, sums, _ = self._global.query(
            np.array(["__all__"] * len(frame)), frame[self.ts_col].to_numpy()
        )
        return counts, sums
