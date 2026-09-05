"""
Report figures: precision-recall and ROC.

Static PNGs embedded in a markdown report, so there is no hover layer - the numbers a
tooltip would carry are in the report's tables instead, which is the table-view fallback
rather than an omission.  The figures commit to a light surface, painted explicitly.

Palette: validated categorical slots 1 and 2 (blue / orange) from the reference
instance - checked with the palette validator rather than eyeballed (all six checks
pass; worst adjacent CVD ΔE 24.7, normal-vision ΔE 33.6).  Reference lines are recessive
grey and dashed: a prevalence baseline and a chance diagonal are not series and must not
compete with one.  Text wears ink tokens, never a series colour; identity comes from the
legend key beside the label.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "SERIES", "plot_pr", "plot_roc", "plot_bootstrap",
    "plot_reliability", "plot_calibrated_distribution", "plot_decision_curve",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8983"

#: Categorical slots 1 and 2.
SERIES = {
    "full": "#2a78d6",
    "baseline": "#eb6834",
}

#: PNG metadata carries a creation date by default, which would make the file differ
#: byte-for-byte between otherwise identical runs.
_NO_DATE = {"Software": None, "Date": None}


def _style(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.set_xlim(0, 1)


def plot_pr(curves: dict[str, tuple], prevalence: float, path: Path, title: str) -> None:
    """
    Precision-recall, with the prevalence baseline drawn.

    The baseline is the whole point of the plot: at 0.78% prevalence a curve that looks
    low in absolute terms may still be far above chance, and a reader cannot tell
    without the line on the chart.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160, facecolor=SURFACE)
    _style(ax, xlabel="Recall", ylabel="Precision", title=title)

    for name, (recall, precision, label) in curves.items():
        ax.plot(
            recall, precision, color=SERIES[name], linewidth=2.0, label=label, zorder=3
        )

    ax.axhline(
        prevalence,
        color=REFERENCE,
        linewidth=1.5,
        linestyle=(0, (5, 4)),
        zorder=2,
        label=f"Prevalence baseline ({prevalence * 100:.3f}%)",
    )

    # At ~0.8% prevalence a PR curve spikes to precision 1.0 at the first ranked row -
    # one true positive at the top of the list.  Scaling to that spike squashes the
    # entire informative range into the bottom 2% of the axis and hides the prevalence
    # baseline, so the y-limit is driven by the curve beyond a small recall floor and
    # the clipping is stated on the figure rather than done quietly.
    floor = 0.02
    body = [
        p[r >= floor].max()
        for _, (r, p, _) in curves.items()
        if (r >= floor).any()
    ]
    top = max(body + [prevalence * 3]) * 1.3
    ax.set_ylim(0, min(1.0, top))
    ax.text(
        0.99, 0.02,
        f"y-axis clipped; curve below recall {floor:g} reaches precision 1.0",
        transform=ax.transAxes, fontsize=7, color=REFERENCE, ha="right", va="bottom",
    )

    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for text in leg.get_texts():
        text.set_color(INK_MUTED)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, metadata=_NO_DATE)
    plt.close(fig)


def plot_roc(curves: dict[str, tuple], path: Path, title: str) -> None:
    """ROC, plotted because omitting it invites the question. Not the headline."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160, facecolor=SURFACE)
    _style(ax, xlabel="False positive rate", ylabel="True positive rate", title=title)

    for name, (fpr, tpr, label) in curves.items():
        ax.plot(fpr, tpr, color=SERIES[name], linewidth=2.0, label=label, zorder=3)

    ax.plot(
        [0, 1], [0, 1],
        color=REFERENCE, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2, label="Chance",
    )
    ax.set_ylim(0, 1)

    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for text in leg.get_texts():
        text.set_color(INK_MUTED)

    ax.text(
        0.02, 0.96,
        "Secondary. At 0.78% prevalence ROC is dominated by the negative class.",
        transform=ax.transAxes, fontsize=7.5, color=INK_MUTED, va="top",
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, metadata=_NO_DATE)
    plt.close(fig)


def plot_bootstrap(series: dict, path: Path, title: str) -> None:
    """
    Bootstrap difference distributions, with zero and each 95% CI marked.

    The reader's question is "does the mass sit clear of zero", so zero is drawn as a
    hard reference line and the CI bounds as ticks beneath each distribution. Histograms
    rather than densities: no smoothing parameter to argue about.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160, facecolor=SURFACE)
    _style(ax, xlabel="AP difference", ylabel="Bootstrap resamples", title=title)
    ax.set_xlim(auto=True)

    lo = min(float(np.percentile(d, 0.2)) for d, _, _ in series.values())
    hi = max(float(np.percentile(d, 99.8)) for d, _, _ in series.values())
    lo = min(lo, -abs(hi) * 0.15)
    bins = np.linspace(lo, hi, 70)

    for name, (diffs, label, res) in series.items():
        ax.hist(
            diffs, bins=bins, color=SERIES[name], alpha=0.55, zorder=3,
            label=f"{label}  95% CI [{res.ci_low:+.4f}, {res.ci_high:+.4f}]",
        )
        for bound in (res.ci_low, res.ci_high):
            ax.axvline(bound, color=SERIES[name], linewidth=1.2, linestyle=(0, (3, 3)),
                       zorder=4)

    ax.axvline(0.0, color=INK, linewidth=1.6, zorder=5)
    ax.text(0.0, ax.get_ylim()[1] * 0.97, "  no difference", color=INK_MUTED,
            fontsize=7.5, va="top", ha="left")
    ax.set_xlim(lo, hi)

    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_MUTED)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, metadata=_NO_DATE)
    plt.close(fig)


def plot_reliability(rep, prevalence: float, path: Path, title: str) -> None:
    """
    Reliability curve with bootstrap CIs on the observed rate.

    Perfect calibration is the diagonal, so it is drawn as a recessive reference line
    rather than a series. The axes are held to the data's own range: at this prevalence
    a 0-1 reliability plot is a dot in the corner.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=160, facecolor=SURFACE)
    _style(ax, xlabel="Mean predicted probability", ylabel="Observed rate", title=title)

    hi = max(float(rep.mean_predicted.max()), float(rep.ci_high.max())) * 1.12
    ax.plot([0, hi], [0, hi], color=REFERENCE, linewidth=1.5, linestyle=(0, (5, 4)),
            zorder=2, label="Perfect calibration")
    ax.axhline(prevalence, color=REFERENCE, linewidth=1.0, linestyle=(0, (1, 3)),
               zorder=2, label=f"Prevalence ({prevalence * 100:.3f}%)")

    ax.errorbar(
        rep.mean_predicted, rep.observed,
        yerr=[rep.observed - rep.ci_low, rep.ci_high - rep.observed],
        fmt="o", markersize=5, color=SERIES["full"], ecolor=SERIES["full"],
        elinewidth=1.4, capsize=3, zorder=4, label="Uniform-mass bins (95% CI)",
    )
    ax.plot(rep.mean_predicted, rep.observed, color=SERIES["full"], linewidth=1.6,
            alpha=0.65, zorder=3)

    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_MUTED)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, metadata=_NO_DATE)
    plt.close(fig)


def plot_calibrated_distribution(p, cuts, prevalence: float, path: Path,
                                 title: str) -> None:
    """
    Where the calibrated mass actually sits, with the provisional band cuts marked.

    Log-scaled counts: at 0.78% prevalence a linear axis shows one bar and nothing else.
    The ceiling is the point of the figure, so the maximum is annotated directly.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160, facecolor=SURFACE)
    _style(ax, xlabel="Calibrated P(fails | ships)", ylabel="Orders (log scale)",
           title=title)

    ax.hist(p, bins=60, color=SERIES["full"], alpha=0.8, zorder=3)
    ax.set_yscale("log")

    # The lowest cuts sit close together, so labels go along the bottom in a staggered
    # row rather than stacked at the top where they collide with each other and with the
    # prevalence marker.
    lo_y = ax.get_ylim()[0]
    for i, c in enumerate(cuts):
        ax.axvline(c, color=REFERENCE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
        ax.annotate(
            f"band {i + 2}",
            xy=(c, lo_y), xytext=(6, 10 + 11 * (len(cuts) - 1 - i)),
            textcoords="offset points", fontsize=7, color=INK_MUTED,
            ha="left", va="bottom",
        )

    ax.axvline(prevalence, color=INK, linewidth=1.4, zorder=5)
    ax.annotate(
        "prevalence", xy=(prevalence, ax.get_ylim()[1]), xytext=(-4, -4),
        textcoords="offset points", fontsize=7.5, color=INK_MUTED,
        ha="right", va="top",
    )

    top = float(np.max(p))
    ax.annotate(
        f"ceiling {top * 100:.2f}%",
        xy=(top, 1.5), xytext=(top * 0.82, ax.get_ylim()[1] * 0.05),
        fontsize=7.5, color=INK_MUTED, ha="right",
        arrowprops=dict(arrowstyle="->", color=REFERENCE, linewidth=1.0),
    )
    ax.set_xlim(0, top * 1.06)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, metadata=_NO_DATE)
    plt.close(fig)


def plot_decision_curve(grid, curves, band, marks, path: Path, title: str) -> None:
    """
    Realised cost against a global treat-above threshold, with the assumption band and
    the four reference policies marked.

    The band is the c_rto +/- 50% range, drawn as a fill rather than two more lines so
    it reads as uncertainty rather than as extra series. Policy markers are labelled
    directly - four points is well under the direct-label budget, and a legend would
    make the reader look twice for something the chart can just say.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160, facecolor=SURFACE)
    _style(ax, xlabel="Treat every order with p >= threshold",
           ylabel="Realised cost", title=title)

    lo, hi = band
    ax.fill_between(grid, lo, hi, color=SERIES["full"], alpha=0.13, zorder=2,
                    label="c_rto +/- 50%")

    palette = {"prepaid_only": SERIES["full"], "confirm": SERIES["baseline"]}
    for name, ys in curves.items():
        ax.plot(grid, ys, color=palette[name], linewidth=2.0, zorder=4,
                label=f"treat at {name}")

    # Markers can land on top of each other - "do nothing" and a policy that treats 26
    # of 19,662 orders sit at almost the same cost and the same threshold. Labels are
    # staggered vertically and flipped to the inside near the right edge so none is
    # clipped or overprinted.
    span = grid.max() - grid.min()
    for i, (label, x, ycost) in enumerate(marks):
        x = float(np.clip(x, grid.min(), grid.max()))
        ax.scatter([x], [ycost], s=34, color=INK, zorder=6)
        near_right = x > grid.min() + 0.75 * span
        ax.annotate(
            label,
            xy=(x, ycost),
            xytext=(-8 if near_right else 8, 9 + 12 * (i % 2)),
            textcoords="offset points", fontsize=7.5, color=INK_MUTED,
            ha="right" if near_right else "left",
        )

    ax.set_xlim(grid.min(), grid.max() * 1.02)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_MUTED)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, metadata=_NO_DATE)
    plt.close(fig)
