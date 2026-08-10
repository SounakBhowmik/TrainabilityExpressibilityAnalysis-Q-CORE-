"""Plots the results DataFrame produced by results.py.

plot_tradeoff/plot_metric_vs_depth (the generic, reusable plotting
functions) now live in pqc_diagnostics.visualize.plots; this module keeps
only the poster figures specific to this paper, plus this paper's exact
visual encoding (registered by importing qcore_expr_train.styles below).

- color        = ansatz_family - the thing held fixed while the feature map
                 varies, so a single color's marker spread shows the feature
                 map's effect on that ansatz directly.
- marker shape = feature_map_family.
- linestyle    = noise_condition or topology, where relevant.
- topology is faceted where the plot's story is about topology, and fixed to
  "linear" (documented in the title) everywhere else.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pqc_diagnostics.visualize.plots import _style_axes, plot_metric_vs_depth, plot_tradeoff  # noqa: F401

# Import side effect: registers this paper's exact colors/markers/linestyles
# into pqc_diagnostics's global StyleRegistry before any plotting happens.
from qcore_expr_train.styles import STYLES


def plot_all(summary: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_tradeoff(summary, out_dir / "tradeoff.png")
    plot_metric_vs_depth(summary, "expressibility_kl", out_dir / "expressibility_vs_depth.png")
    plot_metric_vs_depth(summary, "gradient_variance", out_dir / "trainability_vs_depth.png")


# --- Poster plots: the 3 figures used for the HPC-scale poster presentation ---
# (kernel usefulness is deprioritized for now - the focus is the
# expressibility/trainability tradeoff, not the kernel-discriminability claim)
#
# All three read a `summary` DataFrame that includes a "noise_condition"
# column ("noiseless"/"noisy"), unlike the pilot's noiseless-only summaries
# used by plot_all above.


def poster_tradeoff(summary: pd.DataFrame, out_path: str | Path, topology: str = "linear") -> None:
    """Plot 1 (headline): trainability vs. expressibility, noiseless only,
    one marker per (feature map, ansatz, depth), faceted by qubit count.
    Color = ansatz (held fixed), marker = feature map (varied) - the isolated
    effect of feature-map expressibility on a given ansatz's trainability."""
    noiseless = summary[summary["noise_condition"] == "noiseless"]
    plot_tradeoff(noiseless, out_path, topology=topology)


def poster_saturation(summary: pd.DataFrame, out_path: str | Path, qubits: int | None = None) -> None:
    """Plot 2: expressibility (left) and trainability (right) vs. depth,
    noiseless only, topology fixed to "linear" and qubit count fixed to the
    largest available (or `qubits`). Expressibility only depends on the
    feature map, so the left panel has one line per feature map; trainability
    depends on both, so the right panel repeats those same feature-map colors
    but splits each into 2 linestyles, one per ansatz - so a reader can see,
    for the same feature map, whether the ansatz choice changes where the
    saturation point sits."""
    data = summary[(summary["noise_condition"] == "noiseless") & (summary["topology"] == "linear")]
    qubits = qubits or data["num_qubits"].max()
    data = data[data["num_qubits"] == qubits]
    feature_maps = sorted(data["feature_map_family"].unique())
    ansatze = sorted(data["ansatz_family"].unique())

    fig, (ax_expr, ax_train) = plt.subplots(1, 2, figsize=(10, 4.5))
    for feature_map in feature_maps:
        feature_map_color = STYLES.get("feature_map_family", feature_map).color
        expr_line = data[data["feature_map_family"] == feature_map].drop_duplicates("depth").sort_values("depth")
        ax_expr.errorbar(
            expr_line["depth"],
            expr_line["expressibility_kl_mean"],
            yerr=expr_line["expressibility_kl_std"],
            color=feature_map_color,
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=3,
            label=feature_map,
        )
        for ansatz in ansatze:
            train_line = data[
                (data["feature_map_family"] == feature_map) & (data["ansatz_family"] == ansatz)
            ].sort_values("depth")
            if train_line.empty:
                continue
            ax_train.errorbar(
                train_line["depth"],
                train_line["gradient_variance_mean"],
                yerr=train_line["gradient_variance_std"],
                color=feature_map_color,
                linestyle=STYLES.get("ansatz_family", ansatz).linestyle,
                marker="o",
                markersize=7,
                linewidth=2,
                capsize=3,
            )

    ax_expr.set_title("Expressibility vs. depth")
    ax_expr.set_ylabel("Expressibility (KL to Haar)")
    ax_expr.legend(frameon=False, fontsize=8, loc="best")
    ax_train.set_title("Trainability vs. depth")
    ax_train.set_ylabel("Gradient variance")
    train_handles = [
        plt.Line2D([], [], color=STYLES.get("feature_map_family", f).color, marker="o", markersize=8, label=f)
        for f in feature_maps
    ]
    train_handles += [
        plt.Line2D(
            [], [], color="#52514e", linestyle=STYLES.get("ansatz_family", a).linestyle, linewidth=2, label=a
        )
        for a in ansatze
    ]
    ax_train.legend(handles=train_handles, frameon=False, fontsize=7, loc="best")
    for ax in (ax_expr, ax_train):
        ax.set_xlabel("Depth (reps)")
        _style_axes(ax)

    fig.suptitle(f"Where expressibility saturates ({qubits} qubits, linear topology, noiseless)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def poster_noise_topology(summary: pd.DataFrame, out_path: str | Path, qubits: int | None = None) -> None:
    """Plot 3: retained trainability fraction (noisy / noiseless gradient
    variance) vs. depth, faceted by ansatz - one line per (feature map,
    topology) pair on the SAME axes, so all 3 topologies are directly
    comparable in one panel instead of eyeballing 3 side-by-side facets.
    A ratio near 1 means noise barely touched trainability; near 0 means
    it's been destroyed.

    This replaced an earlier version plotting raw noisy gradient variance on
    a log scale. Once a circuit's noisy density matrix fully decoheres
    (verified directly on this grid: full-topology circuits at depth 8 need
    ~2600 two-qubit gates after routing onto the fixed noise patch, and their
    density matrix comes out with purity ~0.000977 - == 1/2**10, maximally
    mixed, to 4 significant figures), every expectation value collapses to
    ~0 regardless of parameters, so the raw "gradient variance" measured
    there is pure floating-point noise around zero, not a real quantity.
    Plotting that raw on a log scale turned it into a meaningless multi-
    decade zigzag. The ratio form doesn't have that problem: a fully
    decohered cell just reads as ratio -> 0 on an ordinary linear scale, no
    floor/clipping hack required.
    """
    qubits = qubits or summary["num_qubits"].max()
    data = summary[summary["num_qubits"] == qubits]
    index_cols = ["feature_map_family", "ansatz_family", "depth", "topology"]
    noiseless = data[data["noise_condition"] == "noiseless"][index_cols + ["gradient_variance_mean"]]
    noisy = data[data["noise_condition"] == "noisy"][index_cols + ["gradient_variance_mean"]]
    merged = noiseless.merge(noisy, on=index_cols, suffixes=("_noiseless", "_noisy"))
    # Guard a near-zero denominator rather than plot a spurious ratio spike -
    # not observed in this grid (noiseless values stay well clear of zero),
    # but cheap insurance against a divide-by-near-zero artifact.
    merged = merged[merged["gradient_variance_mean_noiseless"] > 1e-9].copy()
    merged["retained_fraction"] = merged["gradient_variance_mean_noisy"] / merged["gradient_variance_mean_noiseless"]

    ansatze = sorted(data["ansatz_family"].unique())
    feature_maps = sorted(data["feature_map_family"].unique())
    topologies = [t for t in ["linear", "circular", "full"] if t in data["topology"].unique()]

    fig, axes = plt.subplots(1, len(ansatze), figsize=(6 * len(ansatze), 4.5), sharey=True)
    axes = [axes] if len(ansatze) == 1 else list(axes)

    for ansatz, ax in zip(ansatze, axes):
        cell = merged[merged["ansatz_family"] == ansatz]
        for feature_map in feature_maps:
            feature_map_color = STYLES.get("feature_map_family", feature_map).color
            feature_map_marker = STYLES.get("feature_map_family", feature_map).marker
            for topology in topologies:
                line = cell[
                    (cell["feature_map_family"] == feature_map) & (cell["topology"] == topology)
                ].sort_values("depth")
                if line.empty:
                    continue
                ax.plot(
                    line["depth"],
                    line["retained_fraction"],
                    color=feature_map_color,
                    linestyle=STYLES.get("topology", topology).linestyle,
                    marker=feature_map_marker,
                    markersize=7,
                    linewidth=2,
                )
        ax.axhline(1.0, color="#999999", linewidth=1, linestyle=":", zorder=0)
        ax.set_ylim(-0.05, 1.3)
        ax.set_title(ansatz)
        ax.set_xlabel("Depth (reps)")
        _style_axes(ax)

    axes[0].set_ylabel("Retained trainability (noisy / noiseless gradient variance) →")
    handles = [
        plt.Line2D(
            [], [], color=STYLES.get("feature_map_family", f).color,
            marker=STYLES.get("feature_map_family", f).marker, markersize=8, label=f,
        )
        for f in feature_maps
    ] + [
        plt.Line2D([], [], color="#52514e", linestyle=STYLES.get("topology", t).linestyle, linewidth=2, label=t)
        for t in topologies
    ]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Noise pulls the trainability ceiling in, by topology ({qubits} qubits)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_poster_set(summary: pd.DataFrame, out_dir: str | Path) -> None:
    """The 3 plots used for the HPC-scale poster - see each function's
    docstring for what claim it carries."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poster_tradeoff(summary, out_dir / "poster_tradeoff.png")
    poster_saturation(summary, out_dir / "poster_saturation.png")
    poster_noise_topology(summary, out_dir / "poster_noise_topology.png")
