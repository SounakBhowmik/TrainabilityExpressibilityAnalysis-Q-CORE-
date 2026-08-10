"""Generic plotting functions over a summarize()'d results DataFrame.

Two independent categorical variables describe a grid cell - which feature
map, and which ansatz - crossed rather than paired 1:1. These functions read
color/marker/linestyle from a StyleRegistry (see styles.py) rather than a
hardcoded dict, so they work on any registered feature map/ansatz, not just a
specific study's fixed set.

Error bars come from the mean/std columns a results.summarize()-shaped
DataFrame provides.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pqc_diagnostics.visualize.styles import DEFAULT_STYLE_REGISTRY, StyleRegistry


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.5, alpha=0.3)


def _legend(
    ax: plt.Axes,
    ansatze: list[str],
    feature_maps: list[str],
    style_registry: StyleRegistry = DEFAULT_STYLE_REGISTRY,
) -> None:
    handles = [
        plt.Line2D(
            [], [], color=style_registry.get("ansatz_family", a).color,
            marker="o", linestyle="", markersize=8, label=a,
        )
        for a in ansatze
    ] + [
        plt.Line2D(
            [], [], color="#52514e", marker=style_registry.get("feature_map_family", f).marker,
            linestyle="", markersize=8, label=f,
        )
        for f in feature_maps
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")


def _facet_by_qubits(summary: pd.DataFrame) -> tuple[plt.Figure, dict[int, plt.Axes]]:
    qubit_counts = sorted(summary["num_qubits"].unique())
    fig, axes = plt.subplots(1, len(qubit_counts), figsize=(5 * len(qubit_counts), 4.5), sharey=True)
    axes = [axes] if len(qubit_counts) == 1 else list(axes)
    return fig, dict(zip(qubit_counts, axes))


def plot_tradeoff(
    summary: pd.DataFrame,
    out_path: str | Path,
    topology: str = "linear",
    style_registry: StyleRegistry = DEFAULT_STYLE_REGISTRY,
) -> None:
    """Gradient variance (trainability) vs. expressibility KL divergence -
    the core expressibility/trainability tradeoff claim, as a connected
    trajectory instead of a scattered cloud: one line per feature map,
    points ordered by depth, marker size growing with depth (the smallest
    marker on each line is the shallowest depth, the largest is the
    deepest) - so the direction circuits move through expressibility/
    trainability space as they get deeper is visible directly, without a
    separate depth legend. Both axes are log-scaled: expressibility can span
    several orders of magnitude across feature maps, and a linear x-axis
    would crush everything except an outlier's values into an unreadable
    cluster. Error bars are dropped here (see summarize()'s std columns /
    the raw CSV for seed variance) to keep the trajectory itself the only
    thing competing for attention.

    Rows = ansatz, columns = qubit count - splitting the ansätze into
    separate rows (rather than overlaying both as a color, with crossing
    lines per panel) means each panel only has to show one line per feature
    map, so trajectories never cross each other from a different ansatz.
    Reading down a column shows exactly how trainability changes as the
    feature map (and its expressibility) varies, with the ansatz held
    fixed - the isolated effect, not a confounded family-level association.
    """
    data = summary[summary["topology"] == topology]
    ansatze = sorted(data["ansatz_family"].unique())
    feature_maps = sorted(data["feature_map_family"].unique())
    qubit_counts = sorted(data["num_qubits"].unique())
    depths = sorted(data["depth"].unique())
    sizes_by_depth = {d: 30 + 25 * i for i, d in enumerate(depths)}

    fig, axes = plt.subplots(
        len(ansatze), len(qubit_counts),
        figsize=(5 * len(qubit_counts), 4 * len(ansatze)),
        sharey="row", squeeze=False,
    )

    for i, ansatz in enumerate(ansatze):
        ansatz_color = style_registry.get("ansatz_family", ansatz).color
        for j, num_qubits in enumerate(qubit_counts):
            ax = axes[i][j]
            cell = data[(data["ansatz_family"] == ansatz) & (data["num_qubits"] == num_qubits)]
            for feature_map in feature_maps:
                line = cell[cell["feature_map_family"] == feature_map].sort_values("depth")
                if line.empty:
                    continue
                ax.plot(
                    line["expressibility_kl_mean"],
                    line["gradient_variance_mean"],
                    color=ansatz_color,
                    linewidth=1.5,
                    zorder=1,
                )
                ax.scatter(
                    line["expressibility_kl_mean"],
                    line["gradient_variance_mean"],
                    s=[sizes_by_depth[d] for d in line["depth"]],
                    color=ansatz_color,
                    marker=style_registry.get("feature_map_family", feature_map).marker,
                    zorder=2,
                )
            ax.set_yscale("log")
            ax.set_xscale("log")
            _style_axes(ax)
            if i == 0:
                ax.set_title(f"{num_qubits} qubits")
            if i == len(ansatze) - 1:
                ax.set_xlabel("Expressibility (KL to Haar) →")
        axes[i][0].set_ylabel(f"{ansatz}\nGradient variance →")

    feature_map_handles = [
        plt.Line2D(
            [], [], color="#52514e", marker=style_registry.get("feature_map_family", f).marker,
            linestyle="", markersize=8, label=f,
        )
        for f in feature_maps
    ]
    axes[0][-1].legend(handles=feature_map_handles, frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Expressibility vs. trainability ({topology} topology)")
    fig.text(
        0.5, 0.01,
        "Marker size grows with depth: " + " → ".join(str(d) for d in depths),
        ha="center", fontsize=8, color="#666666",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_vs_depth(
    summary: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    topology: str = "linear",
    style_registry: StyleRegistry = DEFAULT_STYLE_REGISTRY,
) -> None:
    """Line plot of `metric` (e.g. "expressibility_kl" or "gradient_variance")
    against circuit depth, one line per (ansatz, feature map), faceted by
    qubit count - shows the saturation point as depth increases. Topology is
    fixed (not faceted) to keep every feature map x ansatz combination
    legible on one panel."""
    data = summary[summary["topology"] == topology]
    fig, axes_by_qubits = _facet_by_qubits(data)
    ansatze = sorted(data["ansatz_family"].unique())
    feature_maps = sorted(data["feature_map_family"].unique())

    for num_qubits, ax in axes_by_qubits.items():
        cell = data[data["num_qubits"] == num_qubits]
        for ansatz in ansatze:
            ansatz_color = style_registry.get("ansatz_family", ansatz).color
            for feature_map in feature_maps:
                line = cell[
                    (cell["ansatz_family"] == ansatz) & (cell["feature_map_family"] == feature_map)
                ].sort_values("depth")
                if line.empty:
                    continue
                ax.errorbar(
                    line["depth"],
                    line[f"{metric}_mean"],
                    yerr=line[f"{metric}_std"],
                    color=ansatz_color,
                    marker=style_registry.get("feature_map_family", feature_map).marker,
                    markersize=8,
                    linewidth=2,
                    capsize=3,
                )
        ax.set_title(f"{num_qubits} qubits")
        ax.set_xlabel("Depth (reps)")
        _style_axes(ax)

    axes_by_qubits[min(axes_by_qubits)].set_ylabel(metric.replace("_", " "))
    _legend(list(axes_by_qubits.values())[-1], ansatze, feature_maps, style_registry=style_registry)
    fig.suptitle(f"{metric.replace('_', ' ')} vs. depth ({topology} topology)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
