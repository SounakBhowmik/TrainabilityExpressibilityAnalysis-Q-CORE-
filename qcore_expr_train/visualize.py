"""Plots the results DataFrame produced by results.py.

Two independent categorical variables now describe a grid cell - which
feature map, and which ansatz - crossed rather than paired 1:1 (see
circuits.py). One consistent encoding is used everywhere in this file so a
reader only has to learn it once:

- color        = ansatz_family (2 values) - the thing held fixed while the
                 feature map varies, so a single color's marker spread shows
                 the feature map's effect on that ansatz directly.
- marker shape = feature_map_family (3 values).
- linestyle / marker fill = noise_condition, where relevant.
- topology is faceted where the plot's story is about topology, and fixed to
  "linear" (documented in the title) everywhere else - three categorical
  axes plus qubit-count faceting in one static plot stops being legible.

Error bars come from the mean/std columns summarize() computes across seeds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical order - do not reorder or cycle.
_ANSATZ_COLORS = {"rotation_only": "#2a78d6", "real_amplitudes": "#1baf7a"}
_ANSATZ_LINESTYLES = {"rotation_only": "-", "real_amplitudes": "--"}
_FEATURE_MAP_MARKERS = {"zz": "o", "local_z": "^", "cx_ry": "D"}
_NOISE_LINESTYLES = {"noiseless": "-", "noisy": "--"}


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.5, alpha=0.3)


def _legend(ax: plt.Axes, ansatze: list[str], feature_maps: list[str], noise_conditions: list[str] = ()) -> None:
    handles = [
        plt.Line2D([], [], color=_ANSATZ_COLORS[a], marker="o", linestyle="", markersize=8, label=a)
        for a in ansatze
    ] + [
        plt.Line2D([], [], color="#52514e", marker=_FEATURE_MAP_MARKERS[f], linestyle="", markersize=8, label=f)
        for f in feature_maps
    ]
    if noise_conditions:
        handles += [
            plt.Line2D([], [], color="#52514e", linestyle=_NOISE_LINESTYLES[c], linewidth=2, label=c)
            for c in noise_conditions
        ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")


def _facet_by_qubits(summary: pd.DataFrame) -> tuple[plt.Figure, dict[int, plt.Axes]]:
    qubit_counts = sorted(summary["num_qubits"].unique())
    fig, axes = plt.subplots(1, len(qubit_counts), figsize=(5 * len(qubit_counts), 4.5), sharey=True)
    axes = [axes] if len(qubit_counts) == 1 else list(axes)
    return fig, dict(zip(qubit_counts, axes))


def plot_tradeoff(summary: pd.DataFrame, out_path: str | Path, topology: str = "linear") -> None:
    """Gradient variance (trainability) vs expressibility KL divergence -
    the core expressibility/trainability tradeoff claim. Because color is the
    ansatz and marker is the feature map, reading along one color shows
    exactly how trainability changes as the feature map (and its
    expressibility) varies, with the ansatz held fixed - the isolated effect,
    not a confounded family-level association.
    """
    data = summary[summary["topology"] == topology]
    fig, axes_by_qubits = _facet_by_qubits(data)
    ansatze = sorted(data["ansatz_family"].unique())
    feature_maps = sorted(data["feature_map_family"].unique())

    for num_qubits, ax in axes_by_qubits.items():
        cell = data[data["num_qubits"] == num_qubits]
        for _, row in cell.iterrows():
            ax.errorbar(
                row["expressibility_kl_mean"],
                row["gradient_variance_mean"],
                xerr=row["expressibility_kl_std"],
                yerr=row["gradient_variance_std"],
                color=_ANSATZ_COLORS[row["ansatz_family"]],
                marker=_FEATURE_MAP_MARKERS[row["feature_map_family"]],
                markersize=8,
                linewidth=1.5,
                capsize=3,
            )
        ax.set_yscale("log")
        ax.set_title(f"{num_qubits} qubits")
        ax.set_xlabel("Expressibility (KL to Haar) →")
        _style_axes(ax)

    axes_by_qubits[min(axes_by_qubits)].set_ylabel("Gradient variance (trainability) →")
    _legend(list(axes_by_qubits.values())[-1], ansatze, feature_maps)
    fig.suptitle(f"Expressibility vs. trainability ({topology} topology)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_vs_depth(summary: pd.DataFrame, metric: str, out_path: str | Path, topology: str = "linear") -> None:
    """Line plot of `metric` (e.g. "expressibility_kl" or "gradient_variance")
    against circuit depth, one line per (ansatz, feature map), faceted by
    qubit count - shows the saturation point as depth increases. Topology is
    fixed (not faceted) to keep 4 feature maps x 2 ansatze legible on one
    panel."""
    data = summary[summary["topology"] == topology]
    fig, axes_by_qubits = _facet_by_qubits(data)
    ansatze = sorted(data["ansatz_family"].unique())
    feature_maps = sorted(data["feature_map_family"].unique())

    for num_qubits, ax in axes_by_qubits.items():
        cell = data[data["num_qubits"] == num_qubits]
        for ansatz in ansatze:
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
                    color=_ANSATZ_COLORS[ansatz],
                    marker=_FEATURE_MAP_MARKERS[feature_map],
                    markersize=8,
                    linewidth=2,
                    capsize=3,
                )
        ax.set_title(f"{num_qubits} qubits")
        ax.set_xlabel("Depth (reps)")
        _style_axes(ax)

    axes_by_qubits[min(axes_by_qubits)].set_ylabel(metric.replace("_", " "))
    _legend(list(axes_by_qubits.values())[-1], ansatze, feature_maps)
    fig.suptitle(f"{metric.replace('_', ' ')} vs. depth ({topology} topology)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
# All four read a `summary` DataFrame that includes a "noise_condition" column
# ("noiseless"/"noisy"), unlike the pilot's noiseless-only summaries above.


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
    # Feature maps don't have a dedicated color palette elsewhere in this
    # file (color is reserved for ansatz everywhere else) - assign one here,
    # scoped to this plot only, reusing the same fixed hue order as
    # _ANSATZ_COLORS's slots so it stays consistent with the palette's rule
    # of a fixed categorical order.
    feature_map_colors = dict(zip(feature_maps, ["#2a78d6", "#1baf7a", "#eda100"]))

    fig, (ax_expr, ax_train) = plt.subplots(1, 2, figsize=(10, 4.5))
    for feature_map in feature_maps:
        expr_line = data[data["feature_map_family"] == feature_map].drop_duplicates("depth").sort_values("depth")
        ax_expr.errorbar(
            expr_line["depth"],
            expr_line["expressibility_kl_mean"],
            yerr=expr_line["expressibility_kl_std"],
            color=feature_map_colors[feature_map],
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
                color=feature_map_colors[feature_map],
                linestyle=_ANSATZ_LINESTYLES[ansatz],
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
    train_handles = [plt.Line2D([], [], color=feature_map_colors[f], marker="o", markersize=8, label=f) for f in feature_maps]
    train_handles += [
        plt.Line2D([], [], color="#52514e", linestyle=_ANSATZ_LINESTYLES[a], linewidth=2, label=a)
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
    """Plot 3: trainability vs. depth, noiseless (solid) vs. noisy (dashed),
    faceted by topology instead of qubit count. Color = ansatz, marker =
    feature map - carries the "noise pulls the ceiling in, and it's
    topology-dependent" claim, now also showing whether that effect depends
    on which ansatz is attached to the encoding."""
    qubits = qubits or summary["num_qubits"].max()
    data = summary[summary["num_qubits"] == qubits]
    topologies = [t for t in ["linear", "circular", "full"] if t in data["topology"].unique()]
    ansatze = sorted(data["ansatz_family"].unique())
    feature_maps = sorted(data["feature_map_family"].unique())

    fig, axes = plt.subplots(1, len(topologies), figsize=(5 * len(topologies), 4.5), sharey=True)
    axes = [axes] if len(topologies) == 1 else list(axes)

    for topology, ax in zip(topologies, axes):
        cell = data[data["topology"] == topology]
        for ansatz in ansatze:
            for feature_map in feature_maps:
                for condition in ["noiseless", "noisy"]:
                    line = cell[
                        (cell["ansatz_family"] == ansatz)
                        & (cell["feature_map_family"] == feature_map)
                        & (cell["noise_condition"] == condition)
                    ].sort_values("depth")
                    if line.empty:
                        continue
                    ax.errorbar(
                        line["depth"],
                        line["gradient_variance_mean"],
                        yerr=line["gradient_variance_std"],
                        color=_ANSATZ_COLORS[ansatz],
                        linestyle=_NOISE_LINESTYLES[condition],
                        marker=_FEATURE_MAP_MARKERS[feature_map],
                        markersize=7,
                        linewidth=2,
                        capsize=3,
                    )
        ax.set_yscale("log")
        ax.set_title(topology)
        ax.set_xlabel("Depth (reps)")
        _style_axes(ax)

    axes[0].set_ylabel("Gradient variance (trainability) →")
    _legend(axes[-1], ansatze, feature_maps, noise_conditions=["noiseless", "noisy"])
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
