"""This paper's specific circuit grid, built on top of the generic
pqc_diagnostics library: feature map family x ansatz family x qubit count x
depth x topology.

Each grid cell (a CircuitSpec) crosses one of 3 feature maps with one of 2
ansatze, sharing qubit count/depth/topology between them. The feature map and
ansatz are chosen independently - deliberately, not paired by family - so
that trainability (measured on the two composed together, see results.py) can
be attributed to the feature map's expressibility while the ansatz is held
fixed, rather than confounding "which feature map" with "which ansatz" the
way a fixed 1:1 pairing would.

Family names below are explicit literals, not "whatever's currently
registered" in pqc_diagnostics.registry - this grid is defined once at import
time and must stay exactly this size regardless of what else a shared
interpreter (e.g. a notebook) might register later.
"""

from __future__ import annotations

from pqc_diagnostics.circuits import CircuitSpec, build_grid  # noqa: F401 - re-export, unchanged import path

# Small pilot sweep to validate the pipeline before scaling to the full grid
# (kernel matrices are O(n^2), and the full grid multiplies qubits x depth x
# topology x feature map x ansatz x noise x seeds on top of that). Qubits and
# depths are deliberately the smallest and largest values in FULL_GRID's own
# sweep (4/10 qubits, depth 1/8) rather than adjacent small values, so the
# pilot also smoke-tests the grid's outer bounds before a full HPC run.
PILOT_GRID = build_grid(
    qubits=[4, 10],
    depths=[1, 8],
    topologies=["linear", "circular"],
    feature_map_families=["zz", "local_z", "cx_ry"],
    ansatz_families=["rotation_only", "real_amplitudes"],
)

# Full HPC-scale grid for the poster run - see results.py/run_full.py for how
# this gets sharded one grid cell per SLURM array task. Qubit count is capped
# at 10 to match noise.py's MAX_PATCH_QUBITS (the noisy condition's fixed
# hardware-patch width) - see noise.py for why the noisy condition can't
# currently go wider than that.
_FULL_GRID_QUBITS = [4, 8, 10]
_FULL_GRID_DEPTHS = [1, 2, 4, 8]
_FULL_GRID_TOPOLOGIES = ["linear", "circular", "full"]

# local_z + rotation_only is the one (feature map, ansatz) pair with no
# entangling gate anywhere in the composed circuit (rotation_only discards
# its entanglement argument; z_feature_map has none either) - so topology is
# a no-op for it specifically, and all 3 topologies would evaluate the exact
# same circuit. Every other pair has an entangling gate somewhere (feature
# map, ansatz, or both) and genuinely depends on topology, so only this one
# pair is pulled out and fixed to a single topology instead of crossed with
# all 3 - a free ~11% cut (24 of 216 cells) with no information loss.
FULL_GRID = (
    build_grid(
        qubits=_FULL_GRID_QUBITS,
        depths=_FULL_GRID_DEPTHS,
        topologies=_FULL_GRID_TOPOLOGIES,
        feature_map_families=["zz", "cx_ry"],
        ansatz_families=["rotation_only", "real_amplitudes"],
    )
    + build_grid(
        qubits=_FULL_GRID_QUBITS,
        depths=_FULL_GRID_DEPTHS,
        topologies=_FULL_GRID_TOPOLOGIES,
        feature_map_families=["local_z"],
        ansatz_families=["real_amplitudes"],
    )
    + build_grid(
        qubits=_FULL_GRID_QUBITS,
        depths=_FULL_GRID_DEPTHS,
        topologies=["linear"],
        feature_map_families=["local_z"],
        ansatz_families=["rotation_only"],
    )
)
