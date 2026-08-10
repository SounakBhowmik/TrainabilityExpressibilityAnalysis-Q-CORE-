# Trainability–Expressibility Analysis

Experiment code for a Q-CORE paper (and poster) on the expressibility–trainability
tradeoff in quantum kernels and variational classifiers.

## Core claim being tested

Circuit expressibility governs variational trainability (how well gradients
survive as circuits get more expressive). There's a saturation point beyond
which more expressibility *hurts* trainability, and hardware noise pulls that
saturation point earlier depending on entanglement topology — i.e. the
ceiling is device-specific, not just circuit-specific. (Expressibility's
effect on *kernel discriminability* — how well a quantum kernel separates
classes — is a related, parallel question this codebase can also measure via
`pqc_diagnostics.metrics.kernel_usefulness`, but it isn't wired into this
paper's grid or poster plots; see "Architecture" below.)

## Motivation and application

QML pipelines expose a lot of knobs — feature-map family, depth, entanglement
topology, hardware backend — with no principled way to narrow them down
before paying for a full training/benchmarking run. "More expressible" is
often treated as straightforwardly better for kernel or classifier
performance, but the theory says otherwise past a threshold: gradients
vanish (McClean et al. 2018) and kernel values concentrate, so every pair of
data points starts looking equally (dis)similar and the kernel stops being
useful for classification (Holmes et al. 2022). That failure mode is
usually described in the abstract (random circuit ensembles, no dataset), and
usually noiseless.

This analysis makes the tradeoff concrete and measurable instead: on a real
benchmark dataset (not just an abstract circuit ensemble), tied to an actual
downstream classification accuracy (not just a circuit-only metric), and
under a real calibrated noise model (since near-term hardware, not an
idealized simulator, is the actual deployment target).

**Application**: the resulting expressibility/depth/topology curves become a
practical lookup — given a qubit budget and a target backend, which feature
map, ansatz, depth, and topology are worth training at all, and which are
predictably past the saturation point — rather than exhaustively
benchmarking every combination to find out empirically each time.

## Contribution to Q-CORE

Q-CORE (the parent platform this project sits alongside) benchmarks QML
configurations end-to-end: given a dataset, it sweeps kernel/ansatz/model
configurations, profiles each against real hardware-calibrated fake backends,
and ranks the results. Today that sweep is exhaustive — every configuration
actually gets trained and scored before Q-CORE knows whether it was worth
running.

This project is built on the same shape of building blocks Q-CORE already
uses — a name-keyed feature-map/ansatz registry (mirroring
`qml_pipeline/kernels/registry.py`) and a fake-backend hardware-profiling
approach (mirroring `qml_pipeline/hardware/providers/`) — but run as a
standalone research grid outside Q-CORE's app-level surface (API, database,
dashboard) while the questions being asked are still exploratory.

The intended path back into Q-CORE:
- **Cheaper pre-filtering.** Expressibility, entangling capability, and
  gradient variance are all computable *without training a model* — a
  fraction of the cost of a full QSVM/VQC run. A saturation-point lookup from
  this grid could let Q-CORE prune configurations before spending compute on
  them, instead of discovering post-hoc (via a full training run) that a
  circuit was past its useful depth.
- **A concrete, device-specific noise guideline.** The topology-dependent
  noise-ceiling result gives Q-CORE's hardware-backend selection something
  more specific than "pick the least-noisy backend available" — it says
  *which topology choice costs the least routing overhead, and therefore
  trainability, on a given device's connectivity*.
- **A guided recommendation, not just a leaderboard.** Longer-term, the same
  feature-map/ansatz/depth/topology lookup could surface in the Q-CORE dashboard as a
  proactive suggestion ("for this qubit count and backend, this
  configuration is expected to underperform") rather than only a ranked list
  of configurations that were already run.

## Architecture: a reusable core library + this paper's reproduction layer

The codebase is split into two packages:

```
pqc_diagnostics/     installable, general-purpose PQC diagnostics library -
                     works on any feature map/ansatz/noise backend you
                     register, not just the ones this paper uses
qcore_expr_train/    this paper's specific, reproducible study, built on
                     top of pqc_diagnostics
```

`pqc_diagnostics` is pip-installable on its own (`pip install .` from the
repo root, or `pip install pqc-diagnostics` once published) and has no
dependency on anything in `qcore_expr_train`. It provides:

- **`metrics.py`** — `expressibility_kl`, `meyer_wallach`, `gradient_variance`,
  and `kernel_usefulness` (KTA + SVM accuracy). Every function takes a
  `simulate(circuit, params) -> Statevector | DensityMatrix` callable, so the
  same code covers noiseless and noisy simulation.
- **`circuits.py` + `registry.py`** — `CircuitSpec`, `build_grid()`, and an
  open feature-map/ansatz registry. `register_feature_map("my_fm", factory)` /
  `register_ansatz("my_ansatz", factory)` make a custom PQC usable everywhere
  `CircuitSpec`/`build_grid()` accept a family name, with no need to edit
  this library's source. Three feature maps (`zz`, `local_z`, `cx_ry`) and two
  ansätze (`rotation_only`, `real_amplitudes`) are pre-registered as built-ins.
- **`evaluation.py`** — `evaluate_cell()`/`run_grid()`, a generic per-circuit
  evaluation pipeline. Which metrics to compute (`MetricSpec`), where real
  data samples come from (a `data_sample_provider` callable), and which noise
  backend to use (a `NoiseProvider`) are all supplied by the caller, not
  hardcoded.
- **`hardware/providers.py` + `hardware/registry.py`** — `NoiseProvider`, a
  generic "transpile onto a fixed qubit patch, simulate under calibrated
  noise" interface for any `BackendV2`, plus a registry for naming a specific
  configuration.
- **`visualize/`** — `plots.py`'s `plot_tradeoff`/`plot_metric_vs_depth` read
  color/marker/linestyle from a `StyleRegistry` (`visualize/styles.py`)
  instead of a hardcoded dict, so a custom-registered feature map/ansatz
  still plots (falling back to matplotlib's own color/marker cycling) rather
  than raising `KeyError`.

`qcore_expr_train` is a thin layer on top: `circuits.py` defines
`PILOT_GRID`/`FULL_GRID` as explicit sweeps over the 3 built-in feature maps
and 2 built-in ansätze; `noise.py` instantiates one `NoiseProvider` against
`FakeSherbrooke`; `results.py` supplies this paper's exact `MetricSpec` list,
sample-count budget, and the breast-cancer data-sample provider; `styles.py`
registers this paper's exact color palette; `visualize.py` keeps the
paper-specific `poster_*` figures. None of `run_pilot.py`/`run_full.py`/
`aggregate.py`/the SLURM scripts needed to change for this split — every
name they import is preserved at its existing path.

## Pipeline

```
pqc_diagnostics/       core library (see "Architecture" above)
qcore_expr_train/
  datasets.py    load + PCA-reduce + angle-encode a dataset to `num_qubits` features
  circuits.py    this paper's PILOT_GRID/FULL_GRID, built on pqc_diagnostics.circuits
  noise.py       this paper's NoiseProvider instance (FakeSherbrooke, 10-qubit patch)
  metrics.py     re-exports this paper's 3 metrics from pqc_diagnostics.metrics
  styles.py      registers this paper's exact plot colors/markers/linestyles
  results.py     this paper's MetricSpec list + thin evaluate_cell()/run_grid() wrapper
  visualize.py   this paper's poster_* figures (generic plotting lives in pqc_diagnostics)
  run_pilot.py   entry point: runs the small pilot grid end-to-end, locally
  run_full.py    entry point: evaluates ONE grid cell (one SLURM array task)
  aggregate.py   combines run_full.py's shards + generates the poster plots
```

Each grid cell (`CircuitSpec`) fixes a **feature map**, an **ansatz**, a
**qubit count**, a **depth** (`reps`), and an **entanglement topology**
(`linear`, `circular`, `full`) — all shared between the two circuits. The
feature map and ansatz are chosen **independently and crossed**, not paired
1:1 by family: `build_grid()` takes every combination of
`feature_map_families × ansatz_families`, so every ansatz gets evaluated
against every feature map (with one deliberate exception — see
"Experimental design" below).

This is deliberate, and replaces an earlier version of this project that
paired each feature map with one bespoke ansatz (e.g. `zz` always trained
with a `Ry`+`CRZ` ansatz, `cx_ry` always with `real_amplitudes`). That
1:1-pairing design could only show *association* — "families with more
expressible feature maps tend to pair with harder-to-train composed
circuits" — because the feature map and ansatz always changed together
between families; there was no way to tell whether an observed trainability
difference came from the feature map or the ansatz. Crossing them isolates
the effect: holding one ansatz fixed and sweeping all feature maps shows
the feature map's contribution to trainability on its own, which is what the
core claim ("expressibility governs trainability") actually asserts.

### The 3 feature maps, and why these three

```
local_z          →  zz                       →  cx_ry
no entanglement     diagonal/commuting           non-diagonal, maximal
                     entangling                   entangling
```

- **`local_z`** — single-qubit `Z` rotations only, no entanglement. The
  expressibility/entangling-capability floor.
- **`zz`** — entangles via diagonal (commuting), IQP-style `RZZ`-type phase
  gates. This is the canonical quantum-kernel-advantage circuit style
  (Havlicek et al. 2019). (An earlier version of this grid also included
  `pauli`, generalizing `zz` to higher-order Pauli terms — dropped after the
  pilot showed its points landing on top of `zz`'s in every plot, i.e. no
  distinguishable signal for the extra grid cells it cost.)
- **`cx_ry`** — `Ry` rotations + `CX` entanglers: a non-diagonal, maximally-
  entangling feature map, the "hardware-efficient ansatz" style used in the
  barren-plateau literature (McClean et al. 2018). It's the odd one out
  structurally, and deliberately so — it's what gives the feature-map sweep
  a genuinely different point on the expressibility spectrum instead of a
  near-duplicate of `zz`.

### The 2 ansätze, and why these two

Exactly two, deliberately the two extremes rather than one bespoke ansatz per
feature map:

| Ansatz | Gates | Role |
|---|---|---|
| `rotation_only` | `Ry` | No entanglement — the trainability floor, isolates whatever barren-plateau effect comes from depth/width alone, without any entangling structure to blame. |
| `real_amplitudes` | `Ry` + `CX` | Maximal, non-diagonal entanglement — the standard hardware-efficient-ansatz style, where entangling-related trainability effects should show up most.

## Measurements

Every measurement function (`pqc_diagnostics.metrics`) takes a
`simulate(circuit, params) -> state` callable. The default is exact,
noiseless statevector simulation; the noisy condition passes a
`NoiseProvider.noisy_simulate` instead, which returns a `DensityMatrix`.
`qiskit.quantum_info`'s fidelity/purity/expectation-value functions all
accept either, so the same code computes both conditions.

| Measurement | Function | Circuit it runs on | What it captures |
|---|---|---|---|
| Expressibility | `expressibility_kl` | Feature map only | KL divergence between the circuit's sampled fidelity distribution and the analytic Haar-random one (Sim et al. 2019). Lower = more expressible. |
| Entangling capability | `meyer_wallach` | Feature map only | Meyer-Wallach Q measure, averaged over random parameter draws. 0 = product state, 1 = maximally entangled. |
| Trainability | `gradient_variance` | **Feature map + ansatz, composed** | Variance of a parameter-shift gradient across random ansatz-parameter draws (McClean et al. 2018 barren-plateau protocol), with the feature map bound to a real training sample. Near-zero = vanishing gradients. |
| Kernel usefulness | `kernel_usefulness` (KTA + SVM accuracy) | Feature map only | How well the kernel's similarity structure agrees with class labels, and how well an SVM built on it actually classifies held-out data. **Available in `pqc_diagnostics.metrics`, but not computed by this paper's grid** (`qcore_expr_train.results.METRICS`) or plotted by any poster figure — dropped for cost reasons (its fidelity-matrix Aer calls were roughly half the noisy condition's per-seed cost) and left as a general-purpose diagnostic for other users of the core library. |

### How the feature map and ansatz combine (and why they don't, everywhere)

Expressibility and entangling capability only ever look at the **feature
map** — that's deliberate, not an oversight: a QSVM never trains anything, so
the only circuit whose properties matter for it is the fixed encoding.
Randomizing the feature map's own parameters (rather than binding real data)
is also standard for expressibility/entangling-capability specifically —
those are meant to characterize the circuit family's coverage of Hilbert
space in general, independent of any one dataset.

Trainability is different, because a VQC does not train the ansatz in
isolation — it trains `feature_map.compose(ansatz)`, with the feature map's
parameters bound to a real data point and the ansatz's parameters being what
gradient descent actually updates. `pqc_diagnostics.evaluation.evaluate_cell`
builds exactly that: the feature map and ansatz are composed into one
circuit, the feature map's parameters are bound to a real training sample
(cycled by seed, so different seeds also sample different data points, not
just different ansatz initializations), and `gradient_variance` then runs
its usual parameter-shift sweep over only the ansatz's remaining free
parameters. Measuring the ansatz alone was tried first and produced a
materially different (and less realistic) answer — on one `zz`, 4-qubit,
depth-2 circuit, the ansatz-alone gradient variance was 0.26, versus
0.02–0.09 depending on which real data point the feature map was bound to.
That gap is the encoding's own expressibility leaking into what should be a
trainability measurement, which is exactly what composing the two circuits
fixes.

One consequence worth being explicit about: `poster_tradeoff.png`'s x-axis
(expressibility) and y-axis (trainability) are still computed from two
different circuits — the feature map alone, and the feature map+ansatz
composed — that share depth/topology/qubit-count by construction, plus
whichever ansatz that grid cell crosses the feature map with. The plot isn't
saying "this exact circuit's own expressibility predicts its own
trainability" (a feature map alone has no ansatz to train, so that claim
doesn't type-check); it's saying "encoding data with this feature map, then
training *this specific* ansatz on top of it, produces a composed circuit
this hard to train." Because the grid crosses every feature map with every
ansatz, that claim can be read either way: holding the feature map fixed and
comparing across the 2 ansatz rows shows the ansatz's effect; holding the
ansatz row fixed and comparing across the 3 feature-map markers shows the
feature map's effect in isolation — which is the whole point of crossing
them instead of pairing them 1:1.

## One consistent noise backend

All noisy-condition simulation runs against **one fixed backend**
(`FakeSherbrooke`, a real 127-qubit IBM Eagle r3 device) restricted to a
**fixed, nested set of its lowest-index qubits (0–9)** — the max qubit count
anywhere in the grid. Those 10 qubits happen to form a connected chain on
Sherbrooke's heavy-hex coupling map, so a 4/6/8-qubit circuit just uses a
prefix of the same qubits and edges a 10-qubit circuit uses: one backend, one
fixed hardware patch, for every cell in the grid — qubit count is never
confounded with "which chip." This is `pqc_diagnostics.hardware.providers.NoiseProvider`,
instantiated once in `qcore_expr_train/noise.py` as `SHERBROOKE_PATCH`.

Naively transpiling a small circuit against a large backend (`transpile(circuit,
backend=big_backend)`) pads it out to the *backend's* full qubit count
regardless of the circuit's own width — a 4-qubit circuit becomes 127 qubits
wide, and density-matrix simulation of that is infeasible. `NoiseProvider`
instead transpiles against a `CouplingMap` built from just the fixed patch, so
circuit width always matches the circuit's own qubit count. The (unmodified)
noise model built once from the full backend is still valid here — Aer only
applies noise entries for qubit indices that actually appear in the circuit.

This is also the mechanism that makes the topology story visible: routing a
`full`-topology circuit onto this sparse patch requires more SWAP gates than a
`linear` one does, so noise cost scales with topology mismatch, not just qubit
count. At 10 qubits and higher depths, this cost is severe enough to fully
decohere the circuit (density-matrix purity converging to the maximally-mixed
value `1/2**n`) for `circular`/`full` topology — see `poster_noise_topology.png`,
plotted as a *retained trainability fraction* (noisy / noiseless gradient
variance) specifically because raw noisy gradient variance in that regime is
floating-point noise around zero, not a meaningful quantity.

## Dataset (`datasets.py`)

Breast Cancer Wisconsin (`sklearn.datasets.load_breast_cancer`): standardize
→ PCA to `num_qubits` components → re-standardize → scale to `[-π/2, π/2]`
for angle encoding. All transforms are fit on the training split only.

The DO-EM dataset is the planned second dataset, once the pipeline is
validated on this one.

## Experimental design: why these settings

Every axis of the full grid was chosen to make a specific part of the story
observable, not just for coverage. Feature-map/ansatz choice and the noise
backend are justified in detail above; the rest:

- **3 feature maps × 2 ansätze, crossed, not paired.** This is the setting
  that determines whether the core claim can even be tested causally. A 1:1
  pairing (one ansatz per feature map) can only show association across
  bundled packages; crossing them lets a fixed ansatz's trainability be
  compared across all 3 feature maps, isolating the feature map's
  contribution. One pair, `local_z` + `rotation_only`, is fixed to a single
  topology instead of crossed with all 3: neither circuit has an entangling
  gate, so topology is provably a no-op for that pair (all 3 topologies would
  simulate the identical circuit) — a free ~11% cut in grid size with no
  information loss.
- **Qubits: 4, 8, 10.** This range is where PCA-reduced tabular datasets
  like Breast Cancer Wisconsin (and the planned DO-EM dataset) are commonly
  encoded in near-term QML work. 10 is also a hard ceiling in practice: it
  matches `NoiseProvider`'s fixed hardware-patch width (see "One consistent
  noise backend"), so the noisy condition cannot currently go wider. Spanning
  this range matters specifically because the saturation claim rests on
  Hilbert-space dimension growing exponentially with qubit count — a single
  qubit count can't show that growth; multiple points along it can.
- **Depth: 1, 2, 4, 8 (reps).** 1–2 reps is roughly where currently-runnable
  NISQ circuits sit; deeper reps are where the barren-plateau/saturation
  literature (McClean et al. 2018, Holmes et al. 2022) predicts the onset of
  vanishing gradients. Sweeping a wide depth range (not just 1–4) is what
  turns "there's a saturation point" from a cited claim into an observed
  depth where the curves in `poster_saturation.png` actually bend — the
  earlier 1–4 range only showed a monotonic trend; 8 is where the
  rise-then-fall/collapse actually shows up for several feature-map/ansatz
  pairs.
- **Topology: linear, circular, full.** This is a first-class variable in
  the noise-ceiling claim, not incidental coverage — it needs a genuine
  sparse→dense spread to show a monotonic routing-overhead relationship.
  This was measured, not assumed: transpiling the same circuit onto the fixed
  noise patch cost 12/25/33 two-qubit gates for linear/circular/full
  respectively (4 qubits, depth 2) — that measured spread is the mechanism
  behind `poster_noise_topology.png`.
- **Seeds: 10.** Expressibility, entangling capability, and gradient
  variance are themselves stochastic estimates, built from randomly sampled
  circuit parameters — a single seed's value carries its own sampling noise.
  Ten seeds turn each grid cell into a mean ± std rather than one arbitrary
  draw.

## Running the pilot (local, quick)

```bash
pip install -r requirements.txt
python -m qcore_expr_train.run_pilot
```

Runs `circuits.PILOT_GRID` (4 & 10 qubits × depth 1/8 × linear/circular ×
3 feature maps × 2 ansätze, crossed, 3 seeds, **noiseless only**) and writes
`results/pilot_raw.csv`, `results/pilot_summary.csv`, and 3 exploratory plots
to `results/plots/`. Use this to sanity-check any change before touching the
full grid or a cluster.

## Running the full grid on HPC (SLURM)

The full grid (`circuits.FULL_GRID`) is 4/8/10 qubits × depth 1/2/4/8 ×
linear/circular/full × 3 feature maps × 2 ansätze (crossed, minus the
`local_z`+`rotation_only` single-topology cut described above) = **192 grid
cells**, each evaluated across 10 seeds and both noise conditions (noiseless
+ noisy) — 3,840 measurement rows total.

1. **Dry run one cell first**, before submitting the array:
   ```bash
   python -m qcore_expr_train.run_full --task-id 0
   ```
   This writes `results/shards/cell_000.csv` and prints how long it took.
   Compare that to `scripts/submit_full_grid.slurm`'s `--time` and adjust if
   your cluster is faster/slower than your own dry run. Local timing on a
   laptop is not representative of a compute node's actual performance —
   always dry-run on the target cluster before trusting `--time`.

2. **Submit the array job**:
   ```bash
   sbatch scripts/submit_full_grid.slurm
   ```
   `--array=0-191%20` runs one task per grid cell, throttled to 20 concurrent
   tasks by default — adjust the throttle to your fairshare/allocation.

3. **Aggregate + plot** once the array finishes:
   ```bash
   sbatch --dependency=afterok:<array-job-id> scripts/aggregate_and_plot.slurm
   # or, run locally once shards exist:
   python -m qcore_expr_train.aggregate
   ```
   Writes `results/full_raw.csv`, `results/full_summary.csv`, and the 3
   poster plots to `results/plots/`.

## The 3 poster plots (`visualize.py`)

| File | Claim it carries |
|---|---|
| `poster_tradeoff.png` | Headline: gradient variance vs. expressibility, noiseless, one connected trajectory per feature map ordered by depth (marker size grows with depth), rows = ansatz, columns = qubit count — the rise-then-fall saturation shape, with the ansatz's contribution held apart from the feature map's. |
| `poster_saturation.png` | Expressibility and trainability vs. depth — *where* the saturation point sits, and whether it shifts depending on which ansatz is attached. |
| `poster_noise_topology.png` | Retained trainability (noisy / noiseless gradient variance) vs. depth, faceted by ansatz, one line per (feature map, topology) pair — noise pulls the ceiling in, and by how much depends on topology. |

One consistent encoding across all three: **color = ansatz** (fixed order,
never cycled — the variable held fixed while reading a feature map's
effect), **marker shape = feature map**, **linestyle = topology** where
relevant — so identity never depends on color alone, and topology is faceted
or overlaid where the plot's story needs it, fixed to `linear` elsewhere
(documented in each title). Colors/markers/linestyles are registered in
`qcore_expr_train/styles.py` into `pqc_diagnostics`'s `StyleRegistry`.

## Tests

```bash
python -m pytest pqc_diagnostics/tests qcore_expr_train/tests
```

`pqc_diagnostics/tests/` covers the core library in isolation (registry
round-trips, a custom-feature-map end-to-end smoke test, `NoiseProvider`
against a backend, `StyleRegistry`, and `kernel_usefulness`).
`qcore_expr_train/tests/test_noise.py` covers this paper's specific noisy-
simulation path (small circuits, no SLURM needed); everything else mirrors
the pilot's noiseless-only tests.

## Scaling further

- The DO-EM dataset can be added as a second loader function in `datasets.py`,
  reusing the same PCA/angle-encoding pipeline.
- If qubit counts need to grow past what exact statevector/density-matrix
  simulation can handle, `pqc_diagnostics.metrics`'s `simulate` callable is
  the extension point — a shot-based `Sampler`/`Estimator` primitive slots in
  the same way `NoiseProvider.noisy_simulate` did.
- Bringing your own PQC to `pqc_diagnostics` doesn't require touching this
  repo at all: `register_feature_map`/`register_ansatz`
  (`pqc_diagnostics.registry`) and, for plotting, `register_style`
  (`pqc_diagnostics.visualize.styles`) are the extension points.
