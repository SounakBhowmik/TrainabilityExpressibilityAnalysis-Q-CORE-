# Trainability–Expressibility Analysis

Experiment code for a Q-CORE paper (and poster) on the expressibility–trainability
tradeoff in quantum kernels and variational classifiers.

## Core claim being tested

Circuit expressibility governs both kernel discriminability (how well a
quantum kernel separates classes) and variational trainability (how well
gradients survive as circuits get more expressive). There's a saturation
point beyond which more expressibility *hurts* trainability, and hardware
noise pulls that saturation point earlier depending on entanglement
topology — i.e. the ceiling is device-specific, not just circuit-specific.

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

## Pipeline

```
qcore_expr_train/
  datasets.py    load + PCA-reduce + angle-encode a dataset to `num_qubits` features
  circuits.py    define the (feature map, ansatz, qubits, depth, topology) grid,
                 crossing feature maps and ansatze independently
  noise.py       one consistent real-backend noise model, shared by every grid cell
  metrics.py     the four per-cell measurements (see below), noiseless or noisy
  results.py     evaluate one grid cell (or the whole grid) into a tidy DataFrame
  visualize.py   pilot exploratory plots + the 4-plot poster set
  run_pilot.py   entry point: runs the small pilot grid end-to-end, locally
  run_full.py    entry point: evaluates ONE grid cell (one SLURM array task)
  aggregate.py   combines run_full.py's shards + generates the poster plots
```

Each grid cell (`CircuitSpec`) fixes a **feature map**, an **ansatz**, a
**qubit count**, a **depth** (`reps`), and an **entanglement topology**
(`linear`, `circular`, `full`) — all shared between the two circuits. The
feature map and ansatz are chosen **independently and crossed**, not paired
1:1 by family: `circuits.build_grid()` takes every combination of
`feature_map_families × ansatz_families`, so every ansatz gets evaluated
against every feature map.

This is deliberate, and replaces an earlier version of this project that
paired each feature map with one bespoke ansatz (e.g. `zz` always trained
with a `Ry`+`CRZ` ansatz, `cx_ry` always with `real_amplitudes`). That
1:1-pairing design could only show *association* — "families with more
expressible feature maps tend to pair with harder-to-train composed
circuits" — because the feature map and ansatz always changed together
between families; there was no way to tell whether an observed trainability
difference came from the feature map or the ansatz. Crossing them isolates
the effect: holding one ansatz fixed and sweeping all 4 feature maps shows
the feature map's contribution to trainability on its own, which is what the
core claim ("expressibility governs trainability") actually asserts.

### The 4 feature maps, and why these four

```
local_z          →  zz, pauli               →  cx_ry
no entanglement     diagonal/commuting          non-diagonal, maximal
                     entangling (2 Pauli-order   entangling
                     variants)
```

- **`local_z`** — single-qubit `Z` rotations only, no entanglement. The
  expressibility/entangling-capability floor.
- **`zz`, `pauli`** — entangle via diagonal (commuting), IQP-style `RZZ`-type
  phase gates. This is the canonical quantum-kernel-advantage circuit style
  (Havlicek et al. 2019); `pauli` generalizes `zz` to higher-order Pauli
  terms, but both sit in the same diagonal-entangling class — in the pilot
  data their points landed almost exactly on top of each other in every plot.
- **`cx_ry`** — `Ry` rotations + `CX` entanglers: a non-diagonal, maximally-
  entangling feature map, the "hardware-efficient ansatz" style used in the
  barren-plateau literature (McClean et al. 2018). It's the odd one out
  structurally, and deliberately so — it's what gives the feature-map sweep
  a genuinely different point on the expressibility spectrum instead of two
  near-duplicates.

### The 2 ansätze, and why these two

Exactly two, deliberately the two extremes rather than one bespoke ansatz per
feature map:

| Ansatz | Gates | Role |
|---|---|---|
| `rotation_only` | `Ry` | No entanglement — the trainability floor, isolates whatever barren-plateau effect comes from depth/width alone, without any entangling structure to blame. |
| `real_amplitudes` | `Ry` + `CX` | Maximal, non-diagonal entanglement — the standard hardware-efficient-ansatz style, where entangling-related trainability effects should show up most.

## Measurements (`metrics.py`)

Every measurement function takes a `simulate(circuit, params) -> state`
callable. The default is exact, noiseless statevector simulation; the noisy
condition passes `noise.noisy_simulate` instead, which returns a
`DensityMatrix`. `qiskit.quantum_info`'s fidelity/purity/expectation-value
functions all accept either, so the same code computes both conditions.

| Measurement | Function | Circuit it runs on | What it captures |
|---|---|---|---|
| Expressibility | `expressibility_kl` | Feature map only | KL divergence between the circuit's sampled fidelity distribution and the analytic Haar-random one (Sim et al. 2019). Lower = more expressible. |
| Entangling capability | `meyer_wallach` | Feature map only | Meyer-Wallach Q measure, averaged over random parameter draws. 0 = product state, 1 = maximally entangled. |
| Trainability | `gradient_variance` | **Feature map + ansatz, composed** | Variance of a parameter-shift gradient across random ansatz-parameter draws (McClean et al. 2018 barren-plateau protocol), with the feature map bound to a real training sample. Near-zero = vanishing gradients. |
| Kernel usefulness | `kernel_usefulness` (KTA + SVM accuracy) | Feature map only | How well the kernel's similarity structure agrees with class labels, and how well an SVM built on it actually classifies held-out data. |

### How the feature map and ansatz combine (and why they don't, everywhere)

Expressibility, entangling capability, and kernel usefulness only ever look at
the **feature map** — that's deliberate, not an oversight: a QSVM never trains
anything, so the only circuit whose properties matter for it is the fixed
encoding. Randomizing the feature map's own parameters (rather than binding
real data) is also standard for expressibility/entangling-capability
specifically — those are meant to characterize the circuit family's coverage
of Hilbert space in general, independent of any one dataset.

Trainability is different, because a VQC does not train the ansatz in
isolation — it trains `feature_map.compose(ansatz)`, with the feature map's
parameters bound to a real data point and the ansatz's parameters being what
gradient descent actually updates. `results.py` builds exactly that: the
feature map and ansatz are composed into one circuit, the feature map's
parameters are bound to a real training sample (cycled by seed, so different
seeds also sample different data points, not just different ansatz
initializations), and `gradient_variance` then runs its usual parameter-shift
sweep over only the ansatz's remaining free parameters. Measuring the ansatz
alone was tried first and produced a materially different (and less
realistic) answer — on one `zz`, 4-qubit, depth-2 circuit, the ansatz-alone
gradient variance was 0.26, versus 0.02–0.09 depending on which real data
point the feature map was bound to. That gap is the encoding's own
expressibility leaking into what should be a trainability measurement, which
is exactly what composing the two circuits fixes.

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
comparing across the 2 ansatz colors shows the ansatz's effect; holding the
ansatz color fixed and comparing across the 4 feature-map markers shows the
feature map's effect in isolation — which is the whole point of crossing
them instead of pairing them 1:1.

## One consistent noise backend (`noise.py`)

All noisy-condition simulation runs against **one fixed backend**
(`FakeSherbrooke`, a real 127-qubit IBM Eagle r3 device) restricted to a
**fixed, nested set of its lowest-index qubits (0–9)** — the max qubit count
anywhere in the grid. Those 10 qubits happen to form a connected chain on
Sherbrooke's heavy-hex coupling map, so a 4/6/8-qubit circuit just uses a
prefix of the same qubits and edges a 10-qubit circuit uses: one backend, one
fixed hardware patch, for every cell in the grid — qubit count is never
confounded with "which chip."

Naively transpiling a small circuit against a large backend (`transpile(circuit,
backend=big_backend)`) pads it out to the *backend's* full qubit count
regardless of the circuit's own width — a 4-qubit circuit becomes 127 qubits
wide, and density-matrix simulation of that is infeasible. `noise.py` instead
transpiles against a `CouplingMap` built from just the fixed 10-qubit patch, so
circuit width always matches the circuit's own qubit count. The (unmodified)
noise model built once from the full backend is still valid here — Aer only
applies noise entries for qubit indices that actually appear in the circuit.

This is also the mechanism that makes the topology story visible: routing a
`full`-topology circuit onto this sparse patch requires more SWAP gates than a
`linear` one does, so noise cost scales with topology mismatch, not just qubit
count.

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

- **4 feature maps × 2 ansätze, crossed, not paired.** This is the setting
  that determines whether the core claim can even be tested causally. A 1:1
  pairing (one ansatz per feature map) can only show association across
  bundled packages; crossing them lets a fixed ansatz's trainability be
  compared across all 4 feature maps, isolating the feature map's
  contribution.
- **Qubits: 4, 6, 8, 10.** This range is where PCA-reduced tabular datasets
  like Breast Cancer Wisconsin (and the planned DO-EM dataset) are commonly
  encoded in near-term QML work, and it's also what caps the top end in
  practice: exact statevector/density-matrix simulation stops being
  tractable well before 10 qubits get much larger (see "Scaling further").
  Spanning a 2.5x range matters specifically because the saturation claim
  rests on Hilbert-space dimension growing exponentially with qubit count —
  a single qubit count can't show that growth; four points along it can.
- **Depth: 1–4 (reps).** 1–2 reps is roughly where currently-runnable NISQ
  circuits sit; 3–4 reps is where the barren-plateau/saturation literature
  (McClean et al. 2018, Holmes et al. 2022) predicts the onset of vanishing
  gradients. Sweeping through both regimes is what turns "there's a
  saturation point" from a cited claim into an observed depth where the
  curves in `poster_saturation.png` actually bend.
- **Topology: linear, circular, full.** This is a first-class variable in
  the noise-ceiling claim, not incidental coverage — it needs a genuine
  sparse→dense spread to show a monotonic routing-overhead relationship.
  This was measured, not assumed, while building `noise.py`: transpiling the
  same circuit onto the fixed noise patch cost 12/25/33 two-qubit gates for
  linear/circular/full respectively (4 qubits, depth 2) — that measured
  spread is the mechanism behind `poster_noise_topology.png`.
- **Seeds: 10.** Expressibility, entangling capability, and gradient
  variance are themselves stochastic estimates, built from randomly sampled
  circuit parameters — a single seed's value carries its own sampling noise.
  Ten seeds turn each grid cell into a mean ± std rather than one arbitrary
  draw, which matters concretely here: the pilot's `zz` and `pauli` feature
  maps landed almost exactly on top of each other in every plot, and without
  a confidence interval that's indistinguishable from "these two feature
  maps are the same" rather than "this run happened to land close together."
- **Kernel sample cap: 80 train / 40 test.** The one setting driven by
  compute rather than statistical design — kernel matrices cost
  O(samples²). 80/40 is large enough to move past the pilot's
  visibly-uninformative SVM accuracies (15/7 samples, ~0.4–0.5 regardless of
  configuration) without paying quadratic cost against the full ~455/114
  train/test split.

## Running the pilot (local, quick)

```bash
pip install -r requirements.txt
python -m qcore_expr_train.run_pilot
```

Runs `circuits.PILOT_GRID` (4 & 6 qubits × depth 1/2 × linear/circular ×
4 feature maps × 2 ansätze, crossed, 3 seeds, **noiseless only**) and writes
`results/pilot_raw.csv`, `results/pilot_summary.csv`, and 4 exploratory plots
to `results/plots/`. Use this to sanity-check any change before touching the
full grid or a cluster.

## Running the full grid on HPC (SLURM)

The full grid (`circuits.FULL_GRID`) is 4/6/8/10 qubits × depth 1–4 ×
linear/circular/full × 4 feature maps × 2 ansätze (crossed) = **384 grid
cells**, each evaluated across 10 seeds and both noise conditions (noiseless
+ noisy) — 7,680 measurement rows total. Kernel evaluation (KTA/SVM) uses 80
train / 40 test samples, capped since kernel matrices are O(samples²).

**Sample-count note**: noisy density-matrix simulation costs ≈0.5s per Aer
call (measured, not estimated — see `noise.py`'s docstring), so its sample
counts (`results._NOISY_SAMPLE_COUNTS`) are deliberately much lower than the
noiseless defaults. Don't raise them without checking the resulting per-task
walltime first.

1. **Dry run one cell first**, before submitting the array:
   ```bash
   python -m qcore_expr_train.run_full --task-id 0
   ```
   This writes `results/shards/cell_000.csv` and prints how long it took.
   Compare that to `scripts/submit_full_grid.slurm`'s `--time` and adjust if
   your cluster is faster/slower than the measurement below.

2. **Submit the array job**:
   ```bash
   sbatch scripts/submit_full_grid.slurm
   ```
   `--array=0-383%20` runs one task per grid cell, throttled to 20 concurrent
   tasks by default — adjust the throttle to your fairshare/allocation.

3. **Aggregate + plot** once the array finishes:
   ```bash
   sbatch --dependency=afterok:<array-job-id> scripts/aggregate_and_plot.slurm
   # or, run locally once shards exist:
   python -m qcore_expr_train.aggregate
   ```
   Writes `results/full_raw.csv`, `results/full_summary.csv`, and the 4
   poster plots to `results/plots/`.

**Resource guidance** (measured, not guessed): noisy simulation dominates
runtime. At the default sample counts, one (cell, seed) pair costs ≈165s
noisy + ≈10s noiseless ≈ 175s; ×10 seeds ≈ 29 minutes per array task — this
per-task cost is unchanged by crossing feature maps with ansätze, since each
task still evaluates exactly one grid cell. With `--array=0-383%20`,
wall-clock for the whole grid is roughly `384/20 × 29min ≈ 9.3 hours` (double
the earlier 1:1-pairing design's estimate, since the grid itself doubled in
size from 192 to 384 cells). `scripts/submit_full_grid.slurm` requests 1.5
hours per task as a safety margin — tighten or loosen based on your dry run.

## The 4 poster plots (`visualize.py`)

| File | Claim it carries |
|---|---|
| `poster_tradeoff.png` | Headline: gradient variance vs. expressibility, noiseless, faceted by qubit count — the rise-then-fall saturation shape, with the ansatz's contribution held apart from the feature map's. |
| `poster_saturation.png` | Expressibility and trainability vs. depth — *where* the saturation point sits, and whether it shifts depending on which ansatz is attached. |
| `poster_noise_topology.png` | Trainability vs. depth, noiseless (solid) vs. noisy (dashed), faceted by topology — noise pulls the ceiling in, and by how much depends on topology (and, now, on which ansatz). |
| `poster_kernel_usefulness.png` | SVM accuracy vs. expressibility, noiseless (filled markers) vs. noisy (hollow) — the practical classification payoff. |

One consistent encoding across all four (and the pilot's exploratory plots):
**color = ansatz** (2 values, fixed order, never cycled — the variable held
fixed while reading a feature map's effect), **marker shape = feature map**
(4 values), **linestyle/marker-fill = noise condition** — so identity never
depends on color alone, and topology is faceted where the plot's story
needs it, fixed to `linear` elsewhere (documented in each title).

## Tests

```bash
python -m pytest qcore_expr_train/tests
```

`test_noise.py` covers the noisy-simulation path in isolation (small circuits,
no SLURM needed); everything else mirrors the pilot's noiseless-only tests.

## Scaling further

- The DO-EM dataset can be added as a second loader function in `datasets.py`,
  reusing the same PCA/angle-encoding pipeline.
- If qubit counts need to grow past what exact statevector/density-matrix
  simulation can handle, `metrics.py`'s `simulate` callable is the extension
  point — a shot-based `Sampler`/`Estimator` primitive slots in the same way
  `noise.noisy_simulate` did.
