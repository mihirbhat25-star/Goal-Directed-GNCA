# Code organization

[Documentation index](README.md) · [Experiment details](EXPERIMENT_DETAILS.md)

## Expert dynamics

`modules/boids.py` and `modules/boids_3d.py` define the reference local expert:
separation, alignment, cohesion, goal attraction, limits, and perturbation events.
`modules/waypoints.py` handles dynamic goal sequences and relative-goal inputs.
`modules/perturbations.py` holds shared validated configuration and sample mixtures.

## Neural models

`models/gnn_ca_simple_boids*.py` contain the fixed-goal GNCA; the corresponding
`gnn_ca_goal_conditioned_boids*.py` files add relative-goal input features.
`layers/simple_edge_conv.py` implements relative-position messages. Training
optimizations do not modify this architecture. Original generic GCA utilities
remain available, but Voronoi/fixed-target experiments are not part of this paper.

## Training and data pipelines

`boids/run_boids.py` and `run_boids_3d.py` remain the training entry points. Their
arguments live in `runtime/cli`. The fixed-task runners own the weighted loss,
training orchestration and output naming. Online
orchestration lives in `runtime/common/training_2d.py` and `online_goal_3d.py`;
fixed-2D cloud training also uses the shared 2D trainer. Workers are private process
entry points used to release per-chunk RAM, not commands users need to call.

The two data paths are:

```text
local:  expert -> reusable HDF5 cache -> streamed graph batches -> CPU/GPU fit
cloud:  Numba expert -> shared RAM + bit-packed edges -> lazy TensorFlow batches
```

Both backends support fixed and online goals in 2D and 3D. Online local caches
are managed by `runtime/local/online_cache.py` with bounded in-flight generation
and batch-wise HDF5 reads. They use the same seeded expert configurations and
graph packing contract as cloud execution. Fixed local training retains the
reference expert caches and compact graph shards.
Fixed 2D receives only physical state inputs in both backends, not goal vectors.

Compiled generation uses a CPU thread pool with GIL-free kernels and then
CPU or GPU training according to the selected backend. Cloud mode uses GPU
training. Increasing GPU count does not parallelize the expert kernel itself.
`runtime/cloud/cloud_training.py` handles graph-safe distributed execution;
`runtime/common/timing.py` handles optional time allocation reports.

## Evaluation

`evaluation/paired_rollouts.py` provides a common paired baseline/removal
evaluation for both dimensions and both tasks; `evaluation/parallel.py` splits
its starts across worker processes and merges the results. `evaluation/imitation_error.py`
scores one-step predictions on held-out expert rollouts, and
`evaluation/neighbor_counts.py` counts neighbors along GNCA rollouts at any swarm
size. `evaluation/metrics` contains
general reporting and aggregation; `evaluation/plots.py` draws the plots each
evaluation writes next to its results, and `evaluation/visualization` contains
trajectory plots and the slab-removal filmstrip.
The inference scripts retain their original public filenames and optional
interactive/visualization features. Old threshold-based success reports from
those scripts should not replace the paper launcher's completion metric.

## Presets and command launcher

`experiments/presets` holds the training recipes of the four experiments; common settings can also
be overridden on the launcher's command line. The launcher prints
the low-level commands, uses argument lists rather than shell substitution, and
can be inspected with `--dry-run` before any data is generated. It does not
silently infer a machine's memory capacity or retune the optimizer.
