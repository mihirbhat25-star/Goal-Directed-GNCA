# Experiment details

[Documentation index](README.md) · [Running experiments](USAGE.md)

This guide records the training recipes and the evaluation definitions.

## Four main training configurations

All use 100 agents, perception 0.1, initial scatter 0.325, Adam, weighted
one-step state imitation (`goal_weighted`, critical distance 2.5, weight 2.5), and the
256-hidden-unit GNCA models. The Boids expert's movement limits are
not externally imposed on GNCA predictions; the model's limits MLP is learned.

| Setting | Fixed 2D | Fixed 3D | Dynamic 2D | Dynamic 3D |
| --- | --- | --- | --- | --- |
| Initial learning rate | 0.0003 | 0.0003 | 0.0003 | 0.0003 |
| Global batch size | 256 | 1024 | 512 | 1024 |
| LR reduction factor | 0.1 | 0.1 | 0.1 | 0.1 |
| Minimum LR | 0 | 1e-6 | 1e-6 | 1e-6 |
| LR patience (epochs) | 10 | 3 | 3 | 3 |
| Chunk stopping patience (epochs) | 20 | 10 | 10 | 10 |
| Trajectories/chunk | 200 | 800 | 800 | 800 |
| Unique trajectories | 500 (two repeats) | 8,000 + 4,000 perturbed | 1,000 + 500 perturbed | 8,000 + 4,000 perturbed |
| Validation trajectories | 10 | 16 | 16 | 8 |
| Far-from-goal timestep stride | 5 | 1 | 1 | 1 |
| Learning rate at each new chunk | restarted | restarted | restarted | restarted |
| Exported weights (default) | lowest validation loss over the run | lowest validation loss over the run | lowest validation loss over the run | lowest validation loss over the run |

The fixed-2D recipe subsamples far-from-goal transitions at stride
5, retaining dense transitions near goals (`near_goal_radius=1`). The other
three recipes retain stride 1. Batch size counts graph transitions, not whole
trajectories. Changing the chunk size changes the sequence of training and
early-stopping decisions, not only memory consumption.

The fixed-2D cache generator creates exactly 125 centers in each quadrant, each
repeated twice.

## Chunks and model selection

Within a run, every chunk starts from the previous chunk's best weights. The
fixed-3D, online and all cloud pipelines also carry the Adam state; the local
fixed-2D pipeline starts each chunk with a new optimizer. Every pipeline restarts
the learning-rate schedule at `--lr` for every chunk. By default every chunk
validates on the same episodes, and the exported weights are those of the chunk
with the lowest validation loss over the run. `--last-chunk-selection` instead
exports the best state of the last chunk, each chunk drawing its own validation
episodes. Of the models reported in the paper, the fixed 3D models were selected
over the run at learning rate 0.0003; the fixed 2D, dynamic 2D and dynamic 3D
models were selected from the last chunk at learning rate 0.001 (`--lr 0.001
--last-chunk-selection`).

## Goal switching and backends

Evaluation consistently uses **ordered centroid distance <= 0.5**, stopping
when every requested goal is reached or after at most 5,000 updates. The fixed
waypoints are `(3,-3), (3,3), (0,0)` in 2D and `(-4,-4,-4), (4,4,4)` in 3D.
Online goals are uniform within `[-4.5,4.5]^d`, rejection-sampled to be at least
2.5 units from the previous goal (initial centroid for the first goal).

The expert switches to its next waypoint by the same rule, in training data
and in evaluation, for fixed and for online waypoints: once the **centroid
distance** to the active waypoint is at most a radius (`--goal_arrival_radius`,
which the presets state explicitly). Training data and evaluation use 0.5 for both
controllers in every setting.

The cloud kernels are optimized implementations of the expert equations, not
the `Boids` class running in hundreds of processes. They avoid SciPy graph
construction and inter-process copying, retain bit-packed adjacency, and build
TensorFlow disjoint batches lazily. These are the sole compiled cloud methods
in this artifact.

All four experiments support both backends. Online local training uses the same
seeded compiled experts as online cloud training, saving reusable trajectories
to disk instead of retaining an entire chunk in memory. Fixed-2D cloud generation
keeps the requested unique-center/repeat counts and fixed waypoint order and
draws its own random sample of starts, separate from the local cache's.

## Metrics and failures

For positions p_i and velocities v_i at each observed finite step:

- `r(t)` is the 95th percentile across agents of `||p_i - mean(p)||`.
- Polarization is the norm of the mean unit velocity over moving agents.
- Speed CV is population standard deviation of moving-agent speeds divided by
  their mean. An all-stationary frame has undefined polarization/CV, saved as
  NaN rather than being interpreted as good alignment or zero variation.

Each rollout saves temporal mean, min, max, median, and population standard
deviation. `paper_summary.csv` aggregates temporal **means** across rollouts;
it is not a percentile over pooled agent positions or pooled timesteps.
Median and quartile calculations use NumPy's default linear interpolation.

No compactness cutoff is used in the paper launcher. Failure labels are
`timeout` (goals incomplete at the horizon) and `instability` (a non-finite
predicted state). There is no automatic termination at a spatial boundary.
Metrics for a failed rollout describe only its observed finite prefix; always
read them alongside failure counts and observed steps. Undefined means are
excluded from aggregate numeric calculations and their denominators are saved.

GNCA and expert share initial states/goals but are not forced to reach goals
at the same time. The expert can complete a sequence that the GNCA does not.
Slab removal shares event time and plane normal, not necessarily agent indices.

## Seeds, hardware, and artifacts

Each preset sets the data-generation and weight-initialization seed used when
`--seed` is not given (1 in every preset); `--seed` overrides it. Evaluation and
removal draw their test starts with seed 460 unless `--seed` is given; imitation
scoring draws its held-out expert rollouts with seed 1. Do not choose an evaluation seed based on its score.

The NGC TensorFlow container is the recommended GPU environment; the native
dependency list in `requirements.txt` is an alternative. Full datasets are generated by the
included experts; pretrained checkpoints are the only binary training artifacts
shipped. No training or deployment service credentials are needed.
