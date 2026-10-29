# Detailed usage and checkpoint reference

[Documentation index](README.md) · [Quick start](../README.md)

**In this guide:** [Setup](#installation) · [Evaluation](#evaluate-the-supplied-models) ·
[Training](#retrain) · [Presets](#preset-files) · [Checkpoints](#checkpoint-and-output-names) ·
[Removal](#agent-removal-robustness) · [Attribution](#attribution)

Code supplement for the paper *Learning Collective Goal-Directed Behavior with Graph Neural Cellular Automata*
(Mihir Bhat). It contains the code to run the experiments the paper describes:
training graph neural cellular automata (GNCA) that steer a swarm through
waypoints in 2D and 3D by imitating a goal-conditioned Boids expert, and
evaluating them against that expert. Included are the 2D and 3D Boids experts,
the GNCA models, and the training and evaluation scripts.

The learned update rule is shared by all agents. Fixed-waypoint models receive
positions and velocities; dynamic-waypoint models additionally receive each
agent's displacement to the active goal. Goal selection and arrival detection
are handled outside the neural update rule.

## Repository layout

```text
boids/                     Training entry points and fixed-task cache generation
modules/                   Boids dynamics, initialization, goals, perturbations
models/                    Fixed- and dynamic-waypoint GNCA architectures
layers/                    Relative-position message-passing layer
runtime/
  cli/                     Training arguments and validation
  local/                   Bounded, cache-backed graph loading
  cloud/                   Numba experts, bit-packed datasets, distributed fitting
  common/                  Shared training orchestration, chunking, timing
evaluation/
  metrics/                 Continuous metrics and paper-format aggregation
  visualization/           Trajectory plots and the slab-removal filmstrip
  paired_rollouts.py       Matched GNCA/Boids evaluation and population interventions
  plots.py                 Completion, metric, recovery and swarm-size plots of one evaluation
  parallel.py              Splits an evaluation across worker processes
  imitation_error.py       One-step imitation error on held-out expert rollouts
  neighbor_counts.py       Neighbor counts along GNCA rollouts
  run_inference*.py        Standalone inference and trajectory visualizations
experiments/
  presets/                 Explicit settings for the four main experiments
  run.py                   Cross-platform experiment launcher
environment/               Container and Python dependencies
docs/                      Experiment details and implementation guide
```

The original `boids`, `modules`, `models`, and `layers` entry-point filenames
are retained. Data processing is separated from the neural architectures and
expert dynamics. No trained weights, datasets, results or figures are bundled;
every result is produced by the commands in this guide. No account credentials
or access to the authors' machines are required.

## Installation

Run commands from this repository's root (the directory containing the main README).
The launcher commands work unchanged in Bash and PowerShell; multiline Docker
examples below are **Bash**.

### NVIDIA GPU: recommended container

On a Linux GPU host with Docker and the NVIDIA Container Toolkit already
installed (including a suitably configured cloud instance):

```bash
docker build -f environment/Dockerfile -t gnca-paper .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
  -v "$PWD:/workspace/gnca" -w /workspace/gnca gnca-paper
```

If the host requires elevated Docker access, use `sudo docker ...`. On a Windows
desktop, use a GPU-enabled Linux/WSL2 Docker environment, not native Windows
TensorFlow for modern GPU training. Mount a persistent disk on cloud instances;
the mounted repository holds the resulting checkpoints after the container exits.

Inside the container:

```bash
python -m experiments.check_environment
python -m experiments.check_environment --smoke
```

The Dockerfile uses the NVIDIA TensorFlow `25.02-tf2-py3` image used in the cloud
experiments. Do **not** install the generic TensorFlow wheel over that image;
`environment/requirements-ngc.txt` intentionally omits TensorFlow. A working
`nvidia-smi` alone does not establish that TensorFlow can initialize CUDA: check
the GPU list printed by `check_environment` before training.

### Native Python / CPU training and evaluation

For a fresh Python 3.10 or 3.11 environment:

```bash
python -m venv .venv
# Bash: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m experiments.check_environment --smoke
```

All four experiments can train on CPU with `--backend local`; full training and
rollouts can be slow. `--backend cloud` requires a TensorFlow-visible GPU. The
old Python 3.8 dependency list is retained only as environment provenance in
`environment/requirements-python38.txt`; it is not the recommended new installation.
The smoke check builds each of the four models and runs one forward step; it
does not train a model. Run it on your target machine before starting
a long experiment.

## Evaluate a trained model

```bash
python -m experiments.run evaluate fixed2d --weights saved_models/best_weights_<run tag>
python -m experiments.run evaluate fixed3d --weights saved_models/best_weights_<run tag>
python -m experiments.run evaluate online2d --weights saved_models/best_weights_<run tag>
python -m experiments.run evaluate online3d --weights saved_models/best_weights_<run tag>
```

These run **40** starts in 2D (10/quadrant) and **80** in 3D (10/octant), with
100 agents, perception radius 0.1, initial scatter 0.325, and a **maximum of
5,000 update steps**. A rollout completes only after the centroid reaches every
goal **in order**, within 0.5 units. Fixed models visit the three 2D or two 3D
fixed goals. Dynamic models visit five goals sampled at least 2.5 units from the
previous goal (the first is separated from the initial centroid).

Each GNCA rollout is paired with a Boids rollout using the same initial agent
states and goal sequence. The launcher reports **task completion separately
from continuous flock metrics**; it does not classify completion using an
arbitrary compactness threshold. It saves per-run temporal mean, minimum,
maximum, median, and standard deviation of radial score, polarization, and speed
CV, plus completion and failure information. `paper_summary.csv` additionally
reports the median, quartiles, and IQR across per-rollout temporal means, as used
in the paper. The time-series archive contains per-step radial scores.

Outputs go to `results/evaluate/<experiment>_seed<seed>/`; the folder name also
records `--perception`, `--crowding` or `--init-scatter` when they differ from the
paper's values. Existing outputs are not overwritten. The starts are split
across `--workers` processes (default: one per visible GPU, otherwise one) and
the results merged; every start is seeded by its own index. For a new test sample:

```bash
python -m experiments.run evaluate online3d --weights saved_models/best_weights_<run tag> --seed 456 --output-dir results/online3d_seed456
```

For a quick plumbing check, use `--centers-per-region 1 --max-steps 5`; such a
short run is **not** a task-completion experiment. The standalone inference
scripts also produce trajectory PDFs. Their older fixed-task `success`
reports can include compactness thresholds; use the launcher above for the
paper's unthresholded completion/continuous-metric protocol.

## Retrain

```bash
python -m experiments.run train fixed2d
python -m experiments.run train fixed3d
python -m experiments.run train online2d
python -m experiments.run train online3d
```

First inspect any command without starting expensive work:

```bash
python -m experiments.run train fixed3d --dry-run
```

Fixed 2D creates a 500-center, two-repeat HDF5 cache and trains in 200-trajectory
chunks. The cache can be large; use a disk with ample free space. Cache
generation is a one-time step and a matching cache is reused.

Dynamic 2D trains on 1,000 unperturbed plus 500 position-and-velocity-perturbed
trajectories in 800-trajectory chunks. Fixed 3D and dynamic 3D each train on
8,000 normal plus 4,000 perturbed trajectories, balanced across the octants, in
800-trajectory chunks of batch 1,024. Every experiment restarts each chunk at
learning rate 0.0003.
Perturbed trajectories add independent bounded uniform impulses to a few agents,
with selection probability 0.1 and at most 10 affected agents per event: in 2D
every 250 steps, position coordinates in `[-0.1, 0.1]` and velocity coordinates
in `[-0.005, 0.005]`; in 3D every 500 steps, position coordinates in
`[-0.1, 0.1]` and velocity coordinates in `[-0.01, 0.01]`. The expert computes
the target **from the perturbed state**.
There is no agent-removal training in these four presets.

### Local versus cloud execution

| Experiment | Cache-backed (`local`) | Compiled GPU (`cloud`) |
| --- | --- | --- |
| Fixed 2D | Supported; paper-style pipeline | Supported; compiled fixed-goal expert |
| Fixed 3D | Supported for unperturbed data (`--drift-trajectories 0`) | Supported; the default |
| Dynamic 2D / 3D | Supported; persistent expert caches, streamed batches | Supported on one or multiple visible GPUs |

These names describe **pipelines**, not where the machine lives. Every experiment
accepts either backend. Local training uses one visible GPU, or CPU when no GPU
is available. Cloud mode can also run on a desktop GPU; no cloud-service API is
involved. For example:

```bash
python -m experiments.run train online2d --backend local --label cached
python -m experiments.run train online3d --backend local --label cached
python -m experiments.run train fixed2d --backend cloud --generation-workers 24 --label compiled
python -m experiments.run train fixed3d --backend cloud --generation-workers 24 --label compiled
```

Online local training generates and saves each chunk's expert trajectories to
HDF5, then streams shuffled graph batches. Matching training and validation
caches are reused across runs. Use `--cache-dir path/to/cache` to choose their
location. Cache names fingerprint the expert configuration, generation seeds,
perturbation mixture, and compiled expert source; mismatched recipes are not
silently reused. The same compiled CPU expert and graph representation serve
both online backends. Local execution does not hold the full chunk as TensorFlow
constants; cloud execution keeps it in RAM and does not write an expert cache.

Cloud mode uses all TensorFlow-visible GPUs. `batch_size` is the **global number
of transitions**, not the agent count or a per-GPU batch size. Every transition
contains the complete 100-agent graph in these training presets. The online
presets use global batches of 512 (2D) and 1,024 (3D); the architecture does not
change with the number of GPUs.

On smaller hardware, reduce CPU workers first. `0` selects at most four workers
for online local cache generation and approximately the visible CPU count minus
two for compiled cloud generation. If RAM/VRAM is insufficient, explicitly reduce
chunk or batch size, understanding that this changes the training experiment:

```bash
python -m experiments.run train online3d --generation-workers 8 --chunk-size 200 --batch-size 256 --label small_gpu
```

Every experiment trains in chunks, and `--chunk-patience` (epochs without validation
improvement before a chunk stops) and `--lr-patience` (epochs before the learning
rate is reduced) replace the preset's values for any of the four experiments and
either backend; the learning-rate patience must stay below the chunk patience.

The same holds for every training hyperparameter the launcher exposes:
`--batch-size`, `--chunk-size`, `--epochs`, `--lr`, `--min-lr`, `--lr-red-factor`,
`--lr-patience`, `--chunk-patience`, `--early-stopping-min-delta`, `--val-set-size`,
`--loss-type`, `--critical-distance`, `--distance-weight`, `--timestep-stride`,
`--near-goal-radius`, `--train-set-unique`, `--generation-workers`, `--seed`, the
expert settings `--perception`, `--crowding` and `--init-scatter`, and
`--train-arrival-radius`. Each applies to all four experiments and both backends,
replaces the preset value, and reaches the expert cache of the cached
fixed-waypoint runs, whose size and file name follow the training set.
`--train-set-unique` is split evenly over the preset's quadrants or octants. Drift
perturbation (`--drift-trajectories` and its settings) applies wherever the expert
data are generated during training: both dynamic experiments on either backend and
both fixed experiments on the cloud backend. `--train-set-repeats` applies to the
fixed-waypoint experiments, the only ones that repeat a starting center.

`--lr` sets the learning rate each chunk starts and restarts at. `--train-arrival-radius`
sets the centroid distance at which the expert switches waypoints in the
training data only; evaluation always uses 0.5. By default every chunk
validates on the same episodes and the exported weights are those of the chunk
with the lowest validation loss over the whole run; `--last-chunk-selection`
exports the best state of the last chunk instead, each chunk drawing its own
validation episodes, which is how the models reported in the paper were selected.

Use `CUDA_VISIBLE_DEVICES` before launching Python to restrict GPUs. Do not
blindly increase the batch size: fewer optimizer updates per epoch can change
the fitted model. All cloud global batches must be divisible by the visible GPU
count. Float32 is retained; this artifact does not enable mixed precision.

Training outputs go under `saved_models/` with a run tag that ends in `_retrain`
(or the `--label` value); see [Checkpoint and output names](#checkpoint-and-output-names).
No trained weights are distributed with this repository. `evaluate`, `removal`
and `imitation` take the checkpoint prefix of a trained model via `--weights`.

For example, `--label custom` changes the output suffix to `_custom`. It is
only a run identifier: it does not select a different architecture or alter
training hyperparameters. Give separate runs distinct labels to avoid output
collisions.

## Preset files

Each experiment's recipe is one JSON file in `experiments/presets/`. The launcher
turns it into ordinary command-line arguments of the training script, which
`--dry-run` prints. JSON allows no comments, so the fields are explained here.

**Top level**

| Field | Meaning |
| --- | --- |
| `dimension`, `task` | 2 or 3; `fixed_waypoints` or `online_goals` |
| `backend` | Pipeline used when `--backend` is not given (`local` or `cloud`) |
| `evaluation_centers_per_region` | Starts per quadrant (2D) or octant (3D) for `evaluate` and `removal` |
| `cache` | Fixed tasks only: the cache generator module and its arguments, run once before `local` training |
| `train` | Arguments of the training script (below) |

**`train` fields**

| Field | Meaning |
| --- | --- |
| `train_set_unique`, `train_set_repeats` | Unique expert trajectories (fixed tasks: unique starting centers) and how often each is repeated |
| `normal_trajectories`, `drift_trajectories`, `removal_trajectories` | Online tasks: how the unique trajectories split into unperturbed, drift-perturbed and agent-removal ones |
| `perturbation_interval`, `perturbation_probability` | Steps between perturbation events; chance that an agent is selected at an event |
| `drift_max_per_event`, `drift_bounds`, `drift_perturb_velocity`, `drift_velocity_bounds` | At most this many agents per event, pushed by uniform noise up to these bounds per axis on position and, if enabled, velocity |
| `train_quadrants`, `train_octants`, `octant_unique_counts` | Regions the starting centers are drawn from, and how many come from each |
| `n_boids`, `perception` | Agents per flock and neighbor radius (shared by expert and GNCA) |
| `goal_waypoints_per_episode`, `goal_bounds`, `start_bounds`, `goal_min_distance` | Online tasks: waypoints per training trajectory, the region goals and starts are drawn from, and the minimum spacing between consecutive goals |
| `goal_arrival_radius` | The expert moves on to the next waypoint once the swarm centroid is this close (0.5) |
| `goal_exclusion_size` | Fixed tasks: starting centers keep this distance from the waypoint path |
| `expert_max_steps` | Step limit of one expert trajectory |
| `loss_type`, `critical_distance`, `distance_weight` | `goal_weighted`: mean squared error with transitions within `critical_distance` of a goal weighted by `distance_weight`; `mse`: unweighted |
| `lr`, `lr_red_factor`, `lr_patience`, `min_lr` | Adam learning rate, reduced by the factor after `lr_patience` epochs without validation improvement, down to `min_lr` |
| `batch_size`, `epochs` | Transitions per optimizer step (global across GPUs); epoch limit per chunk |
| `chunk_size`, `chunk_patience`, `val_set_size` | Trajectories per training chunk; epochs without improvement before a chunk stops; validation trajectories |
| `timestep_stride`, `near_goal_radius` | Use every n-th transition, but every transition within `near_goal_radius` of a goal |
| `generation_workers` | CPU workers for expert generation (0: automatic) |
| `steps_per_execution` | Optimizer steps per compiled call on the GPU (`cloud`) |
| `seed`, `generation_seed`, `weight_init_seed` | Data-generation seed (`seed` in the 2D scripts, `generation_seed` in the 3D ones) and weight-initialization seed; where only the first is set it seeds both. `--seed` sets all of them |
| `run_tag_suffix` | Name part of the outputs; `--label` is appended to it |
| `skip_evaluation` | Skip the training script's own post-training evaluation; use `experiments.run evaluate` instead |

Training runs in chunks: each chunk is a fresh process that generates or loads
its share of the expert trajectories, trains until its validation loss stops
improving, and hands its best weights to the next chunk. Every pipeline
restarts the learning-rate schedule at `lr` for every chunk. The `cloud`
pipelines and the `local` online and fixed-3D pipelines also hand over the
optimizer state; the `local` fixed-2D pipeline starts each chunk with a fresh
optimizer.

## Checkpoint and output names

New training runs write `saved_models/best_weights_<run tag>`, where the run tag
is `<experiment>_<unique>x<repeats>_<loss>[_<expert>]_<suffix>_<label>`: the
experiment name, the number of unique training trajectories and their repeats,
the loss (`goal_weighted_cd2.5_dw2.5` for the near-goal weighted MSE with its
critical distance and weight, or `mse`), the expert's perception radius, crowding
radius and starting box (`p<..>_c<..>_s<..>`) when they differ from the paper's,
the preset's `run_tag_suffix` (the stage name for fixed 3D, `perturbed` for the
online presets), and the `--label` value (`retrain` by default). For example,
`python -m experiments.run train fixed2d --label custom` produces
`saved_models/best_weights_fixed2d_500x2_goal_weighted_cd2.5_dw2.5_custom`, and the
launcher prints the expected path before training starts. The same tag names
the training history under `saved_history/` and the training trajectories or
generation manifest under `saved_training_data/`.

## Agent-removal robustness

```bash
python -m experiments.run removal fixed2d --weights saved_models/best_weights_<run tag>
python -m experiments.run removal fixed3d --weights saved_models/best_weights_<run tag>
python -m experiments.run removal online2d --weights saved_models/best_weights_<run tag>
python -m experiments.run removal online3d --weights saved_models/best_weights_<run tag>
```

The paper's final experiment is **slab removal**, not independent random removal:
at one uniformly sampled step from 500 to 1,000, remove the 10, 25, or 50 agents
nearest a plane through the current centroid. The same random plane normal and
event time are used for GNCA and Boids; the selected indices can differ because
their flocks have evolved differently. An unmodified baseline is also run for
the event-aligned compactness comparison. Radial series are saved. If a rollout
finishes before its scheduled event, the report marks the event as not applied.
Random removal and insertion remain available in the underlying evaluator as
optional extensions, but are not part of the paper presets.

## Swarm-size scaling

```bash
python -m experiments.run scaling fixed2d --weights saved_models/best_weights_<run tag> --seed 457
```

Evaluates a model trained on 100 agents, without retraining, at other swarm
sizes. `--sizes` (default 25 500) start at the training density: the side
of the starting box grows as `0.325 * (N / 100) ** (1 / d)`. `--training-box-sizes`
(default 200 500) place N agents in the training box of side 0.325, at N / 100
times the training density. Each size is a baseline evaluation of the GNCA alone
(`--gnca_only`, no Boids rollouts) with the protocol above, written to `results/scaling/<experiment>_seed<seed>/constant_density/n<N>/`
or `.../training_box/n<N>/`; sizes with an existing `per_run.csv` are skipped, so
an interrupted command can be run again. Evaluate the same model with
`python -m experiments.run evaluate` for N = 100.

## Imitation fidelity

```bash
python -m experiments.run imitation fixed2d --weights saved_models/best_weights_<run tag>
python -m experiments.run imitation online3d --weights saved_models/best_weights_<tag>
```

`evaluation.imitation_error` has two stages. `generate` samples new starts with
the evaluation protocol (balanced over quadrants or octants, fixed-waypoint
starts outside the goal-path exclusion zone) from a random stream separate from
the evaluation starts, and lets the Boids expert drive every rollout, switching
waypoints when the flock centroid comes within 0.5 of the active one. `score` applies the GNCA one step ahead at every
tenth state of each expert rollout, from the expert's state and neighborhood
graph, and compares the prediction with the expert's actual next state; the
GNCA never drives the trajectory, so errors do not accumulate. Dynamic-waypoint
models receive the displacement to the waypoint the expert was pursuing at that
step. The first three lines of the report are the one-step root mean squared
error over the next state, the direction error (angle between predicted and
actual next velocity) and the relative speed error, each as the mean over all
agent-steps with the standard deviation across rollouts. Further lines give
medians, the position error, and the share of agent-steps with a neighbor
inside the crowding radius, with the direction error split by that condition.
Outputs go to
`results/imitation/<experiment>_seed<seed>/`, with the expert rollouts kept in
`expert_rollouts/` so other checkpoints can be scored on the same data with
`python -m evaluation.imitation_error score --rollout_dir ...`.

## Plots

Every evaluation command plots its own results next to them:

| Command | Plots |
| --- | --- |
| `evaluate` | `completion.pdf`, `metrics.pdf`: task completion and per-rollout flock metrics of both controllers |
| `removal` | the same for every removal count, and `recovery.pdf`: median GNCA flock radius around the removal step |
| `scaling` | `scaling.pdf`: GNCA completion and median flock radius against the swarm size |

The plots are drawn by `evaluation/plots.py`; `python -m evaluation.plots --scaling <folder>`
redraws the swarm-size plot of an existing scaling folder. The flock around one
slab removal of a trained fixed 2D model is drawn by

```bash
python -m evaluation.visualization.slab_filmstrip --weights saved_models/best_weights_<run tag> --removed 50
```

which simulates one evaluation start (`--seed`, `--run`) and writes
`results/figures/slab_removal_filmstrip.pdf`. Each command concerns one model;
results of several models are obtained by repeating it with each model's weights.

## Attribution

This work builds on Grattarola, Livi, and Alippi, *Learning Graph Cellular
Automata*, NeurIPS 2021. The original architecture and experiment organization
are available in the [upstream GNCA repository](https://github.com/danielegrattarola/GNCA).
The architecture, relative-message layer, and original Boids/GCA experiment
scaffolding derive from that work. This supplement extends the flocking
experiment with goal-directed navigation, 3D dynamics, changing goals, state
perturbations, efficient data processing, and matched evaluation.

```bibtex
@article{grattarola2021learning,
  title={Learning Graph Cellular Automata},
  author={Grattarola, Daniele and Livi, Lorenzo and Alippi, Cesare},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={20983--20994},
  year={2021}
}
```

The original code of this repository is released under the MIT License
([LICENSE](../LICENSE)). The upstream GNCA repository has no license, so the
code derived from it (the architecture, the relative-message layer, and the
original Boids/GCA experiment scaffolding) remains the work of its authors and is
not relicensed here. Installed dependencies and NVIDIA's base container retain
their own licenses.
