# Usage

[Project quick start](../README.md) · [Experiment details](EXPERIMENT_DETAILS.md) · [Code map](ARCHITECTURE.md)

Run every command from the repository root.

## Installation

### NVIDIA GPU container

On a Linux GPU host with Docker and the NVIDIA Container Toolkit:

```bash
docker build -f environment/Dockerfile -t gnca .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
  -v "$PWD:/workspace/gnca" -w /workspace/gnca gnca
```

Inside the container:

```bash
python -m experiments.check_environment --smoke
```

The Dockerfile uses the NVIDIA TensorFlow `25.02-tf2-py3` image. Do not install the
generic TensorFlow wheel over that image; `environment/requirements-ngc.txt` omits
TensorFlow on purpose. Check the GPU list printed by `check_environment` before training.

### Native Python

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m experiments.check_environment --smoke
```

Without a GPU, training and evaluation run on one CPU device, which suits evaluation
and short tests. `environment/requirements-python38.txt` records the TensorFlow 2.4 stack
the code is also tested on.

## Training

```bash
python -m experiments.run train fixed2d
python -m experiments.run train fixed2d --dry-run
```

| Option | Meaning |
| --- | --- |
| `--label L` | Appended to the run names |
| `--stage both\|onestep\|bptt` | Run both stages (default), only pretraining, or only the fine-tune |
| `--init-weights P` | With `--stage bptt`: the checkpoint prefix of the one-step model |
| `--generation-workers N` | CPU threads of the expert generation (default: CPU count minus two) |
| `--dry-run` | Print each stage's plan without running it |

Training runs on all TensorFlow-visible GPUs; use `CUDA_VISIBLE_DEVICES` to restrict
them. The batch sizes are global and must be divisible by the GPU count. Each chunk is a
fresh process that generates its expert trajectories into RAM, so its exit releases all
memory.

Outputs, for `fixed3d`:

| Path | Contents |
| --- | --- |
| `saved_models/fixed3d_s0_onestep` | Stage 1 checkpoint: the best epoch of its last chunk |
| `saved_models/fixed3d_s0` | Final checkpoint, after stage 2 |
| `saved_history/<run name>.json` | The stage's settings, and per chunk its loss and learning-rate history and generated trajectories |

Existing checkpoints are not overwritten; choose a new `--label`.

## Preset files

Each experiment's hyperparameters are one JSON file in `experiments/presets/`. Edit it to
change a run. JSON allows no comments, so the fields are explained here.

**`expert`**

| Field | Meaning |
| --- | --- |
| `n_boids`, `perception`, `crowding` | Agents per flock, neighbor radius (shared by expert and GNCA), separation radius |
| `init_scatter` | Side of the starting box around each sampled center |
| `start_bounds` | Box the starting centers are drawn from, split into quadrants or octants |
| `goal_arrival_radius` | The expert moves on to the next waypoint once the centroid is this close |
| `expert_max_steps` | Step limit of one expert trajectory |
| `region_order` | How a chunk's tasks are dealt to the regions: `cycle` in turn, `blocks` one region after another. Counts are equal either way |
| `goal_exclusion_size` | Fixed tasks: starting centers keep this distance from the waypoint path |
| `waypoints_per_episode`, `goal_bounds`, `goal_min_distance` | Online tasks: waypoints per training trajectory, the box goals are drawn from, the minimum spacing between consecutive goals |

**`drift`**: the perturbation of perturbed trajectories. Every `interval` steps each agent
is selected with `probability`, at most `max_agents` per event, and receives uniform noise
up to `position_bound` and `velocity_bound` per axis.

**`training`**: shared by both stages.

| Field | Meaning |
| --- | --- |
| `epochs` | Epoch limit per chunk |
| `critical_distance`, `distance_weight` | Goal-weighted loss: the radius and the upweighting factor |
| `lr_red_factor`, `lr_patience`, `min_lr` | The learning rate is multiplied by the factor after this many epochs without validation improvement, down to the floor |
| `chunk_patience` | Epochs without improvement before a chunk stops |
| `early_stopping_min_delta` | Smallest validation improvement that counts |
| `val_set_size` | Validation trajectories per chunk |
| `steps_per_execution` | Optimizer steps per compiled call |

**`onestep`, `bptt`**: one block per stage.

| Field | Meaning |
| --- | --- |
| `trajectories`, `perturbed` | Expert trajectories of the stage, and how many of them are perturbed |
| `chunk_size`, `batch_size` | Trajectories per chunk; samples per optimizer step, global across GPUs |
| `lr`, `grad_clip_norm` | Learning rate every chunk starts at; global gradient norm limit (0: none) |
| `timestep_stride`, `near_goal_radius` | `onestep`: use every n-th transition, but every transition within this radius of the active or just-left goal |
| `window`, `data_seed_offset` | `bptt`: rollout steps K per sample; keeps this stage's data separate from stage 1's |

**`evaluation`**: defaults of the evaluation commands, `centers_per_region`, `max_steps`,
`seed`, `success_threshold`, `online_goal_count`, `goal_min_distance`.

## Evaluation

```bash
python -m experiments.run evaluate fixed2d --weights saved_models/fixed2d_s0
python -m experiments.run removal fixed2d --weights saved_models/fixed2d_s0
python -m experiments.run scaling fixed2d --weights saved_models/fixed2d_s0
python -m experiments.run imitation fixed2d --weights saved_models/fixed2d_s0
```

`--weights` takes a checkpoint prefix, without `.index`. All four commands take the expert
settings from the experiment's preset and accept `--max-steps` and `--output-dir`;
all but `imitation` accept `--centers-per-region` and `--workers`. For a quick plumbing check
use `--centers-per-region 1 --max-steps 5`; such a short run is not a task-completion
experiment.

`evaluate` writes to `results/evaluate/<experiment>_seed460/`. Every GNCA rollout is
paired with a Boids rollout from the same initial states and goal sequence. Task completion
is reported separately from the continuous flock metrics; no compactness threshold decides
completion. Besides the mean, minimum, maximum, median and standard deviation over time of
the radial score, polarization and speed CV, each rollout records the mean neighbor count
(`gnca_mean_neighbors`), its peak (`gnca_max_neighbors`) and its mean over the last 20% of the
frames (`gnca_late_mean_neighbors`). `series_report.txt` reports the radius and neighbor count
over time, by step and by fifth of each rollout, and is also printed;
`python -m evaluation.series_report <folder>/radius_series.npz` writes it again.

`removal` adds the slab removal conditions: at one uniformly sampled step from 500 to 1,000,
remove the 50 agents nearest a plane through the current centroid. The same random
plane normal and event time are used for GNCA and Boids; the selected agents can differ
because their flocks have evolved differently. If a rollout finishes before its event, the
report marks the event as not applied. `evaluation.paired_rollouts` itself also offers
uniformly random removal (`--removal_pattern random`) and insertion
(`--conditions baseline insertion`).

`scaling` evaluates the GNCA alone at other swarm sizes; `--with-boids` also rolls out the expert from the same starts. `--sizes` (default 25 200) start at the
training density: the side of the starting box grows as `init_scatter * (N / 100) ** (1 / d)`.
`--training-box-sizes` (default 250) place N agents in the training box. Sizes with an
existing `per_run.csv` are skipped, so an interrupted command can be run again.
`python -m evaluation.plots --scaling <folder>` redraws `scaling.pdf`.

`imitation` runs both stages of `evaluation.imitation_error`. `generate` samples new starts from
a random stream separate from the evaluation starts and lets the Boids expert drive every
rollout. `score` applies the GNCA one step ahead at every tenth state, from the expert's state
and neighborhood graph, so errors do not accumulate. The report opens with the one-step root mean
squared error over the next state, the angle between predicted and actual next velocity, and the
relative speed error, each as the mean over all agent-steps with the standard deviation across
rollouts. The expert rollouts stay in `expert_rollouts/`, so other checkpoints can be scored on
the same data with `python -m evaluation.imitation_error score --rollout_dir ...`.

## Figures

| Command | Output |
| --- | --- |
| `python -m evaluation.visualization.trajectories <experiment> --weights <prefix> --mode <mode>` | GNCA trajectories in `results/figures/`. Modes: `tubular` (mean path with tube; fixed tasks), `per_boid`, `multi_tubular`, `individual` (ranked by tube radius) |
| `python -m evaluation.visualization.slab_filmstrip --weights <prefix> --removed 50` | The flock around one slab removal of a fixed 2D model |
| `python -m evaluation.neighbor_counts --experiment <experiment> --weights <prefix> --n_boids 25 --output_dir <folder>` | Neighbor counts along GNCA rollouts at a given swarm size |
| `python -m evaluation.visualization.show_boids --experiment <experiment>` | Live animation of the Boids expert, with the preset's settings; `--waypoints` and `--start` are optional |

The two animations in `docs/media` illustrate the README.

## Attribution

This work builds on Grattarola, Livi, and Alippi, *Learning Graph Cellular
Automata*, NeurIPS 2021. The original architecture and experiment organization
are available in the [upstream GNCA repository](https://github.com/danielegrattarola/GNCA).
The architecture, relative-message layer, and original Boids/GCA experiment
scaffolding derive from that work. This code extends the flocking experiment with
goal-directed navigation, 3D dynamics, changing goals, state perturbations, efficient
data processing, and matched evaluation.

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
