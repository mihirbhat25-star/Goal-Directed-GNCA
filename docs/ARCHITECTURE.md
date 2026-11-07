# Code map

[Project quick start](../README.md) · [Usage](USAGE.md) · [Experiment details](EXPERIMENT_DETAILS.md)

2D and 3D share every piece of code except the expert implementations, which stay a pair
behind one interface. They differ only in the values of `experiments/presets`.

```text
experiments/
  run.py                  list, train, evaluate, removal, scaling, imitation
  check_environment.py    Dependencies, GPU visibility, one forward step of each model
  presets/                The hyperparameters and expert settings of the four experiments
models/gnca.py            The GNCA, and building it and loading a checkpoint into it
layers/pairwise_edge_conv.py   Its relative-position messages, summed or averaged
modules/
  boids.py, boids_3d.py   The reference Python experts
  experts.py              One interface over that pair
  waypoints.py            Random waypoint sequences, goal-conditioned inputs
  perturbations.py        Drift settings per episode, exact counts per chunk
  regions.py              Quadrants and octants as the 2^d orthants of a box
  presets.py, run_names.py
training/
  train.py                The stage driver and chunk loop
  worker.py               One chunk in its own process
  data.py                 Chunk allocation, episode configurations, compiled generation
  compiled_expert_2d.py, compiled_expert_3d.py   The Numba experts
  batches.py              Bit-packed one-step batches and rollout windows
  loss.py                 The goal-weighted loss
  rollout.py              The K-step objective, with graphs rebuilt from predictions
  fit.py                  The synchronous multi-GPU loop, with a CPU fallback for tests
  native_output.py        Removes known library noise from stderr
evaluation/
  paired_rollouts.py      Matched GNCA and Boids rollouts, removal and insertion
  parallel.py             Splits an evaluation across worker processes
  series_report.py        Radius and neighbor count over time
  plots.py                Completion, metrics, recovery and swarm-size plots
  imitation_error.py      One-step error on held-out expert rollouts
  neighbor_counts.py      Neighbor counts along GNCA rollouts
  metrics/                Ordered waypoint scoring, median and quartile summaries
  visualization/          Trajectory figures, the slab filmstrip, the expert animation
environment/              Container and Python dependencies
```

## The path of one training run

`experiments.run train` asks `training.train` for one settings dictionary per stage, built from
the preset, and runs the stages in order. For each chunk of a stage, `training.train` starts
`training.worker`, which

1. generates the chunk's trajectories and validation episodes with `training.data`, which picks
   the compiled expert of the dimension;
2. packs them into batches with `training.batches`, one-step or rollout according to the stage;
3. restores the previous chunk's best weights and Adam state, or the stage's starting weights,
   and restarts the learning rate;
4. trains with `training.fit`, on the objective of `training.loss` and, in stage 2, `training.rollout`;
5. exports its best epoch to the run's checkpoint, so the last chunk's best epoch is the
   stage's model.

Data is laid out alike for all experiments. Every trajectory carries the active goal, the previous
goal and the running maximum distance from the previous goal, whether or not the model sees a goal.
Fixed-task models receive positions and velocities; online models also receive goal minus position.
The one loss reads those columns.

## Expert implementations

`training/compiled_expert_*.py` generate the training data. They mirror the equations of
`modules/boids*.py`, which the evaluation rolls out next to the GNCA. Their kernels avoid
inter-process copying, retain bit-packed adjacency and release the GIL, so a thread pool shares the
results in RAM. Generation stays on the CPU; adding GPUs does not speed it up.
