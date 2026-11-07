# Experiment details

[Project quick start](../README.md) · [Usage](USAGE.md) · [Code map](ARCHITECTURE.md)

## The recipe

All four experiments use 100 agents, Adam, the dual-aggregation GNCA with 256 hidden units, and
two stages. The Boids expert's movement limits are not imposed on GNCA predictions; the model's
final MLP learns them.

| Setting | `fixed2d` | `online2d` | `fixed3d` | `online3d` |
| --- | --- | --- | --- | --- |
| Expert perception / crowding / start box | 0.05 / 0.02 / 0.325 | 0.05 / 0.02 / 0.325 | 0.105 / 0.06 / 0.4 | 0.105 / 0.06 / 0.4 |
| Stage 1 trajectories (perturbed) | 2,500 (833) | 2,500 (833) | 15,000 (5,000) | 15,000 (5,000) |
| Stage 1 chunk size / batch | 500 / 256 | 500 / 256 | 1,000 / 1,024 | 1,000 / 1,024 |
| Stage 1 timestep stride | 1 | 1 | 1 | 1 |
| Stage 2 trajectories (perturbed) / chunk size | 500 (0) / 500 | 500 (0) / 500 | 3,000 (0) / 1,000 | 3,000 (0) / 1,000 |
| Stage 2 window K / batch | 16 / 256 | 16 / 256 | 64 / 512 | 32 / 512 |
| Validation trajectories per chunk | 10 | 10 | 10 | 10 |
| Minimum learning rate | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| Drift interval / velocity bound | 250 / 0.005 | 250 / 0.005 | 500 / 0.01 | 500 / 0.01 |

In every experiment and stage: chunk patience 10 epochs, learning-rate patience 4 epochs, learning-rate
factor 0.1, at most 10,000 epochs per chunk. Stage 1 runs at learning rate 3e-4 without gradient clipping.
Stage 2 runs at 3e-5 with the global gradient norm clipped to 1.0.

**Chunks.** A chunk holds equal counts per start region (quadrants in 2D, octants in 3D), the perturbed
share split proportionally over the chunks, and draws its own validation episodes. It starts from the
previous chunk's best weights and Adam moments, with the learning rate restarted at the stage's rate: the
annealed rate encodes a plateau on the previous chunk's data, not convergence. The exported model is the
best epoch of the last chunk. Batch size counts graphs (stage 1) or windows (stage 2), not trajectories,
and changing the chunk size changes the sequence of early-stopping decisions, not only memory use.

**One-step stage.** A sample is a single expert transition. Where the stride exceeds 1, transitions
further than 1.0 from the active and the just-left goal are subsampled by it.

**BPTT stage.** The stage starts from the one-step weights with a fresh optimizer. The scale of the summed
term is a weight of the checkpoint, so it stays at the value computed on stage 1's first chunk. A sample is
a window of K consecutive transitions: the model starts from the expert state, feeds on its own
predictions, the graph is rebuilt from them each step, and the loss is averaged over the window with
gradients through all steps. Windows tile each trajectory without overlap. Windows with a perturbation
after their first step are dropped. In fixed-waypoint tasks windows end at the waypoint switches, which the
model cannot observe; online models see the switch through their goal input.

**Loss.** Mean squared error between predicted and expert next state, multiplied by 2.5 for
a graph whose flock centroid is within 2.5 of the active goal, or of the previous goal while the running
maximum distance from it since the switch stays below 2.5.

**Model.** The state branch averages neighbor features of the full input. The relative branch has two
message functions over each neighbor's offset and distance, one summed and one averaged. The summed
term is scaled by one over the mean neighbor count of the training graphs, computed on the first chunk.

**Perturbations.** Each selected agent receives independent uniform noise on position and velocity; the
expert then evolves the disturbed state, so the target is its response from there.

## Goals

Evaluation and the expert use the same rule: a waypoint is reached when the **centroid distance** to it is at
most 0.5. The fixed waypoints are `(3,-3), (3,3), (0,0)` in 2D and `(-4,-4,-4), (4,4,4)` in 3D. Online goals are
uniform within `[-4.5,4.5]^d`, rejection-sampled to be at least 2.5 units from the previous goal (the initial
centroid for the first goal). Training trajectories visit two online goals, evaluation rollouts five.

## Metrics and failures

For positions p_i and velocities v_i at each observed finite step:

- `r(t)` is the 95th percentile across agents of `||p_i - mean(p)||`.
- Polarization is the norm of the mean unit velocity over moving agents.
- Speed CV is the population standard deviation of moving-agent speeds divided by their mean. An
  all-stationary frame has undefined polarization and CV, saved as NaN rather than interpreted as good
  alignment or zero variation.
- The neighbor count of a frame is the number of agents within the perception radius of each agent,
  itself excluded, averaged over agents.

Each rollout saves temporal mean, minimum, maximum, median and population standard deviation.
`paper_summary.csv` aggregates temporal means across rollouts; it is not a percentile over pooled agent
positions or timesteps.

No compactness cutoff is used. Failure labels are `timeout` (goals incomplete at the horizon) and
`instability` (a non-finite predicted state). Metrics for a failed rollout describe only its observed finite
prefix; read them alongside failure counts and observed steps. GNCA and expert share initial states and goals
but are not forced to reach goals at the same time.
