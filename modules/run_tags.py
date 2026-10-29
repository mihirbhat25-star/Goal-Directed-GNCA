"""Names of training outputs, shared by the training scripts and the launcher.

A run tag reads
``<experiment>_<unique>x<repeats>_<loss>[_rollout<K>][_p<..>_c<..>_s<..>][_<suffix>]``,
for example ``fixed2d_500x2_goal_weighted_cd2.5_dw2.5_seed1``: the experiment, the
number of unique training trajectories and their repeats, the loss (the
near-goal weighted MSE with its critical distance and weight, or the plain MSE),
the rollout length when the model was trained on multi-step rollouts, the Boids
expert's perception radius, crowding radius and starting-box side when they
differ from the paper's (0.1, 0.02, 0.325), and an optional suffix made of the
preset's stage name and the launcher's ``--label``.
The best validation weights of a run are saved as
``saved_models/best_weights_<run tag>``. This module has no third-party
dependencies.
"""

GOAL_WEIGHTED, MSE = "goal_weighted", "mse"
LOSS_TYPES = (MSE, GOAL_WEIGHTED)
PAPER_EXPERT = {"perception": 0.1, "crowding": 0.02, "init_scatter": 0.325}


def expert_tag(perception=0.1, crowding=0.02, init_scatter=0.325):
    """Empty for the paper's expert settings, otherwise ``p<..>_c<..>_s<..>``."""
    values = (float(perception), float(crowding), float(init_scatter))
    if values == tuple(PAPER_EXPERT.values()):
        return ""
    return f"p{values[0]:g}_c{values[1]:g}_s{values[2]:g}"


def loss_tag(loss_type, critical_distance, distance_weight):
    if loss_type == GOAL_WEIGHTED:
        return f"{GOAL_WEIGHTED}_cd{float(critical_distance):g}_dw{float(distance_weight):g}"
    return MSE


def run_tag(experiment, unique, repeats, loss_type, critical_distance, distance_weight,
            suffix="", expert=""):
    parts = [experiment, f"{int(unique)}x{int(repeats)}",
             loss_tag(loss_type, critical_distance, distance_weight),
             str(expert or ""),
             str(suffix or "").strip("_")]
    return "_".join(part for part in parts if part)
