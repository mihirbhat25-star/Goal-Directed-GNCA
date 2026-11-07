"""Names of training outputs.

One recipe exists, so a name only tells the experiment, the seed and the stage:
``fixed3d_s0_onestep`` after one-step pretraining and ``fixed3d_s<S>`` after the BPTT
fine-tune, followed by the launcher's ``--label`` when given. Weights are saved as
``saved_models/<name>`` and the training record as ``saved_history/<name>.json``.
"""


def run_name(experiment, seed, stage, label=""):
    parts = [experiment, f"s{int(seed)}", "onestep" if stage == "onestep" else "", label]
    return "_".join(part for part in parts if part)
