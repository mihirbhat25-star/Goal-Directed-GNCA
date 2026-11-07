"""The experiment presets. Standard library only, so the launcher runs before TensorFlow is installed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("fixed2d", "online2d", "fixed3d", "online3d")


def load_preset(experiment):
    with (ROOT / "experiments" / "presets" / f"{experiment}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def goal_conditioned(preset):
    """Online-goal models receive the goal displacement; fixed-waypoint models do not."""
    return preset["task"] == "online_goals"
