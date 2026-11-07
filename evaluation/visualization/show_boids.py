"""
Shows an animation of the Boids system with the expert settings of an experiment's preset.

The flock visits the fixed waypoints unless ``--waypoints`` lists other ones, each as
comma-separated coordinates inside the canvas [-5, 5]:

    python -m evaluation.visualization.show_boids                              # 2D, fixed waypoints
    python -m evaluation.visualization.show_boids --experiment fixed3d         # 3D, fixed waypoints
    python -m evaluation.visualization.show_boids --waypoints 2,-3 -2,3 0,0    # 2D, your waypoints
    python -m evaluation.visualization.show_boids --experiment fixed3d --waypoints 3,3,3 -3,0,2 --start -4,-4,-4
"""
import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from modules.experts import reference_expert
from modules.presets import NAMES, load_preset


COORDINATES = re.compile(r"^-?(\d+\.?\d*|\.\d+)(,-?(\d+\.?\d*|\.\d+))+$")


def point(text):
    try:
        return np.array([float(value) for value in text.strip().split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Show an animation of the Boids system.")
    parser.add_argument("--experiment", choices=NAMES, default="fixed2d",
                        help="Selects the dimension and the expert settings from its preset.")
    parser.add_argument("--waypoints", type=point, nargs="+",
                        help="Waypoints visited in order, e.g. 2,-3 -2,3 0,0 (default: the fixed waypoints)")
    parser.add_argument("--start", type=point,
                        help="Corner of the starting box of the flock (default: random)")
    # A point such as -2,3 would otherwise be read as an option; a leading space keeps it a value.
    args = parser.parse_args([" " + arg if COORDINATES.match(arg) else arg for arg in argv])
    args.dimension = load_preset(args.experiment)["dimension"]
    for name, values in (("waypoint", args.waypoints or []), ("start", [args.start] if args.start is not None else [])):
        for value in values:
            if value.shape != (args.dimension,):
                parser.error(f"each {name} needs {args.dimension} coordinates, got {value.tolist()}")
            if np.any(np.abs(value) > 5):
                parser.error(f"{name} {value.tolist()} lies outside the canvas [-5, 5]")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    expert = load_preset(args.experiment)["expert"]
    boids = reference_expert(args.dimension, n_boids=expert["n_boids"], perception=expert["perception"],
                             crowding=expert["crowding"], init_scatter=expert["init_scatter"], show=True)
    if args.waypoints:
        boids.canonical_goal_positions = np.array(args.waypoints)
    start = args.start if args.start is not None else np.random.uniform(-5.0, 5.0 - expert["init_scatter"], args.dimension)
    boids.generate_trajectory(*boids.initial_state(start))
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
