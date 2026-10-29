"""
Shows an animation of the Boids system.

The flock visits the fixed waypoints of the paper unless ``--waypoints`` lists
other ones, each as comma-separated coordinates inside the canvas [-5, 5]:

    python -m boids.show_boids                                  # 2D, fixed waypoints
    python -m boids.show_boids --dimension 3                    # 3D, fixed waypoints
    python -m boids.show_boids --waypoints 2,-3 -2,3 0,0        # 2D, your waypoints
    python -m boids.show_boids --dimension 3 --waypoints 3,3,3 -3,0,2 --start -4,-4,-4
"""
import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from modules.boids import Boids
from modules.boids_3d import Boids3D


COORDINATES = re.compile(r"^-?(\d+\.?\d*|\.\d+)(,-?(\d+\.?\d*|\.\d+))+$")


def point(text):
    try:
        return np.array([float(value) for value in text.strip().split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}")


parser = argparse.ArgumentParser(description="Show an animation of the Boids system.")
parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
parser.add_argument("--waypoints", type=point, nargs="+",
                    help="Waypoints visited in order, e.g. 2,-3 -2,3 0,0 (default: the fixed waypoints)")
parser.add_argument("--start", type=point,
                    help="Corner of the starting box of the flock (default: random)")
# A point such as -2,3 would otherwise be read as an option; a leading space keeps it a value.
args = parser.parse_args([" " + arg if COORDINATES.match(arg) else arg for arg in sys.argv[1:]])

for name, values in (("waypoint", args.waypoints or []), ("start", [args.start] if args.start is not None else [])):
    for value in values:
        if value.shape != (args.dimension,):
            parser.error(f"each {name} needs {args.dimension} coordinates, got {value.tolist()}")
        if np.any(np.abs(value) > 5):
            parser.error(f"{name} {value.tolist()} lies outside the canvas [-5, 5]")

Simulator = Boids if args.dimension == 2 else Boids3D
boids = Simulator(
    min_speed=0.0001,
    max_speed=0.01,
    max_force=0.1,
    max_turn=5 if args.dimension == 2 else 15,
    perception=0.1,
    crowding=0.02,
    n_boids=100,
    dt=1,
    canvas_scale=1,
    boundary_size_pctg=0.2,
    wrap=False,
    show=True,
)
if args.waypoints:
    boids.canonical_goal_positions = np.array(args.waypoints)

start = args.start if args.start is not None else boids.get_random_init(boids.n_boids, save_config=False)[3]
boids.generate_trajectory(save_config=False, random_init=start)
plt.ioff()
plt.show()
