"""Start regions: the 2^d orthants of a box, the quadrants in 2D and the octants in 3D.

Regions 0 to 3 go counterclockwise from the positive corner of the x-y plane; in 3D
they lie above the z midpoint and regions 4 to 7 repeat them below it.
"""

import numpy as np


def region_count(dimension):
    return 2 ** int(dimension)


def region_bounds(bounds, region):
    """Flat lower/upper pairs of one region of the box given by flat lower/upper pairs."""
    pairs = np.asarray(bounds, dtype=np.float64).reshape(-1, 2)
    region = int(region)
    upper_half = [region % 4 in (0, 3), region % 4 in (0, 1), region < 4][: len(pairs)]
    selected = []
    for (lower, upper), positive in zip(pairs, upper_half):
        middle = (lower + upper) / 2.0
        selected.extend((middle, upper) if positive else (lower, middle))
    return tuple(float(value) for value in selected)
