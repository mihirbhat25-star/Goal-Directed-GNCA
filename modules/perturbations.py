"""Shared configuration helpers for supervised Boids perturbation episodes.

The canonical perturbation semantics live in :mod:`modules.boids` and
:mod:`modules.boids_3d`. The opt-in compiled cloud experts mirror those rules
inside Numba kernels so trajectory generation remains GIL-free. This module
only normalizes public configuration and allocates exact trajectory counts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


NORMAL_MODE = "normal"
REMOVAL_MODE = "removal"
DRIFT_MODE = "drift"
PERTURBATION_MODES = (NORMAL_MODE, REMOVAL_MODE, DRIFT_MODE)


@dataclass(frozen=True)
class PerturbationConfig:
    mode: str = NORMAL_MODE
    interval: int = 1
    probability: float = 0.0
    removal_max_per_event: int | None = None
    removal_max_total: int | None = None
    drift_max_per_event: int | None = None
    drift_bounds: tuple[float, ...] = ()
    drift_perturb_velocity: bool = False
    drift_velocity_bounds: tuple[float, ...] | None = None


def normalize_perturbation_config(config, spatial_dims):
    """Validate one episode configuration without imposing an arbitrary cap."""
    if config is None:
        config = {}
    if isinstance(config, PerturbationConfig):
        values = config
    else:
        values = PerturbationConfig(
            mode=str(config.get("mode", NORMAL_MODE)),
            interval=int(config.get("interval", 1)),
            probability=float(config.get("probability", 0.0)),
            removal_max_per_event=_optional_nonnegative_int(
                config.get("removal_max_per_event")
            ),
            removal_max_total=_optional_nonnegative_int(
                config.get("removal_max_total")
            ),
            drift_max_per_event=_optional_nonnegative_int(
                config.get("drift_max_per_event")
            ),
            drift_bounds=tuple(float(value) for value in config.get("drift_bounds", ())),
            drift_perturb_velocity=bool(
                config.get("drift_perturb_velocity", False)
            ),
            drift_velocity_bounds=(
                None if config.get("drift_velocity_bounds") is None else
                tuple(float(value) for value in config["drift_velocity_bounds"])
            ),
        )
    if values.mode not in PERTURBATION_MODES:
        raise ValueError(f"Unknown perturbation mode: {values.mode!r}.")
    if values.interval < 1:
        raise ValueError("Perturbation interval must be at least 1.")
    if not 0.0 <= values.probability <= 1.0:
        raise ValueError("Perturbation probability must be in [0, 1].")
    if values.mode == DRIFT_MODE:
        if len(values.drift_bounds) != spatial_dims:
            raise ValueError(
                f"Drift mode requires {spatial_dims} nonnegative coordinate bounds."
            )
        if any(not np.isfinite(bound) or bound < 0 for bound in values.drift_bounds):
            raise ValueError("Drift bounds must be finite and nonnegative.")
        if values.drift_velocity_bounds is not None:
            if not values.drift_perturb_velocity:
                raise ValueError("Velocity bounds require drift_perturb_velocity=True.")
            if len(values.drift_velocity_bounds) != spatial_dims or any(
                not np.isfinite(bound) or bound < 0 for bound in values.drift_velocity_bounds
            ):
                raise ValueError(f"Velocity drift requires {spatial_dims} finite nonnegative bounds.")
    return values


def _optional_nonnegative_int(value):
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError("Optional perturbation caps must be nonnegative or None.")
    return value


def perturbation_counts(args, trajectory_total=None):
    """Return exact trajectory totals; by default every trajectory is unperturbed."""
    total = int(
        args.train_set_unique if trajectory_total is None else trajectory_total
    )
    supplied = any(
        getattr(args, name, None) is not None
        for name in (
            "normal_trajectories",
            "removal_trajectories",
            "drift_trajectories",
        )
    )
    if not supplied:
        return {NORMAL_MODE: total, REMOVAL_MODE: 0, DRIFT_MODE: 0}
    counts = {
        NORMAL_MODE: int(getattr(args, "normal_trajectories", 0) or 0),
        REMOVAL_MODE: int(getattr(args, "removal_trajectories", 0) or 0),
        DRIFT_MODE: int(getattr(args, "drift_trajectories", 0) or 0),
    }
    if any(count < 0 for count in counts.values()):
        raise ValueError("Perturbation trajectory counts cannot be negative.")
    if sum(counts.values()) != total:
        raise ValueError(
            "--normal_trajectories + --removal_trajectories + "
            "--drift_trajectories must equal the total number of generated "
            "trajectories (unique x repeats for fixed-waypoint training)."
        )
    return counts


def config_for_mode(args, mode):
    """Build the small config passed to either Python or compiled experts."""
    spatial_dims = len(getattr(args, "drift_bounds", ()))
    return normalize_perturbation_config(
        {
            "mode": mode,
            "interval": args.perturbation_interval,
            "probability": args.perturbation_probability,
            "removal_max_per_event": args.removal_max_per_event,
            "removal_max_total": args.removal_max_total,
            "drift_max_per_event": args.drift_max_per_event,
            "drift_bounds": tuple(args.drift_bounds),
            "drift_perturb_velocity": bool(args.drift_perturb_velocity),
            "drift_velocity_bounds": getattr(args, "drift_velocity_bounds", None),
        },
        spatial_dims,
    )


def allocate_mode_chunks(counts, chunk_sizes):
    """Split exact mode totals over predetermined chunk sizes proportionally."""
    remaining = {mode: int(counts.get(mode, 0)) for mode in PERTURBATION_MODES}
    allocations = []
    for requested_size in chunk_sizes:
        requested_size = int(requested_size)
        total_remaining = sum(remaining.values())
        if requested_size < 0 or requested_size > total_remaining:
            raise ValueError("Invalid perturbation chunk size.")
        if requested_size == 0:
            allocations.append({})
            continue
        exact = {
            mode: requested_size * count / total_remaining
            for mode, count in remaining.items()
        }
        allocation = {
            mode: min(remaining[mode], math.floor(exact[mode]))
            for mode in remaining
        }
        while sum(allocation.values()) < requested_size:
            candidates = [
                mode
                for mode in PERTURBATION_MODES
                if allocation[mode] < remaining[mode]
            ]
            selected = max(
                candidates,
                key=lambda mode: (
                    exact[mode] - allocation[mode],
                    remaining[mode],
                ),
            )
            allocation[selected] += 1
        allocation = {mode: count for mode, count in allocation.items() if count}
        allocations.append(allocation)
        for mode, count in allocation.items():
            remaining[mode] -= count
    if sum(remaining.values()):
        raise ValueError("Chunk sizes did not consume every perturbation trajectory.")
    return allocations


def add_perturbation_arguments(parser, spatial_dims):
    """Add the same trajectory-mixture interface to every training entry point."""
    group = parser.add_argument_group("expert perturbations")
    group.add_argument(
        "--normal_trajectories", type=int, default=None,
        help="Exact number of unperturbed expert trajectories.",
    )
    group.add_argument(
        "--removal_trajectories", type=int, default=None,
        help="Exact number of permanent agent-removal trajectories.",
    )
    group.add_argument(
        "--drift_trajectories", type=int, default=None,
        help="Exact number of trajectories with bounded state disturbances and expert recovery.",
    )
    group.add_argument(
        "--perturbation_interval", type=int, default=100,
        help="Apply perturbation trials every this many expert steps.",
    )
    group.add_argument(
        "--perturbation_probability", type=float, default=0.1,
        help="Independent probability that each active agent is selected at an event.",
    )
    group.add_argument(
        "--removal_max_per_event", type=int, default=None,
        help="Optional per-event removal cap; omit for no per-event cap.",
    )
    group.add_argument(
        "--removal_max_total", type=int, default=None,
        help="Optional trajectory-wide removal cap; omit for no total cap.",
    )
    group.add_argument(
        "--drift_max_per_event", type=int, default=None,
        help="Optional per-event drift cap; omit for no per-event cap.",
    )
    group.add_argument(
        "--drift_bounds",
        type=float,
        nargs=spatial_dims,
        default=[0.0] * spatial_dims,
        metavar=tuple(f"D{index}_MAX" for index in range(spatial_dims)),
        help=(
            "Per-coordinate absolute bounds for zero-mean uniform state "
            "position drift. With --drift_perturb_velocity these also bound "
            "velocity noise unless --drift_velocity_bounds is specified."
        ),
    )
    group.add_argument(
        "--drift_perturb_velocity",
        action="store_true",
        help=(
            "Apply independently sampled bounded drift to velocity inputs as "
            "well as position inputs. Defaults to --drift_bounds; override "
            "with --drift_velocity_bounds. Targets are expert updates from the disturbed "
            "state, with no additional label noise."
        ),
    )
    group.add_argument(
        "--drift_velocity_bounds", type=float, nargs=spatial_dims, default=None,
        metavar=tuple(f"V{index}_MAX" for index in range(spatial_dims)),
        help="Per-axis uniform velocity noise bounds; requires --drift_perturb_velocity.",
    )
    return group


def validate_perturbation_args(args, spatial_dims, trajectory_total=None):
    counts = perturbation_counts(args, trajectory_total=trajectory_total)
    if args.perturbation_interval < 1:
        raise ValueError("--perturbation_interval must be at least 1.")
    if not 0.0 <= args.perturbation_probability <= 1.0:
        raise ValueError("--perturbation_probability must be in [0, 1].")
    for name in (
        "removal_max_per_event",
        "removal_max_total",
        "drift_max_per_event",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name} must be nonnegative or omitted.")
    if len(args.drift_bounds) != spatial_dims or any(
        not np.isfinite(value) or value < 0 for value in args.drift_bounds
    ):
        raise ValueError(
            f"--drift_bounds requires {spatial_dims} finite nonnegative values."
        )
    velocity_bounds = getattr(args, "drift_velocity_bounds", None)
    if velocity_bounds is not None:
        if not args.drift_perturb_velocity:
            raise ValueError("--drift_velocity_bounds requires --drift_perturb_velocity.")
        if len(velocity_bounds) != spatial_dims or any(
            not np.isfinite(value) or value < 0 for value in velocity_bounds
        ):
            raise ValueError(f"--drift_velocity_bounds requires {spatial_dims} finite nonnegative values.")
    has_velocity_noise = args.drift_perturb_velocity and any(
        args.drift_bounds if velocity_bounds is None else velocity_bounds
    )
    if counts[DRIFT_MODE] and not (any(args.drift_bounds) or has_velocity_noise):
        raise ValueError("Drift trajectories require at least one nonzero drift bound.")
    return counts
