"""Shared continuous flock metrics and online-evaluation reporting."""

from __future__ import annotations

import csv
import json
import os

import numpy as np


TEMPORAL_STATISTICS = ("mean", "max", "min", "median", "std")
TEMPORAL_SERIES = (
    ("radius", "r(t), 95th-agent radius"),
    ("polarization", "polarization"),
    ("speed_cv", "speed CV"),
)
TEMPORAL_METRICS = tuple(
    (f"{stat}_{key}", f"temporal {stat} {label}")
    for key, label in TEMPORAL_SERIES
    for stat in TEMPORAL_STATISTICS
)
MOTION_METRICS = (
    ("max_goal_tube_radius", "maximum per-goal tube radius r"),
    *TEMPORAL_METRICS,
    ("p05_polarization", "5th temporal percentile polarization"),
    ("p95_speed_cv", "95th temporal percentile speed CV"),
)


class OrderedWaypointProgress:
    """Count centroid visits only while the corresponding goal is active.

    At most one goal is credited per frame, so arrival steps are strictly
    increasing even when goal neighborhoods overlap. Passing an inactive
    waypoint does not give credit that can be redeemed later.
    """

    def __init__(self, goals, threshold=0.5):
        self.goals = np.asarray(goals, dtype=np.float64)
        if (self.goals.ndim != 2 or len(self.goals) == 0
                or not np.all(np.isfinite(self.goals))):
            raise ValueError("goals must be a nonempty, finite coordinate array")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("goal threshold must be finite and positive")
        self.threshold = float(threshold)
        self.reached = 0
        self.arrival_steps = []
        self.first_hit_steps = np.full(len(self.goals), -1, dtype=int)
        self.closest_distances = np.full(len(self.goals), np.inf)
        self._last_step = -1

    @property
    def completed(self):
        return self.reached == len(self.goals)

    def update(self, centroid, step):
        if self.completed:
            return True
        if step <= self._last_step:
            raise ValueError("Waypoint progress needs strictly increasing frame indices")
        self._last_step = int(step)
        centroid = np.asarray(centroid, dtype=np.float64)
        if centroid.shape != self.goals.shape[1:]:
            raise ValueError("Centroid and goal dimensions differ")
        if not np.all(np.isfinite(centroid)):
            return False
        active = self.reached
        distance = float(np.linalg.norm(centroid - self.goals[active]))
        self.closest_distances[active] = min(self.closest_distances[active], distance)
        if distance <= self.threshold:
            self.first_hit_steps[active] = int(step)
            self.arrival_steps.append(int(step))
            self.reached += 1
        return self.completed


def ordered_waypoint_progress(trajectory, goals, threshold=0.5):
    """Score an observed physical-state trajectory using ordered centroid visits."""
    progress = OrderedWaypointProgress(goals, threshold)
    dimension = progress.goals.shape[1]
    for step, state in enumerate(np.asarray(trajectory)):
        if not np.all(np.isfinite(state)):
            break
        if progress.update(state[:, :dimension].mean(axis=0), step):
            break
    return progress


def robust_tube_radius(trajectory, spatial_dims=2):
    """Return the 99th temporal percentile of the 95th agent radius."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 3 or trajectory.shape[-1] < spatial_dims:
        raise ValueError(
            "trajectory must have shape (steps, agents, features) with at "
            f"least {spatial_dims} features; got {trajectory.shape}"
        )
    positions = trajectory[..., :spatial_dims]
    with np.errstate(invalid="ignore"):
        centroid = np.nanmean(positions, axis=1)
        distances = np.linalg.norm(positions - centroid[:, None, :], axis=-1)
        frame_radii = np.nanpercentile(distances, 95, axis=1)
        return float(np.nanpercentile(frame_radii, 99))


def goal_segment_tube_radii(trajectory, arrival_steps, spatial_dims=2):
    """Return one robust tube radius for each active-goal segment."""
    final_frame = len(trajectory) - 1
    segment_start = 0
    radii = []
    for arrival_step in arrival_steps:
        segment_stop = min(max(int(arrival_step), segment_start), final_frame)
        radii.append(
            robust_tube_radius(
                trajectory[segment_start : segment_stop + 1],
                spatial_dims=spatial_dims,
            )
        )
        segment_start = segment_stop
    if segment_start < final_frame or not radii:
        radii.append(
            robust_tube_radius(
                trajectory[segment_start:], spatial_dims=spatial_dims
            )
        )
    return radii


def load_statistics(cache_path, run_tag, dimensions, goals):
    """Load cached trajectories and compute per-run tube-radius statistics."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Trajectory cache not found: {cache_path}")
    with np.load(cache_path, allow_pickle=True) as cache:
        trajectories = cache["trajs"]
        centers = cache["centers"] if "centers" in cache.files else None

    if trajectories.ndim != 4 or trajectories.shape[-1] < dimensions:
        raise ValueError(
            "Expected trajectories shaped (runs, steps, boids, features), "
            f"got {trajectories.shape}"
        )
    if len(trajectories) == 0:
        raise ValueError(f"Trajectory cache is empty: {cache_path}")
    if centers is not None and len(centers) != len(trajectories):
        raise ValueError(
            "Trajectory cache has different numbers of centers and trajectories"
        )

    run_stats = []
    for run_index, trajectory in enumerate(trajectories):
        centroid = trajectory[:, :, :dimensions].mean(axis=1)
        distances = np.linalg.norm(
            trajectory[:, :, :dimensions] - centroid[:, None, :], axis=-1
        )
        radius = robust_tube_radius(trajectory, spatial_dims=dimensions)
        center = centers[run_index] if centers is not None else centroid[0]
        run_stats.append((run_index, radius, centroid, distances, center))
    run_stats.sort(key=lambda item: item[1], reverse=True)
    radii = np.asarray([item[1] for item in run_stats], dtype=np.float64)
    return {
        "tag": run_tag,
        "dimensions": dimensions,
        "goals": np.asarray(goals, dtype=np.float32),
        "trajs": trajectories,
        "run_stats": run_stats,
        "min": float(radii.min()),
        "max": float(radii.max()),
        "mean": float(radii.mean()),
        "median": float(np.median(radii)),
        "std": float(radii.std()),
    }


def nearest_runs(target_run, all_runs, count=5):
    """Return the nearest starts to a target run, excluding that run."""
    target_index, _, target_centroid, _, _ = target_run
    target_start = target_centroid[0]
    candidates = []
    for run_index, radius, centroid, distances, center in all_runs:
        if run_index == target_index:
            continue
        separation = float(np.linalg.norm(target_start - centroid[0]))
        candidates.append(
            (separation, run_index, radius, centroid, distances, center)
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[:count]


def print_summary(stats, threshold):
    """Print a concise tube-radius report."""
    radii = np.asarray([run[1] for run in stats["run_stats"]])
    blowups = int(np.count_nonzero(radii >= threshold))
    print(f"Run: {stats['tag']}")
    print(
        f"r min/mean/median/max/std: {stats['min']:.6f} / "
        f"{stats['mean']:.6f} / {stats['median']:.6f} / "
        f"{stats['max']:.6f} / {stats['std']:.6f}"
    )
    print(f"Blowups (r >= {threshold:g}): {blowups}/{len(radii)}")


def write_blowup_csv(stats, output_path, threshold=2.0, neighbors=5):
    """Write blowups and nearest-start radii to a CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    neighbor_columns = [f"neighbor_{index + 1}_r" for index in range(neighbors)]
    fieldnames = [
        "run", "start_center", "r_score", "nearest_start_center",
        "nearest_distance", *neighbor_columns,
    ]
    rows = []
    for run in stats["run_stats"]:
        run_index, radius, centroid, _, _ = run
        if radius < threshold:
            continue
        nearest = nearest_runs(run, stats["run_stats"], count=neighbors)
        row = {
            "run": run_index,
            "start_center": np.array2string(centroid[0], precision=6),
            "r_score": f"{radius:.6f}",
            "nearest_start_center": "",
            "nearest_distance": "",
        }
        if nearest:
            row["nearest_start_center"] = np.array2string(
                nearest[0][3][0], precision=6
            )
            row["nearest_distance"] = f"{nearest[0][0]:.6f}"
        for column, neighbor in zip(neighbor_columns, nearest):
            row[column] = f"{neighbor[2]:.6f}"
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} blowup rows to {output_path}")
    return output_path


def trajectory_metric_series(trajectory, spatial_dims):
    """Return instantaneous flock metrics, not per-goal/time quantiles.

    A non-finite frame is excluded for the affected metric. Agents with speed
    <= 1e-12 are excluded from polarization and speed CV; a frame with no
    moving agents has undefined motion metrics.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 3 or trajectory.shape[-1] < 2 * spatial_dims:
        raise ValueError("Expected (frames, agents, position+velocity features).")
    series = {key: np.full(len(trajectory), np.nan) for key, _ in TEMPORAL_SERIES}
    if trajectory.shape[1] == 0:
        return series
    positions = trajectory[:, :, :spatial_dims]
    finite_positions = np.all(np.isfinite(positions), axis=(1, 2))
    p = positions[finite_positions]
    if len(p):
        distances = np.linalg.norm(p - p.mean(axis=1)[:, None, :], axis=-1)
        series["radius"][finite_positions] = np.percentile(distances, 95, axis=1)
    velocity = trajectory[:, :, spatial_dims : 2 * spatial_dims]
    speed = np.linalg.norm(velocity, axis=-1)
    valid = np.all(np.isfinite(velocity), axis=-1) & (speed > 1e-12)
    finite_velocity = np.all(np.isfinite(velocity), axis=(1, 2))
    usable = finite_velocity & np.any(valid, axis=1)
    if np.any(usable):
        v, s, mask = velocity[usable], speed[usable], valid[usable]
        unit_velocity = np.divide(v, s[..., None], out=np.zeros_like(v),
                                  where=mask[..., None])
        series["polarization"][usable] = np.linalg.norm(
            unit_velocity.sum(axis=1) / mask.sum(axis=1)[:, None], axis=-1
        )
        valid_speed = np.where(mask, s, np.nan)
        series["speed_cv"][usable] = (
            np.nanstd(valid_speed, axis=1) / np.nanmean(valid_speed, axis=1)
        )
    return series


def trajectory_motion_metrics(trajectory, spatial_dims):
    """Temporal statistics on each frame, including the initial state once."""
    series = trajectory_metric_series(trajectory, spatial_dims)
    metrics = {"observed_frames": len(trajectory)}
    for key, values in series.items():
        stats = _finite_summary(values)
        metrics[f"valid_{key}_frames"] = stats["n"]
        metrics.update({f"{stat}_{key}": stats[stat] for stat in TEMPORAL_STATISTICS})
    for key, percentile, name in (
        ("polarization", 5, "p05_polarization"),
        ("speed_cv", 95, "p95_speed_cv"),
    ):
        values = series[key][np.isfinite(series[key])]
        metrics[name] = float(np.percentile(values, percentile)) if len(values) else np.nan
    return metrics


def attach_paired_temporal_metrics(result, spatial_dims):
    """Keep a separate equal-duration prefix comparison, without padding.

    This controls observed duration, not goal progress, and cannot undo
    censoring from early failures. Cache only summaries so reconstruction can
    release the trajectory arrays after archiving each pair.
    """
    if "paired_temporal_metrics" in result:
        return
    expert = result["expert_benchmark"]
    if "trajectory" not in result or "trajectory" not in expert:
        return
    frames = min(len(result["trajectory"]), len(expert["trajectory"]))
    result["paired_temporal_metrics"] = {
        "frames": frames,
        "gnca": trajectory_motion_metrics(result["trajectory"][:frames], spatial_dims),
        "boids": trajectory_motion_metrics(expert["trajectory"][:frames], spatial_dims),
    }


def online_outcome(result):
    """Return one mutually exclusive navigation outcome."""
    if result["reached"] == result["requested"] and not result["diverged"]:
        return "completed"
    if result["diverged"]:
        return "instability"
    return "timeout"


def _finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, **{stat: np.nan for stat in TEMPORAL_STATISTICS}}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
    }


def _metric_value(result, key):
    if key == "max_goal_tube_radius":
        return float(result[key])
    return float(result["motion_metrics"][key])


def _stats_line(source, label, values):
    stats = _finite_summary(values)
    return (
        f"  {source:<18} | {label:<42} | n={stats['n']:3d} | "
        f"mean={stats['mean']:.6f} | min={stats['min']:.6f} | "
        f"max={stats['max']:.6f} | median={stats['median']:.6f} | std={stats['std']:.6f}"
    )


def _json_array(values):
    return json.dumps(np.asarray(values).tolist(), separators=(",", ":"))


def write_online_benchmark_files(
    *, output_dir, run_tag, centers, regions, region_name, results,
    goal_threshold, max_steps, task="online_goals",
):
    """Write continuous online/fixed benchmarks, retaining the existing API."""
    if task not in ("online_goals", "fixed_waypoints"):
        raise ValueError(f"Unknown benchmark task: {task}")
    prefix = "online" if task == "online_goals" else "fixed"
    os.makedirs(output_dir, exist_ok=True)
    spatial_dims = np.asarray(centers).shape[1]
    for result in results:
        attach_paired_temporal_metrics(result, spatial_dims)
    outcomes = [online_outcome(result) for result in results]
    completed = np.asarray([value == "completed" for value in outcomes])
    failed = ~completed
    expert_completed = np.asarray(
        [result["expert_benchmark"]["completed"] for result in results],
        dtype=bool,
    )
    paired_completed = completed & expert_completed

    csv_path = os.path.join(output_dir, f"{prefix}_per_run_metrics.csv")
    fieldnames = [
        "run", region_name, "center", "goals", "goals_requested",
        "goal_visit_order", "gnca_arrival_steps", "boids_arrival_steps",
        "gnca_goals_reached", "gnca_completed_all_goals", "failure_type",
        "instability_step", "gnca_steps", "gnca_goal_radii", "gnca_max_r",
        "gnca_median_polarization", "gnca_p05_polarization",
        "gnca_median_speed_cv", "gnca_p95_speed_cv",
        "boids_goals_reached", "boids_completed_all_goals", "boids_steps",
        "boids_goal_radii", "boids_max_r", "boids_median_polarization",
        "boids_p05_polarization", "boids_median_speed_cv",
        "boids_p95_speed_cv",
    ]
    for source in ("gnca", "boids"):
        fieldnames.extend(
            f"{source}_{key}" for key, _ in TEMPORAL_METRICS
            if f"{source}_{key}" not in fieldnames
        )
        fieldnames.extend([f"{source}_observed_frames"] + [
            f"{source}_valid_{key}_frames" for key, _ in TEMPORAL_SERIES
        ])
    fieldnames.append("paired_observed_steps")
    fieldnames.extend(
        f"paired_{source}_{key}"
        for source in ("gnca", "boids", "delta") for key, _ in TEMPORAL_METRICS
    )
    fieldnames.extend(
        f"paired_{source}_valid_{key}_frames"
        for source in ("gnca", "boids") for key, _ in TEMPORAL_SERIES
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_index, (center, region, result, outcome) in enumerate(
            zip(centers, regions, results, outcomes)
        ):
            expert = result["expert_benchmark"]
            gnca_motion = result["motion_metrics"]
            expert_motion = expert["motion_metrics"]
            row = {
                "run": run_index,
                region_name: region,
                "center": _json_array(center),
                "goals": _json_array(result["goals"]),
                "goals_requested": result["requested"],
                "goal_visit_order": "sequential",
                "gnca_arrival_steps": _json_array(result.get("switch_steps", [])),
                "boids_arrival_steps": _json_array(expert.get("switch_steps", [])),
                "gnca_goals_reached": result["reached"],
                "gnca_completed_all_goals": outcome == "completed",
                "failure_type": "none" if outcome == "completed" else outcome,
                "instability_step": (
                    "" if result["divergence_step"] is None
                    else result["divergence_step"]
                ),
                "gnca_steps": (result["steps"] if "steps" in result
                               else len(result["trajectory"]) - 1),
                "gnca_goal_radii": _json_array(result["goal_tube_radii"]),
                "gnca_max_r": result["max_goal_tube_radius"],
                "gnca_median_polarization": gnca_motion["median_polarization"],
                "gnca_p05_polarization": gnca_motion["p05_polarization"],
                "gnca_median_speed_cv": gnca_motion["median_speed_cv"],
                "gnca_p95_speed_cv": gnca_motion["p95_speed_cv"],
                "boids_goals_reached": expert["reached"],
                "boids_completed_all_goals": expert["completed"],
                "boids_steps": expert["steps"],
                "boids_goal_radii": _json_array(expert["goal_tube_radii"]),
                "boids_max_r": expert["max_goal_tube_radius"],
                "boids_median_polarization": expert_motion["median_polarization"],
                "boids_p05_polarization": expert_motion["p05_polarization"],
                "boids_median_speed_cv": expert_motion["median_speed_cv"],
                "boids_p95_speed_cv": expert_motion["p95_speed_cv"],
            }
            for source, motion in (("gnca", gnca_motion), ("boids", expert_motion)):
                row.update({f"{source}_{key}": motion[key] for key, _ in TEMPORAL_METRICS})
                row[f"{source}_observed_frames"] = motion["observed_frames"]
                for key, _ in TEMPORAL_SERIES:
                    row[f"{source}_valid_{key}_frames"] = motion[f"valid_{key}_frames"]
            paired = result.get("paired_temporal_metrics")
            if paired:
                row["paired_observed_steps"] = max(0, paired["frames"] - 1)
                for key, _ in TEMPORAL_METRICS:
                    row[f"paired_gnca_{key}"] = paired["gnca"][key]
                    row[f"paired_boids_{key}"] = paired["boids"][key]
                    row[f"paired_delta_{key}"] = paired["gnca"][key] - paired["boids"][key]
                for source in ("gnca", "boids"):
                    for key, _ in TEMPORAL_SERIES:
                        row[f"paired_{source}_valid_{key}_frames"] = paired[source][f"valid_{key}_frames"]
            writer.writerow(row)

    total = len(results)
    lines = [
        ("Online-goal" if task == "online_goals" else "Fixed-waypoint")
        + " GNCA benchmark report",
        f"run_tag: {run_tag}",
        f"goal completion: centroid distance <= {goal_threshold:g} for every requested goal",
        "goals checked sequentially against the active waypoint; earlier visits to later goals do not count",
        f"evaluation horizon: {max_steps} steps",
        "instability: non-finite model state",
        "instantaneous r(t): 95th percentile of agent distances to the flock centroid at each frame",
        "temporal metrics: mean, max, min, median and population std (ddof=0) over frames, including the initial state once",
        "max_r: maximum across per-goal robust tube radii (95th agent percentile, then 99th temporal percentile); NOT temporal max r(t)",
        "polarization: norm of the mean unit-velocity vector (higher is more aligned)",
        "speed CV: within-flock speed standard deviation divided by mean speed (lower is more uniform)",
        "motion metrics exclude agents with speed <= 1e-12; frames with no moving agents are undefined, not zero",
        "non-finite metric values are excluded, never replaced by zero; valid-frame counts are saved per run",
        "full observed windows end at each controller's own completion, timeout or last finite state",
        "equal-duration comparison: both controllers truncated to min(observed GNCA frames, observed Boids frames); no extrapolation",
        "equal duration does not imply equal goal progress and does not correct selection bias from failures",
        "aggregate rows summarize per-run temporal statistics with equal weight per run; n counts finite run-level values",
        "",
        "Navigation outcomes",
        f"  completed all goals: {int(completed.sum())}/{total} = {100 * completed.mean():.2f}%",
        f"  failed: {int(failed.sum())}/{total} = {100 * failed.mean():.2f}%",
        f"    instability: {outcomes.count('instability')}",
        f"    timeout: {outcomes.count('timeout')}",
        f"  mean goals reached: {np.mean([result['reached'] for result in results]):.4f}",
        f"  matched Boids completed all goals: {int(expert_completed.sum())}/{total} = {100 * expert_completed.mean():.2f}%",
        f"  completed GNCA--Boids pairs used for primary metric comparison: {int(paired_completed.sum())}",
        "",
        f"Completion by {region_name}",
    ]
    region_array = np.asarray(regions)
    for region in sorted(set(regions)):
        mask = region_array == region
        region_completed = completed[mask]
        lines.append(
            f"  {region_name} {region}: {int(region_completed.sum())}/{int(mask.sum())} "
            f"= {100 * region_completed.mean():.2f}%"
        )

    all_indices = list(range(total))
    completed_indices = np.flatnonzero(paired_completed).tolist()
    failed_indices = np.flatnonzero(failed).tolist()

    def append_metric_section(title, indices, include_expert):
        lines.extend(["", title])
        for key, label in MOTION_METRICS:
            gnca_values = [_metric_value(results[index], key) for index in indices]
            lines.append(_stats_line("GNCA", label, gnca_values))
            if include_expert:
                expert_values = [
                    _metric_value(results[index]["expert_benchmark"], key)
                    for index in indices
                ]
                differences = [
                    gnca - expert
                    for gnca, expert in zip(gnca_values, expert_values)
                ]
                lines.append(_stats_line("matched Boids", label, expert_values))
                lines.append(_stats_line("GNCA - Boids", label, differences))

    append_metric_section(
        "Continuous metrics across all finite observed rollouts",
        all_indices,
        include_expert=True,
    )
    append_metric_section(
        "Pairs in which both GNCA and matched Boids completed all goals",
        completed_indices,
        include_expert=True,
    )
    append_metric_section(
        "Failed GNCA runs (observed before termination)",
        failed_indices,
        include_expert=True,
    )

    for title, indices in (
        ("All runs", all_indices),
        ("Both controllers completed", completed_indices),
        ("GNCA failed", failed_indices),
    ):
        lines.extend(["", f"Equal-duration observed-prefix comparison: {title}"])
        pairs = [results[i]["paired_temporal_metrics"] for i in indices
                 if "paired_temporal_metrics" in results[i]]
        lines.append(_stats_line("paired window", "observed update steps", [max(0, p["frames"] - 1) for p in pairs]))
        for key, label in TEMPORAL_METRICS:
            # Compare precisely the same runs for both controllers and delta.
            valid_pairs = [p for p in pairs if np.isfinite(p["gnca"][key]) and np.isfinite(p["boids"][key])]
            gnca = [p["gnca"][key] for p in valid_pairs]
            boids = [p["boids"][key] for p in valid_pairs]
            lines.append(_stats_line("GNCA", label, gnca))
            lines.append(_stats_line("matched Boids", label, boids))
            lines.append(_stats_line("GNCA - Boids", label, [g - b for g, b in zip(gnca, boids)]))

    lines.extend(["", "Per-run temporal statistics (each controller's full observed window)"])
    for index, result in enumerate(results):
        lines.append(f"  run {index:03d} | outcome={outcomes[index]}")
        for source, controller in (("GNCA", result), ("Boids", result["expert_benchmark"])):
            metrics = controller["motion_metrics"]
            lines.append(f"    {source}: observed_frames={metrics['observed_frames']} | observed_steps={max(0, metrics['observed_frames'] - 1)}")
            for key, label in TEMPORAL_SERIES:
                stats = " | ".join(f"{stat}={metrics[f'{stat}_{key}']:.6g}" for stat in TEMPORAL_STATISTICS)
                lines.append(f"      {label} | valid_frames={metrics[f'valid_{key}_frames']} | {stats}")

    lines.extend(["", f"Per-run metrics: {os.path.basename(csv_path)}"])
    summary_path = os.path.join(output_dir, f"{prefix}_evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Saved benchmark summary: {summary_path}")
    print(f"Saved per-run metrics: {csv_path}")
    return summary_path, csv_path
