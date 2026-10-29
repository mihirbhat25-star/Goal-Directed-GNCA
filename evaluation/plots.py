"""Plots of one model's paired GNCA--Boids evaluation.

* ``completion.pdf`` and ``metrics.pdf``: task completion and the per-rollout
  temporal means of the flock metrics for every evaluated condition (no removal,
  and each removal or insertion count).
* ``recovery.pdf``: median radial score r(t) of the GNCA around the removal
  step, with the same rollouts without removal dashed (removal runs with saved
  radius series).
* ``scaling.pdf``: GNCA completion and median flock radius against the swarm size,
  from the folders written by ``python -m experiments.run scaling``:

      python -m evaluation.plots --scaling <scaling output folder>
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

COLORS = {"gnca": "#228833", "boids": "#2369A0"}
LABELS = {"gnca": "GNCA", "boids": "Boids"}
METRICS = ("radius", "polarization", "speed_cv")
METRIC_LABELS = {
    "radius": r"Flock compactness $\overline{r}$",
    "polarization": r"Polarization $\overline{P}$",
    "speed_cv": r"Speed variation $\overline{CV}_v$",
}
RECOVERY_WINDOW = (-250, 1000)


def _condition_label(condition):
    if condition == "baseline":
        return "No removal"
    kind, count = condition.rsplit("_", 1)
    return f"{'Remove' if kind == 'remove' else 'Insert'} {count}"


def _save(figure, output_dir, stem):
    path = os.path.join(output_dir, f"{stem}.pdf")
    figure.savefig(path, bbox_inches="tight", pad_inches=0.05, metadata={"Creator": "Matplotlib", "Author": ""})
    figure.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return path


def rollout_plots(rows, aggregates, output_dir):
    """Completion bars and metric distributions for every condition of one evaluation."""
    conditions = list(aggregates)
    controllers = [who for who in ("gnca", "boids") if f"{who}_completed" in rows[0]]
    labels = [_condition_label(condition) for condition in conditions]
    x = np.arange(len(conditions))

    figure, axis = plt.subplots(figsize=(max(3.0, 1.2 * len(conditions) + 1.5), 3.0))
    width = 0.76 / len(controllers)
    for index, who in enumerate(controllers):
        offset = (index - (len(controllers) - 1) / 2) * width
        axis.bar(x + offset, [aggregates[key][f"{who}_completion_rate"] for key in conditions], width,
                 label=LABELS[who], color=COLORS[who])
    axis.set_ylabel("Task completion (%)")
    axis.set_ylim(0, 105)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    completion_path = _save(figure, output_dir, "completion")

    rng = np.random.default_rng(1)
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.85))
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.84, wspace=0.36)
    any_incomplete = False
    for axis_index, (axis, metric) in enumerate(zip(axes, METRICS)):
        for condition_index, condition in enumerate(conditions):
            selected = [row for row in rows if row["condition"] == condition]
            for controller_index, who in enumerate(controllers):
                position = condition_index + (controller_index - (len(controllers) - 1) / 2) * 0.3
                values = np.asarray([row[f"{who}_mean_{metric}"] for row in selected], dtype=float)
                incomplete = np.asarray([not bool(row[f"{who}_completed"]) for row in selected])
                any_incomplete = any_incomplete or bool(incomplete.any())
                finite = np.isfinite(values)
                box = axis.boxplot([values[finite]], positions=[position], widths=0.24, patch_artist=True,
                                   showfliers=False, medianprops={"color": "black", "linewidth": 1.0},
                                   whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7})
                box["boxes"][0].set(facecolor=COLORS[who], alpha=0.18, edgecolor=COLORS[who])
                jitter = rng.uniform(-0.065, 0.065, len(values))
                axis.scatter(position + jitter[~incomplete], values[~incomplete], s=7, c=COLORS[who],
                             alpha=0.55, edgecolors="none", zorder=3)
                axis.scatter(position + jitter[incomplete], values[incomplete], s=22, c="black", marker="x",
                             linewidths=0.9, zorder=4)
        axis.set_title(f"({chr(97 + axis_index)}) {METRIC_LABELS[metric]}", fontsize=8)
        axis.set_xticks(np.arange(len(conditions), dtype=float))
        axis.set_xticklabels(labels, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
        if metric in ("radius", "speed_cv"):
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.set_ylabel("Log scale", fontsize=7)
        else:
            axis.set_ylim(0.45, 1.01)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    handles = [Line2D([], [], color=COLORS[who], lw=5, alpha=0.65, label=LABELS[who]) for who in controllers]
    if any_incomplete:
        handles.append(Line2D([], [], marker="x", color="black", linestyle="none", markersize=5,
                              label="Incomplete rollout"))
    figure.legend(handles=handles, ncol=len(handles), frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, 1.01), fontsize=8)
    metrics_path = _save(figure, output_dir, "metrics")
    return completion_path, metrics_path


def _aligned(series, condition, who, event_condition):
    """Radius series of every rollout of a condition, aligned on the event step of ``event_condition``."""
    offsets = np.arange(RECOVERY_WINDOW[0], RECOVERY_WINDOW[1] + 1)
    runs = sorted({key.split("_run")[1][:3] for key in series if key.startswith(f"{condition}_run")})
    stacked = []
    for run in runs:
        values = np.asarray(series[f"{condition}_run{run}_{who}"], dtype=float)
        index = int(series[f"{event_condition}_run{run}_event_step"]) + offsets
        row = np.full(offsets.shape, np.nan)
        valid = (index >= 0) & (index < len(values))
        row[valid] = values[index[valid]]
        stacked.append(row)
    with np.errstate(all="ignore"):
        return offsets, np.nanmedian(np.vstack(stacked), axis=0)


def recovery_plot(series, output_dir):
    """Median GNCA r(t) around the removal step for each removal count; None without removal series."""
    removals = sorted({key.split("_run")[0] for key in series if key.startswith("remove_")},
                      key=lambda key: int(key.rsplit("_", 1)[1]))
    if not removals or not any(key.startswith("baseline_run") for key in series):
        return None
    figure, axes = plt.subplots(1, len(removals), figsize=(max(3.4, 2.3 * len(removals)), 2.4), sharey=True,
                                squeeze=False)
    for axis, condition in zip(axes[0], removals):
        offsets, median = _aligned(series, "baseline", "gnca", condition)
        axis.plot(offsets, median, color=COLORS["gnca"], lw=1.0, ls="--")
        offsets, median = _aligned(series, condition, "gnca", condition)
        axis.plot(offsets, median, color=COLORS["gnca"], lw=1.1)
        axis.axvline(0, color="0.4", lw=0.8, ls=":")
        axis.set_title(_condition_label(condition), fontsize=8)
        axis.set_xlabel("Steps from removal", fontsize=7)
        axis.grid(alpha=0.25, lw=0.5)
        axis.tick_params(labelsize=6)
    axes[0][0].set_ylabel(r"Median $r(t)$", fontsize=7)
    handles = [Line2D([], [], color=COLORS["gnca"], lw=1.1, label="Removal"),
               Line2D([], [], color=COLORS["gnca"], lw=1.0, ls="--", label="No removal")]
    figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=6.5,
                  bbox_to_anchor=(0.5, 1.08))
    figure.tight_layout()
    return _save(figure, output_dir, "recovery")


def _scaling_points(folder, regime):
    points = []
    root = Path(folder) / regime
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        csv_path = child / "per_run.csv"
        if not (child.name.startswith("n") and child.name[1:].isdigit() and csv_path.is_file()):
            continue
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["condition"] == "baseline"]
        point = {"n": int(child.name[1:])}
        point["gnca_completion"] = 100.0 * np.mean([row["gnca_completed"] == "True" for row in rows])
        radii = [float(row["gnca_mean_radius"]) for row in rows if row["gnca_mean_radius"] not in ("", "nan")]
        point["gnca_radius"] = float(np.median(radii)) if radii else float("nan")
        points.append(point)
    return sorted(points, key=lambda point: point["n"])


def scaling_plot(folder):
    """GNCA completion and median flock radius against N for one model's scaling runs."""
    density, box = _scaling_points(folder, "constant_density"), _scaling_points(folder, "training_box")
    if not density and not box:
        raise SystemExit(f"No scaling runs under {folder}")
    figure, axes = plt.subplots(1, 2, figsize=(5.0, 2.3))
    for axis, key in zip(axes, ("completion", "radius")):
        if density:
            axis.plot([p["n"] for p in density], [p[f"gnca_{key}"] for p in density], "o-",
                      color=COLORS["gnca"], ms=3.5, lw=1.2)
        if box:
            axis.plot([p["n"] for p in box], [p[f"gnca_{key}"] for p in box], "s", mfc="none",
                      mec=COLORS["gnca"], ms=5, mew=1.2)
    sizes = sorted({p["n"] for p in density + box})
    for axis, label in zip(axes, ("Task completion (%)", r"Median $\overline{r}$")):
        axis.set_xscale("log")
        axis.set_xticks(sizes)
        axis.set_xticklabels([f"{n:,}" for n in sizes])
        axis.minorticks_off()
        axis.set_xlabel("$N$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylim(0, 105)
    handles = [Line2D([], [], marker="o", color=COLORS["gnca"], ms=3.5, lw=1.2, label="Training density"),
               Line2D([], [], marker="s", mfc="none", mec=COLORS["gnca"], ls="none", ms=5, mew=1.2,
                      label="Training box")]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.06))
    figure.tight_layout()
    return _save(figure, str(folder), "scaling")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scaling", type=Path, required=True,
                        help="Output folder of python -m experiments.run scaling")
    args = parser.parse_args()
    print(f"Saved {scaling_plot(args.scaling)}")


if __name__ == "__main__":
    main()
