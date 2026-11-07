"""Radius and neighbor count over time, GNCA against the matched Boids run.

Reads the series of the baseline rollouts of one evaluation. ``evaluation.paired_rollouts``
writes this report as ``series_report.txt``; for an existing evaluation folder run

    python -m evaluation.series_report <folder>/radius_series.npz
"""
import argparse
import os

import numpy as np

STEPS = (0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 5001)
FIFTHS = 5
# A rollout counts as collapsed above this many neighbors per agent.
CROWDED_NEIGHBORS = 15


def _series(data, runs, controller, suffix):
    series = [np.asarray(data[f"baseline_run{run}_{controller}{suffix}"], dtype=float) for run in runs]
    # Rollouts that diverged are saved as all NaN.
    return [values for values in series if np.isfinite(values).any()]


def _median(values):
    return float(np.median(values)) if len(values) else float("nan")


def _table(lines, kind, gnca, boids):
    lines.append(f"\n== {kind}: median over runs, GNCA / Boids (runs still going) ==")
    lines.append("by step:")
    for start, stop in zip(STEPS[:-1], STEPS[1:]):
        g = [x[start:stop].mean() for x in gnca if len(x) > start]
        e = [x[start:stop].mean() for x in boids if len(x) > start]
        if g:
            lines.append(f"  steps {start:>4}-{stop - 1:<4}  {_median(g):7.3f} / {_median(e):7.3f}  (n={len(g)})")
    lines.append("by fifth of each run's own length:")
    for fifth in range(FIFTHS):
        g = [part.mean() for part in (np.array_split(x, FIFTHS)[fifth] for x in gnca) if len(part)]
        e = [part.mean() for part in (np.array_split(x, FIFTHS)[fifth] for x in boids) if len(part)]
        lines.append(f"  fifth {fifth + 1}        {_median(g):7.3f} / {_median(e):7.3f}")


def report_lines(data):
    """The report of the baseline series in ``data`` (a mapping); None without baseline rollouts."""
    keys = list(data.keys())
    runs = sorted({key.split("_run")[1][:3] for key in keys if key.startswith("baseline_run")})
    if not runs:
        return None
    lines = [f"{len(runs)} baseline runs"]
    has_boids = f"baseline_run{runs[0]}_boids" in keys
    boids = _series(data, runs, "boids", "") if has_boids else []
    _table(lines, "radius", _series(data, runs, "gnca", ""), boids)
    gnca = _series(data, runs, "gnca", "_neighbors")
    boids = _series(data, runs, "boids", "_neighbors") if has_boids else []
    _table(lines, "neighbor count", gnca, boids)
    peak_step = np.array([int(np.argmax(x)) for x in gnca])
    peak_share = np.array([np.argmax(x) / max(1, len(x) - 1) for x in gnca])
    above = [float(np.mean(x > CROWDED_NEIGHBORS)) for x in gnca]
    first = [int(np.argmax(x > CROWDED_NEIGHBORS)) for x in gnca if (x > CROWDED_NEIGHBORS).any()]
    lines.append(
        f"\nGNCA peak neighbor step: median {int(np.median(peak_step))}, "
        f"quartiles {np.percentile(peak_step, [25, 75]).astype(int).tolist()}; "
        f"as a fraction of the run: median {np.median(peak_share):.2f}"
    )
    lines.append(
        f"GNCA share of steps above {CROWDED_NEIGHBORS} neighbors: median {np.median(above):.1%}, "
        f"max {max(above):.1%}; first step above {CROWDED_NEIGHBORS}: "
        f"median {int(np.median(first)) if first else 'never'}"
    )
    if has_boids:
        lines.append(f"Boids peak neighbor step: median {int(np.median([np.argmax(x) for x in boids]))}")
    return lines


def write_report(data, output_dir):
    """Print the report and write ``series_report.txt``; returns its path, or None without baseline rollouts."""
    lines = report_lines(data)
    if lines is None:
        return None
    print("\n".join(lines), flush=True)
    path = os.path.join(output_dir, "series_report.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series", help="radius_series.npz of an evaluation.")
    args = parser.parse_args()
    with np.load(args.series) as archive:
        write_report({key: archive[key] for key in archive.files}, os.path.dirname(args.series) or ".")


if __name__ == "__main__":
    main()
