"""Aggregate per-rollout temporal means into the paper's median/IQR format."""
import argparse
import csv
from pathlib import Path

import numpy as np


def summarize_csv(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for condition in dict.fromkeys(row["condition"] for row in rows):
        subset = [row for row in rows if row["condition"] == condition]
        for controller in (c for c in ("gnca", "boids") if f"{c}_completed" in subset[0]):
            for metric in ("radius", "polarization", "speed_cv"):
                values = np.array([float(row[f"{controller}_mean_{metric}"]) for row in subset])
                valid = values[np.isfinite(values)]
                quartiles = np.percentile(valid, [25, 50, 75]) if len(valid) else [np.nan] * 3
                result.append({
                    "condition": condition, "controller": controller, "metric": metric,
                    "n_rollouts": len(subset), "n_finite_temporal_means": len(valid),
                    "completed": sum(row[f"{controller}_completed"].lower() in ("true", "1") for row in subset),
                    "instability": sum(row[f"{controller}_failure_type"] == "instability" for row in subset),
                    "timeout": sum(row[f"{controller}_failure_type"] == "timeout" for row in subset),
                    "median": quartiles[1], "q25": quartiles[0], "q75": quartiles[2],
                    "iqr": quartiles[2] - quartiles[0],
                    "mean": float(valid.mean()) if len(valid) else np.nan,
                    "std": float(valid.std()) if len(valid) else np.nan,
                    "min": float(valid.min()) if len(valid) else np.nan,
                    "max": float(valid.max()) if len(valid) else np.nan,
                })
    output = path.with_name("paper_summary.csv")
    if not result:
        raise ValueError("No per-run rows to summarize")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"Saved paper-format aggregates: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    summarize_csv(parser.parse_args().csv)
