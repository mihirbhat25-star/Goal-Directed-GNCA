"""Small, process-safe timing utilities for long-running experiments."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict


TIMEOUT_EXIT_CODE = 124


def append_timing_event(path, category, seconds, **metadata):
    """Append one completed timing event to a JSON-lines file."""
    if not path:
        return
    event = {
        "category": str(category),
        "seconds": float(seconds),
        "recorded_at": time.time(),
        **metadata,
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def deadline_reached(deadline):
    """Return whether an absolute Unix wall-clock deadline has passed."""
    return bool(deadline and time.time() >= float(deadline))


def write_timing_report(event_path, report_path, total_seconds, status):
    """Aggregate timing events, print their allocation, and save JSON."""
    events = []
    if event_path and os.path.exists(event_path):
        with open(event_path, encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]

    totals = defaultdict(float)
    for event in events:
        totals[event["category"]] += float(event["seconds"])

    accounted = sum(totals.values())
    orchestration = max(0.0, float(total_seconds) - accounted)
    totals["pipeline_orchestration"] += orchestration
    denominator = max(float(total_seconds), 1e-12)
    allocation = {
        category: {
            "seconds": seconds,
            "hours": seconds / 3600.0,
            "percent": 100.0 * seconds / denominator,
        }
        for category, seconds in totals.items()
    }
    report = {
        "status": status,
        "total_seconds": float(total_seconds),
        "total_hours": float(total_seconds) / 3600.0,
        "allocation": allocation,
        "events": events,
    }
    if report_path:
        directory = os.path.dirname(os.path.abspath(report_path))
        os.makedirs(directory, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    print("\n" + "=" * 72)
    print(f"3D PIPELINE TIMING REPORT ({status})")
    print("=" * 72)
    for category, values in sorted(
        allocation.items(), key=lambda item: item[1]["seconds"], reverse=True
    ):
        label = category.replace("_", " ").title()
        print(
            f"{label:30s} {values['hours']:8.3f} h  "
            f"({values['percent']:6.2f}%)"
        )
    print(f"{'Total wall time':30s} {total_seconds / 3600.0:8.3f} h")
    if report_path:
        print(f"Detailed JSON: {report_path}")
    print("=" * 72, flush=True)
    return report
