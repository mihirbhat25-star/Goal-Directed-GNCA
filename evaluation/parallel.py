"""Run an evaluation script's starts in parallel worker processes and merge the results.

The evaluator is re-invoked on itself, once per worker, with a contiguous range
of start indices. Every start derives its random draws from its own index. Workers
take the visible GPUs in turn (sharing them when there are more workers than
GPUs) or run on the CPU when none is visible.
"""

import os
import pickle
import shutil
import subprocess
import sys
import threading

import numpy as np


def visible_gpus():
    """GPU ids as the CUDA runtime names them, honoring an existing CUDA_VISIBLE_DEVICES."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return [item.strip() for item in configured.split(",") if item.strip()]
    import tensorflow as tf
    return [str(index) for index in range(len(tf.config.list_physical_devices("GPU")))]


def _stream(prefix, pipe):
    for line in iter(pipe.readline, ""):
        print(f"[{prefix}] {line.rstrip()}", flush=True)
    pipe.close()


def run_with_workers(module, args, n_runs, condition_order, write_outputs):
    """Shard ``n_runs`` starts over ``args.workers`` processes of ``module`` and merge.

    Each worker receives the parent's command line plus ``--workers 1``, its
    start range, its own output directory and ``--_shard``, and writes
    ``rows.pkl`` (and ``radius_series.npz`` when requested) there. The merged
    rows are sorted by condition, in ``condition_order``, then by start index,
    and handed to ``write_outputs(args, rows, radius_series)``.
    """
    workers = min(int(args.workers), int(n_runs))
    bounds = np.linspace(0, n_runs, workers + 1).astype(int)
    shards_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    gpus = visible_gpus()
    print(
        f"Evaluating {n_runs} starts in {workers} worker processes"
        + (f" over {len(gpus)} GPU(s)" if gpus else " on the CPU") + ".",
        flush=True,
    )
    processes = []
    for shard, (lower, upper) in enumerate(zip(bounds[:-1], bounds[1:])):
        shard_dir = os.path.join(shards_dir, f"{shard:02d}")
        command = [
            sys.executable, "-m", module, *sys.argv[1:],
            "--workers", "1", "--run_start", str(lower), "--run_stop", str(upper),
            "--output_dir", shard_dir, "--_shard",
        ]
        environment = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true")
        if gpus:
            environment["CUDA_VISIBLE_DEVICES"] = gpus[shard % len(gpus)]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=environment,
        )
        thread = threading.Thread(
            target=_stream, args=(f"worker {shard}", process.stdout), daemon=True
        )
        thread.start()
        processes.append((shard, shard_dir, process, thread))
    failed = []
    for shard, _, process, thread in processes:
        process.wait()
        thread.join()
        if process.returncode:
            failed.append(shard)
    if failed:
        raise RuntimeError(f"Evaluation worker(s) {failed} failed; see their output above.")

    order = {key: index for index, key in enumerate(condition_order)}
    rows, radius_series = [], {}
    for _, shard_dir, _, _ in processes:
        with open(os.path.join(shard_dir, "rows.pkl"), "rb") as handle:
            rows.extend(pickle.load(handle))
        series_path = os.path.join(shard_dir, "radius_series.npz")
        if os.path.isfile(series_path):
            with np.load(series_path) as archive:
                radius_series.update({key: archive[key] for key in archive.files})
    rows.sort(key=lambda row: (order[row["condition"]], row["run"]))
    write_outputs(args, rows, radius_series)
    shutil.rmtree(shards_dir, ignore_errors=True)


def save_shard(output_dir, rows, radius_series):
    """Write one worker's raw results for the parent to merge."""
    with open(os.path.join(output_dir, "rows.pkl"), "wb") as handle:
        pickle.dump(rows, handle)
    if radius_series:
        np.savez_compressed(os.path.join(output_dir, "radius_series.npz"), **radius_series)
