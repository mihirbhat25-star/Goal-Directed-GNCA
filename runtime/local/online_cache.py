"""Persistent online-goal expert caches and bounded-memory graph batches.

The disk and RAM backends use the same seeded expert configurations. Local
execution writes one trajectory at a time and reads only each requested batch;
the full training chunk is never materialized as TensorFlow constants.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np


FIELDS = ("states", "input_states", "active_masks", "active_goals",
          "previous_goals", "max_previous_goal_distances", "goal_segment_ids",
          "adjacency_bits", "waypoints", "sampled_center")


@dataclass(frozen=True)
class CachedChunk:
    path: str


def _bounded_results(generator, configs, workers):
    """Keep at most one pending trajectory per worker, including completed ones."""
    def timed(config):
        started = time.perf_counter()
        trajectory = generator(config)
        return trajectory, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        iterator = iter(configs)
        for _ in range(workers):
            config = next(iterator, None)
            if config is not None:
                pending.append(pool.submit(timed, config))
        while pending:
            yield pending.popleft().result()
            config = next(iterator, None)
            if config is not None:
                pending.append(pool.submit(timed, config))


def generate_cached_chunk(args, chunk_index, allocation, *, spatial_dims,
                          validation=False, mode_counts=None):
    """Reuse a matching cache, otherwise atomically create it from expert episodes."""
    if spatial_dims == 2:
        from runtime.cloud.cloud_compiled_2d import build_chunk_configs
        from runtime.cloud import compiled_expert_2d as expert
        configs = build_chunk_configs(args, chunk_index, allocation,
                                      validation=validation, mode_counts=mode_counts)
        warm_up = expert.warm_up_compiled_expert_2d
    else:
        from runtime.cloud.cloud_compiled_3d import build_chunk_configs
        from runtime.cloud import compiled_expert_3d as expert
        configs = build_chunk_configs(args, chunk_index, allocation, task=args.task,
                                      validation=validation, mode_counts=mode_counts)
        warm_up = expert.warm_up_compiled_expert_3d
    recipe = json.dumps({"schema": 1, "dimension": spatial_dims, "task": args.task,
                         "expert_sha256": hashlib.sha256(Path(expert.__file__).read_bytes()).hexdigest(),
                         "configs": configs}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(recipe.encode()).hexdigest()[:20]
    directory = Path(args.cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    split = "validation" if validation else "training"
    path = directory / f"online{spatial_dims}d_{split}_{digest}.h5"
    if path.exists():
        with h5py.File(path, "r") as cache:
            if cache["recipe"].asstr()[()] != recipe or not cache.attrs.get("complete", False):
                raise ValueError(f"Cache is incomplete or has a mismatched recipe: {path}")
        print(f">>> Local {spatial_dims}D: reusing {split} cache {path}", flush=True)
    else:
        workers = max(1, min(int(args.generation_workers) or 4, len(configs)))
        print(f">>> Local {spatial_dims}D: caching {len(configs)} {split} episodes "
              f"with {workers} CPU workers to {path}", flush=True)
        warm_up()
        generator = expert.generate_online_trajectory
        if spatial_dims == 2 and args.task == "fixed_waypoints":
            generator = expert.generate_fixed_trajectory
        started = time.perf_counter()
        descriptor, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".partial", dir=directory)
        os.close(descriptor)
        try:
            with h5py.File(temporary, "w") as cache:
                cache.create_dataset("recipe", data=recipe, dtype=h5py.string_dtype())
                centers, records = [], []
                for index, (trajectory, seconds) in enumerate(_bounded_results(generator, configs, workers)):
                    group = cache.create_group(f"trajectories/{index}")
                    for name in FIELDS:
                        value = np.asarray(getattr(trajectory, name))
                        group.create_dataset(name, data=value, compression="lzf")
                    centers.append(trajectory.sampled_center.tolist())
                    records.append({"seed": configs[index]["seed"], "steps": trajectory.steps,
                                    "seconds": seconds,
                                    "start_region": configs[index].get("octant", configs[index].get("start_quadrant"))})
                    del trajectory
                cache.create_dataset("centers", data=np.asarray(centers, dtype=np.float32))
                cache.create_dataset("records", data=json.dumps(records), dtype=h5py.string_dtype())
                cache.attrs["complete"] = True
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        print(f">>> Local {spatial_dims}D: {split} cache ready in {time.perf_counter()-started:.1f}s", flush=True)
    with h5py.File(path, "r") as cache:
        centers = cache["centers"][:]
        records = json.loads(cache["records"].asstr()[()])
    result = (CachedChunk(str(path)), centers)
    if spatial_dims == 3:
        result += (np.asarray([config["octant"] for config in configs], dtype=np.int8),)
    return result + (records,)


def load_cached_trajectories(chunk):
    """Read every trajectory of a cached chunk into memory as field dictionaries."""
    trajectories = []
    with h5py.File(chunk.path, "r") as cache:
        for index in range(len(cache["trajectories"])):
            group = cache[f"trajectories/{index}"]
            trajectories.append({name: group[name][:] for name in FIELDS})
    return trajectories


def dataset_from_cache(chunk, batch_size, n_boids, timestep_stride, near_goal_radius,
                       seed, *, shuffle_batches=True, goal_conditioned=True,
                       spatial_dims=3, retain_previous_goal=False):
    """Shuffle transitions across trajectories and stream complete graph batches."""
    import tensorflow as tf
    from runtime.cloud.cloud_bitpacked_3d import _selected_steps, _make_model_batch

    if batch_size < 1 or timestep_stride < 1:
        raise ValueError("Batch size and timestep stride must be positive.")
    references = []
    with h5py.File(chunk.path, "r") as cache:
        for index in range(len(cache["trajectories"])):
            group = cache[f"trajectories/{index}"]
            if group["states"].shape[1:] != (n_boids, 2 * spatial_dims):
                raise ValueError("Cached trajectory dimensions do not match the requested model.")
            selected = _selected_steps(group["input_states"][:], group["active_masks"][:],
                                       group["active_goals"][:], timestep_stride,
                                       near_goal_radius, spatial_dims,
                                       group["previous_goals"][:] if retain_previous_goal else None,
                                       group["max_previous_goal_distances"][:] if retain_previous_goal else None)
            references.append(np.column_stack((np.full(len(selected), index, dtype=np.int32), selected)))
    references = np.concatenate(references)
    order = np.random.default_rng(seed).permutation(len(references))
    usable = len(order) // batch_size * batch_size
    if not usable:
        raise ValueError("Cached trajectories produced no complete training batch; decrease --batch_size.")
    batches = references[order[:usable]].reshape(-1, batch_size, 2)
    steps = len(batches)
    print(f">>> Local: indexed {len(references)} transitions as {steps} streamed batches.", flush=True)

    names = ("input_states", "states", "active_masks", "active_goals", "previous_goals",
             "max_previous_goal_distances", "adjacency_bits")
    shapes = ((batch_size, n_boids, 2 * spatial_dims),) * 2 + (
        (batch_size, n_boids), (batch_size, spatial_dims), (batch_size, spatial_dims),
        (batch_size,), (batch_size, (n_boids * n_boids + 7) // 8))
    dtypes = (np.float32,) * 6 + (np.uint8,)

    def iterator():
        rng = np.random.default_rng(seed)
        with h5py.File(chunk.path, "r") as cache:
            while True:
                batch_order = rng.permutation(steps) if shuffle_batches else range(steps)
                for batch_index in batch_order:
                    entries = batches[batch_index]
                    arrays = [np.empty(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes)]
                    for trajectory_id in np.unique(entries[:, 0]):
                        positions = np.flatnonzero(entries[:, 0] == trajectory_id)
                        group = cache[f"trajectories/{trajectory_id}"]
                        # h5py requires increasing row indices. Restore shuffled order afterwards.
                        sort = np.argsort(entries[positions, 1])
                        positions = positions[sort]
                        rows = entries[positions, 1]
                        for name, array in zip(names, arrays):
                            array[positions] = group[name][rows + (1 if name == "states" else 0)]
                    yield tuple(arrays)

    signature = tuple(tf.TensorSpec(shape, tf.as_dtype(dtype)) for shape, dtype in zip(shapes, dtypes))
    dataset = tf.data.Dataset.from_generator(iterator, output_signature=signature)

    def pack(inputs, outputs, masks, goals, previous, distances, bits):
        states = tf.reshape(tf.stack((inputs, outputs), axis=1),
                            [2 * batch_size, n_boids, 2 * spatial_dims])
        ids = tf.range(batch_size, dtype=tf.int32)
        return _make_model_batch(ids, states, 2 * ids, inputs, masks, goals, previous,
                                 distances, bits, batch_size, n_boids,
                                 goal_conditioned, spatial_dims)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    return dataset.map(pack).with_options(options).prefetch(1), steps
