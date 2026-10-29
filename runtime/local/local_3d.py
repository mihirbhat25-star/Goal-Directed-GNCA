"""Local backend: bounded-memory packing of precomputed HDF5 caches."""

import os
from contextlib import ExitStack

import h5py
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from spektral.data import Graph

from modules.boids_3d import (
    BOIDS_3D_ACTIVE_MASK_INDEX,
    BOIDS_3D_MASKED_TARGET_FEATURES,
    BOIDS_GOAL_POSITIONS_3D,
    BoidsDataset3D,
    ensure_active_mask_targets_3d,
)


def save_validation_dataset(dataset, path, n_boids):
    """Serialize a small Spektral dataset for an isolated chunk worker."""
    graphs = list(dataset.graphs)
    max_edges = max((graph.a.nnz for graph in graphs), default=0)
    x = np.stack([graph.x for graph in graphs], axis=0).astype(np.float32)
    y = np.stack([graph.y for graph in graphs], axis=0).astype(np.float32)
    a_row = np.full((len(graphs), max_edges), -1, dtype=np.int32)
    a_col = np.full((len(graphs), max_edges), -1, dtype=np.int32)
    a_len = np.zeros(len(graphs), dtype=np.int32)
    for index, graph in enumerate(graphs):
        adjacency = graph.a.tocoo()
        a_row[index, : adjacency.nnz] = adjacency.row
        a_col[index, : adjacency.nnz] = adjacency.col
        a_len[index] = adjacency.nnz
    np.savez_compressed(
        path,
        x=x,
        y=y,
        a_row=a_row,
        a_col=a_col,
        a_len=a_len,
        n_boids=np.array([n_boids], dtype=np.int32),
    )


def load_validation_dataset(path):
    """Rebuild a small validation dataset inside an isolated local worker."""
    with np.load(path) as data:
        x = data["x"]
        y = data["y"]
        rows = data["a_row"]
        cols = data["a_col"]
        lengths = data["a_len"]
        n_boids = int(data["n_boids"][0])
        graphs = []
        for graph_index in range(len(x)):
            edge_count = int(lengths[graph_index])
            adjacency = sp.coo_matrix(
                (
                    np.ones(edge_count, dtype=np.float32),
                    (
                        rows[graph_index, :edge_count],
                        cols[graph_index, :edge_count],
                    ),
                ),
                shape=(n_boids, n_boids),
            )
            graphs.append(
                Graph(x=x[graph_index], a=adjacency, y=y[graph_index])
            )
    return BoidsDataset3D(graphs)


def _make_disjoint_batch(
    x_list, y_list, row_list, col_list, length_list, n_boids
):
    x = np.concatenate(x_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    edge_rows = []
    edge_cols = []
    for batch_index, (rows, cols, edge_count) in enumerate(
        zip(row_list, col_list, length_list)
    ):
        edge_count = int(edge_count)
        if edge_count <= 0:
            continue
        offset = batch_index * n_boids
        edge_rows.append(rows[:edge_count].astype(np.int64) + offset)
        edge_cols.append(cols[:edge_count].astype(np.int64) + offset)
    indices = (
        np.stack(
            [np.concatenate(edge_rows), np.concatenate(edge_cols)], axis=1
        )
        if edge_rows
        else np.zeros((0, 2), dtype=np.int64)
    )
    return x, y, indices


def _load_selected_samples(
    cache, start, end, timestep_stride, near_goal_radius
):
    if timestep_stride <= 1 or near_goal_radius <= 0:
        selection = slice(start, end, timestep_stride)
        return (
            cache["x"][selection],
            ensure_active_mask_targets_3d(cache["y"][selection]),
            cache["a_row"][selection],
            cache["a_col"][selection],
            cache["a_len"][selection],
        )

    x_full = cache["x"][start:end]
    y_full = ensure_active_mask_targets_3d(cache["y"][start:end])
    active = y_full[..., BOIDS_3D_ACTIVE_MASK_INDEX]
    mean_pos = np.sum(y_full[..., :3] * active[..., None], axis=1)
    mean_pos /= np.maximum(active.sum(axis=1, keepdims=True), 1.0)
    distance = np.min(
        np.linalg.norm(
            mean_pos[:, None, :] - BOIDS_GOAL_POSITIONS_3D[None, :, :],
            axis=-1,
        ),
        axis=1,
    )
    keep = np.zeros(end - start, dtype=bool)
    keep[::timestep_stride] = True
    keep |= distance < near_goal_radius
    absolute_indices = np.flatnonzero(keep) + start
    return (
        x_full[keep],
        y_full[keep],
        cache["a_row"][absolute_indices],
        cache["a_col"][absolute_indices],
        cache["a_len"][absolute_indices],
    )


def _validate_adjacency(cache, sample_indices, perception):
    """Reject caches whose adjacency is stale or uses another perception."""
    for sample_index in sample_indices:
        sample_index = int(sample_index)
        x = cache["x"][sample_index]
        edge_count = int(cache["a_len"][sample_index])
        cached_pairs = np.column_stack((
            cache["a_row"][sample_index, :edge_count],
            cache["a_col"][sample_index, :edge_count],
        ))
        distances = np.linalg.norm(
            x[:, None, :3] - x[None, :, :3], axis=-1
        )
        neighbors = distances < perception
        np.fill_diagonal(neighbors, False)
        raw_y = cache["y"][sample_index]
        if raw_y.shape[-1] == BOIDS_3D_MASKED_TARGET_FEATURES:
            active = raw_y[:, BOIDS_3D_ACTIVE_MASK_INDEX] > 0.5
            neighbors &= active[:, None] & active[None, :]
        expected_pairs = np.column_stack(np.nonzero(neighbors))
        if set(map(tuple, cached_pairs)) != set(map(tuple, expected_pairs)):
            raise ValueError(
                "Cached adjacency does not match the current state at sample "
                f"{sample_index} for perception={perception}."
            )
        if edge_count > 1:
            order = np.lexsort((cached_pairs[:, 1], cached_pairs[:, 0]))
            if not np.array_equal(order, np.arange(edge_count)):
                raise ValueError(
                    f"Cached adjacency at sample {sample_index} is not "
                    "row-major sorted."
                )


def write_batch_shards(
    cache_paths,
    trajectory_refs,
    boundaries_by_cache,
    n_boids,
    batch_size,
    output_dir,
    *,
    timestep_stride=1,
    near_goal_radius=1.0,
    perception=0.1,
    shuffle_buffer_size=4096,
    batches_per_shard=16,
):
    """Write timestep-shuffled disjoint batches to bounded NPZ shards."""
    os.makedirs(output_dir, exist_ok=True)
    shard_paths = []
    shard_batches = []
    total_batches = 0
    x_batch, y_batch, row_batch, col_batch, length_batch = [], [], [], [], []
    shuffle_buffer = []
    shuffle_buffer_size = max(batch_size, int(shuffle_buffer_size))
    batches_per_shard = max(1, int(batches_per_shard))
    rng = np.random.default_rng()

    def flush_shard():
        nonlocal shard_batches
        if not shard_batches:
            return
        node_lengths = np.asarray(
            [len(batch[0]) for batch in shard_batches], dtype=np.int64
        )
        edge_lengths = np.asarray(
            [len(batch[2]) for batch in shard_batches], dtype=np.int64
        )
        path = os.path.join(output_dir, f"shard_{len(shard_paths):05d}.npz")
        np.savez(
            path,
            x=np.concatenate([batch[0] for batch in shard_batches], axis=0),
            y=np.concatenate([batch[1] for batch in shard_batches], axis=0),
            indices=np.concatenate(
                [batch[2] for batch in shard_batches], axis=0
            ).astype(np.int32, copy=False),
            node_offsets=np.concatenate((
                np.zeros(1, dtype=np.int64), np.cumsum(node_lengths)
            )),
            edge_offsets=np.concatenate((
                np.zeros(1, dtype=np.int64), np.cumsum(edge_lengths)
            )),
            n_boids=np.asarray(n_boids, dtype=np.int32),
        )
        shard_paths.append(path)
        shard_batches = []

    def flush_batch():
        nonlocal total_batches
        nonlocal x_batch, y_batch, row_batch, col_batch, length_batch
        shard_batches.append(_make_disjoint_batch(
            x_batch,
            y_batch,
            row_batch,
            col_batch,
            length_batch,
            n_boids,
        ))
        total_batches += 1
        x_batch, y_batch, row_batch, col_batch, length_batch = [], [], [], [], []
        if len(shard_batches) >= batches_per_shard:
            flush_shard()

    def emit_sample(sample):
        x, y, row, col = sample
        x_batch.append(x)
        y_batch.append(y)
        row_batch.append(row)
        col_batch.append(col)
        length_batch.append(len(row))
        if len(x_batch) == batch_size:
            flush_batch()

    def emit_random_sample():
        sample_index = int(rng.integers(len(shuffle_buffer)))
        sample = shuffle_buffer[sample_index]
        shuffle_buffer[sample_index] = shuffle_buffer[-1]
        shuffle_buffer.pop()
        emit_sample(sample)

    with ExitStack() as stack:
        caches = [
            stack.enter_context(h5py.File(path, "r")) for path in cache_paths
        ]
        refs = np.asarray(trajectory_refs, dtype=np.int64).reshape(-1, 2)
        for cache_index, cache in enumerate(caches):
            local_refs = refs[refs[:, 0] == cache_index, 1]
            if len(local_refs) == 0:
                continue
            boundaries = boundaries_by_cache[cache_index]
            probe_refs = local_refs[np.linspace(
                0, len(local_refs) - 1, num=min(3, len(local_refs)), dtype=int
            )]
            _validate_adjacency(
                cache,
                [int(boundaries[int(ref)]) for ref in probe_refs],
                perception,
            )

        for cache_index, trajectory_index in rng.permutation(refs):
            cache = caches[int(cache_index)]
            boundaries = boundaries_by_cache[int(cache_index)]
            start = int(boundaries[int(trajectory_index)])
            end = int(boundaries[int(trajectory_index) + 1])
            x, y, rows, cols, lengths = _load_selected_samples(
                cache, start, end, timestep_stride, near_goal_radius
            )
            for timestep in rng.permutation(len(x)):
                edge_count = int(lengths[timestep])
                shuffle_buffer.append((
                    x[timestep].copy(),
                    y[timestep].copy(),
                    rows[timestep, :edge_count].copy(),
                    cols[timestep, :edge_count].copy(),
                ))
                if len(shuffle_buffer) >= shuffle_buffer_size:
                    emit_random_sample()

    while shuffle_buffer:
        emit_random_sample()
    if x_batch:
        flush_batch()
    flush_shard()
    if total_batches < 1:
        raise ValueError("Local backend produced no training batches.")
    return shard_paths, total_batches


def dataset_from_shards(shard_paths):
    """Stream shuffled NPZ shards without retaining a complete chunk in RAM."""
    def generator():
        for path in np.random.permutation(shard_paths):
            with np.load(path) as shard:
                x_all = shard["x"]
                y_all = shard["y"]
                indices_all = shard["indices"]
                node_offsets = shard["node_offsets"]
                edge_offsets = shard["edge_offsets"]
                n_boids = int(shard["n_boids"])
                for batch_index in np.random.permutation(len(node_offsets) - 1):
                    node_start = int(node_offsets[batch_index])
                    node_end = int(node_offsets[batch_index + 1])
                    edge_start = int(edge_offsets[batch_index])
                    edge_end = int(edge_offsets[batch_index + 1])
                    n_nodes = node_end - node_start
                    graph_ids = np.repeat(
                        np.arange(n_nodes // n_boids, dtype=np.int64), n_boids
                    )
                    adjacency = tf.SparseTensor(
                        indices=indices_all[edge_start:edge_end].astype(
                            np.int64, copy=False
                        ),
                        values=np.ones(edge_end - edge_start, dtype=np.float32),
                        dense_shape=np.asarray([n_nodes, n_nodes], dtype=np.int64),
                    )
                    yield (
                        (x_all[node_start:node_end], adjacency, graph_ids),
                        y_all[node_start:node_end],
                    )

    output_signature = (
        (
            tf.TensorSpec(shape=(None, 6), dtype=tf.float32),
            tf.SparseTensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ),
        tf.TensorSpec(shape=(None, 16), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(
        generator, output_signature=output_signature
    ).repeat()
