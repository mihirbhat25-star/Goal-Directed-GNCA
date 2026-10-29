"""Local backend: bounded-memory packing of precomputed 2D HDF5 caches."""

import os

import h5py
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from spektral.data import Graph

from modules.boids import (
    BOIDS_ACTIVE_MASK_INDEX,
    BOIDS_MASKED_TARGET_FEATURES,
    BOIDS_STATE_FEATURES,
    BOIDS_TRANSITION_TARGET_FEATURES,
    BoidsDataset,
    ensure_goal_transition_metadata,
    ensure_active_mask_targets,
    goal_transition_proximity,
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
    """Rebuild a small validation dataset inside an isolated worker."""
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
    return BoidsDataset(graphs)


def _make_disjoint_batch(
    x_list, y_list, row_list, col_list, length_list, n_boids
):
    batch_size = len(x_list)
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

    if edge_rows:
        indices = np.stack(
            [np.concatenate(edge_rows), np.concatenate(edge_cols)], axis=1
        )
        order = np.lexsort((indices[:, 1], indices[:, 0]))
        indices = indices[order]
        order = np.lexsort((indices[:, 1], indices[:, 0]))
        indices = indices[order]
    else:
        indices = np.zeros((0, 2), dtype=np.int64)
    n_nodes = batch_size * n_boids
    graph_ids = np.repeat(
        np.arange(batch_size, dtype=np.int64), n_boids
    )
    return x, y, indices, graph_ids, n_nodes


def _load_selected_samples(
    cache, start, end, timestep_stride, near_goal_radius
):
    cached_target_features = cache["y"].shape[-1]
    if (
        cached_target_features == BOIDS_TRANSITION_TARGET_FEATURES
        and (timestep_stride <= 1 or near_goal_radius <= 0)
    ):
        selection = slice(start, end, timestep_stride)
        return cache["x"][selection], ensure_active_mask_targets(
            cache["y"][selection]
        )

    x = cache["x"][start:end]
    y = ensure_active_mask_targets(
        ensure_goal_transition_metadata(x, cache["y"][start:end])
    )
    keep = np.zeros(end - start, dtype=bool)
    keep[::timestep_stride] = True
    if near_goal_radius > 0:
        keep |= goal_transition_proximity(x, y, near_goal_radius)
    return x[keep], y[keep]


def _adjacency_indices(state, perception, active_mask=None):
    positions = state[:, :2]
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    neighbors = distances < perception
    np.fill_diagonal(neighbors, False)
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
        neighbors &= active[:, None] & active[None, :]
    return np.nonzero(neighbors)


def write_batch_files(
    cache_path,
    trajectory_indices,
    boundaries,
    n_boids,
    batch_size,
    output_dir,
    *,
    timestep_stride=1,
    near_goal_radius=1.0,
    perception=0.1,
    shuffle_buffer_size=4096,
):
    """Write timestep-shuffled disjoint batches with adjacency rebuilt from x."""
    os.makedirs(output_dir, exist_ok=True)
    batch_paths = []
    x_batch, y_batch, row_batch, col_batch, length_batch = [], [], [], [], []
    shuffle_buffer = []
    shuffle_buffer_size = max(batch_size, int(shuffle_buffer_size))
    rng = np.random.default_rng()

    def flush_batch():
        nonlocal x_batch, y_batch, row_batch, col_batch, length_batch
        x, y, indices, graph_ids, n_nodes = _make_disjoint_batch(
            x_batch,
            y_batch,
            row_batch,
            col_batch,
            length_batch,
            n_boids,
        )
        path = os.path.join(output_dir, f"batch_{len(batch_paths):05d}.npz")
        np.savez(
            path,
            x=x,
            y=y,
            indices=indices,
            graph_ids=graph_ids,
            n_nodes=np.asarray(n_nodes, dtype=np.int64),
        )
        batch_paths.append(path)
        x_batch, y_batch, row_batch, col_batch, length_batch = [], [], [], [], []

    def emit_sample(sample):
        x, y, rows, cols = sample
        x_batch.append(x)
        y_batch.append(y)
        row_batch.append(rows)
        col_batch.append(cols)
        length_batch.append(len(rows))
        if len(x_batch) == batch_size:
            flush_batch()

    def emit_random_sample():
        sample_index = int(rng.integers(len(shuffle_buffer)))
        sample = shuffle_buffer[sample_index]
        shuffle_buffer[sample_index] = shuffle_buffer[-1]
        shuffle_buffer.pop()
        emit_sample(sample)

    with h5py.File(cache_path, "r") as cache:
        for trajectory_index in rng.permutation(
            np.asarray(trajectory_indices, dtype=np.int64)
        ):
            start = int(boundaries[trajectory_index])
            end = int(boundaries[trajectory_index + 1])
            x, y = _load_selected_samples(
                cache, start, end, timestep_stride, near_goal_radius
            )
            for timestep in rng.permutation(len(x)):
                rows, cols = _adjacency_indices(
                    x[timestep],
                    perception,
                    y[timestep, :, BOIDS_ACTIVE_MASK_INDEX],
                )
                shuffle_buffer.append((
                    x[timestep].copy(),
                    y[timestep].copy(),
                    rows,
                    cols,
                ))
                if len(shuffle_buffer) >= shuffle_buffer_size:
                    emit_random_sample()

    while shuffle_buffer:
        emit_random_sample()
    if x_batch:
        flush_batch()
    if not batch_paths:
        raise ValueError("Local 2D backend produced no training batches.")
    return batch_paths, len(batch_paths)


def dataset_from_batch_files(batch_paths):
    """Stream repeatable TensorFlow batches from compact local NPZ files."""
    def generator():
        for path in np.random.permutation(batch_paths):
            with np.load(path) as batch:
                n_nodes = int(batch["n_nodes"])
                indices = batch["indices"]
                adjacency = tf.SparseTensor(
                    indices=indices,
                    values=np.ones(len(indices), dtype=np.float32),
                    dense_shape=np.asarray([n_nodes, n_nodes], dtype=np.int64),
                )
                yield (
                    (batch["x"], adjacency, batch["graph_ids"]),
                    batch["y"],
                )

    output_signature = (
        (
            tf.TensorSpec(
                shape=(None, BOIDS_STATE_FEATURES), dtype=tf.float32
            ),
            tf.SparseTensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ),
        tf.TensorSpec(
            shape=(None, BOIDS_MASKED_TARGET_FEATURES),
            dtype=tf.float32,
        ),
    )
    return tf.data.Dataset.from_generator(
        generator, output_signature=output_signature
    ).repeat()
