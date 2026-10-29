"""Train the 2D GNCA to imitate the 2D Boids GCA.

This remains the public experiment entry point. Bounded cache packing and
streaming live in :mod:`runtime.local.local_2d`; model and experiment semantics stay
here.
"""
from runtime.cli.cli_2d import build_parser
from runtime.common import run_best
import atexit
import gc
import os
import sys
import json
import shutil
import subprocess
import tempfile

if __name__ == "__main__" and any(flag in sys.argv for flag in ("-h", "--help")):
    build_parser().parse_args()

# Silence TensorFlow's benign native CUDA/NUMA registration diagnostics.  Real
# Python exceptions and fatal native errors remain visible.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

from runtime.common.subprocess_utils import install_filtered_stderr

install_filtered_stderr()

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

from spektral.data import DisjointLoader
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from evaluation.evaluate_boids import evaluate
from models.gnn_ca_simple_boids import GNNCASimpleBoids
from modules.boids import (
    BOIDS_ACTIVE_MASK_INDEX,
    BOIDS_MASKED_TARGET_FEATURES,
    CURRENT_GOAL_SLICE,
    PREVIOUS_GOAL_MAX_DISTANCE_INDEX,
    PREVIOUS_GOAL_SLICE,
    Boids,
    make_dataset,
)
from modules.callbacks import ComplexityCallback
from modules.run_tags import expert_tag
from modules.perturbations import (
    config_for_mode,
    perturbation_counts,
    validate_perturbation_args,
)
from runtime.local.local_2d import (
    dataset_from_batch_files,
    load_validation_dataset,
    save_validation_dataset,
    write_batch_files,
)
import h5py
import psutil

# tf.config.run_functions_eagerly(True)

def load_init_centers_from_npz(npz_path):
    """Load start-center configs from an NPZ file (expects a 'centers' array)."""
    if not npz_path:
        return None
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Initial-center npz not found: {npz_path}")

    cache = np.load(npz_path, allow_pickle=True)
    if "centers" not in cache.files:
        raise KeyError(
            f"NPZ '{npz_path}' does not contain a 'centers' key. Available keys: {cache.files}"
        )

    centers = np.array(cache["centers"], dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
        raise ValueError(
            f"Invalid centers shape in '{npz_path}': {centers.shape}. Expected (N, 2) with N > 0."
        )

    print(f"Loaded {len(centers)} initial start centers from {npz_path}")
    return [np.array(c, dtype=np.float32) for c in centers]

def custom_weighted_mse(y_true, y_pred):
    # y_true is [current_state, next_state, active_goal, previous_goal,
    #            max_distance_from_previous_since_switch, graph_id].
    n_features = tf.shape(y_pred)[-1]
    current_state = y_true[..., :n_features]
    next_state = y_true[..., n_features:2*n_features]
    mse = tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)
    active_mask = tf.cast(y_true[..., BOIDS_ACTIVE_MASK_INDEX], mse.dtype)
    graph_ids = tf.cast(y_true[..., BOIDS_MASKED_TARGET_FEATURES], tf.int32)
    n_graphs = tf.reduce_max(graph_ids) + 1
    active_count = tf.math.unsorted_segment_sum(
        active_mask, graph_ids, n_graphs
    )
    total_count = tf.math.unsorted_segment_sum(
        tf.ones_like(active_mask), graph_ids, n_graphs
    )
    mse *= active_mask * tf.gather(
        total_count / tf.maximum(active_count, 1.0), graph_ids
    )

    if UPWEIGHT:
        avg_pos = tf.math.unsorted_segment_sum(
            current_state[..., :2] * active_mask[:, None], graph_ids, n_graphs
        ) / tf.maximum(active_count[:, None], 1.0)
        active_goals = tf.math.unsorted_segment_mean(
            y_true[..., CURRENT_GOAL_SLICE], graph_ids, n_graphs
        )
        previous_goals = tf.math.unsorted_segment_mean(
            y_true[..., PREVIOUS_GOAL_SLICE], graph_ids, n_graphs
        )
        max_previous_distances = tf.math.unsorted_segment_max(
            y_true[..., PREVIOUS_GOAL_MAX_DISTANCE_INDEX],
            graph_ids,
            n_graphs,
        )
        active_distances = tf.norm(avg_pos - active_goals, axis=-1)
        previous_distances = tf.norm(avg_pos - previous_goals, axis=-1)
        near_active_goal = active_distances < args.critical_distance
        near_eligible_previous_goal = (
            (max_previous_distances >= 0.0)
            & (max_previous_distances < args.critical_distance)
            & (previous_distances < args.critical_distance)
        )
        near_transition_goal = near_active_goal | near_eligible_previous_goal
        graph_weight = tf.where(
            near_transition_goal,
            tf.cast(args.distance_weight, mse.dtype),
            tf.ones_like(active_distances, dtype=mse.dtype),
        )
        goal_weight = tf.gather(graph_weight, graph_ids)
    else:
        goal_weight = 1.0
    return mse * goal_weight


def add_graph_ids_to_targets(inputs, targets):
    """Append Spektral's disjoint graph id for graph-specific loss weights."""
    graph_ids = tf.cast(inputs[2], targets.dtype)
    targets = tf.concat([targets, graph_ids[:, None]], axis=-1)
    targets = tf.ensure_shape(
        targets, [None, BOIDS_MASKED_TARGET_FEATURES + 1]
    )
    return inputs, targets


def build_run_tag(unique_reps, tr_repeats, upweight, critical_distance, distance_weight, run_tag_suffix,
                  expert=""):
    from modules.run_tags import run_tag as make_run_tag
    return make_run_tag("fixed2d", unique_reps, tr_repeats, "goal_weighted" if upweight else "mse",
                        critical_distance, distance_weight, run_tag_suffix, expert=expert)

def print_run_parameters(args, upweight, run_tag_suffix, effective_unique_reps, run_tag):
    print("\n>>> Full run parameters (before training):")
    for key in sorted(vars(args).keys()):
        print(f"  {key}: {getattr(args, key)}")
    print(f"  UPWEIGHT: {upweight}")
    print(f"  run_tag_suffix: {run_tag_suffix}")
    print(f"  effective_unique_reps: {effective_unique_reps}")
    print(f"  run_tag: {run_tag}")

class BestAfterEpochCallback(tf.keras.callbacks.Callback):
    """Saves weights whenever val_loss improves after min_epoch."""
    def __init__(self, save_path, min_epoch=50):
        super().__init__()
        self.save_path = save_path
        self.min_epoch = min_epoch
        self.best_val_loss = float('inf')
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.min_epoch:
            val_loss = logs.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.model.save_weights(self.save_path)
                print(f"\nNew best val_loss after epoch 50: {val_loss:.6f} at epoch {self.best_epoch}")

class _BestInfo:
    """Minimal checkpoint-info object compatible with the post-training restore block."""
    def __init__(self, best_epoch, best_val_loss):
        self.best_epoch    = best_epoch
        self.best_val_loss = best_val_loss


class _HistoryProxy:
    """Wraps a plain dict so it has the same .history attribute as a Keras History."""
    def __init__(self, history_dict):
        self.history = history_dict


def _build_model(lr, weight_init_seed=None):
    """Construct and compile a fresh GNNCASimpleBoids. Called once per chunk."""
    if weight_init_seed is not None:
        tf.random.set_seed(int(weight_init_seed))
    m = GNNCASimpleBoids(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    m.compile(optimizer=Adam(learning_rate=lr), loss=custom_weighted_mse, run_eagerly=True)
    return m


def _build_model_variables(model, n_boids):
    """Run a dummy forward pass so Keras variables exist before load_weights."""
    x_dummy = tf.zeros((n_boids, 4), dtype=tf.float32)
    a_dummy = tf.SparseTensor(
        indices=tf.zeros((0, 2), dtype=tf.int64),
        values=tf.zeros((0,), dtype=tf.float32),
        dense_shape=(n_boids, n_boids),
    )
    model([x_dummy, tf.sparse.reorder(a_dummy), tf.constant(0)], training=False)


def run_chunked_subprocess(boids_cache_path, flat_traj_indices, boundaries, cache_info, val_data, run_tag):
    """Chunked 2D training with one subprocess per chunk to release RAM."""
    chunk_size = args.chunk_size
    chunk_patience = args.chunk_patience
    n_boids_c = cache_info['n_boids']
    stride = cache_info['timestep_stride']
    near_goal_radius = cache_info.get('near_goal_radius', 1.0)
    cache_repeats = cache_info.get('cache_repeats', 1)
    centers = cache_info.get('centers', None)
    checkpoint_path = f"saved_models/best_weights_{run_tag}"

    from boids.generate_boids_cache import QUADRANTS

    def _center_quadrant(flat_idx):
        if centers is None:
            return 0
        ci = int(flat_idx) // cache_repeats
        cx, cy = centers[ci]
        for q, (xmn, xmx, ymn, ymx, _) in enumerate(QUADRANTS):
            if xmn <= cx <= xmx and ymn <= cy <= ymx:
                return q
        return 0

    q_buckets = [[] for _ in range(4)]
    for fi in flat_traj_indices:
        q_buckets[_center_quadrant(fi)].append(fi)
    for q in range(4):
        np.random.shuffle(q_buckets[q])

    active_quads = [q for q in range(4) if q_buckets[q]]
    if active_quads and chunk_size < len(active_quads):
        raise ValueError(f"--chunk_size must be at least {len(active_quads)} so every chunk holds each training quadrant.")
    n_per_q = chunk_size // len(active_quads) if active_quads else chunk_size
    remainder = chunk_size % len(active_quads) if active_quads else 0
    q_ptrs = [0, 0, 0, 0]
    chunks = []
    while any(q_ptrs[q] < len(q_buckets[q]) for q in active_quads):
        chunk = []
        for i, q in enumerate(active_quads):
            take = n_per_q + (1 if i < remainder else 0)
            end = min(q_ptrs[q] + take, len(q_buckets[q]))
            chunk.extend(q_buckets[q][q_ptrs[q]:end])
            q_ptrs[q] = end
        if chunk:
            chunks.append(np.array(chunk, dtype=np.int64))

    print(f"\n>>> Chunked training: {sum(len(c) for c in chunks)} trajectories, "
          f"chunk_size={chunk_size} ({len(chunks)} balanced chunks), "
          f"chunk_patience={chunk_patience} (subprocess per chunk)")

    current_lr = args.lr
    with tempfile.TemporaryDirectory(prefix="gnca_2d_chunks_") as tmpdir:
        boundaries_file = os.path.join(tmpdir, "boundaries.npy")
        val_npz_file = os.path.join(tmpdir, "validation.npz")
        lr_state_file = os.path.join(tmpdir, "learning_rate.npy")
        run_best_path = os.path.join(tmpdir, "run_best", "weights")
        np.save(boundaries_file, boundaries)
        save_validation_dataset(val_data, val_npz_file, n_boids_c)

        for chunk_num, chunk_idxs in enumerate(chunks, start=1):
            q_counts = [0, 0, 0, 0]
            for fi in chunk_idxs:
                q_counts[_center_quadrant(fi)] += 1
            q_summary = "  ".join(f"{QUADRANTS[q][4]}:{q_counts[q]}" for q in range(4))

            chunk_file = os.path.join(tmpdir, f"chunk_{chunk_num:04d}.json")
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump([int(i) for i in chunk_idxs], f)

            try:
                rss_mb = psutil.Process().memory_info().rss / 1e6
            except ImportError:
                rss_mb = float("nan")
            print(f"\n--- Chunk {chunk_num}/{len(chunks)}  [{q_summary}] | parent RAM: {rss_mb:.0f} MB ---")

            cmd = [
                sys.executable,
                "-m",
                "boids.run_boids",
                "--_chunk_worker",
                "--_chunk_indices_file", chunk_file,
                "--_cache_path", boids_cache_path,
                "--_checkpoint_path", checkpoint_path,
                "--_n_boids_cache", str(n_boids_c),
                "--_boundaries_file", boundaries_file,
                "--_val_npz_file", val_npz_file,
                "--_lr_state_file", lr_state_file,
                "--lr", str(args.lr),
                "--_run_best_path", run_best_path,
                "--_chunk_number", str(chunk_num - 1),
                "--batch_size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--es_patience", str(chunk_patience),
                "--lr_patience", str(args.lr_patience),
                "--early_stopping_min_delta", str(args.early_stopping_min_delta),
                "--lr_red_factor", str(args.lr_red_factor),
                "--min_lr", str(args.min_lr),
                "--weight_init_seed", str(args.weight_init_seed),
                "--n_boids", str(args.n_boids),
                "--loss_type", args.loss_type,
                "--critical_distance", str(args.critical_distance),
                "--distance_weight", str(args.distance_weight),
                "--timestep_stride", str(stride),
                "--near_goal_radius", str(near_goal_radius),
                "--perception", str(args.perception),
                "--shuffle_buffer_size", str(args.shuffle_buffer_size),
            ]
            if not run_best.enabled(args):
                cmd.append("--last_chunk_selection")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    f"2D chunk worker {chunk_num}/{len(chunks)} failed with exit code {result.returncode}"
                )
            if not os.path.exists(lr_state_file):
                raise RuntimeError(
                    f"2D chunk worker {chunk_num}/{len(chunks)} did not save its learning rate"
                )
            current_lr = float(np.load(lr_state_file))
            # The next chunk continues from this chunk's best weights and restarts
            # the learning-rate schedule at --lr, like every other trainer.
            print(f"  Chunk ended at learning rate {current_lr:g}; the next chunk restarts at {args.lr:g}")
            gc.collect()
            try:
                rss_mb = psutil.Process().memory_info().rss / 1e6
                print(f"  Parent RAM after chunk process exited: {rss_mb:.0f} MB")
            except ImportError:
                pass

        final_model = _build_model(current_lr, args.weight_init_seed)
        _build_model_variables(final_model, n_boids_c)
        if run_best.enabled(args):
            final_model.load_weights(run_best_path).expect_partial()
            final_model.save_weights(checkpoint_path)
            source = "the chunk with the lowest validation loss over the run"
        else:
            final_model.load_weights(checkpoint_path).expect_partial()
            source = "the last chunk's best weights"
    print(f"\nChunked training done ({len(chunks)} chunks). Final model = {source}.")
    return _HistoryProxy({'loss': []}), final_model, _BestInfo(len(chunks), float('nan'))


def run(train_data, val_data, run_tag):
    tf.random.set_seed(args.weight_init_seed)
    model = GNNCASimpleBoids(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    optimizer = Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss=custom_weighted_mse)

    train_loader = DisjointLoader(train_data, node_level=True, batch_size=args.batch_size)
    val_loader = DisjointLoader(val_data, node_level=True, batch_size=args.batch_size)

    checkpoint_path = f"saved_models/best_weights_{run_tag}"
    best_after50_cb = BestAfterEpochCallback(save_path=checkpoint_path, min_epoch=args.best_after_epoch)

    history = model.fit(
        train_loader.load().map(add_graph_ids_to_targets),
        steps_per_epoch=train_loader.steps_per_epoch,
        epochs=args.epochs, # Modified to use args
        validation_data=val_loader.load().map(add_graph_ids_to_targets),
        validation_steps=val_loader.steps_per_epoch,
        callbacks=[
            EarlyStopping(
                patience=args.es_patience, min_delta=args.early_stopping_min_delta,
                restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(patience=args.lr_patience, factor=args.lr_red_factor,
                              min_lr=args.min_lr, min_delta=args.early_stopping_min_delta, verbose=1),
            # RESTORED: Complexity Callback
            ComplexityCallback(test_every=args.test_complexity_every),
            best_after50_cb,
        ],
    )

    return history, model, best_after50_cb

####################################################################################
# Configuration
####################################################################################
parser = build_parser()

####################################################################################
# Training
####################################################################################

args = parser.parse_args()
np.random.seed(args.seed)
_perturbation_total = (
    args.train_set_unique
    if args.task == "online_goals"
    else args.train_set_unique * args.train_set_repeats
)
validate_perturbation_args(
    args, 2, trajectory_total=_perturbation_total
)

UPWEIGHT = args.loss_type == "goal_weighted"
print(
    f"\n>>> Loss config: {args.loss_type} (UPWEIGHT={UPWEIGHT}, critical_distance={args.critical_distance}, distance_weight={args.distance_weight})"
)

# Online goals and the cloud backend use the chunked generated-data pipeline;
# only cache-backed fixed-waypoint training stays here.
if args.task == "online_goals" or args.backend == "cloud":
    from runtime.common.training_2d import train_chunked_2d

    train_chunked_2d(args)
    sys.exit(0)

if args._chunk_worker:
    chunk_flat = np.array(json.load(open(args._chunk_indices_file)), dtype=np.int64)
    boundaries = np.load(args._boundaries_file)

    worker_tmpdir = tempfile.mkdtemp(prefix="gnca_2d_worker_batches_")
    atexit.register(shutil.rmtree, worker_tmpdir, ignore_errors=True)
    print(
        f">>> Worker setup: {len(chunk_flat)} trajectories, "
        f"batch_size={args.batch_size}, stride={args.timestep_stride}, "
        f"near_goal_radius={args.near_goal_radius}",
        flush=True,
    )
    print(">>> Worker setup: writing compact train batch files...", flush=True)
    train_batch_paths, train_steps = write_batch_files(
        args._cache_path,
        chunk_flat,
        boundaries,
        args._n_boids_cache,
        args.batch_size,
        os.path.join(worker_tmpdir, "train_batches"),
        timestep_stride=args.timestep_stride,
        near_goal_radius=args.near_goal_radius,
        perception=args.perception,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    print(
        f">>> Worker setup: wrote {len(train_batch_paths)} train batches "
        f"({train_steps} steps)",
        flush=True,
    )
    train_data = dataset_from_batch_files(train_batch_paths).map(add_graph_ids_to_targets)
    data_val = load_validation_dataset(args._val_npz_file)

    model = _build_model(args.lr, args.weight_init_seed)
    val_loader = DisjointLoader(data_val, node_level=True, batch_size=args.batch_size)

    if os.path.exists(args._checkpoint_path + ".index"):
        _build_model_variables(model, args._n_boids_cache)
        model.load_weights(args._checkpoint_path).expect_partial()
        val_loader = DisjointLoader(data_val, node_level=True, batch_size=args.batch_size)
        print(f">>> Worker initialized from stage checkpoint: {args._checkpoint_path}")

    chunk_history = model.fit(
        train_data,
        steps_per_epoch=train_steps,
        epochs=args.epochs,
        validation_data=val_loader.load().map(add_graph_ids_to_targets),
        validation_steps=val_loader.steps_per_epoch,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=args.es_patience, min_delta=args.early_stopping_min_delta,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=args.lr_patience,
                              factor=args.lr_red_factor, min_lr=args.min_lr,
                              min_delta=args.early_stopping_min_delta, verbose=1),
        ],
    )
    model.save_weights(args._checkpoint_path)
    if args._run_best_path:
        # The validation set of this path is built once per run, so chunk losses compare directly.
        os.makedirs(os.path.dirname(args._run_best_path), exist_ok=True)
        run_best.export_if_best(args, os.path.dirname(args._run_best_path), args._chunk_number,
                                min(chunk_history.history["val_loss"]),
                                lambda: model.save_weights(args._run_best_path))
    if args._lr_state_file is None:
        raise RuntimeError("2D chunk worker requires --_lr_state_file")
    final_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    np.save(args._lr_state_file, np.array(final_lr, dtype=np.float64))
    print(f">>> Worker final learning rate: {final_lr:g}", flush=True)
    shutil.rmtree(worker_tmpdir, ignore_errors=True)
    sys.exit(0)

run_tag_suffix = args.run_tag_suffix
train_init_centers = load_init_centers_from_npz(args.init_centers_npz)
effective_unique_reps = len(train_init_centers) if train_init_centers is not None else args.train_set_unique
run_tag = build_run_tag(
    unique_reps=effective_unique_reps,
    tr_repeats=args.train_set_repeats,
    upweight=UPWEIGHT,
    critical_distance=args.critical_distance,
    distance_weight=args.distance_weight,
    run_tag_suffix=run_tag_suffix,
    expert=expert_tag(args.perception, args.crowding, args.init_scatter),
)
print_run_parameters(args, UPWEIGHT, run_tag_suffix, effective_unique_reps, run_tag)

print(f"\n>>> Generating dataset...")
boids_cache  = args.boids_cache if args.boids_cache else None
use_chunked  = args.chunk_size > 0 and boids_cache is not None

val_data = make_dataset(
    unique_reps=args.val_set_size,
    repeat_reps=1,
    save_config=False,
    n_boids=args.n_boids,
    perception=args.perception,
    crowding=args.crowding,
    init_scatter=args.init_scatter,
    random_init=True,
    goal_arrival_radius=args.goal_arrival_radius,
)

if use_chunked:
    # ------------------------------------------------------------------ #
    # Chunked path: extract metadata without loading all graphs into RAM. #
    # ------------------------------------------------------------------ #
    training_boids = Boids(n_boids=args.n_boids, perception=args.perception,
                           crowding=args.crowding, init_scatter=args.init_scatter)
    with h5py.File(boids_cache, 'r') as _f:
        _centers       = _f['centers'][:]
        _cache_unique  = int(_f.attrs['unique_reps'])
        _cache_repeats = int(_f.attrs['repeats'])
        _n_boids_cache = int(_f.attrs['n_boids'])
        _traj_lengths  = _f['traj_lengths'][:] if 'traj_lengths' in _f else None

    _n_train     = min(effective_unique_reps, _cache_unique)

    # Filter to selected quadrants first if specified, then select from filtered set
    if args.train_quadrants is not None:
        from boids.generate_boids_cache import QUADRANTS
        # Group all centers by quadrant
        _idxs_by_quad = {q: [] for q in args.train_quadrants}
        for ti in range(_cache_unique):
            cx, cy = _centers[ti]
            for q in args.train_quadrants:
                xmn, xmx, ymn, ymx, _ = QUADRANTS[q]
                if xmn <= cx <= xmx and ymn <= cy <= ymx:
                    _idxs_by_quad[q].append(ti)
                    break
        # Balance selection across quadrants
        n_per_q = _n_train // len(args.train_quadrants)
        remainder = _n_train % len(args.train_quadrants)
        _train_idxs_list = []
        for i, q in enumerate(args.train_quadrants):
            take = n_per_q + (1 if i < remainder else 0)
            take = min(take, len(_idxs_by_quad[q]))
            _train_idxs_list.extend(np.random.choice(_idxs_by_quad[q], size=take, replace=False))
        _train_idxs = np.array(_train_idxs_list, dtype=np.int64)
        _n_train = len(_train_idxs)
        print(f">>> Filtered to quadrants {args.train_quadrants}: {_n_train} unique centers selected")
    else:
        _train_idxs = np.random.choice(_cache_unique, size=_n_train, replace=False)
    _unseen_idxs = np.array([i for i in range(_cache_unique) if i not in set(_train_idxs)])
    training_boids.rand_configs   = [np.array(_centers[i], dtype=np.float32) for i in _train_idxs]
    training_boids.unseen_configs = [np.array(_centers[i], dtype=np.float32) for i in _unseen_idxs]

    _n_reps = min(args.train_set_repeats, _cache_repeats)
    _flat_traj_idxs = np.array([ti * _cache_repeats + rj
                                 for ti in _train_idxs
                                 for rj in range(_n_reps)], dtype=np.int64)

    if _traj_lengths is not None:
        _boundaries = np.concatenate([[0], np.cumsum(_traj_lengths)])
    else:
        _total_samples    = int(h5py.File(boids_cache, 'r')['x'].shape[0])
        _samples_per_traj = _total_samples // (_cache_unique * _cache_repeats)
        _boundaries       = np.arange(_cache_unique * _cache_repeats + 1) * _samples_per_traj

    _cache_info = {'n_boids': _n_boids_cache, 'timestep_stride': args.timestep_stride,
                   'near_goal_radius': args.near_goal_radius,
                   'cache_repeats': _cache_repeats,
                   'centers': _centers}
    print(f">>> Chunked training: cache={boids_cache}, "
          f"{_n_train} unique x {_n_reps} reps = {len(_flat_traj_idxs)} trajectories, "
          f"chunk_size={args.chunk_size}, chunk_patience={args.chunk_patience}")

    history, model, best_after50_cb = run_chunked_subprocess(
        boids_cache, _flat_traj_idxs, _boundaries, _cache_info, val_data, run_tag
    )

else:
    # ------------------------------------------------------------------ #
    # Standard path: load all training graphs into RAM, then train.      #
    # ------------------------------------------------------------------ #
    if boids_cache:
        with h5py.File(boids_cache, 'r') as _f:
            cache_total_unique = int(_f.attrs["unique_reps"])
        print(f">>> Using precomputed cache: {boids_cache} "
              f"({cache_total_unique} total unique centers, training on {effective_unique_reps})")

    _perturbation_configs = None
    if not boids_cache:
        _counts = perturbation_counts(
            args,
            trajectory_total=(
                effective_unique_reps * args.train_set_repeats
            ),
        )
        _modes = [mode for mode, count in _counts.items() for _ in range(count)]
        np.random.default_rng(args.seed).shuffle(_modes)
        _perturbation_configs = [
            config_for_mode(args, mode) for mode in _modes
        ]
    train_data, training_boids = make_dataset(
        unique_reps=effective_unique_reps,
        repeat_reps=args.train_set_repeats,
        save_config=train_init_centers is None and not boids_cache,
        n_boids=args.n_boids,
        perception=args.perception,
        crowding=args.crowding,
        init_scatter=args.init_scatter,
        return_boids=True,
        random_init=train_init_centers if train_init_centers is not None else True,
        boids_cache_npz=boids_cache,
        timestep_stride=args.timestep_stride,
        near_goal_radius=args.near_goal_radius,
        perturbation_configs=_perturbation_configs,
        goal_arrival_radius=args.goal_arrival_radius,
    )

    if train_init_centers is not None:
        # Preserve reused center list for downstream evaluate()
        training_boids.rand_configs = [np.array(c, dtype=np.float32) for c in train_init_centers]

    history, model, best_after50_cb = run(train_data, val_data, run_tag)

print(f"\n>>> Saving model and history with run_tag='{run_tag}'...")
os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_training_data", exist_ok=True)
os.makedirs("saved_history", exist_ok=True)
model.save(f"saved_models/gnca_model_{run_tag}", save_format="tf")
joblib.dump(history.history, f"saved_history/history_{run_tag}.pkl")
joblib.dump(training_boids, f"saved_training_data/training_boids_{run_tag}.pkl")

# Restore best weights onto the existing model object
checkpoint_path = f"saved_models/best_weights_{run_tag}"
if best_after50_cb.best_epoch is not None:
    print(f"\nRestoring best weights (epoch {best_after50_cb.best_epoch}, val_loss={best_after50_cb.best_val_loss:.6f})...")
    model.load_weights(checkpoint_path)
else:
    print(f"\nWarning: No best-weights checkpoint found. Using early-stopped model.")

if args.skip_evaluation:
    print(">>> Skipping all post-training inference and visualization.")
    sys.exit(0)

####################################################################################
# Evaluation
####################################################################################
max_trajectory_len = 2000
n_boids = args.n_boids

# Assemble test centers for robustness visualization
if args.viz_centers_source == "trained":
    test_centers = training_boids.rand_configs[:args.viz_n_centers] if training_boids.rand_configs else None
    print(f">>> Viz test centers: {len(test_centers) if test_centers else 0} trained centers")
elif args.viz_centers_source == "unseen_cache":
    if not training_boids.unseen_configs:
        print("Warning: No unseen_cache centers available. Falling back to random.")
        test_centers = None
    else:
        test_centers = training_boids.unseen_configs[:args.viz_n_centers]
elif args.viz_centers_source == "fresh_quadrant":
    # Sample brand-new centers per quadrant - not from the cache at all.
    # Verified not to match any training center (continuous space -> exact collision ~ 0).
    from boids.generate_boids_cache import QUADRANTS, build_exclusion_zone
    _excl       = build_exclusion_zone(training_boids.goal_positions, args.exclusion_size)
    _selected_q = args.selected_quadrants if args.selected_quadrants is not None else [0, 1, 2, 3]
    _n_per_q    = max(1, args.viz_n_centers // len(_selected_q))
    _train_set  = {tuple(np.round(c, 6)) for c in training_boids.rand_configs}
    _sampler    = Boids(n_boids=args.n_boids, perception=args.perception,
                        crowding=args.crowding, init_scatter=args.init_scatter)
    test_centers = []
    for _q_idx in _selected_q:
        _xmn, _xmx, _ymn, _ymx, _qlabel = QUADRANTS[_q_idx]
        _q_centers = []
        while len(_q_centers) < _n_per_q:
            _, _, _, _c = _sampler.get_random_init(
                args.n_boids, save_config=False,
                bounds=(_xmn, _xmx, _ymn, _ymx),
                exclusion_zone=_excl,
            )
            if tuple(np.round(_c, 6)) not in _train_set:
                _q_centers.append(_c)
        test_centers.extend(_q_centers)
    print(f">>> Viz test centers: {len(test_centers)} fresh per-quadrant centers "
          f"({_n_per_q} per quadrant, {len(_selected_q)} quadrants), none in training set")
else:  # random
    test_centers = None
    print(f">>> Viz test centers: random (fresh)")

evaluate(
    model,
    max_trajectory_len,
    n_boids,
    saved_boids=training_boids,
    viz_mode=args.viz_mode,
    traj_cache_path=f"saved_trajectory/viz_trajectories_{run_tag}.npz",
    run_tag=run_tag,
    n_show=args.viz_n_show if args.viz_n_show is not None else effective_unique_reps,
    specific_runs=args.viz_runs,
    output_dir=".",
    test_centers=test_centers,
    viz_trained=args.viz_trained,
)

####################################################################################
# Plot SampEn and Correlation Dimension
####################################################################################
if args.test_complexity_every > 0:
    # This assumes complexities.npz was generated by the callback
    if os.path.exists("complexities.npz"):
        c = np.load("complexities.npz")["complexities"]
        means = c[:, 0, :]
        stds = c[:, 1, :]
        x = np.arange(means.shape[0] - 1) * 10

        plt.figure(figsize=(4.5, 2))
        plt.subplot(121)
        plt.axhline(means[:, 0].mean(), label="True")
        plt.plot(x, means[:-1, 1], " x", label="GNCA")
        plt.xlabel("Epoch")
        plt.ylabel("SampEn")
        plt.xticks(x[::2])
        plt.legend()

        plt.subplot(122)
        plt.axhline(means[:, 2].mean(), label="True")
        plt.plot(x, means[:-1, 3], " x", label="GNCA")
        plt.xlabel("Epoch")
        plt.ylabel("CD")
        plt.xticks(x[::2])
        plt.legend()
        plt.tight_layout()

        plt.savefig("complexities.pdf", bbox_inches="tight")
