"""Chunked 2D training with cached or compiled expert data.

Every sample is a single expert transition, learned one step ahead.
"""

import gc
import json
import math
import os
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models.gnn_ca_goal_conditioned_boids import GoalConditionedGNNCABoids
from models.gnn_ca_simple_boids import GNNCASimpleBoids
from modules.boids import (
    BOIDS_ACTIVE_MASK_INDEX,
    BOIDS_MASKED_TARGET_FEATURES,
    CURRENT_GOAL_SLICE,
    PREVIOUS_GOAL_MAX_DISTANCE_INDEX,
    PREVIOUS_GOAL_SLICE,
)
from modules.perturbations import allocate_mode_chunks, perturbation_counts
from runtime.common import run_best
from runtime.cloud.cloud_training import fit_distributed
from runtime.common.backends import data_backend, training_strategy


from runtime.common.subprocess_utils import (
    BENIGN_NATIVE_DIAGNOSTICS, run_with_filtered_stderr,
)


def _run_with_clean_tensorflow_stderr(command):
    return run_with_filtered_stderr(command, BENIGN_NATIVE_DIAGNOSTICS)


def training_run_tag(args):
    from modules.run_tags import expert_tag, run_tag as make_run_tag
    online = args.task == "online_goals"
    return make_run_tag("online2d" if online else "fixed2d", args.train_set_unique,
                        1 if online else args.train_set_repeats, args.loss_type,
                        args.critical_distance, args.distance_weight, args.run_tag_suffix,
                        expert=expert_tag(args.perception, getattr(args, "crowding", 0.02),
                                          getattr(args, "init_scatter", 0.325)))


def make_transition_loss(args):
    """Create the existing transition-aware state imitation loss."""
    upweight = args.loss_type == "goal_weighted"
    critical_distance = float(args.critical_distance)
    distance_weight = float(args.distance_weight)

    def loss(y_true, y_pred):
        active_mask = tf.cast(y_true[..., BOIDS_ACTIVE_MASK_INDEX], y_pred.dtype)
        graph_ids = tf.cast(y_true[..., BOIDS_MASKED_TARGET_FEATURES], tf.int32)
        current_state = y_true[..., :4]
        next_state = y_true[..., 4:8]
        node_mse = tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)
        graph_count = tf.reduce_max(graph_ids) + 1
        active_count = tf.math.unsorted_segment_sum(
            active_mask, graph_ids, graph_count
        )
        total_count = tf.math.unsorted_segment_sum(
            tf.ones_like(active_mask), graph_ids, graph_count
        )
        node_mse *= active_mask * tf.gather(
            total_count / tf.maximum(active_count, 1.0), graph_ids
        )
        if not upweight:
            return node_mse

        centroid = tf.math.unsorted_segment_sum(
            current_state[..., :2] * active_mask[:, None], graph_ids, graph_count
        ) / tf.maximum(active_count[:, None], 1.0)
        active_goal = tf.math.unsorted_segment_mean(
            y_true[..., CURRENT_GOAL_SLICE], graph_ids, graph_count
        )
        previous_goal = tf.math.unsorted_segment_mean(
            y_true[..., PREVIOUS_GOAL_SLICE], graph_ids, graph_count
        )
        max_previous_distance = tf.math.unsorted_segment_max(
            y_true[..., PREVIOUS_GOAL_MAX_DISTANCE_INDEX],
            graph_ids,
            graph_count,
        )
        active_distance = tf.norm(centroid - active_goal, axis=-1)
        previous_distance = tf.norm(centroid - previous_goal, axis=-1)
        near_active = active_distance < critical_distance
        near_eligible_previous = (
            (max_previous_distance >= 0.0)
            & (max_previous_distance < critical_distance)
            & (previous_distance < critical_distance)
        )
        graph_weight = tf.where(
            near_active | near_eligible_previous,
            tf.cast(distance_weight, node_mse.dtype),
            tf.ones_like(active_distance, dtype=node_mse.dtype),
        )
        return node_mse * tf.gather(graph_weight, graph_ids)

    return loss


def build_model(learning_rate, goal_conditioned=True):
    model_type = GoalConditionedGNNCABoids if goal_conditioned else GNNCASimpleBoids
    model = model_type(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model


def build_model_variables(model, n_boids, goal_conditioned=True):
    x = tf.zeros((n_boids, 6 if goal_conditioned else 4), dtype=tf.float32)
    adjacency = tf.SparseTensor(
        indices=tf.zeros((0, 2), dtype=tf.int64),
        values=tf.zeros((0,), dtype=tf.float32),
        dense_shape=(n_boids, n_boids),
    )
    model([x, tf.sparse.reorder(adjacency), tf.constant(0)], training=False)


class _BestCompleteState(tf.keras.callbacks.Callback):
    """Save model and Adam state together whenever validation improves."""

    def __init__(self, manager, min_delta):
        super().__init__()
        self.manager = manager
        self.min_delta = float(min_delta)
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        value = float((logs or {}).get("val_loss", np.inf))
        if value < self.best - self.min_delta:
            self.best = value
            path = self.manager.save()
            print(
                f">>> Saved best complete training state at epoch "
                f"{epoch + 1}: val_loss={value:.6g} ({path})",
                flush=True,
            )


def _worker(args, chunk_index, trajectory_count, state_dir, result_path, run_tag):
    """One isolated chunk worker; process exit releases all host/GPU memory."""
    np.random.seed(args.seed + chunk_index)
    tf.random.set_seed((args.seed if args.task == "online_goals" else args.weight_init_seed) + chunk_index)
    generate_online_goal_chunk, dataset_from_trajectories = data_backend(args, 2)
    mode_counts = args._perturbation_allocations[chunk_index]
    train_trajectories, centers, train_records = generate_online_goal_chunk(
        args, chunk_index, trajectory_count, mode_counts=mode_counts
    )
    validation_index = run_best.validation_chunk_index(args, chunk_index)
    validation_trajectories, _, validation_records = generate_online_goal_chunk(
        args, validation_index, args.val_set_size, validation=True
    )

    strategy = training_strategy(args.backend)
    replicas = strategy.num_replicas_in_sync
    if args.batch_size % replicas:
        raise ValueError(
            f"--batch_size {args.batch_size} must be divisible by {replicas} GPUs."
        )
    per_replica_batch = args.batch_size // replicas
    conditioned = args.task == "online_goals"
    train_data, train_steps = dataset_from_trajectories(
        train_trajectories,
        per_replica_batch,
        args.n_boids,
        args.timestep_stride,
        args.near_goal_radius,
        args.seed + chunk_index,
        shuffle_batches=True,
        goal_conditioned=True,
        retain_previous_goal=args.task == "fixed_waypoints",
    )
    validation_data, validation_steps = dataset_from_trajectories(
        validation_trajectories,
        per_replica_batch,
        args.n_boids,
        args.timestep_stride,
        args.near_goal_radius,
        args.seed + 50_000_021 + validation_index,
        shuffle_batches=False,
        goal_conditioned=True,
        retain_previous_goal=args.task == "fixed_waypoints",
    )

    if not conditioned:
        # Preserve transition metadata for the existing loss, but fixed models
        # receive positions and velocities only (no explicit goal input).
        def physical_inputs(inputs, targets):
            return (inputs[0][:, :4], inputs[1], inputs[2]), targets
        train_data = train_data.map(physical_inputs)
        validation_data = validation_data.map(physical_inputs)

    with strategy.scope():
        model = build_model(args.lr, conditioned)
        build_model_variables(model, args.n_boids, conditioned)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        manager = tf.train.CheckpointManager(
            checkpoint, state_dir, max_to_keep=1
        )
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print(
                f">>> Restored model + Adam state: {manager.latest_checkpoint}",
                flush=True,
            )
            # The restored state holds the best weights and Adam moments of the
            # previous chunk together with its annealed learning rate. The rate
            # encodes a plateau on that chunk's data, not convergence, so the
            # schedule restarts at --lr and this chunk is learned at full rate.
            tf.keras.backend.set_value(model.optimizer.learning_rate, args.lr)
            print(f">>> Learning rate reset to {args.lr:g} for this chunk", flush=True)
        else:
            init_weights = getattr(args, "init_weights", None)
            if init_weights:
                model.load_weights(init_weights).expect_partial()
                print(f">>> Initialized model from: {init_weights}")

    best_state = _BestCompleteState(manager, args.early_stopping_min_delta)
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=args.lr_patience,
            factor=args.lr_red_factor,
            min_lr=args.min_lr,
            min_delta=args.early_stopping_min_delta,
            verbose=1,
        ),
        best_state,
        EarlyStopping(
            monitor="val_loss",
            patience=args.chunk_patience,
            min_delta=args.early_stopping_min_delta,
            restore_best_weights=False,
            verbose=1,
        ),
    ]
    loss_fn = make_transition_loss(args)
    history = fit_distributed(
        model,
        strategy,
        train_data,
        train_steps,
        args.epochs,
        callbacks,
        args.batch_size * args.n_boids,
        args.steps_per_execution,
        loss_fn,
        validation_data,
        validation_steps,
        execution_label=args.backend.capitalize(),
    )
    if not manager.latest_checkpoint:
        raise RuntimeError("Training worker did not save a validation checkpoint.")
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    os.makedirs("saved_models", exist_ok=True)
    weights_path = f"saved_models/best_weights_{run_tag}"
    saved_model_path = f"saved_models/gnca_model_{run_tag}"

    def export():
        model.save_weights(weights_path)
        model.save(saved_model_path, save_format="tf")

    run_best.export_if_best(args, state_dir, chunk_index, best_state.best, export)
    result = {
        "chunk": chunk_index,
        "trajectory_count": trajectory_count,
        "perturbation_counts": mode_counts,
        "centers": centers.tolist(),
        "train_generation": train_records,
        "validation_generation": validation_records,
        "history": history,
        "weights_path": weights_path,
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)

    del train_data, validation_data, train_trajectories, validation_trajectories
    tf.keras.backend.clear_session()
    gc.collect()


def validate_chunked_args_2d(args):
    if args.boids_cache:
        raise ValueError("Use --cache_dir for online caches; --boids_cache is for fixed local training.")
    if args.task == "online_goals" and args.train_set_repeats != 1:
        raise ValueError("Online training requires --train_set_repeats 1.")
    if args.train_set_repeats < 1 or args.batch_size < 1 or args.epochs < 1 or args.n_boids < 1:
        raise ValueError("Repeats, batch size, epochs and agent count must be positive.")
    if args.init_centers_npz:
        raise ValueError("This generated-data pipeline does not accept --init_centers_npz.")
    if args.chunk_size < 1:
        raise ValueError("Chunked training requires --chunk_size > 0.")
    if args.train_set_unique < 1 or args.val_set_size < 1:
        raise ValueError("Training and validation episode counts must be positive.")
    if args.generation_workers < 0:
        raise ValueError("--generation_workers cannot be negative.")
    if args.timestep_stride < 1:
        raise ValueError("--timestep_stride must be at least 1.")
    if args.lr_patience >= args.chunk_patience:
        raise ValueError("--lr_patience must be smaller than --chunk_patience.")
    if args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("--min_lr must be nonnegative and no greater than --lr.")
    if args.task == "online_goals" and args.goal_waypoints_per_episode != 2:
        raise ValueError(
            "This experiment intentionally requires exactly two waypoint "
            "terminations per training episode."
        )
    if len(args.goal_bounds) != 4:
        raise ValueError("--goal_bounds requires x_min x_max y_min y_max.")
    if len(args.start_bounds) != 4:
        raise ValueError("--start_bounds requires x_min x_max y_min y_max.")
    if not (
        args.goal_bounds[0] < args.goal_bounds[1]
        and args.goal_bounds[2] < args.goal_bounds[3]
        and args.start_bounds[0] < args.start_bounds[1]
        and args.start_bounds[2] < args.start_bounds[3]
    ):
        raise ValueError("Goal and start bounds must have positive width and height.")
    if args.goal_min_distance < 0 or args.goal_arrival_radius <= 0:
        raise ValueError("Goal distance must be nonnegative and arrival radius positive.")
    if args.perception <= 0:
        raise ValueError("--perception must be positive.")
    if args.train_quadrants is not None and (
        not args.train_quadrants
        or any(q not in range(4) for q in args.train_quadrants)
        or len(set(args.train_quadrants)) != len(args.train_quadrants)
    ):
        raise ValueError("--train_quadrants must contain distinct IDs from 0 to 3.")
    if args.expert_max_steps < 1 or args.steps_per_execution < 1:
        raise ValueError("Step limits must be positive.")
    if args.eval_online_goal_count < 1 or args.eval_max_steps < 1:
        raise ValueError("Online evaluation goal and step counts must be positive.")
    if args.eval_max_tube_radius <= 0:
        raise ValueError("--eval_max_tube_radius must be positive.")
    if args.eval_expert_r_padding < 0:
        raise ValueError("--eval_expert_r_padding cannot be negative.")
    if args.eval_expert_calibration_max_steps < 1:
        raise ValueError("--eval_expert_calibration_max_steps must be positive.")
    visible_gpus = tf.config.list_physical_devices("GPU")
    if args.backend == "cloud" and not visible_gpus:
        raise ValueError("--backend cloud requires at least one visible GPU.")
    if args.backend == "cloud" and args.batch_size % len(visible_gpus):
        raise ValueError(
            "Cloud --batch_size is global and must be divisible by the "
            f"{len(visible_gpus)} visible GPUs."
        )


def train_chunked_2d(args):
    """Train generated expert chunks, caching them for the local backend."""
    validate_chunked_args_2d(args)
    run_tag = training_run_tag(args)
    total = args.train_set_unique * args.train_set_repeats
    chunks = math.ceil(total / args.chunk_size)
    chunk_sizes = [
        min(args.chunk_size, total - start)
        for start in range(0, total, args.chunk_size)
    ]
    args._perturbation_allocations = allocate_mode_chunks(
        perturbation_counts(args, trajectory_total=total), chunk_sizes
    )
    print(
        f">>> 2D {args.task} training: {total} episodes, "
        f"{chunks} isolated chunks, backend={args.backend}.",
        flush=True,
    )
    print(">>> Data: " + ("expert trajectories cached on disk and streamed in batches" if args.backend == "local"
                           else "compiled expert trajectories and bit-packed graphs held in memory"), flush=True)
    print(">>> Objective: one-step transitions.", flush=True)
    print(f">>> Learning rate across chunks: schedule restarted at {args.lr:g} for every chunk, "
          "from the previous chunk's best weights and optimizer moments.", flush=True)
    all_results = []
    with tempfile.TemporaryDirectory(prefix="gnca_2d_training_") as temp_dir:
        state_dir = os.path.join(temp_dir, "training_state")
        os.makedirs(state_dir)
        args_file = os.path.join(temp_dir, "args.pkl")
        joblib.dump(args, args_file)
        remaining = total
        for chunk_index in range(chunks):
            count = min(args.chunk_size, remaining)
            result_path = os.path.join(temp_dir, f"chunk_{chunk_index:04d}.json")
            print(
                f"\n{'=' * 64}\n  Training chunk {chunk_index + 1}/{chunks} | "
                f"{count} episodes\n{'=' * 64}",
                flush=True,
            )
            returncode = _run_with_clean_tensorflow_stderr(
                [
                    sys.executable,
                    "-m",
                    "runtime.common.training_2d_worker",
                    "--args_file",
                    args_file,
                    "--chunk_index",
                    str(chunk_index),
                    "--trajectory_count",
                    str(count),
                    "--state_dir",
                    state_dir,
                    "--result_path",
                    result_path,
                    "--run_tag",
                    run_tag,
                ]
            )
            if returncode != 0:
                raise RuntimeError(
                    f"Training chunk {chunk_index + 1}/{chunks} failed "
                    f"with exit code {returncode}."
                )
            with open(result_path, encoding="utf-8") as handle:
                all_results.append(json.load(handle))
            remaining -= count

    os.makedirs("saved_history", exist_ok=True)
    os.makedirs("saved_training_data", exist_ok=True)
    joblib.dump(
        {"task": args.task, "chunks": all_results},
        f"saved_history/history_{run_tag}.pkl",
    )
    manifest = {
        "task": args.task,
        "run_tag": run_tag,
        "training_episodes": total,
        "loss_type": args.loss_type,
        "perturbation_counts": perturbation_counts(args, trajectory_total=total),
        "waypoints_per_episode": 2 if args.task == "online_goals" else 3,
        "cacheless": args.backend == "cloud",
        "backend": args.backend,
        "data_pipeline": "compiled_bitpacked" if args.backend == "cloud" else "cached_bitpacked",
        "goal_bounds": list(args.goal_bounds),
        "start_bounds": list(args.start_bounds),
        "goal_min_distance": args.goal_min_distance,
        "goal_arrival_radius": args.goal_arrival_radius,
        "perception": args.perception,
        "crowding": getattr(args, "crowding", 0.02),
        "init_scatter": getattr(args, "init_scatter", 0.325),
        "chunks": all_results,
    }
    with open(
        f"saved_training_data/generation_manifest_{run_tag}.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nSaved weights to saved_models/best_weights_{run_tag}")
    print(f"Saved model to saved_models/gnca_model_{run_tag}")
    if args.backend == "local":
        print(f"Expert caches are reusable in {args.cache_dir}")
    if not args.skip_evaluation:
        output_dir = os.path.join("results", f"inference_{run_tag}")
        command = [
            sys.executable,
            "-m",
            "evaluation.run_inference",
            "--task",
            args.task,
            "--mode",
            "single",
            "--run_tag",
            run_tag,
            "--output_dir",
            output_dir,
            "--n_centers",
            str(args.te_set_size),
            "--quadrants",
            "0",
            "1",
            "2",
            "3",
            "--online_goal_count",
            str(args.eval_online_goal_count),
            "--goal_bounds",
            *[str(value) for value in args.goal_bounds],
            "--goal_min_distance",
            str(args.goal_min_distance),
            "--success_threshold",
            str(args.goal_arrival_radius),
            "--max_tube_radius",
            str(args.eval_max_tube_radius),
            "--r_threshold_mode",
            args.eval_r_threshold_mode,
            "--expert_r_padding",
            str(args.eval_expert_r_padding),
            "--expert_calibration_max_steps",
            str(args.eval_expert_calibration_max_steps),
            "--max_steps",
            str(args.eval_max_steps),
            "--n_boids",
            str(args.n_boids),
            "--perception",
            str(args.perception),
            "--seed",
            str(args.eval_seed),
        ]
        print(f">>> Running post-training evaluation in {output_dir}")
        subprocess.run(command, check=True)
    return run_tag
