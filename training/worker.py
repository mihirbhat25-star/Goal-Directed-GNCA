"""One training chunk in its own process, whose exit releases all host and GPU memory.

The chunk generates its expert trajectories and validation episodes, restores the
previous chunk's best weights and Adam state with the learning rate reset to the
stage's rate, trains until validation stalls, and exports its best epoch to the run's
checkpoint, so the last chunk's best epoch is the stage's model. Private entry point of
``training.train``.
"""

import argparse
import gc
import json
import os

# Configure native libraries before TensorFlow/ABSL are imported. NVIDIA containers may
# predefine these at a noisier level, so assigning is intentional.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

from training.native_output import install_filtered_stderr

install_filtered_stderr()

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models.gnca import build_gnca
from training.batches import transition_dataset, window_dataset
from training.data import VALIDATION_SEED_OFFSET, balanced_region_counts, chunk_plan, generate_chunk
from training.fit import fit_distributed, training_strategy
from training.loss import make_transition_loss
from training.rollout import make_rollout_forward


def mean_degree(dataset, batches=32):
    """Mean neighbor count over the graphs of the first training batches."""
    edges = nodes = 0
    for inputs, _ in dataset.take(batches):
        edges += int(tf.shape(inputs[1].indices)[0])
        nodes += int(inputs[1].dense_shape[0])
    return edges / max(nodes, 1)


class BestCompleteState(tf.keras.callbacks.Callback):
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


def chunk_datasets(settings, chunk_index, train_trajectories, validation_trajectories, per_replica_batch, replicas):
    """Training and validation batches of the stage's objective, with their batch counts."""
    dimension, n_boids = settings["dimension"], settings["expert"]["n_boids"]
    conditioned = settings["goal_conditioned"]
    train_seed = settings["seed"] + chunk_index
    validation_seed = settings["seed"] + VALIDATION_SEED_OFFSET + chunk_index
    if settings["stage"] == "bptt":
        window = settings["window"]
        return (
            *window_dataset(train_trajectories, per_replica_batch, n_boids, dimension, conditioned, window,
                            train_seed, shuffle_batches=True),
            *window_dataset(validation_trajectories, per_replica_batch, n_boids, dimension, conditioned, window,
                            validation_seed, shuffle_batches=False, min_batches=replicas),
        )
    stride, radius = settings["timestep_stride"], settings["near_goal_radius"]
    return (
        *transition_dataset(train_trajectories, per_replica_batch, n_boids, dimension, conditioned, stride, radius,
                            train_seed, shuffle_batches=True),
        *transition_dataset(validation_trajectories, per_replica_batch, n_boids, dimension, conditioned, stride,
                            radius, validation_seed, shuffle_batches=False),
    )


def run_chunk(settings, chunk_index, state_dir, result_path):
    np.random.seed(settings["seed"] + chunk_index)
    tf.random.set_seed(settings["seed"] + chunk_index)
    dimension, n_boids = settings["dimension"], settings["expert"]["n_boids"]
    conditioned = settings["goal_conditioned"]
    lr = settings["lr"]

    region_counts, mode_counts = chunk_plan(settings)[chunk_index]
    train_trajectories, centers, train_records = generate_chunk(
        settings, chunk_index, region_counts, mode_counts=mode_counts
    )
    validation_trajectories, _, validation_records = generate_chunk(
        settings, chunk_index, balanced_region_counts(settings["val_set_size"], dimension), validation=True
    )

    strategy = training_strategy()
    replicas = strategy.num_replicas_in_sync
    if settings["batch_size"] % replicas:
        raise ValueError(f"The batch size {settings['batch_size']} must be divisible by {replicas} replicas.")
    per_replica_batch = settings["batch_size"] // replicas
    train_data, train_steps, validation_data, validation_steps = chunk_datasets(
        settings, chunk_index, train_trajectories, validation_trajectories, per_replica_batch, replicas
    )
    del train_trajectories, validation_trajectories
    gc.collect()

    with strategy.scope():
        model = build_gnca(dimension, conditioned, n_boids, learning_rate=lr)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        manager = tf.train.CheckpointManager(checkpoint, state_dir, max_to_keep=1)
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print(f">>> Restored model + Adam state: {manager.latest_checkpoint}", flush=True)
            # The restored state holds the best weights and Adam moments of the
            # previous chunk together with its annealed learning rate. The rate
            # encodes a plateau on that chunk's data, not convergence, so the
            # schedule restarts at the stage's rate and this chunk is learned at full rate.
            tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
            print(f">>> Learning rate reset to {lr:g} for this chunk", flush=True)
        elif settings["init_weights"]:
            # Weights only: the checkpoint also holds the stage it came from's Adam
            # state and annealed learning rate, which would replace the stage's rate.
            # Loading into a scratch model and copying the weights leaves this optimizer fresh.
            scratch = build_gnca(dimension, conditioned, n_boids, learning_rate=lr)
            scratch.load_weights(settings["init_weights"]).assert_existing_objects_matched().expect_partial()
            model.set_weights(scratch.get_weights())
            del scratch
            # The scale is a weight of the checkpoint; recomputing it would change the warm-started model.
            print(f">>> Initialized model from: {settings['init_weights']} (weights only; fresh optimizer "
                  f"at lr {lr:g}); summed term scaled by {float(model.sum_scale.numpy()):.4f}", flush=True)
        else:
            degree = mean_degree(train_data)
            model.set_sum_scale(degree)
            print(f">>> Mean neighbor count of the training graphs {degree:.2f}; "
                  f"summed term scaled by {float(model.sum_scale.numpy()):.4f}", flush=True)

    best_state = BestCompleteState(manager, settings["early_stopping_min_delta"])
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=settings["lr_patience"],
            factor=settings["lr_red_factor"],
            min_lr=settings["min_lr"],
            min_delta=settings["early_stopping_min_delta"],
            verbose=1,
        ),
        best_state,
        EarlyStopping(
            monitor="val_loss",
            patience=settings["chunk_patience"],
            min_delta=settings["early_stopping_min_delta"],
            restore_best_weights=False,
            verbose=1,
        ),
    ]
    loss_fn = make_transition_loss(dimension, settings["critical_distance"], settings["distance_weight"])
    forward_loss = None
    if settings["stage"] == "bptt":
        forward_loss = make_rollout_forward(model, loss_fn, settings["window"], settings["expert"]["perception"],
                                            conditioned, dimension)
    history = fit_distributed(
        model,
        strategy,
        train_data,
        train_steps,
        settings["epochs"],
        callbacks,
        settings["batch_size"] * n_boids,
        settings["steps_per_execution"],
        loss_fn,
        validation_data,
        validation_steps,
        forward_loss=forward_loss,
        grad_clip_norm=settings["grad_clip_norm"],
    )
    if not manager.latest_checkpoint:
        raise RuntimeError("Training worker did not save a validation checkpoint.")
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    model.save_weights(settings["weights_path"])

    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump({
            "chunk": chunk_index,
            "region_counts": {str(region): count for region, count in region_counts.items()},
            "perturbation_counts": mode_counts,
            "centers": centers.tolist(),
            "train_generation": train_records,
            "validation_generation": validation_records,
            "history": history,
        }, handle)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--chunk", required=True, type=int)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--result", required=True)
    cli = parser.parse_args()
    with open(cli.settings, encoding="utf-8") as handle:
        settings = json.load(handle)
    run_chunk(settings, cli.chunk, cli.state_dir, cli.result)


if __name__ == "__main__":
    main()
