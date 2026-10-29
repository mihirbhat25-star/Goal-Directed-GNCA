"""Shared synchronous multi-GPU training loop for cloud backends."""

from __future__ import annotations

import time

import tensorflow as tf

from runtime.common.timing import deadline_reached


def fit_distributed(
    model,
    strategy,
    train_data,
    train_local_steps,
    n_epochs,
    callbacks,
    global_node_count,
    steps_per_execution,
    loss_fn,
    val_data=None,
    val_local_steps=0,
    wall_deadline=0.0,
    execution_label="Cloud",
):
    """Train complete graph batches synchronously on all visible GPUs.

    ``loss_fn(targets, predictions)`` scores one model application.
    A batch whose loss or gradient is not finite never reaches the optimizer and
    is left out of the epoch averages; the number of such batches is reported.
    """
    def forward_loss(inputs, targets, training):
        return loss_fn(targets, model(inputs, training=training))

    replicas = strategy.num_replicas_in_sync
    train_steps = train_local_steps // replicas
    if train_steps < 1:
        raise ValueError(
            f"Need at least {replicas} train batches for {replicas} GPUs."
        )
    dropped_train = train_local_steps - train_steps * replicas
    if dropped_train:
        print(
            f">>> {execution_label}: dropping {dropped_train} incomplete batch(es) per "
            "epoch to keep replicas synchronized."
        )
    val_steps = val_local_steps // replicas if val_data is not None else 0
    if val_data is not None and val_steps < 1:
        raise ValueError(
            f"Need at least {replicas} validation batches for {replicas} GPUs."
        )

    distributed_train = strategy.distribute_datasets_from_function(
        lambda _context: train_data
    )
    distributed_val = (
        strategy.distribute_datasets_from_function(lambda _context: val_data)
        if val_data is not None
        else None
    )

    def replica_train_step(batch):
        inputs, targets = batch
        with tf.GradientTape() as tape:
            per_node_loss = forward_loss(inputs, targets, True)
            local_loss_sum = tf.reduce_sum(per_node_loss)
            local_node_count = tf.cast(tf.size(per_node_loss), tf.float32)
            loss = local_loss_sum / tf.cast(global_node_count, tf.float32)
        gradients = tape.gradient(loss, model.trainable_variables)
        gradient_norm = tf.linalg.global_norm(gradients)
        finite = tf.math.is_finite(local_loss_sum) & tf.math.is_finite(gradient_norm)
        gradients = [tf.where(finite, gradient, tf.zeros_like(gradient)) for gradient in gradients]
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        zero = tf.constant(0.0, dtype=tf.float32)
        return (
            tf.where(finite, local_loss_sum, zero),
            tf.where(finite, local_node_count, zero),
            tf.cast(tf.logical_not(finite), tf.float32),
        )

    def replica_validation_step(batch):
        inputs, targets = batch
        per_node_loss = forward_loss(inputs, targets, False)
        local_loss_sum = tf.reduce_sum(per_node_loss)
        finite = tf.math.is_finite(local_loss_sum)
        zero = tf.constant(0.0, dtype=tf.float32)
        return (
            tf.where(finite, local_loss_sum, zero),
            tf.where(finite, tf.cast(tf.size(per_node_loss), tf.float32), zero),
            tf.cast(tf.logical_not(finite), tf.float32),
        )

    def compiled(function):
        try:
            return tf.function(function, reduce_retracing=True)
        except TypeError:  # TensorFlow before 2.9 spells the option differently.
            return tf.function(function, experimental_relax_shapes=True)

    def run_block(step_function, iterator, block_steps):
        total_loss = tf.constant(0.0, dtype=tf.float32)
        total_nodes = tf.constant(0.0, dtype=tf.float32)
        total_skipped = tf.constant(0.0, dtype=tf.float32)
        for _ in tf.range(block_steps):
            loss_sums, node_counts, skipped = strategy.run(
                step_function, args=(next(iterator),)
            )
            total_loss += strategy.reduce(
                tf.distribute.ReduceOp.SUM, loss_sums, axis=None
            )
            total_nodes += strategy.reduce(
                tf.distribute.ReduceOp.SUM, node_counts, axis=None
            )
            total_skipped += strategy.reduce(
                tf.distribute.ReduceOp.SUM, skipped, axis=None
            )
        return total_loss, total_nodes, total_skipped

    @compiled
    def train_block(iterator, block_steps):
        return run_block(replica_train_step, iterator, block_steps)

    @compiled
    def validation_block(iterator, block_steps):
        return run_block(replica_validation_step, iterator, block_steps)

    callback_list = tf.keras.callbacks.CallbackList(
        callbacks,
        add_history=False,
        add_progbar=False,
        model=model,
        epochs=n_epochs,
        steps=train_steps,
        verbose=1,
        metrics=["loss"] + (["val_loss"] if val_data is not None else []),
    )
    execution_steps = max(1, min(int(steps_per_execution), train_steps))
    print(
        f">>> {execution_label} execution: {replicas} replicas, "
        f"{train_steps} global steps/epoch, "
        f"{execution_steps} compiled steps/dispatch.",
        flush=True,
    )

    def mean_or_inf(loss_sum, node_count):
        return loss_sum / node_count if node_count > 0 else float("inf")

    model.stop_training = False
    callback_list.on_train_begin()
    train_iterator = iter(distributed_train)
    timed_out = False
    epochs_completed = 0
    losses = []
    validation_losses = []
    learning_rates = []
    skipped_batches = []
    try:
        for epoch in range(n_epochs):
            if model.stop_training or deadline_reached(wall_deadline):
                timed_out = deadline_reached(wall_deadline)
                break
            epoch_t0 = time.time()
            callback_list.on_epoch_begin(epoch)
            epoch_loss_sum = 0.0
            epoch_node_count = 0.0
            epoch_skipped = 0.0
            for block_start in range(0, train_steps, execution_steps):
                block_steps = min(execution_steps, train_steps - block_start)
                loss_sum, node_count, skipped = train_block(
                    train_iterator, tf.constant(block_steps, dtype=tf.int32)
                )
                epoch_loss_sum += float(loss_sum.numpy())
                epoch_node_count += float(node_count.numpy())
                epoch_skipped += float(skipped.numpy())
                if deadline_reached(wall_deadline):
                    timed_out = True
                    break

            if timed_out:
                break

            logs = {"loss": mean_or_inf(epoch_loss_sum, epoch_node_count)}
            val_skipped = 0.0
            if distributed_val is not None:
                val_iterator = iter(distributed_val)
                val_loss_sum = 0.0
                val_node_count = 0.0
                val_execution_steps = max(1, min(execution_steps, val_steps))
                for block_start in range(0, val_steps, val_execution_steps):
                    block_steps = min(
                        val_execution_steps, val_steps - block_start
                    )
                    loss_sum, node_count, skipped = validation_block(
                        val_iterator, tf.constant(block_steps, dtype=tf.int32)
                    )
                    val_loss_sum += float(loss_sum.numpy())
                    val_node_count += float(node_count.numpy())
                    val_skipped += float(skipped.numpy())
                    if deadline_reached(wall_deadline):
                        timed_out = True
                        break
                if timed_out:
                    break
                logs["val_loss"] = mean_or_inf(val_loss_sum, val_node_count)

            callback_list.on_epoch_end(epoch, logs)
            learning_rate = float(
                tf.keras.backend.get_value(model.optimizer.learning_rate)
            )
            losses.append(logs["loss"])
            if "val_loss" in logs:
                validation_losses.append(logs["val_loss"])
            learning_rates.append(learning_rate)
            skipped_batches.append(int(epoch_skipped + val_skipped))
            summary = (
                f"Epoch {epoch + 1}/{n_epochs} - "
                f"{time.time() - epoch_t0:.1f}s - loss: {logs['loss']:.6g}"
            )
            if "val_loss" in logs:
                summary += f" - val_loss: {logs['val_loss']:.6g}"
            summary += f" - lr: {learning_rate:.6g}"
            if epoch_skipped or val_skipped:
                summary += (
                    f" - non-finite batches skipped: {int(epoch_skipped)} train"
                    + (f", {int(val_skipped)} validation" if val_skipped else "")
                )
            print(summary, flush=True)
            epochs_completed += 1
    finally:
        callback_list.on_train_end()

    return {
        "loss": losses,
        "val_loss": validation_losses,
        "lr": learning_rates,
        "skipped_batches": skipped_batches,
        "steps_per_epoch": train_steps,
        "epochs_completed": epochs_completed,
        "timed_out": timed_out,
    }
