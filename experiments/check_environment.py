"""Check dependencies, GPU visibility, and optionally one forward step of each GNCA model."""
import argparse
import importlib.metadata
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Build each of the four GNCA models, untrained, and take one forward step.")
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf
    for name in ("tensorflow", "spektral", "numpy", "scipy", "numba", "h5py"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed (required for compiled generation)" if name == "numba" else "not installed"
        print(f"{name}: {version}")
    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("TensorFlow GPUs:", devices)
    if not devices:
        print("CPU training and evaluation are available. Use --backend local for cache-based training; --backend cloud requires a visible GPU.")
    if args.smoke:
        from evaluation import paired_rollouts as rollouts
        from modules.boids import Boids
        from modules.boids_3d import Boids3D
        _to_tf_sparse = rollouts._to_tf_sparse
        classes = {(2, "fixed_waypoints"): rollouts.GNNCASimpleBoids,
                   (2, "online_goals"): rollouts.GoalConditionedGNNCABoids,
                   (3, "fixed_waypoints"): rollouts.GNNCASimpleBoids3D,
                   (3, "online_goals"): rollouts.GoalConditionedGNNCABoids3D}
        for name in ("fixed2d", "fixed3d", "online2d", "online3d"):
            dim = 2 if name.endswith("2d") else 3
            task = "online_goals" if name.startswith("online") else "fixed_waypoints"
            model = classes[(dim, task)](activation="linear", batch_norm=False, hidden=256,
                                         hidden_activation="relu", connectivity="cat", aggregate="mean")
            rng = np.random.RandomState(0)
            positions = rng.uniform(-0.1, 0.1, (100, dim)).astype("float32")
            velocities = np.full((100, dim), 0.001, dtype="float32")
            state = np.concatenate((positions, velocities), axis=1)
            inputs = np.concatenate((state, 2.0 - positions), axis=1) if task == "online_goals" else state
            boids = (Boids if dim == 2 else Boids3D)(n_boids=100)
            adjacency = _to_tf_sparse(boids.get_neighbors(positions))
            output = model([tf.constant(inputs), adjacency, tf.constant(0)], training=False).numpy()
            assert output.shape == (100, 2 * dim) and np.isfinite(output).all(), name
            print(f"{name}: model built, finite one-step output {output.shape}")


if __name__ == "__main__":
    main()
