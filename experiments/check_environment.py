"""Check dependencies, GPU visibility, and optionally one forward step of each experiment's model."""
import argparse
import importlib.metadata
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Build the model of each of the four experiments, untrained, and take one forward step.")
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf
    for name in ("tensorflow", "spektral", "numpy", "scipy", "numba"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{name}: {version}")
    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("TensorFlow GPUs:", devices)
    if not devices:
        print("Training and evaluation will run on one CPU device, which suits short tests; full training needs GPUs.")
    if args.smoke:
        from evaluation.paired_rollouts import to_tf_sparse
        from models.gnca import build_gnca
        from modules.experts import reference_expert
        from modules.presets import NAMES, goal_conditioned, load_preset
        for name in NAMES:
            preset = load_preset(name)
            dim, conditioned = preset["dimension"], goal_conditioned(preset)
            model = build_gnca(dim, conditioned)
            rng = np.random.RandomState(0)
            positions = rng.uniform(-0.1, 0.1, (100, dim)).astype("float32")
            velocities = np.full((100, dim), 0.001, dtype="float32")
            state = np.concatenate((positions, velocities), axis=1)
            inputs = np.concatenate((state, 2.0 - positions), axis=1) if conditioned else state
            adjacency = to_tf_sparse(reference_expert(dim, n_boids=100).get_neighbors(positions))
            output = model([tf.constant(inputs), adjacency, tf.constant(0)], training=False).numpy()
            assert output.shape == (100, 2 * dim) and np.isfinite(output).all(), name
            print(f"{name}: model built, finite one-step output {output.shape}")


if __name__ == "__main__":
    main()
