"""Backend selection shared by fixed- and online-goal training."""

import tensorflow as tf


def training_strategy(backend):
    gpus = tf.config.list_physical_devices("GPU")
    if backend == "cloud":
        if not gpus:
            raise ValueError("--backend cloud requires a TensorFlow-visible GPU; use --backend local for CPU training.")
        return tf.distribute.MirroredStrategy()
    return tf.distribute.OneDeviceStrategy("/GPU:0" if gpus else "/CPU:0")


def data_backend(args, spatial_dims):
    if args.backend == "local":
        from runtime.local.online_cache import generate_cached_chunk, dataset_from_cache

        def generate(*a, **kw):
            return generate_cached_chunk(*a, spatial_dims=spatial_dims, **kw)

        def dataset(*a, **kw):
            return dataset_from_cache(*a, spatial_dims=spatial_dims, **kw)
    elif spatial_dims == 2:
        from runtime.cloud.cloud_2d import generate_online_goal_chunk as generate
        from runtime.cloud.cloud_2d import dataset_from_trajectories as dataset
    else:
        from runtime.cloud.cloud_3d import generate_online_goal_chunk as generate
        from runtime.cloud.cloud_3d import dataset_from_trajectories as dataset
    return generate, dataset
