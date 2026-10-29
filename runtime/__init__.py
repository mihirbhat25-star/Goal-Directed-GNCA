"""Execution support for GNCA experiments.

Local pipelines stream caches; cloud pipelines use compiled experts,
bit-packed graphs, and distributed training. Supported task/backend
combinations are listed in the README. The experiment scripts remain the
public entry points.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

LOCAL_BACKEND = "local"
CLOUD_BACKEND = "cloud"
SUPPORTED_BACKENDS = (LOCAL_BACKEND, CLOUD_BACKEND)
