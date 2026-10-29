"""Subprocess helpers that preserve diagnostics while filtering known noise."""

from __future__ import annotations

import os
import subprocess
import sys
import threading


BENIGN_NATIVE_DIAGNOSTICS = (
    ("successful NUMA node read from SysFS had negative value",),
    ("could not open file to read NUMA node",),
    ("Your kernel may have been built without NUMA support",),
    ("Unable to register cuFFT factory",),
    ("Unable to register cuDNN factory",),
    ("Unable to register cuBLAS factory",),
    ("All log messages before absl::InitializeLog()",),
    ("ptx85", "not a recognized feature"),
)

_STDERR_FILTER_STATE = []


def _matches_ignored_rule(line, ignored_fragments):
    return any(
        all(fragment in line for fragment in rule)
        for rule in ignored_fragments
    )


def install_filtered_stderr(ignored_fragments=BENIGN_NATIVE_DIAGNOSTICS):
    """Filter known native-library noise from the current process in real time.

    TensorFlow can emit CUDA diagnostics directly to file descriptor 2 before
    Python or absl logging is initialized.  Redirecting that descriptor through
    a lightweight thread lets us remove only explicitly allow-listed messages;
    all other stderr output, including tracebacks and native failures, is
    forwarded unchanged and without waiting for process exit.
    """
    if _STDERR_FILTER_STATE:
        return

    try:
        sys.stderr.flush()
        original_fd = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.set_inheritable(original_fd, False)
        os.set_inheritable(read_fd, False)
        os.dup2(write_fd, 2, inheritable=True)
        os.close(write_fd)
    except (AttributeError, OSError):
        return

    rules = tuple(tuple(rule) for rule in ignored_fragments)

    def _forward():
        pending = b""
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                pending += chunk
                while True:
                    newline = pending.find(b"\n")
                    carriage = pending.find(b"\r")
                    endings = [index for index in (newline, carriage) if index >= 0]
                    if not endings:
                        break
                    end = min(endings) + 1
                    raw_line, pending = pending[:end], pending[end:]
                    line = raw_line.decode("utf-8", errors="replace")
                    if not _matches_ignored_rule(line, rules):
                        os.write(original_fd, raw_line)
            if pending:
                line = pending.decode("utf-8", errors="replace")
                if not _matches_ignored_rule(line, rules):
                    os.write(original_fd, pending)
        finally:
            os.close(read_fd)

    thread = threading.Thread(
        target=_forward,
        name="filtered-native-stderr",
        daemon=True,
    )
    thread.start()
    _STDERR_FILTER_STATE.append((thread, original_fd))


def run_with_filtered_stderr(command, ignored_fragments):
    """Run a command and hide stderr lines matching every fragment in a rule."""
    environment = os.environ.copy()
    # Override NVIDIA image defaults rather than inheriting a noisier value.
    environment["TF_CPP_MIN_LOG_LEVEL"] = "3"
    process = subprocess.Popen(
        command,
        env=environment,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stderr is not None
    for line in process.stderr:
        should_ignore = _matches_ignored_rule(line, ignored_fragments)
        if should_ignore:
            continue
        sys.stderr.write(line)
        sys.stderr.flush()
    return process.wait()
