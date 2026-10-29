"""Best-over-run model selection for chunked training (``--select_best_over_run``).

This is the default. Every chunk validates on the same episodes (those of
chunk 0), so validation losses are comparable across chunks, and the exported
weights are replaced only when a chunk's best validation loss improves on the
best of all earlier chunks. With ``--last_chunk_selection`` every chunk draws
its own validation episodes and the exported weights are the best state of
the last chunk.
Training itself is unchanged: each chunk still continues from the previous
chunk's best state.
"""
from __future__ import annotations

import json
import os


def enabled(args) -> bool:
    return bool(getattr(args, "select_best_over_run", True))


def validation_chunk_index(args, chunk_index: int) -> int:
    """Chunk index that seeds the validation episodes and their batches."""
    return 0 if enabled(args) else int(chunk_index)


def export_if_best(args, state_dir: str, chunk_index: int, value: float, save) -> bool:
    """Call ``save()`` unless an earlier chunk of this run validated better.

    ``state_dir`` persists across the chunks of one run; the record of the best
    chunk lives there. Returns whether this chunk's weights were exported.
    """
    if not enabled(args):
        save()
        return True
    record_path = os.path.join(state_dir, "run_best.json")
    previous = None
    if os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as handle:
            previous = json.load(handle)
    value = float(value)
    if previous is None or value < float(previous["val_loss"]):
        save()
        with open(record_path, "w", encoding="utf-8") as handle:
            json.dump({"chunk": int(chunk_index), "val_loss": value}, handle)
        print(f">>> Run best: chunk {chunk_index + 1} val_loss={value:.6g}; exported these weights.", flush=True)
        return True
    print(
        f">>> Run best stays chunk {int(previous['chunk']) + 1} "
        f"(val_loss={float(previous['val_loss']):.6g}); this chunk reached {value:.6g} and is not exported.",
        flush=True,
    )
    return False
