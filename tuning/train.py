"""mlx_lm.lora with memory guards for a 16 GB Mac: same arguments as `mlx_lm.lora`.

mlx_lm wires the GPU's recommended working set (12.7 GB of 16 GB on an M1) so macOS can't page it
out; with long examples that starved the kernel and the watchdog panicked the laptop. Here MLX wires
at most WIRED_GB, the loss builds the 152k-vocabulary logits CHUNK positions at a time (recomputed in
the backward pass instead of kept), and a watcher stops the run while macOS still has free memory.
"""

import os
import subprocess
import sys
import threading
import time
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import lora
from mlx_lm.tuner import trainer

WIRED_GB = 6
MIN_FREE_PERCENT = 15  # kern.memorystatus_level: the share of memory macOS counts as available
CHUNK = 512


def free_percent() -> int:
    return int(subprocess.check_output(["sysctl", "-n", "kern.memorystatus_level"]))


def watch() -> None:
    while True:
        level = free_percent()
        if level < MIN_FREE_PERCENT:
            peak = mx.get_peak_memory() / 1024**3
            print(f"\n[train] stopping: free memory {level}% < {MIN_FREE_PERCENT}% (MLX peak {peak:.1f} GB)",
                  file=sys.stderr, flush=True)
            os._exit(3)  # a hard exit: the main thread is inside an MLX kernel and won't see a signal soon
        time.sleep(1)


def wire_less() -> int:
    return mx.set_wired_limit(WIRED_GB * 1024**3)


def chunked_loss(model: Any, batch: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
    """mlx_lm's default_loss without the full (sequence x vocabulary) logits in memory."""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    hidden = model.model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    head = model.lm_head if hasattr(model, "lm_head") else model.model.embed_tokens.as_linear

    def piece(i: int) -> mx.array:
        t, m = targets[:, i:i + CHUNK], mask[:, i:i + CHUNK]  # captured, so checkpoint differentiates h only
        loss = mx.checkpoint(lambda h: (nn.losses.cross_entropy(head(h), t) * m).astype(mx.float32).sum())
        return loss(hidden[:, i:i + CHUNK])

    total = sum(piece(i) for i in range(0, targets.shape[1], CHUNK))
    ntoks = mask.sum()
    return total / ntoks, ntoks


if __name__ == "__main__":
    setattr(trainer, "maybe_set_recommended_wired_limit", wire_less)  # the module calls it at train start
    setattr(lora, "train", partial(trainer.train, loss=chunked_loss))
    threading.Thread(target=watch, daemon=True).start()
    sys.argv[0] = "mlx_lm.lora"
    lora.main()
