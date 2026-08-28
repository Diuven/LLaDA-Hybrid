"""Opt-in JSONL component timing for LLaDA/LLaDAFast SGLang forwards."""

from __future__ import annotations

import atexit
import json
import os
import time
from contextlib import contextmanager
from typing import Callable, TypeVar

import torch

T = TypeVar("T")


def profile_path() -> str | None:
    return os.environ.get("SGLANG_PROFILE_COMPONENT_JSONL")


# Cached append-mode file handles, keyed by path.
#
# The previous code did open()/write()/close() on *every* timed call. That
# host-side I/O runs inside the enclosing timed region's CUDA-event window, so
# a nested parent (block.attention wraps 5 child timers; block.mlp wraps 0)
# absorbs ~5x the I/O stall as GPU-idle time — inflating it and skewing any
# parent/parent ratio. A cached handle keeps the inode stable, so the runner
# may *truncate* the file (not unlink it) to drop warmup records while these
# handles stay valid; append-mode writes then resume at offset 0.
_HANDLES: dict = {}


def _emit(path: str, record: dict) -> None:
    f = _HANDLES.get(path)
    if f is None or f.closed:
        f = open(path, "a", encoding="utf-8")
        _HANDLES[path] = f
    f.write(json.dumps(record, sort_keys=True) + "\n")
    f.flush()  # no open/close churn; flush keeps it crash-safe under SIGKILL


@atexit.register
def _close_handles() -> None:
    for f in _HANDLES.values():
        try:
            f.flush()
            f.close()
        except Exception:
            pass


def timed_call(component: str, fn: Callable[[], T], **metadata) -> T:
    path = profile_path()
    if not path:
        return fn()

    use_cuda_timer = torch.cuda.is_available()
    if use_cuda_timer:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    wall_start = time.perf_counter()
    result = fn()

    if use_cuda_timer:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start_event.elapsed_time(end_event))
    else:
        elapsed_ms = (time.perf_counter() - wall_start) * 1000.0

    record = {
        "event": "sglang_component",
        "component": component,
        "elapsed_ms": elapsed_ms,
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
        "pid": os.getpid(),
    }
    record.update(metadata)
    _emit(path, record)

    return result


@contextmanager
def timed_region(component: str, **metadata):
    path = profile_path()
    if not path:
        yield
        return

    use_cuda_timer = torch.cuda.is_available()
    if use_cuda_timer:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    wall_start = time.perf_counter()
    yield

    if use_cuda_timer:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start_event.elapsed_time(end_event))
    else:
        elapsed_ms = (time.perf_counter() - wall_start) * 1000.0

    record = {
        "event": "sglang_component",
        "component": component,
        "elapsed_ms": elapsed_ms,
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
        "pid": os.getpid(),
    }
    record.update(metadata)
    _emit(path, record)
