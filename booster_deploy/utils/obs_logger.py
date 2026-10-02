"""Per-step log of what a policy saw and did, robust to the run ending at any moment.

Layout of a log directory:

    meta.json     written on the first step: task, config, file hashes, record layout, obs term layout
    obs.bin       one fixed-size binary record per policy step, appended as the run goes
    events.jsonl  one line per event (start, motion_released, safety_stop, stop, close), flushed at once

Real runs often end early (a fall trips the safety fallback, Ctrl-C, the portal kills the inference
process), and episodes are short, so nothing is held back in memory: ``record`` packs the step and queues
it, and a writer thread appends it to ``obs.bin`` and flushes within ~20 ms. Flushed data sits in the OS
page cache, which survives the process being killed or crashing; an fsync every second bounds a power cut.
``meta.json`` is written up front, so a log that was cut short is still self-describing, and
``load_obs_log`` drops a partially written trailing record.

    python -m booster_deploy.utils.obs_logger <log dir>     # print a summary
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

_DRAIN_PERIOD_S = 0.02
_FSYNC_PERIOD_S = 1.0


class RecordingModel:
    """Wraps a policy network and keeps its last input and output, so the log holds exactly what the
    network received, whatever the policy did to build it (e.g. locomotion's flattened history)."""

    def __init__(self, model):
        self.model = model
        self.last_input = None
        self.last_output = None

    def __call__(self, x, *args, **kwargs):
        out = self.model(x, *args, **kwargs)
        self.last_input = x.detach()
        self.last_output = out.detach()
        return out

    def __getattr__(self, name):
        return getattr(self.model, name)


def file_sha256(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict[str, Any]:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True,
                                    text=True, timeout=5).stdout.strip())
        return {"commit": commit or None, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


class ObsLogger:
    """Appends one record per policy step to ``<log_dir>/obs.bin``; see the module docstring."""

    def __init__(self, log_dir: str, meta: dict[str, Any]):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._meta = dict(meta)
        self._dtype: np.dtype | None = None
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._closed = False
        self._stop = threading.Event()
        self._bin = open(os.path.join(log_dir, "obs.bin"), "ab")
        self._events = open(os.path.join(log_dir, "events.jsonl"), "a")
        self._writer = threading.Thread(target=self._write_loop, name="obs_logger", daemon=True)
        self._writer.start()

    # ---- recording (called from the control loop; no I/O here) ----

    def record(self, fields: dict[str, Any]) -> None:
        if self._closed:
            return
        if self._dtype is None:
            self._start_layout(fields)
        rec = np.zeros((), dtype=self._dtype)
        for name in self._dtype.names:
            rec[name] = fields[name]
        self._queue.put(rec.tobytes())

    def _start_layout(self, fields: dict[str, Any]) -> None:
        spec = []
        for name, value in fields.items():
            arr = np.asarray(value)
            if arr.dtype.kind == "f":
                dt = "f8" if name.startswith("t_") else "f4"
            elif arr.dtype.kind == "b":
                dt = "?"
            else:
                dt = "i8"
            spec.append((name, dt, tuple(arr.shape)))
        self._dtype = np.dtype([(n, d, s) for n, d, s in spec])

        layout = self._meta.get("obs_layout")
        width = int(np.prod(np.asarray(fields["obs"]).shape))
        if layout is not None and sum(w for _, w in layout) != width:
            print(f"[obs_logger] observation layout covers {sum(w for _, w in layout)} values but the network "
                  f"input has {width}; logging the raw vector only.")
            self._meta["obs_layout_mismatch"] = layout
            self._meta["obs_layout"] = None
        self._meta["fields"] = [[n, d, list(s)] for n, d, s in spec]
        self._meta["record_bytes"] = self._dtype.itemsize
        with open(os.path.join(self.log_dir, "meta.json"), "w") as f:
            json.dump(self._meta, f, indent=1, default=str)

    def event(self, name: str, **info: Any) -> None:
        if self._closed:
            return
        line = {"event": name, "t_wall": time.time(), **info}
        self._events.write(json.dumps(line, default=str) + "\n")
        self._events.flush()

    # ---- writing ----

    def _drain(self) -> None:
        chunks = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if chunks:
            self._bin.write(b"".join(chunks))
            self._bin.flush()

    def _write_loop(self) -> None:
        last_sync = time.monotonic()
        while not self._stop.is_set():
            self._stop.wait(_DRAIN_PERIOD_S)
            self._drain()
            if time.monotonic() - last_sync >= _FSYNC_PERIOD_S:
                os.fsync(self._bin.fileno())
                last_sync = time.monotonic()

    def close(self) -> None:
        """Write out everything queued. Idempotent; safe to call from a signal handler's exit path."""
        if self._closed:
            return
        self.event("close")
        self._closed = True
        self._stop.set()
        self._writer.join(timeout=2.0)
        self._drain()
        os.fsync(self._bin.fileno())
        self._bin.close()
        self._events.close()


def load_obs_log(log_dir: str) -> dict[str, Any]:
    """Read a log written by ObsLogger, including one that was cut short.

    Returns the per-step fields as arrays (``out["obs"]``, ``out["joint_pos"]``, ...), plus ``terms``
    (observation slices by name, if the policy declared a layout), ``events`` and ``meta``.
    """
    with open(os.path.join(log_dir, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.dtype([(n, d, tuple(s)) for n, d, s in meta["fields"]])
    path = os.path.join(log_dir, "obs.bin")
    count = os.path.getsize(path) // dtype.itemsize   # drop a partially written trailing record
    data = np.fromfile(path, dtype=dtype, count=count)
    out: dict[str, Any] = {name: data[name] for name in dtype.names}
    out["meta"] = meta
    out["terms"] = {}
    if meta.get("obs_layout"):
        obs = data["obs"].reshape(count, -1)
        start = 0
        for name, width in meta["obs_layout"]:
            out["terms"][name] = obs[:, start:start + width]
            start += width
    events_path = os.path.join(log_dir, "events.jsonl")
    out["events"] = []
    if os.path.isfile(events_path):
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out["events"].append(json.loads(line))
                    except json.JSONDecodeError:
                        pass   # a line cut short by a crash
    return out


def _summary(log_dir: str) -> None:
    log = load_obs_log(log_dir)
    meta, n = log["meta"], len(log["step"])
    dur = (log["t_wall"][-1] - log["t_wall"][0]) if n else 0.0
    print(f"{log_dir}\n  task {meta.get('task')}  ({meta.get('policy_class')}, {meta.get('backend')})")
    print(f"  {n} steps ({n * meta.get('policy_dt', 0):.2f}s policy time, {dur:.2f}s wall), obs {log['obs'].shape[-1] if n else '?'} wide, action "
          f"{log['action'].shape[-1] if n else '?'} wide")
    if log["terms"]:
        print(f"  terms: {', '.join(f'{k}[{v.shape[1]}]' for k, v in list(log['terms'].items())[:12])}"
              f"{' ...' if len(log['terms']) > 12 else ''} ({len(log['terms'])} total)")
    for e in log["events"]:
        extra = {k: v for k, v in e.items() if k not in ("event", "t_wall")}
        print(f"  event {e['event']:16s} {extra if extra else ''}")
    if not log["events"] or log["events"][-1]["event"] != "close":
        print("  no clean close: the run was killed or crashed; data is complete up to the last flush")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        _summary(d)
