"""Bounded asynchronous checkpoint persistence through a writable OSS mount.

The mount MUST report upload completion/errors at close (e.g. ossfs2 with
sync_upload=true). No rename, append, or overwrite is used on this mount.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Callable

import torch

from persistence.files import atomic_path, sha256_file


class _HashWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data):
        written = self.stream.write(data)
        if written != len(data):
            raise OSError("Short checkpoint write")
        self.digest.update(data)
        return written

    def flush(self):
        self.stream.flush()


def write_new(path: Path, value: dict, *, checkpoint: bool = False) -> str:
    """Close and verify a new object before publishing any reference to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        writer = _HashWriter(stream)
        if checkpoint:
            torch.save(value, writer)
        else:
            writer.write((json.dumps(value, allow_nan=False, indent=2) + "\n").encode())
        writer.flush()
        os.fsync(stream.fileno())
    digest = writer.digest.hexdigest()
    if sha256_file(path) != digest:
        raise OSError(f"Read-back SHA256 mismatch: {path.name}")
    return digest


def reference_path(runs_dir: Path, reference: dict) -> Path:
    if not isinstance(reference, dict):
        raise ValueError("Invalid checkpoint reference")
    name = reference.get("path")
    if not isinstance(name, str):
        raise ValueError("Invalid checkpoint path")
    relative = PurePosixPath(name)
    digest = reference.get("sha256", "")
    if (relative.is_absolute() or ".." in relative.parts or "\\" in name
            or not relative.parts or str(relative) != name
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("Invalid checkpoint path or SHA256")
    root = runs_dir.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Checkpoint reference escapes runs directory")
    return path


def run_path(runs_dir: Path, run_id: str) -> Path:
    """Resolve a run directory without allowing traversal outside its root."""
    if not run_id or Path(run_id).name != run_id or run_id in (".", "..") or "\\" in run_id or "\0" in run_id:
        raise ValueError("Invalid run directory name")
    root = runs_dir.resolve()
    path = (root / run_id).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Run path escapes runs directory")
    return path


def restore_run(runs_dir: Path, run_id: str, destination: Path, *, best: bool = False) -> dict:
    """Copy the newest committed checkpoint to local disk and verify it."""
    root = runs_dir.resolve()
    run = run_path(root, run_id)
    markers = [p for p in (run / "epochs").glob("*/complete.json")
               if p.parent.name.isdigit()]
    if not markers:
        raise FileNotFoundError("No completed checkpoint in the requested run")
    marker = max(markers, key=lambda p: int(p.parent.name))
    index = json.loads(marker.read_text())
    epoch = int(marker.parent.name)
    if index.get("format_version") != 1 or index.get("epoch") != epoch:
        raise ValueError("Invalid completion marker")
    for key, filename in (("checkpoint", "checkpoint.pt"), ("metrics", "metrics.json")):
        ref = index.get(key)
        path = reference_path(root, ref)
        if path != marker.parent / filename or sha256_file(path) != ref["sha256"]:
            raise ValueError(f"Completed {key} path or SHA256 mismatch")
    reference = index.get("best") if best else index["checkpoint"]
    source = reference_path(root, reference)
    with atomic_path(destination) as temporary:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != reference["sha256"]:
            raise ValueError("Checkpoint SHA256 mismatch")
    return {**index, "checkpoint": reference}


class AsyncRunWriter:
    """One ordered blocking-I/O worker behind a bounded submission interface.

    max_pending includes the in-flight write. Backpressure is applied BEFORE
    taking a CPU snapshot, so memory cannot grow with a slow mount.
    """
    def __init__(self, runs_dir: Path, run_id: str, *, best: dict | None = None, max_pending: int = 1):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.runs_dir = runs_dir.resolve()
        self.run_dir = run_path(self.runs_dir, run_id)
        if best is not None:
            reference_path(self.runs_dir, best)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.best = copy.deepcopy(best)
        self.max_pending = max_pending
        self.pending = deque()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-io")
        self.closed = False
        self.failure = None

    def _execute(self, operation, *args):
        # A failed save prevents queued epochs/status markers from being committed.
        if self.failure is not None:
            raise self.failure
        try:
            return operation(*args)
        except BaseException as error:
            self.failure = error
            raise

    def check(self):
        if self.failure is not None:
            raise self.failure
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()

    def _space(self):
        if self.closed:
            raise RuntimeError("Writer is closed")
        self.check()
        while len(self.pending) >= self.max_pending:
            self.pending.popleft().result()

    def publish_json(self, name: str, value: dict):
        self._space()
        self.pending.append(self.executor.submit(self._execute, write_new, self.run_dir / name, copy.deepcopy(value)))

    def publish_config(self, config: dict):
        self.publish_json("config.json", config)

    def publish_epoch(self, snapshot: Callable[[], dict], metrics: dict, is_best: bool):
        self._space()
        # Runs on the training thread; completed CPU copies own all their storage.
        payload = snapshot()
        self.pending.append(self.executor.submit(self._execute, self._save_epoch, payload, copy.deepcopy(metrics), is_best))

    def _save_epoch(self, payload, metrics, is_best):
        epoch = metrics["epoch"]
        directory = self.run_dir / "epochs" / f"{epoch:04d}"
        checkpoint = directory / "checkpoint.pt"
        digest = write_new(checkpoint, payload, checkpoint=True)
        ref = {"path": checkpoint.relative_to(self.runs_dir).as_posix(), "sha256": digest, "epoch": epoch}
        metrics_path = directory / "metrics.json"
        metrics_ref = {"path": metrics_path.relative_to(self.runs_dir).as_posix(),
                       "sha256": write_new(metrics_path, metrics)}
        best = ref if is_best else self.best
        write_new(directory / "complete.json", {
            "format_version": 1, "epoch": epoch, "checkpoint": ref,
            "metrics": metrics_ref, "best": best,
        })
        self.best = best

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            for future in self.pending:
                future.result()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
