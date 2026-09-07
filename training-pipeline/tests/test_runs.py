import json
from threading import Event, Thread

import pytest
import torch

from persistence.runs import AsyncRunWriter, restore_run


def submit(writer, epoch=1, best=True):
    writer.publish_epoch(lambda: {"epoch": epoch}, {"epoch": epoch}, best)


def test_completed_epochs_resume_and_keep_parent_best(tmp_path):
    with AsyncRunWriter(tmp_path, "first") as writer:
        submit(writer)
    with AsyncRunWriter(tmp_path, "second", best=writer.best) as other:
        submit(other, 2, False)
    index = restore_run(tmp_path, "second", tmp_path / "resume.pt", best=True)
    assert index["checkpoint"]["path"] == "first/epochs/0001/checkpoint.pt"
    assert torch.load(tmp_path / "resume.pt", weights_only=True)["epoch"] == 1
    (tmp_path / "second/epochs/0003").mkdir()
    (tmp_path / "second/epochs/0003/checkpoint.pt").write_bytes(b"incomplete")
    index = restore_run(tmp_path, "second", tmp_path / "resume.pt")
    assert index["checkpoint"]["epoch"] == 2
    (tmp_path / index["checkpoint"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        restore_run(tmp_path, "second", tmp_path / "resume.pt")
    assert torch.load(tmp_path / "resume.pt", weights_only=True)["epoch"] == 2


def test_async_write_overlaps_compute_and_bounds_snapshot_memory(tmp_path, monkeypatch):
    started, release, captured, submitted = Event(), Event(), Event(), Event()
    original = AsyncRunWriter._save_epoch
    def slow(self, *args):
        started.set()
        assert release.wait(5)
        return original(self, *args)
    monkeypatch.setattr(AsyncRunWriter, "_save_epoch", slow)
    writer = AsyncRunWriter(tmp_path, "run", max_pending=1)
    submit(writer)
    assert started.wait(2)  # submit has returned while disk I/O is still blocked
    def second():
        submitted.set()
        writer.publish_epoch(lambda: captured.set() or {"epoch": 2}, {"epoch": 2}, False)
    thread = Thread(target=second)
    thread.start()
    try:
        assert submitted.wait(2)
        assert not captured.wait(0.1)  # no second CPU snapshot while queue is full
        assert not list(tmp_path.rglob("complete.json"))
    finally:
        release.set()
        thread.join(5)
        writer.close()
    assert not thread.is_alive()
    assert captured.is_set()
    assert (tmp_path / "run/epochs/0002/complete.json").exists()


def test_metrics_write_failure_has_no_completion_marker(tmp_path, monkeypatch):
    import persistence.runs as runs
    original = runs.write_new
    def fail(path, *args, **kwargs):
        if path.name == "metrics.json":
            raise OSError("mount disconnected")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(runs, "write_new", fail)
    with pytest.raises(OSError, match="disconnected"):
        with AsyncRunWriter(tmp_path, "run") as writer:
            submit(writer)
    assert not list(tmp_path.rglob("complete.json"))


def test_close_waits_for_final_save(tmp_path, monkeypatch):
    started, release, closed = Event(), Event(), Event()
    original = AsyncRunWriter._save_epoch
    def slow(self, *args):
        started.set()
        assert release.wait(5)
        return original(self, *args)
    monkeypatch.setattr(AsyncRunWriter, "_save_epoch", slow)
    writer = AsyncRunWriter(tmp_path, "run")
    submit(writer)
    assert started.wait(2)
    thread = Thread(target=lambda: (writer.close(), closed.set()))
    thread.start()
    try:
        assert not closed.wait(0.1)
    finally:
        release.set()
        thread.join(5)
    assert closed.is_set()
    assert (tmp_path / "run/epochs/0001/complete.json").exists()


def test_resume_rejects_reference_outside_runs(tmp_path):
    with AsyncRunWriter(tmp_path, "run") as writer:
        submit(writer)
    path = tmp_path / "run/epochs/0001/complete.json"
    index = json.loads(path.read_text())
    index["checkpoint"]["path"] = "../escape.pt"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="path"):
        restore_run(tmp_path, "run", tmp_path / "resume.pt")


def test_failure_stops_already_queued_status(tmp_path, monkeypatch):
    started, release = Event(), Event()
    def fail(*args):
        started.set()
        assert release.wait(5)
        raise OSError("save failed")
    monkeypatch.setattr(AsyncRunWriter, "_save_epoch", fail)
    writer = AsyncRunWriter(tmp_path, "run", max_pending=2)
    submit(writer)
    assert started.wait(2)
    writer.publish_json("succeeded.json", {})
    release.set()
    with pytest.raises(OSError, match="save failed"):
        writer.close()
    assert not (tmp_path / "run/succeeded.json").exists()


def test_mount_sync_failure_cannot_publish_completion(tmp_path, monkeypatch):
    def fail(_):
        raise OSError("mount fsync failed")
    monkeypatch.setattr("persistence.runs.os.fsync", fail)
    with pytest.raises(OSError, match="fsync failed"):
        with AsyncRunWriter(tmp_path, "run") as writer:
            submit(writer)
    assert not list(tmp_path.rglob("complete.json"))
