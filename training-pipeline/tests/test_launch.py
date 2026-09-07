import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def staged_data(tmp_path):
    from cloud.resources import _pack
    source = tmp_path / "dataset.txt"
    source.write_text("tiny local fixture")
    resource = tmp_path / "resources"
    _pack([("dataset.txt", source)], resource, "data")
    return resource


@pytest.fixture
def small_job(monkeypatch):
    import train

    def loaders(data_dir, batch_size, num_workers, download=True):
        assert download is False
        assert (data_dir / "dataset.txt").read_text() == "tiny local fixture"
        dataset = TensorDataset(torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24,
                                torch.tensor([0, 1] * 4))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), DataLoader(dataset, batch_size=batch_size)
    monkeypatch.setattr(train, "create_dataloaders", loaders)
    monkeypatch.setattr(train, "SimpleCNN", lambda: nn.Linear(3, 2))


def job_args(tmp_path, resource, epochs=1, *extra):
    from cloud.launch import parse_args
    config = tmp_path / f"training-{epochs}.json"
    config.write_text(json.dumps({"device": "cpu", "epochs": epochs, "batch_size": 4}))
    return parse_args([
        "--config", str(config), "--data-resource-dir", str(resource),
        "--work-dir", str(tmp_path / "work"), "--runs-dir", str(tmp_path / "runs"), *extra,
    ])


def test_full_job_then_resume_publishes_two_distinct_runs(tmp_path, staged_data, small_job, monkeypatch):
    from cloud.launch import run_job
    monkeypatch.setattr(torch, "__version__", "2.8.0+cpu")
    first = run_job(job_args(tmp_path, staged_data, 1, "--run-label", "my_smoke", "--save-queue-size", "3"))
    second = run_job(job_args(tmp_path, staged_data, 2, "--resume-run", first["run_id"]))
    assert first["run_id"] != second["run_id"]
    latest = json.loads((Path(second["output_dir"]) / "epochs/0002/complete.json").read_text())
    checkpoint = torch.load(tmp_path / "runs" / latest["checkpoint"]["path"], weights_only=True)
    assert checkpoint["epoch"] == 2
    from test_train import assert_tree_equal
    full = run_job(job_args(tmp_path, staged_data, 2))
    continuous = torch.load(Path(full["output_dir"]) / "epochs/0002/checkpoint.pt", weights_only=True)
    for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "best_accuracy"):
        assert_tree_equal(continuous[key], checkpoint[key])
    config = json.loads((Path(second["output_dir"]) / "config.json").read_text())
    assert config["metadata"]["torch_version"] == "2.8.0+cpu"
    assert "gpu" not in config["metadata"]
    assert config["device"] == "cpu" and config["gpu"] is None
    assert config["metadata"]["resume_checkpoint"] == checkpoint["config"]["metadata"]["resume_checkpoint"]
    assert config["metadata"]["resume_run"] == first["run_id"]
    assert len(config["metadata"]["data_sha256"]) == 64
    assert config["output_dir"] == second["output_dir"]
    assert len(list((tmp_path / "work/data").iterdir())) == 1
    assert not list((tmp_path / "work/runs").glob("*/data"))
    assert (Path(second["output_dir"]) / "succeeded.json").is_file()
    assert not (Path(second["local_dir"]) / "outputs/last.pt").exists()


def test_job_failure_is_reported_without_success_marker(tmp_path, staged_data, small_job, monkeypatch):
    from cloud.launch import run_job
    def fail(*args):
        raise OSError("simulated mounted write failure")
    monkeypatch.setattr("persistence.runs.AsyncRunWriter._save_epoch", fail)
    with pytest.raises(OSError, match="simulated"):
        run_job(job_args(tmp_path, staged_data))
    assert list((tmp_path / "runs").glob("*/failed.json"))
    assert not list((tmp_path / "runs").glob("*/succeeded.json"))
    assert not list((tmp_path / "runs").rglob("complete.json"))


def test_missing_dataset_stops_before_any_output_write(tmp_path, monkeypatch):
    from cloud.launch import run_job
    target_file = tmp_path / "log-target"
    monkeypatch.setenv("PIPELINE_LOG_TARGET_FILE", str(target_file))
    with pytest.raises(FileNotFoundError):
        run_job(job_args(tmp_path, tmp_path / "missing"))
    assert Path(target_file.read_text()).parent.parent == tmp_path / "runs"
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("damage", ["modified", "extra", "missing-marker"])
def test_bash_runs_copied_code_and_rejects_incomplete_release(tmp_path, damage):
    from cloud.prepare import prepare_code
    project = tmp_path / "project"
    (project / "cloud").mkdir(parents=True)
    for name in ("train.py", "evaluate.py", "model.py", "data.py", "engine.py", "checkpoint.py"):
        (project / name).write_text("pass")
    (project / "cloud/__init__.py").write_text("")
    (project / "cloud/launch.py").write_text(
        'import os, sys\nfrom pathlib import Path\n'
        'result = Path(sys.argv[sys.argv.index("--result") + 1])\n'
        'result.write_text("bundled code")\n'
        'Path(os.environ["PIPELINE_LOG_TARGET_FILE"]).write_text(str(result.parent / "run/train.log"))\n'
        'print("training stdout", flush=True)\n'
        'print("training stderr", file=sys.stderr, flush=True)\n'
        'sys.exit(int(os.environ.get("TEST_TRAIN_EXIT", "0")))\n'
    )
    resource = tmp_path / "resources"
    (project / "dlc").mkdir()
    root = Path(__file__).resolve().parents[1]
    (project / "dlc/bootstrap.sh").write_text((root / "dlc/bootstrap.sh").read_text())
    prepare_code(project, resource)
    result = tmp_path / "result"
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path)
    process = subprocess.run([
        "bash", str(resource / "bootstrap.sh"),
        "--work-dir", str(tmp_path / "work"),
        "--result", str(result),
    ], env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert result.read_text() == "bundled code"
    archived = tmp_path / "run/train.log"
    assert "training stdout" in archived.read_text()
    assert "training stderr" in archived.read_text()
    assert "training stdout" in process.stdout
    # Reuse this entrypoint test for failure exit and archive-error behavior.
    archived.unlink()
    env["TEST_TRAIN_EXIT"] = "7"
    process = subprocess.run([
        "bash", str(resource / "bootstrap.sh"), "--result", str(result),
    ], env=env, capture_output=True, text=True)
    assert process.returncode == 7
    assert "training stderr" in archived.read_text()
    archived.unlink()
    archived.parent.rmdir()
    archived.parent.write_text("unwritable archive destination")
    process = subprocess.run([
        "bash", str(resource / "bootstrap.sh"), "--result", str(result),
    ], env=env, capture_output=True, text=True)
    assert process.returncode == 7
    assert "Log archive failed" in process.stderr
    result.unlink()
    if damage == "modified":
        (resource / "cloud/launch.py").write_text("corrupt")
    elif damage == "extra":
        (resource / "torch.py").write_text("unexpected module")
    else:
        (resource / "SHA256SUMS").unlink()
    process = subprocess.run([
        "bash", str(resource / "bootstrap.sh"), "--work-dir", str(tmp_path / "work"),
        "--result", str(result),
    ], env=env, capture_output=True, text=True)
    assert process.returncode != 0
    assert not result.exists()


def test_real_cifar_loader_cnn_and_evaluation_work_offline(tmp_path, monkeypatch):
    """Use tiny valid CIFAR-shaped files; replace only official checksum metadata."""
    import hashlib
    import pickle
    import numpy as np
    from torchvision.datasets import CIFAR10
    from checkpoint import load_checkpoint
    from data import create_dataloaders
    from engine import evaluate
    from model import SimpleCNN
    from cloud.launch import run_job
    from cloud.resources import pack_dataset

    root = tmp_path / "data" / CIFAR10.base_folder
    root.mkdir(parents=True)

    def write_pickle(name, payload):
        content = pickle.dumps(payload)
        (root / name).write_bytes(content)
        return hashlib.md5(content).hexdigest()

    samples = np.random.default_rng(123).integers(0, 256, size=(4, 3072), dtype=np.uint8)
    train_md5 = write_pickle("data_batch_1", {"data": samples, "labels": [0, 1, 2, 3]})
    test_md5 = write_pickle("test_batch", {"data": samples, "labels": [0, 1, 2, 3]})
    meta_md5 = write_pickle("batches.meta", {"label_names": [str(index) for index in range(10)]})
    monkeypatch.setattr(CIFAR10, "train_list", [("data_batch_1", train_md5)])
    monkeypatch.setattr(CIFAR10, "test_list", [("test_batch", test_md5)])
    monkeypatch.setattr(CIFAR10, "meta", {"filename": "batches.meta", "key": "label_names", "md5": meta_md5})
    resources = tmp_path / "resources"
    pack_dataset(tmp_path / "data", resources)
    first = run_job(job_args(tmp_path, resources))
    second = run_job(job_args(tmp_path, resources, 2, "--resume-run", first["run_id"]))
    model = SimpleCNN()
    state = load_checkpoint(Path(second["output_dir"]) / "epochs/0002/checkpoint.pt", model, None, torch.device("cpu"))
    _, test_loader = create_dataloaders(tmp_path / "data", 4, download=False)
    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert state.epoch == 2
    assert metrics.samples == 4 and 0 <= metrics.accuracy <= 1


def test_invalid_run_label_cannot_create_paths(tmp_path, staged_data):
    from cloud.launch import run_job
    args = job_args(tmp_path, staged_data)
    args.run_label = "../../escape"
    with pytest.raises(ValueError, match="run"):
        run_job(args)
    assert not (tmp_path / "work").exists()
