import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def archive_fixture(tmp_path):
    def create(members):
        archive = tmp_path / "resource.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for name, content, kind in members:
                member = tarfile.TarInfo(name)
                if kind == "link":
                    member.type, member.linkname = tarfile.SYMTYPE, "/tmp/outside"
                    bundle.addfile(member)
                else:
                    member.size = len(content)
                    bundle.addfile(member, io.BytesIO(content))
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"format_version": 1, "archive": archive.name,
                                        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
        return archive, manifest
    return create


def test_extract_verifies_and_can_reuse_completed_stage(tmp_path, archive_fixture):
    from cloud.resources import extract_archive
    archive, manifest = archive_fixture([("dataset/value", b"data", "file")])
    destination = tmp_path / "stage"
    extract_archive(archive, manifest, destination)
    assert (destination / "dataset/value").read_bytes() == b"data"
    extract_archive(archive, manifest, destination)
    assert (destination / ".prepared.json").is_file()


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("/tmp/escape", "file"), ("linked", "link")])
def test_extract_rejects_unsafe_members(tmp_path, archive_fixture, name, kind):
    from cloud.resources import extract_archive
    archive, manifest = archive_fixture([(name, b"evil", kind)])
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(archive, manifest, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()
    assert not (tmp_path / "escape").exists()


def test_extract_checksum_failure_does_not_publish_stage(tmp_path, archive_fixture):
    from cloud.resources import extract_archive
    archive, manifest = archive_fixture([("file", b"data", "file")])
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        extract_archive(archive, manifest, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_extract_checks_uncompressed_space(tmp_path, archive_fixture, monkeypatch):
    from cloud.resources import extract_archive
    archive, manifest = archive_fixture([("file", b"data", "file")])
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("cloud.resources.shutil.disk_usage", lambda _: usage(100, 99, 1))
    with pytest.raises(OSError, match="space"):
        extract_archive(archive, manifest, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_pack_only_ships_runtime_files(tmp_path):
    from cloud.prepare import prepare_code
    project = tmp_path / "project"
    project.mkdir()
    for name in ("train.py", "evaluate.py", "model.py", "data.py", "engine.py", "checkpoint.py", "README.md", ".env", "secret.json"):
        (project / name).write_text("content")
    for name in ("cloud", "persistence", "venv", "data", "outputs"):
        (project / name).mkdir()
        (project / name / "launch.py").write_text("content")
    for name in ("__init__.py", "files.py", "runs.py"):
        (project / "persistence" / name).write_text("content")
    (project / "cloud/.env").write_text("credential")
    (project / "cloud/prepare.py").write_text("local publishing tool")
    (project / "cloud/__pycache__").mkdir()
    (project / "cloud/__pycache__/cache.py").write_text("cache")
    (project / "dlc").mkdir()
    (project / "dlc/bootstrap.sh").write_text("entry")
    release = prepare_code(project, tmp_path / "packed")
    names = {p.relative_to(release).as_posix() for p in release.rglob("*") if p.is_file()}
    assert {"train.py", "evaluate.py", "cloud/launch.py", "model.py", "data.py", "engine.py", "checkpoint.py", "persistence/files.py", "persistence/runs.py"} <= names
    assert "README.md" not in names
    assert "cloud/prepare.py" not in names
    assert not any("venv/" in name or "data/" in name or "outputs/" in name or ".env" in name or "secret" in name or "__pycache__" in name for name in names)
    assert (release / "SHA256SUMS").is_file()


def test_pack_rejects_runtime_symlink(tmp_path):
    from cloud.prepare import prepare_code
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").symlink_to("/etc/passwd")
    (project / "evaluate.py").write_text("pass")
    with pytest.raises(ValueError, match="symlink"):
        prepare_code(project, tmp_path / "packed")


def test_dataset_pack_requires_all_cifar_files(tmp_path):
    from cloud.resources import pack_dataset
    with pytest.raises(FileNotFoundError, match="CIFAR"):
        pack_dataset(tmp_path / "missing", tmp_path / "packed")


def test_reused_stage_rejects_extra_executable_files(tmp_path, archive_fixture):
    from cloud.resources import extract_archive
    archive, manifest = archive_fixture([("train.py", b"pass", "file")])
    destination = tmp_path / "stage"
    extract_archive(archive, manifest, destination)
    (destination / "torch.py").write_text("unverified executable")
    with pytest.raises(FileExistsError, match="changed"):
        extract_archive(archive, manifest, destination)


def test_reused_stage_does_not_trust_modified_completion_marker(tmp_path, archive_fixture):
    from cloud.resources import extract_archive
    archive, manifest = archive_fixture([("train.py", b"pass", "file")])
    destination = tmp_path / "stage"
    extract_archive(archive, manifest, destination)
    (destination / "train.py").write_text("modified")
    marker = json.loads((destination / ".prepared.json").read_text())
    marker["files"] = {"train.py": hashlib.sha256(b"modified").hexdigest()}
    (destination / ".prepared.json").write_text(json.dumps(marker))
    with pytest.raises(FileExistsError, match="changed"):
        extract_archive(archive, manifest, destination)
