"""Deterministic resource bundles and checked extraction onto a local disk."""

import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from persistence.files import atomic_path, sha256_file, write_json

SPACE_RESERVE = 64 * 1024 * 1024


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name and name != ".prepared.json" and str(path) != "."


def _matches_archive(bundle: tarfile.TarFile, members: list[tarfile.TarInfo], destination: Path) -> bool:
    expected = {".prepared.json"}
    for member in members:
        path = PurePosixPath(member.name)
        expected.add(str(path))
        expected.update(str(parent) for parent in path.parents if str(parent) != ".")
    paths = list(destination.rglob("*"))
    if any(path.is_symlink() for path in paths):
        return False
    if {path.relative_to(destination).as_posix() for path in paths} != expected:
        return False
    for member in members:
        path = destination / member.name
        if member.isdir():
            if not path.is_dir():
                return False
        else:
            if not path.is_file():
                return False
            with bundle.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha256_file(path) != digest:
                return False
    return (destination / ".prepared.json").is_file()


def extract_archive(archive: Path, manifest_path: Path, destination: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1 or manifest.get("archive") != archive.name:
        raise ValueError("Invalid resource manifest version or archive name")
    digest = sha256_file(archive)
    if digest != manifest.get("sha256"):
        raise ValueError("Resource SHA256 mismatch")
    if destination.is_symlink():
        raise ValueError("Unsafe symlink destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = set()
        for member in members:
            name = str(PurePosixPath(member.name))
            if not _safe_name(member.name) or not (member.isfile() or member.isdir()) or name in names:
                raise ValueError(f"Unsafe archive member: {member.name}")
            names.add(name)
        required = sum(member.size for member in members if member.isfile())
        if not members or len(members) > 100000:
            raise ValueError("Archive is empty or has too many members")
        if destination.exists():
            if _matches_archive(bundle, members, destination):
                return manifest
            raise FileExistsError("Existing resource stage is incomplete or changed; choose a new work directory")
        if shutil.disk_usage(destination.parent).free < required + SPACE_RESERVE:
            raise OSError("Insufficient local disk space for extracted resource")
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as scratch:
            staged = Path(scratch) / "content"
            staged.mkdir()
            bundle.extractall(staged, members=members, filter="data")
            write_json(staged / ".prepared.json", {"sha256": digest})
            staged.rename(destination)
    return manifest


def _pack(files: list[tuple[str, Path]], output_dir: Path, name: str, metadata: dict | None = None) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{name}.tar.gz"
    for _, path in files:
        if path.is_symlink():
            raise ValueError(f"Cannot package runtime symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    with atomic_path(archive) as temporary:
        with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as bundle:
                for relative, path in sorted(files):
                    info = bundle.gettarinfo(str(path), arcname=relative)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o755 if path.suffix == ".sh" else 0o644
                    with path.open("rb") as stream:
                        bundle.addfile(info, stream)
    manifest_path = output_dir / f"{name}.manifest.json"
    write_json(manifest_path, {
        "format_version": 1, "archive": archive.name, "sha256": sha256_file(archive),
        "bytes": archive.stat().st_size, "unpacked_bytes": sum(path.stat().st_size for _, path in files),
        "file_count": len(files), **(metadata or {}),
    })
    return archive, manifest_path


def pack_dataset(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    from torchvision.datasets import CIFAR10
    from torchvision.datasets.utils import check_integrity

    root = data_dir / CIFAR10.base_folder
    entries = [*CIFAR10.train_list, *CIFAR10.test_list,
               (CIFAR10.meta["filename"], CIFAR10.meta["md5"])]
    files = []
    if root.is_symlink():
        raise ValueError("Cannot package CIFAR symlink directory")
    for name, md5 in entries:
        path = root / name
        if path.is_symlink() or not check_integrity(path, md5):
            raise FileNotFoundError(f"CIFAR-10 file missing or checksum invalid: {path}; prepare the official dataset first")
        files.append((f"{CIFAR10.base_folder}/{name}", path))
    return _pack(files, output_dir, "data", {"dataset": "cifar10"})


def stage_bundle(resource_dir: Path, name: str, work_dir: Path) -> tuple[Path, Path]:
    """Copy a mounted bundle to local storage before verification and extraction."""
    if name != "data":
        raise ValueError("Unknown resource bundle")
    source_archive = resource_dir / f"{name}.tar.gz"
    source_manifest = resource_dir / f"{name}.manifest.json"
    manifest = json.loads(source_manifest.read_text())
    digest = manifest.get("sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid resource SHA256")
    local = work_dir / "bundles" / name / digest
    local.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(local).free < source_archive.stat().st_size + SPACE_RESERVE:
        raise OSError("Insufficient local disk space to copy resource archive")
    archive, manifest_path = local / source_archive.name, local / source_manifest.name
    with atomic_path(archive) as temporary:
        shutil.copyfile(source_archive, temporary)
    write_json(manifest_path, manifest)
    destination = work_dir / name / digest
    extract_archive(archive, manifest_path, destination)
    return destination, manifest_path
