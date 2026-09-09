"""Publish a local code directory and optional dataset archive for OSS mounts."""

import argparse
from pathlib import Path
import hashlib
import shutil
import subprocess

from cloud.resources import pack_dataset
from persistence.files import sha256_file, write_json


def prepare_code(project_dir: Path, output_dir: Path) -> Path:
    # Explicit runtime allowlist avoids collecting local credentials, datasets or venv.
    files = [(name, project_dir / name) for name in ("train.py", "evaluate.py", "model.py", "data.py", "engine.py", "checkpoint.py")]
    for directory, patterns in {
        "persistence": ("__init__.py", "files.py", "runs.py"),
        "cloud": ("__init__.py", "launch.py", "resources.py"),
        "operators": ("__init__.py", "soft_medoid_pool.py"),
        "configs": ("*.json",),
    }.items():
        root = project_dir / directory
        if root.is_symlink():
            raise ValueError(f"Cannot package runtime symlink: {root}")
        for pattern in patterns:
            # Runtime packages are flat. Do not recurse into caches or local folders.
            for path in sorted(root.glob(pattern)):
                if not path.name.startswith("."):
                    files.append((path.relative_to(project_dir).as_posix(), path))
    files.append(("bootstrap.sh", project_dir / "dlc/bootstrap.sh"))
    output_dir.mkdir(parents=True, exist_ok=False)
    for relative, source in files:
        if source.is_symlink():
            raise ValueError(f"Cannot publish runtime symlink: {source}")
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    checksums = "".join(f"{sha256_file(output_dir / name)}  {name}\n" for name, _ in sorted(files))
    metadata = {"sha256": hashlib.sha256(checksums.encode()).hexdigest()}
    try:
        metadata["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_dir, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        metadata["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "."], cwd=project_dir, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        metadata["git_revision"] = None
    write_json(output_dir / "code.manifest.json", metadata)
    checksums += f"{sha256_file(output_dir / 'code.manifest.json')}  code.manifest.json\n"
    # Last file copied to OSS: identifies a fully published code directory.
    (output_dir / "SHA256SUMS").write_text(checksums)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args(argv)
    # Fresh release directories prevent obsolete binaries or manifests leaking in.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty; use a new release directory")
    code = prepare_code(args.project_dir, args.output_dir / "code")
    print(f"Prepared code directory {code}", flush=True)
    if not args.code_only:
        archive, manifest = pack_dataset(args.data_dir, args.output_dir)
        print(f"Prepared {archive}\nManifest {manifest}", flush=True)


if __name__ == "__main__":
    main()
