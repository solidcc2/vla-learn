"""Run one offline single-worker job; each invocation gets a fresh run ID."""

import argparse
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import sys
import uuid

import torch
import torchvision

from persistence.runs import AsyncRunWriter, restore_run, run_path, write_new
from cloud.resources import stage_bundle
from persistence.files import write_json
import train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--data-resource-dir", type=Path, required=True, help="Read-only mount with data.tar.gz and data.manifest.json")
    parser.add_argument("--work-dir", type=Path, required=True, help="Writable container-local disk, not an OSS mount")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Writable OSS runs mount")
    parser.add_argument("--resume-run", help="Previous run directory name under runs-dir")
    parser.add_argument("--resume-best", action="store_true", help="Resume the best checkpoint referenced by that run")
    parser.add_argument("--save-queue-size", type=int, default=1, help="Maximum pending snapshots, including the active write")
    parser.add_argument("--run-label", default="cifar10")
    parser.add_argument("--code-manifest", type=Path, help="Verified manifest supplied by bootstrap")
    args = parser.parse_args(argv)
    if args.resume_best and not args.resume_run:
        parser.error("resume-best requires resume-run")
    return args


def run_job(args: argparse.Namespace) -> dict:
    training_args = train.parse_args(["--config", str(args.config), "--no-download"])
    if training_args.resume:
        raise ValueError("Use --resume-run for cloud resume; remove resume from the training JSON")
    run_id = f"{args.run_label}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
    run_dir = run_path(args.work_dir / "runs", run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    output_dir = run_path(args.runs_dir, run_id)
    # The entry script archives its local stdout/stderr capture after Python exits.
    if target_file := os.environ.get("PIPELINE_LOG_TARGET_FILE"):
        Path(target_file).write_text(str(output_dir / "train.log"))
    data_dir, data_manifest_path = stage_bundle(args.data_resource_dir, "data", args.work_dir.resolve())
    data_manifest = json.loads(data_manifest_path.read_text())
    code_manifest = json.loads(args.code_manifest.read_text()) if args.code_manifest else {}
    best = None
    resume_checkpoint = None
    if args.resume_run:
        training_args.resume = run_dir / "resume.pt"
        index = restore_run(args.runs_dir, args.resume_run, training_args.resume, best=args.resume_best)
        best = index.get("best")
        resume_checkpoint = index["checkpoint"]
    training_args.data_dir = data_dir
    training_args.data_version = data_manifest["sha256"]
    training_args.output_dir = output_dir
    # Construct startup metadata once, after input and resume references are known.
    metadata = {
        "run_id": run_id, "data_sha256": data_manifest["sha256"],
        "code_sha256": code_manifest.get("sha256"),
        "git_revision": code_manifest.get("git_revision"),
        "git_dirty": code_manifest.get("git_dirty"),
        "resume_run": args.resume_run, "resume_checkpoint": resume_checkpoint,
        "image_uri": os.environ.get("PIPELINE_IMAGE_URI"),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "cuda_version": torch.version.cuda,
    }
    summary = {"run_id": run_id, "output_dir": str(output_dir), "local_dir": str(run_dir)}
    print(json.dumps(summary), flush=True)
    try:
        with AsyncRunWriter(args.runs_dir, run_id, best=best, max_pending=args.save_queue_size) as writer:
            writer.publish_json("started.json", summary)
            train.run_training(training_args, publisher=writer, metadata=metadata)
            # Ordered after every checkpoint; close joins the queue before success.
            writer.publish_json("succeeded.json", summary)
    except Exception as error:
        failed = {**summary, "state": "failed", "error_type": type(error).__name__}
        write_json(run_dir / "failed.json", failed)
        try:
            write_new(output_dir / "failed.json", failed)
        except Exception:
            print("Could not persist failure status; see DLC process logs.", file=sys.stderr, flush=True)
        raise
    return summary


def main() -> None:
    run_job(parse_args())


if __name__ == "__main__":
    main()
