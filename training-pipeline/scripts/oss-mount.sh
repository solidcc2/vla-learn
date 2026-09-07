#!/usr/bin/env bash
# This project's storage layout; mounting mechanics are shared by the repository.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLA_OSS_PROJECT=training-pipeline
export VLA_OSS_DATA_PREFIX=datasets/cifar10
exec bash "$script_dir/../../scripts/oss-mount.sh" "$@"
