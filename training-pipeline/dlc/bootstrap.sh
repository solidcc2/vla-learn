#!/usr/bin/env bash
set -euo pipefail
# SHA256SUMS must be copied last when publishing this trusted release directory.
resource_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(mktemp -d "${TMPDIR:-/tmp}/vla-bootstrap.XXXXXXXX")"
staged_code="$workspace/code"
log_file="$workspace/train.log"
export PIPELINE_LOG_TARGET_FILE="$workspace/log-target"
keep_log=false
trap 'if [[ "$keep_log" == true ]]; then rm -rf -- "$staged_code"; else rm -rf -- "$workspace"; fi' EXIT

run() (
  set -euo pipefail
  test -f "$resource_dir/SHA256SUMS"
  mkdir -- "$staged_code"
  cp -R -- "$resource_dir/." "$staged_code/"
  cd -- "$staged_code"
  [[ -z "$(find . ! -type d ! -type f -print -quit)" ]] || { echo 'Code release contains links or special files' >&2; exit 1; }
  sha256sum --check --strict SHA256SUMS
  # Reject stray modules as well as missing/modified files.
  diff <(find . -type f ! -path './SHA256SUMS' -printf '%P\n' | LC_ALL=C sort) \
       <(cut -c 67- SHA256SUMS | LC_ALL=C sort)
  unset PYTHONPATH PYTHONHOME
  python -u -m cloud.launch --code-manifest "$staged_code/code.manifest.json" "$@"
)

# Wait for tee as well as Python so the archived file includes the final traceback.
set +e
run "$@" 2>&1 | tee -- "$log_file"
statuses=("${PIPESTATUS[@]}")
set -e
training_status="${statuses[0]}"
if [[ "${statuses[1]}" != 0 ]]; then
  echo 'Local log capture failed; the log may be incomplete.' >&2
fi
if [[ -s "$PIPELINE_LOG_TARGET_FILE" ]]; then
  log_target="$(cat -- "$PIPELINE_LOG_TARGET_FILE")"
  # One copy after completion, never append to an OSS object. Close waits for
  # upload on the configured sync_upload=true mount.
  if mkdir -p -- "$(dirname -- "$log_target")" && cp -- "$log_file" "$log_target"; then
    echo "Log archived: $log_target"
  else
    keep_log=true
    echo "Log archive failed; local copy: $log_file" >&2
  fi
elif [[ "$training_status" != 0 ]]; then
  keep_log=true
  echo "Run was not initialized; local log: $log_file" >&2
fi
exit "$training_status"
