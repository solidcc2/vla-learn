#!/usr/bin/env bash
# Mount project storage on ECS using its instance RAM role.
set -euo pipefail

project="${VLA_OSS_PROJECT:?Set VLA_OSS_PROJECT or use a subproject mount script}"
[[ "$project" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || { echo "Invalid project name" >&2; exit 2; }
data_prefix="${VLA_OSS_DATA_PREFIX:-datasets}"
[[ "$data_prefix" =~ ^[a-zA-Z0-9_-]+(/[a-zA-Z0-9_-]+)*$ ]] || { echo "Invalid data prefix" >&2; exit 2; }
bucket="${VLA_OSS_BUCKET:?Set VLA_OSS_BUCKET}"
region="${VLA_OSS_REGION:?Set VLA_OSS_REGION}"
role="${VLA_OSS_ROLE:?Set VLA_OSS_ROLE}"
mount_root="${VLA_OSS_MOUNT_ROOT:-/mnt/oss/$project}"
endpoint="https://oss-${region}-internal.aliyuncs.com"
action="${1:-mount}"
case "$action" in
  mount|unmount|status) ;;
  *) echo "Usage: $0 [mount|unmount|status]" >&2; exit 2 ;;
esac
[[ "$mount_root" == /* && "$mount_root" != / ]] || { echo 'Mount root must be an absolute non-root path' >&2; exit 2; }
mount_root="${mount_root%/}"

for kind in code data runs; do
  case "$kind" in
    code) prefix="$project/code"; access=rw ;;
    data) prefix="$project/$data_prefix"; access=rw ;;
    runs) prefix="$project/runs"; access=ro ;;
  esac
  target="$mount_root/$kind"
  expected="${bucket}.oss-${region}-internal.aliyuncs.com:/$prefix"
  if findmnt -rn --mountpoint "$target" >/dev/null; then
    source_name=$(findmnt -rn --mountpoint "$target" -o SOURCE)
    filesystem=$(findmnt -rn --mountpoint "$target" -o FSTYPE)
    options=$(findmnt -rn --mountpoint "$target" -o OPTIONS)
    if [[ "$source_name" != "$expected" || "$filesystem" != fuse.ossfs2 ]]; then
      echo "Refusing to change unexpected mount at $target: $source_name ($filesystem)" >&2
      exit 1
    fi
    if [[ "$action" == unmount ]]; then
      sudo umount "$target"
      echo "Unmounted $target"
    else
      [[ ",$options," == *",$access,"* ]] || { echo "Unexpected permissions at $target" >&2; exit 1; }
      echo "$target -> oss://$bucket/$prefix/ ($access, mounted)"
    fi
    continue
  fi
  if [[ "$action" != mount ]]; then
    echo "$target: not mounted"
    continue
  fi
  sudo mkdir -p "$target"
  [[ -z "$(ls -A "$target")" ]] || { echo "Mount point is not empty: $target" >&2; exit 1; }
  # Internal OSS and instance metadata must bypass any shell HTTP proxy.
  sudo env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    /usr/local/bin/ossfs2 mount "$target" \
    --oss_endpoint="$endpoint" --oss_region="$region" --oss_bucket="$bucket" \
    --oss_bucket_prefix="$prefix/" --ram_role="$role" --ro="$([[ "$access" == ro ]] && echo true || echo false)" \
    --sync_upload=true --close_to_open=true \
    --uid="$(id -u)" --gid="$(id -g)" --dir_mode=0755 --file_mode=0644 \
    --total_mem_limit=134217728 --upload_concurrency=2 \
    --prefetch_concurrency=2 --prefetch_concurrency_per_file=2
  findmnt -rn --mountpoint "$target" -o TARGET,SOURCE,OPTIONS
done
