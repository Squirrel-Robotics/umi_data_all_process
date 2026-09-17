#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="/home/dzq/data_deal/data_choose/runtime"

mkdir -p "${RUNTIME_DIR}"
exec 9>"${RUNTIME_DIR}/server.lock"
if ! flock -n 9; then
  echo "data_choose 已经在运行；请不要重复启动。" >&2
  exit 1
fi

exec /home/dzq/openpi/.venv/bin/python "${SCRIPT_DIR}/app.py" \
  --root /mnt/data/dzq/umi/data/task_v1_new \
  --host 127.0.0.1 \
  --port 8091 \
  --cache "${RUNTIME_DIR}/cache" \
  --state-dir "${RUNTIME_DIR}" \
  --token-file "${RUNTIME_DIR}/review_token" \
  "$@"
