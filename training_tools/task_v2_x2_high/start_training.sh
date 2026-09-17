#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec /home/dzq/openpi/.venv/bin/python launch_training.py \
  --config pi05_umi_task_v2_x2_high_10hz_h50_masked_with_full_head_state_10k_b64_w32_8gpu_v1 \
  --exp-name task_v2_x2_high_10hz_h50_full_head_state_10k_b64_w32_8gpu_20260909 \
  --smoke-script /home/dzq/data_deal/training_tools/task_v2_x2_high/smoke_training.py \
  --memory-fraction 0.90 --wait "$@"
