#!/usr/bin/env bash
set -euo pipefail
cd -- /home/dzq/data_deal/training_tools/task_v2_x2_high_right_eye_v2
exec /home/dzq/openpi/.venv/bin/python \
  /home/dzq/data_deal/training_tools/task_v2_x2_high_right_eye_v2/launch_training.py \
  --config pi05_umi_task_v2_x2_high_10hz_h50_right_eye_state_10k_b64_w32_8gpu_v2 \
  --exp-name task_v2_x2_high_right_eye_10hz_h50_state_10k_b64_w32_8gpu_20260909_v2 \
  --smoke-script /home/dzq/data_deal/training_tools/task_v2_x2_high_right_eye_v2/smoke_training.py \
  --memory-fraction 0.90 --wait "$@"
