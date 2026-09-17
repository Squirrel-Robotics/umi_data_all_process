#!/usr/bin/env bash
set -euo pipefail
cd -- /home/dzq/data_deal/training_tools/task_v2_x2_high_right_eye_no_state_v1
exec /home/dzq/openpi/.venv/bin/python \
  /home/dzq/data_deal/training_tools/task_v2_x2_high_right_eye_no_state_v1/launch_training.py \
  --config pi05_umi_task_v2_x2_high_10hz_h50_right_eye_no_state_10k_b64_w32_8gpu_v1 \
  --exp-name task_v2_x2_high_right_eye_10hz_h50_no_state_10k_b64_w32_8gpu_20260910_v1 \
  --smoke-script /home/dzq/data_deal/training_tools/task_v2_x2_high_right_eye_no_state_v1/smoke_training.py \
  --memory-fraction 0.90 --wait "$@"
