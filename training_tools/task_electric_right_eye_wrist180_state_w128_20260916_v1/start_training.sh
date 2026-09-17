#!/usr/bin/env bash
set -euo pipefail
cd /home/dzq/data_deal/training_tools/task_electric_right_eye_wrist180_state_w128_20260916_v1
exec /home/dzq/openpi/.venv/bin/python launch_training.py \
  --config pi05_umi_task_electric_10hz_h50_wrist180_state_20k_b64_w128_8gpu_v1 \
  --exp-name task_electric_right_eye_wrist180_state_10hz_h50_20k_b64_w128_8gpu_20260916_v1 \
  --smoke-script /home/dzq/data_deal/training_tools/task_electric_right_eye_wrist180_state_w128_20260916_v1/smoke_training.py \
  --memory-fraction 0.90 --wait "$@"
