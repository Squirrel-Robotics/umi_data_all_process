#!/usr/bin/env bash
set -euo pipefail
cd /home/dzq/data_deal/training_tools/task_v2_x2_high_wrist180_state_ablation_4gpu_v1
exec /home/dzq/openpi/.venv/bin/python launch_pair.py
