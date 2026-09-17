#!/usr/bin/env bash
set -euo pipefail
exec nice -n 10 env \
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dzq/openpi/.venv/bin/python \
  /home/dzq/data_deal/training_tools/compute_masked_norm_stats.py \
  --dataset /mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50 \
  --assets-base-dir /mnt/data/dzq/openpi/data/assets \
  --asset-id umi_task_v2_x2_high_hand_pose_10hz_h50_masked_v1 \
  --expected-fps 10 --expected-horizon 50 \
  --expected-task 'Put the two objects into the box.'
