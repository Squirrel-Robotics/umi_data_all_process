#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# CPU-only conversion, limited to 24 CPUs at lower priority on this shared host.
# The strict converter checks all inputs and refuses an existing output dataset.
exec nice -n 10 taskset -c 0-23 env \
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dzq/openpi/.venv/bin/python converter_strict.py convert \
  --source /mnt/data/dzq/umi_v2/data/task_v2_x2 \
  --target /mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50 \
  --episode-list selected_85_episodes.txt \
  --repo-id dzq/task_v2_x2_high_lerobot_10hz_h50 \
  --task 'Put the two objects into the box.' \
  --fps 10 --action-horizon 50 \
  --max-alignment-ms 100 --max-hand-age-ms 100 --hand-alignment nearest \
  --video-width 640 --video-height 480 --resize-mode letterbox \
  --crf 20 --video-workers 4 --confirm CREATE_LEROBOT_DATASET
