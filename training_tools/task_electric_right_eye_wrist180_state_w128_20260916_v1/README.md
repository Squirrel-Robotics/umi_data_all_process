# task_electric：128 workers 重启版

用户要求将 `num_workers=32` 改成 **128** 并立即重启。旧实验未保存首个 1k checkpoint，因此新实验从 Pi0.5 基础模型、step 0 开始；旧日志和目录保留，不覆盖。

除实验名称和 `num_workers` 外，配置与 32-worker 版本完全相同：八卡 0–7、全局 batch 64、有 state (`discrete_state_input=True`)、10 Hz / H50、20,000 步，每 1,000 步保存并保留、32维模型接口、学习率/优化器不变、显存预分配 0.90。

数据不重新转换，不改变数值、视频或归一化统计。仍使用 E6 右眼、cam1→左腕、cam0→右腕、双腕各旋转 180° 的已验证数据集。

- 数据：`/mnt/data/dzq/umi_v2/datasets/task_electric_lerobot_10hz_h50_right_eye_wrist180_v1`
- 配置：`pi05_umi_task_electric_10hz_h50_wrist180_state_20k_b64_w128_8gpu_v1`
- 实验：`task_electric_right_eye_wrist180_state_10hz_h50_20k_b64_w128_8gpu_20260916_v1`
- 工具：`/home/dzq/data_deal/training_tools/task_electric_right_eye_wrist180_state_w128_20260916_v1`

查看训练：

```bash
tail -f /home/dzq/openpi/logs/task_electric_right_eye_wrist180_state_10hz_h50_20k_b64_w128_8gpu_20260916_v1.log
```

Checkpoint：

```text
/mnt/data/dzq/openpi/checkpoints/pi05_umi_task_electric_10hz_h50_wrist180_state_20k_b64_w128_8gpu_v1/task_electric_right_eye_wrist180_state_10hz_h50_20k_b64_w128_8gpu_20260916_v1/
```

`start_training.sh` 先执行 CPU 核验，再等待八卡空闲并启动一次，不会重复启动、自动重试或覆盖旧 checkpoint。启动后不要再次执行它。`launcher.log` 和训练日志旁的 `.queue.json`/`.launch.json` 记录进度。

128 个 spawn worker 的首次启动会比 32 个更久，吞吐是否提升应以进入训练后的稳定窗口实测为准。
