# task_v2_x2 完整右眼：无 state 输入版

复用已修复的 85 条、9,995 帧数据，不重新转换、不修改 state/action 或三路视频。模型使用 Pi0.5，`discrete_state_input=False`，条件输入只有三路图像与任务文本；保留 `observation.state` 字段供数据格式、归一化和 batch 形状接口使用，其数值不参与模型条件输入。

任务文本：`Put the two objects into the box.`

## 固定配置

- 配置：`pi05_umi_task_v2_x2_high_10hz_h50_right_eye_no_state_10k_b64_w32_8gpu_v1`
- 数据：`/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2`
- 10 Hz、H50、GPU 0–7、全局 batch 64、workers 32、随机种子 42。
- 10,000 步，每 1,000 步保存并保留 checkpoint；XLA 显存比例 0.90。
- 与带 state 版使用相同的归一化、动作 padding mask、学习率及图像增强，不添加固定 ROI。
- 从 `/mnt/data/checkpoints/pi05_base/params` 重新开始，不恢复带 state 版 checkpoint。

## 启动与验证

```bash
bash /home/dzq/data_deal/training_tools/task_v2_x2_high_right_eye_no_state_v1/start_training.sh
```

入口只允许全新实验，先执行 CPU 验证，再等 8 卡全部空闲后启动；已有同名日志或 checkpoint 会拒绝启动，失败不自动重试。不杀其他任务，也不覆盖旧实验。

验证保留全部右眼、85 条选择、100 ms 对齐、动作及归一化检查，并增加真实样本 state 扰动测试：改变 state 数值不得改变图像、prompt tokens、mask 或动作；检查 Pi0.5 没有另一路连续 state 投影。

日志和启动收据使用同一前缀：

```text
/home/dzq/openpi/logs/task_v2_x2_high_right_eye_10hz_h50_no_state_10k_b64_w32_8gpu_20260910_v1
```

`.log` 是实际训练进度；`.smoke.log` 是验证结果；`.queue.json` 是启动状态；`.launch.json` 记录实际训练 PID、启动时间和配置/数据校验值。启动状态不会代替最终训练日志。

Checkpoint 目录：

```text
/mnt/data/dzq/openpi/checkpoints/pi05_umi_task_v2_x2_high_10hz_h50_right_eye_no_state_10k_b64_w32_8gpu_v1/task_v2_x2_high_right_eye_10hz_h50_no_state_10k_b64_w32_8gpu_20260910_v1
```

本目录中的 `backups/openpi_config.before_no_state.py` 为添加独立配置前的备份。原有带 state 的配置、数据和 checkpoint 保持原样。
