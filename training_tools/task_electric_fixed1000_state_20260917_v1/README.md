# task_electric 固定手指缩放重训

修复近似静止手指的经验 q01/q99 区间过窄、稀有动作被放大到 800 多的问题。

- 保持全部 115 条数据；不重写 Parquet、视频或原始记录，不裁剪手指动作。
- state/action 的最后 12 维仍是绝对角度（degree）。串口 0～1000 计数对应 0～100°。
- 仅对这 12 维使用固定边界：`normalized = 2 * degrees / 100 - 1`；EEF 前 18 维归一化不变。
- OpenPI 兼容资产的 `q01/q99` 在手指维度表示固定 0/100 边界，不再表示经验百分位。
- 推理应使用新 checkpoint 内随模型保存的 norm stats，反归一化输出仍为绝对角度。不要再额外缩放一次；机器人 SDK 的 Normalized 0～1000 命令是另一套接口单位，不可直接当作串口 0.1° 计数。
- 全部历史实验、原 norm stats 和旧 1000 步 checkpoint 保留。本实验从 Pi0.5 base 的 step 0 开始，不接着旧 checkpoint 训练。

配置：八卡 0–7、global batch 64、128 workers、有 state (`discrete_state_input=True`)、10 Hz / H50、20k 步、每 1k 步保存并保留。学习率、优化器、模型、显存预分配 0.90 均不变。

配置名：`pi05_umi_task_electric_10hz_h50_wrist180_state_fixed1000_20k_b64_w128_8gpu_v1`

资产：`/mnt/data/dzq/openpi/data/assets/umi_task_electric_10hz_h50_right_eye_wrist180_fixed1000_v1`

一次性启动（拒绝覆盖和重复启动）：

```bash
bash /home/dzq/data_deal/training_tools/task_electric_fixed1000_state_20260917_v1/start_training.sh
```

查看日志：

```bash
tail -f /home/dzq/openpi/logs/task_electric_right_eye_wrist180_state_fixed1000_10hz_h50_20k_b64_w128_8gpu_20260917_v1.log
```

Checkpoint 根目录：

```text
/mnt/data/dzq/openpi/checkpoints/pi05_umi_task_electric_10hz_h50_wrist180_state_fixed1000_20k_b64_w128_8gpu_v1/task_electric_right_eye_wrist180_state_fixed1000_10hz_h50_20k_b64_w128_8gpu_20260917_v1/
```

`build_fixed_norm.py` 重新核对所有有效串口反馈和数值目标；`finger_scale_audit.json` 保存全量计算结果。`smoke_training.py` 验证全部数值、视频来源/哈希、固定缩放、反归一化、state token 和 H50 padding mask 后才允许启动。
