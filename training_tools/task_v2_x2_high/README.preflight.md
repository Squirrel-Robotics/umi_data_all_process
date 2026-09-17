# task_v2_x2 高质量子集：转换前检查

状态：等待确认断点处理策略。**还未生成 LeRobot 数据集、归一化统计或启动训练。**

英文任务指令：`Put the two objects into the box.`

`training_recipe.json` 保存沿用 task_v3_high 的训练参数：10 Hz、H50、三路未裁剪图像、本体 state 输入、8 卡、全局 batch 64、workers 32、训练 10k 步、每 1k 步保存并保留 checkpoint。JSON 为参数记录，不是可直接传给 OpenPI 的启动配置。

## 数据检查

`track_pass_episodes.txt` 中 95 条数据均存在，必需文件完整。当前完整时间线转换器允许 115 个输出采样点不满足对齐条件，且两条轨迹在中间发生追踪失效，重建后存在 identity 重置边界，不能跨边界生成 action。

严格沿用之前不超过 100 ms 的对齐策略时：85 条能输出 9,995 帧，另 10 条出现多个连续有效片段，原严格转换器会阻止整体发布，不会自动删除或拼接它们。详情见 `data_preflight.json`。

待确认：按连续有效片段拆分（保留有效部分，action 不跨断点），或整条排除这 10 条。无论选择哪一种，都需要在最终转换后重新计算本数据集的 masked norm stats，并验证真实数据加载、state token 和 padding mask。

检查时 8 张 GPU 被其他用户训练占用；未停止任何进程，也未启动排队训练。
