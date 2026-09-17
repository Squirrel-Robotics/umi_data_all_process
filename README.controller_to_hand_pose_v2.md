# controller_to_hand_pose_v2 使用说明

2026-09-09 版内置已确认的 **test_true 坐标定义**，并默认将两只手的原点分别沿其最终 Hand 局部 X 轴移动 **−42 mm**。这是独立的 Python 3.8+ 脚本，只用标准库；复制 `controller_to_hand_pose_v2.py` 到另一台机器即可运行，不需要 NumPy、ROS、旧脚本或 `umi_convert` 文件夹。

## 按新标定转换

推荐对 `task_v2_x2` 另存新文件，保留已有结果：

```bash
python3 /home/dzq/data_deal/controller_to_hand_pose_v2.py \
  /mnt/data/dzq/umi_v2/data/task_v2_x2 \
  --input-glob '**/camera/e6_rgb_controller_poses.csv' \
  --output-name hand_pose_test_true_minus42mm.csv \
  --execute
```

每个 episode 的 `camera/` 下生成 `hand_pose_test_true_minus42mm.csv`，内容是偏移后末端的逐帧相对增量。需要先预览时去掉 `--execute`；预览只列出匹配和输出计划，不读取校验 CSV 内容，也不写文件。加 `--verbose` 可显示每个文件。

不指定 `--output-name` 时，默认输出仍为每个输入旁的 `camera/hand_pose.csv`。脚本更新不会自动重算已有数据，本次更新本身也不批量替换任何数据 CSV。已有目标默认拒绝覆盖；确认要替换时显式使用 `--overwrite`。

**重新应用本次标定时不要直接使用 `--skip-existing`**：它只根据目标是否存在跳过，不检查矩阵、左右映射或 −42 mm 偏移，可能把旧结果保留下来。`--skip-existing` 只适合已核对同一配置的批次续跑，不能与 `--overwrite` 同用。

取消 −42 mm 偏移、另存原始 Hand 原点的结果：

```bash
python3 /home/dzq/data_deal/controller_to_hand_pose_v2.py \
  /mnt/data/dzq/umi_v2/data/task_v2_x2 \
  --target-origin-x-m 0 \
  --output-name hand_pose_test_true_no_offset.csv \
  --execute
```

同一命令可将输入目录替换为 `/mnt/data/dzq/umi_v2/data/test_true`。脚本不会按目录名称判断硬件版本或左右通道；不同硬件应分别使用正确标定及通道映射。

另存目录并生成报告：

```bash
python3 controller_to_hand_pose_v2.py /path/to/dataset \
  --output-root /path/to/converted \
  --output-name hand_pose_test_true_minus42mm.csv \
  --report /path/to/conversion_report.json \
  --verbose --execute
```

`--output-root` 保留原目录层级。已有报告也需要另取名字或明确 `--overwrite`。

单文件：

```bash
python3 controller_to_hand_pose_v2.py /path/to/e6_rgb_controller_poses.csv \
  --output /path/to/hand_pose_test_true_minus42mm.csv --execute
```

## 已确认的旋转矩阵与通道

内置矩阵与 `confirmed_hand_calibration.json` 的 `computed_transforms` 一致。下面 `left/right` 指物理手别：

| 物理手（输出列） | 默认原始输入通道 | 偏移前 Hand 原点在 controller 中的位置，m |
| --- | --- | --- |
| 左手 `left_*` | 原始 `left_*` | `(-0.007435, -0.015192, -0.053839)` |
| 右手 `right_*` | 原始 `right_*` | `(+0.007435, -0.015192, -0.053839)` |

默认 `--side-map identity`。若某批数据的原始通道与物理手别相反，显式使用 `--side-map swapped`：物理左手读取原始 `right_*`，物理右手读取原始 `left_*`。先选择来源通道，再按物理手别应用外参。

`R_C_H` 将 Hand 中的列向量映射到原生 controller 坐标系；每列分别是 Hand 的 +X、+Y、+Z 轴在 controller 中的方向：

```text
R_left =
[ 0                    0                   -1 ]
[-0.5439932518055669   0.8390895911581822     0 ]
[ 0.839089591158182    0.5439932518055669     0 ]

R_right =
[ 0                    0                    1 ]
[-0.5439932518055669  -0.8390895911581822     0 ]
[ 0.839089591158182   -0.5439932518055669     0 ]
```

两者满足 `R.T @ R = I`、`det(R) = +1`。三点定义为双手 `O→P1 = +X`，左手 `O→P2 = −Y`、右手 `O→P2 = +Y`，点坐标如下（单位 mm，左右分别取负、正的 x）：

```text
O  = (±7.435,  -15.192,  -53.839)
P1 = (±7.435,  -20.632,  -45.448)
P2 = (±7.435, -100.782, -109.321)
```

第一方向严格保留，第二方向正交化，最后一轴按右手规则补齐。以上是最终确认的旋转矩阵，不再叠加早期的 Y 轴 90°、左右 Rx(±90°) 或轴交换。

网页 JSON 中的独立世界显示变换 `B` 不应用到本脚本的 controller→Hand 标定。输入使用原生 UMI 世界坐标；网页的显示变换也不能当作局部外参右乘。

## −42 mm 如何应用

定义 `C` 为 controller，`H` 为已确认的 Hand 坐标系，`F` 为偏移后的末端，`W` 为输入世界坐标系。默认处理顺序是：

```text
r = (-0.042, 0, 0) m
T_H_F = [I, r]
T_W_F[t] = T_W_C[t] @ T_C_H @ Trans(-0.042, 0, 0)
D_F[t] = inverse(T_W_F[t-1]) @ T_W_F[t]
```

平移只在各手的最终 Hand 局部 X 方向执行一次，旋转矩阵保持不变。等效 controller→末端平移为 `t_C_F = t_C_H + R_C_H @ r`；默认两侧分别为：

```text
left:  (-0.007435, 0.007655716575833811, -0.08908076282864365) m
right: (+0.007435, 0.007655716575833811, -0.08908076282864365) m
```

省略 `--target-origin-x-m` 即使用 `-0.042`；显式传 `--target-origin-x-m 0` 取消该偏移。参数单位始终是米，`--position-unit mm` 只影响输入 controller 位置。不要写成 `-42`，那表示 42 米。

对于偏移前的相对增量 `D_H`，有：

```text
D_F = inverse(T_H_F) @ D_H @ T_H_F
R_F_delta = R_H_delta
p_F_delta = p_H_delta + (R_H_delta - I) @ r
```

因此静止或纯平移时增量不变；旋转时会出现由原点偏移引起的平移差。绝对原点距离为 42 mm，不代表每一行 `dx` 减少 42 mm，也不是沿世界 X 轴移动。

该偏移假定目标轴方向与 Hand 一致。报告中的 `robot_flange_transform_applied` 只标识启用了此平移，`robot_flange_calibration_verified` 为假，不表示已验证机器人法兰的完整标定。

## 输出含义

默认输出是**相对于上一帧偏移后末端自身坐标系**的逐帧增量 `D_F`。取消偏移后，输出为相对于上一帧 Hand 自身坐标系的 `D_H`。它不是世界系位置差，也不以第一帧作为所有行的统一参考。重建使用右乘：`T[t] = T[t-1] @ D[t]`。

固定世界基变换在相对计算中消去，所以直接使用原生绝对位姿求相对增量，与先左乘网页的世界显示变换再求相对增量一致。

输出共 20 列：四个时间/帧号列，以及每只手各八列：

```text
frame_number, previous_frame_number, timestamp_ns, dt_ns
left_relative_valid, left_local_dpx, left_local_dpy, left_local_dpz,
left_local_dqx, left_local_dqy, left_local_dqz, left_local_dqw
right_relative_valid, right_local_dpx, right_local_dpy, right_local_dpz,
right_local_dqx, right_local_dqy, right_local_dqz, right_local_dqw
```

位置单位为米，四元数顺序 XYZW，时间戳为整数纳秒。四元数不是欧拉角；`dp` 不是速度，不除以 `dt`。第一行是单位增量且 `relative_valid=0`，前一帧号和时间差留空。

如果后续还需要从已偏移的 `F` 转到另一个机器人坐标系，必须使用相对于 `F` 的完整刚体变换，不能再次叠加同一个 −42 mm，也不能直接把固定平移加到每行 `dp` 上。

## 输入约定和无效数据

- 输入必须是原生 UMI controller 绝对位姿，列为 `left/right_px,py,pz,qx,qy,qz,qw`，以及各自的 `active`。默认位置为米；毫米输入加 `--position-unit mm`。
- 自动识别 `frame_number / frame_id / sequence_number`。时间优先 `timestamp_ns`，然后 E6 realtime/UTC/boot、controller realtime/boot。可用 `--frame-column` 和 `--timestamp-column` 明确指定。
- 有 `pose_matched` 时检查该列；可用 `--matched-column NAME` 指定其他列。`none` 禁用全局匹配标志检查。
- 检查来源通道的 `active`；若有 `position_valid`、`orientation_valid`，它们也必须为真。这里不额外要求 `position_tracked` 或 `orientation_tracked`。
- 默认帧号必须连续加一。断帧对应增量标为无效，不跨过去拼接；若确实需要计算相邻 CSV 行的变化，可用 `--require-frame-step 0`。`--max-dt-ns` 可另设最大时间间隔。
- 前一帧或当前帧无效时，该手增量为单位变换且有效标记为零，不跨无效帧连接。消费端必须读取有效标记；恢复后如需重建绝对轨迹，要用新的有效绝对位姿重新初始化。
- 时间戳重复/倒序、有效位姿含 NaN/Inf、零四元数或长度误差大于 0.05 都报错。仅对允许范围内的四元数做归一化。不平滑、不插值、不删跳变行；本脚本不是异常运动过滤器。

## 更换硬件标定

本次确认的矩阵已内置，直接运行即可，**不要给本脚本传入刚导出的 `confirmed_hand_calibration.json`**。该文件是 `umi.three_point_calibration.v1`，供 `umi_convert.py --config` 使用；本脚本的 `--calibration-json` 只接受下述 `umi.controller_to_hand.direct.v2` 直接矩阵格式。

更换硬件时，可通过 `--calibration-json /path/to/new_calibration.json` 提供 controller→Hand 外参。下面示例是两侧单位外参，不是本次硬件参数，不能直接用于本次数据：

```json
{
  "schema_version": "umi.controller_to_hand.direct.v2",
  "coordinate_system": "umi_native",
  "translation_unit": "m",
  "transforms": {
    "left": {
      "translation": [0, 0, 0],
      "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    },
    "right": {
      "translation": [0, 0, 0],
      "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    }
  }
}
```

自定义外参不改变 `--side-map`，也不取消默认的局部 X 偏移。若自定义外参的平移已包含目标原点偏移，必须同时传 `--target-origin-x-m 0`，避免重复偏移。

## 写入与验证

脚本不修改原始 controller CSV。所有待处理输入转换、逐帧右乘重建验证、暂存 CSV 读回验证都通过后，才逐个发布最终文件。输入处理期间变化会取消提交。已有结果默认拒绝覆盖，并拒绝输出覆盖源数据、脚本、配置或指向它们的硬链接/符号链接。

单文件发布是原子的；整批不是跨文件事务。如果提交过程中磁盘或权限报错，可能已有部分文件成功，脚本会列出已保存文件。修复后，核对已保存结果采用同一配置，再使用 `--skip-existing` 续跑。

测试命令（测试文件不是运行依赖）：

```bash
python3 -m unittest discover -s tests -p test_controller_to_hand_pose_v2.py -v
```

验证最终矩阵与确认 JSON 一致、右手性、左右映射、默认及取消原点偏移、180° 旋转、纳秒精度、断帧和无效位姿、防覆盖及单文件迁移。
