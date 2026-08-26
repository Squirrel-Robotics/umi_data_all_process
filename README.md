1.data_deal/flatten_data.py
将数据展开到/data里面 使用教程
通用
```/home/dzq/data_deal/flatten_data.py \
  /path/to/source \
  --target /path/to/target \
  --parent-pattern 'run_*' \
  --mode copy \
  --conflict rename \
  --execute```
特例：
```
/home/dzq/data_deal/flatten_data.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --parent-pattern 'collector_run_*' \
  --remove-empty-parents \
  --execute
```

2.将controler_pose转为hand_pose
通用：
```/home/dzq/data_deal/controller_to_hand_pose.py \
  /任意/数据根目录 \
  --input-glob '**/camera/e6_rgb_controller_poses.csv' \
  --skip-existing \
  --execute```
特例：
```
/home/dzq/data_deal/controller_to_hand_pose.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --expected-count 317 \
  --skip-existing
```
不同硬件外参可以改变超参：
```controller_to_hand_pose.py /data/root \
  --input-glob '**/任意输入文件.csv' \
  --output-name 任意输出名.csv \
  --frame-column 帧号字段 \
  --timestamp-column 时间字段 \
  --output-root /可选/输出目录 \
  --skip-existing \
  --execute```
