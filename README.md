1.data_deal/flatten_data.py
将数据展开到/data里面 使用教程
/home/dzq/data_deal/flatten_data.py \
  /path/to/source \
  --target /path/to/target \
  --parent-pattern 'run_*' \
  --mode copy \
  --conflict rename \
  --execute
