# 原创数据与字段字典

四份数据：train/audit同方差，hetero_train/hetero_audit异方差。x0至x3是四项读数；z0,z1是人工生成潜真值，不输入PPCA/FA，只用于训练对齐与审计恢复评价。每行id唯一，所有坐标是人工单位。

id为唯一字符串对象标识，不输入模型。所有数值保留12位小数。generation.json记种子、来源与每份CSV的SHA-256；加载器还把冻结字节与生成器输出比对。无真实个人或业务数据，无外部下载。

运行`python generate_data.py --directory outputs/generated-data`重建到新目录。Python/NumPy版本变化可能改变随机数实现或末位；若摘要不同，应先记录版本差异，不要删除校验。
