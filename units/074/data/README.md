# 原创数据与字段字典

train与audit来自一维三成分Gaussian混合。x是唯一模型输入，source记录生成来源，不输入EM。权重0.35/0.4/0.25，均值-3/0/3，标准差0.45/0.8/0.45。

id为唯一字符串对象标识，不输入模型。所有数值保留12位小数。generation.json记种子、来源与每份CSV的SHA-256；加载器还把冻结字节与生成器输出比对。无真实个人或业务数据，无外部下载。

运行`python generate_data.py --directory outputs/generated-data`重建到新目录。Python/NumPy版本变化可能改变随机数实现或末位；若摘要不同，应先记录版本差异，不要删除校验。
