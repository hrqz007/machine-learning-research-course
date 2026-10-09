# 原创数据与字段字典

development和audit来自三个Gaussian来源；null来自一个标准Gaussian。null中的source是与坐标无关的占位编号，不是其生成类别。outcome按2*x0-x1加标准差0.5的独立噪声生成。所有x同人工单位，source和outcome不进入聚类。

id为唯一字符串对象标识，不输入模型。所有数值保留12位小数。generation.json记种子、来源与每份CSV的SHA-256；加载器还把冻结字节与生成器输出比对。无真实个人或业务数据，无外部下载。

运行`python generate_data.py --directory outputs/generated-data`重建到新目录。Python/NumPy版本变化可能改变随机数实现或末位；若摘要不同，应先记录版本差异，不要删除校验。
