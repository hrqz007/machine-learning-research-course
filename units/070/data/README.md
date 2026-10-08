# 原创连续表面与负对照

roll.csv：320个连续Swiss roll点，输入x0=t*cos(t)、x1=height、x2=t*sin(t)，t在[1.5pi,4.5pi]均匀，height在[0,21]均匀。arc_length=.5*(t*sqrt(1+t*t)+asinh(t))为真实展开横坐标。t仅用作连续颜色，arc_length与height用于事后内在距离核验；模型输入严格限三列x。

gaussian.csv：240个六维标准Gaussian独立样本，单一生成分布，无预设类别。种子70021。ID全部是非个人合成标识。generate_data.py可在新目录恢复所有CSV字节，读取核验摘要和生成器。全部点参与探索性映射，邻域诊断不称作新样本测试性能。
