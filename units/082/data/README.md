# 数据字典与来源

generate_data.py固定种子82021。dataset.json保存全部原始数据，generation.json保存字节数与SHA256。默认重建到outputs/generated-data；实验通过相同make_data生成，不下载数据。

## 原创三高斯真值

训练600、验证200、测试400，分别独立抽样。先抽隐藏z，再按分量高斯抽二维x。权重[0.45,0.35,0.2]，均值[[-2,-0.7],[0.9,1.6],[2,-1.2]]，协方差完整写入truth字段。z只是合成世界已知生成来源，不拿来拟合或选K；truth参数只在冻结选择后计算恢复及oracle对照。

## 原创单高斯负对照

训练600、验证200、测试400，二维标准正态；没有混合隐藏真类别。negative原始第一维另供完整共轭参数后验预测基准：观测方差1已知、未知均值θ先验N(0,4)。该基准使用原始坐标而非标准化坐标，每份复制400行共享一次后验θ抽样。它与主GMM固定参数复制分开记录，不替代GMM参数后验。

## 真实Wine

随scikit-learn.datasets.load_wine提供178行历史葡萄酒化学测量，原数据13特征、3个品种来源标签。本项目预声明只用列0 alcohol与列6 flavanoids；单位沿用来源，不擅自补全原页面未给出的单位。随机划分106训练、36验证、36测试；不按标签分层，标签不参与划分或拟合。

字段x为两项原始观测；row_id保留随包原行号；cultivar为原始品种类别编号0至2，仅冻结模型后的外部ARI审计。训练均值/标准差由experiment.py独立拟合，原始数据文件不预先用全量数据标准化。

出处：Aeberhard, S. & Forina, M. (1992). Wine. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5PC7J 。页面：https://archive.ics.uci.edu/dataset/109/wine 。许可CC BY 4.0：https://creativecommons.org/licenses/by/4.0/ 。本课程修改：预选两列、保存行号、内部随机划分；不修改品种标签。随包接口：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html 。

边界：真实品种是外部来源分类，不是高斯分量真值；K与品种数可以不同。没有时间、产地批次或新生产商外部验证，不能推出化学因果机制或跨地区泛化。
