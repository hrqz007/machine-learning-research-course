# 真实Wine小样本数据

wine_tiny.json保存30条真实测量，每类10条，共13个原始化学特征。它不是合成数据，也不是随机重标记任务。数据取自scikit-learn随包Wine副本；运行时仅使用此JSON，不导入scikit-learn，不联网。

## 来源、许可与修改

Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. DOI：https://doi.org/10.24432/C5PC7J 。官方页：https://archive.ics.uci.edu/dataset/109/wine 。许可为CC BY 4.0：https://creativecommons.org/licenses/by/4.0/ 。来源页核验于2026-10-10。

原库178行、13特征、3个品种来源，研究对象为意大利同一区域不同品种葡萄酒的化学分析。课程不推断原页面未明确列出的单位，不声称这些特征构成因果机制。原始标签1–3在随包副本中编码为0–2。

本课程修改：根据已知类别，对每一类取原始顺序前10个行号；保留所有13个原始特征与标签；转换为JSON；row_ids使用从0开始的随包原始行号。未依据训练效果挑行，未添加噪声，未改写测量值。

具体行号：0–9、59–68、130–139。第一组类别0，第二组1，第三组2。标签参与构造一个类别均衡的刻意记忆集合，因此不能宣称无标签抽样或总体随机样本。

## 字段

- features：30×13浮点数组，每行一条酒样，每列按feature_names排列
- labels：30个整数，取值0、1、2
- row_ids：30个唯一零起始原始行号，便于追溯
- feature_names：13列名称，顺序为alcohol、malic_acid、ash、alcalinity_of_ash、magnesium、total_phenols、flavanoids、nonflavanoid_phenols、proanthocyanins、color_intensity、hue、od280/od315_of_diluted_wines、proline
- source、source_url、license、selection：来源与固定选择规则

## 预处理与使用责任

文件保存原始数值。load_wine只用这30条训练记录计算每列均值和总体标准差（ddof=0），在训练时做z-score。零标准差列会显式报错。数据仅用于检查网络能否记住一个真实小集合，训练和评价使用同一批样本。

本课没有验证集、测试集或外部评估；不得把100%训练准确率写成泛化准确率。标签抽取、原始顺序和小样本规模都限制代表性。需要评价现实预测时，应另行建立隔离训练/验证/测试或交叉验证协议。

## 完整性与离线复现

manifest.json记录该JSON的字节数、SHA256、形状与行号。数据固定随课程交付，experiment-result.json也保存相同哈希。实验中的正态随机输入和上游探针不是Wine数据的一部分，而由明确种子的NumPy生成器在本地生成。
