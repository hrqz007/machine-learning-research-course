# 第087课 一手来源与核验范围

核验日期：2026-10-10。下列页面和论文正文已通过公开网页读取核对，不以搜索摘要代替公式核验。教材推导、练习、实验设计和图形均为课程原创表述；没有复制论文图表。

1. Xavier Glorot与Yoshua Bengio，Understanding the difficulty of training deep feedforward neural networks，AISTATS 2010，PMLR 9:249–256。
   - 官方出版页：https://proceedings.mlr.press/v9/glorot10a.html
   - 正文：https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
   - 已核对第4.2节、正文第253页公式10–12：前向/反向尺度条件及折中初始化。该推导以初始化时近线性激活等假设为前提。
   - 本课使用正态同方差版本并自行推导对应均匀边界；不声称原论文中的每个实验都由本课复现。
2. Kaiming He、Xiangyu Zhang、Shaoqing Ren、Jian Sun，Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification，2015。
   - 作者论文：https://arxiv.org/pdf/1502.01852
   - 摘要与版本页：https://arxiv.org/abs/1502.01852
   - 已核对第2.2节公式7–10及反向公式12–14：ReLU作用后的平方期望、二阶矩与方差区别、前后向fan条件。
   - CVF出版页本次返回403，未声称已读该页；公式核验使用成功读取的作者arXiv版本。未复制图片，未运行ImageNet实验。
3. Kaiming He等，Identity Mappings in Deep Residual Networks，2016。
   - 正文：https://arxiv.org/pdf/1603.05027
   - 版本页：https://arxiv.org/abs/1603.05027
   - 已核对第2节公式及第3–4节条件：恒等跳连、相加后恒等映射对直接传播路径的重要性；投影、缩放、门和激活会改变路径。
   - 本课h + α tanh(hW)仅是原创诊断模型，不是原论文网络的复现；抵消/放大标量反例为本课自行计算。
4. Aeberhard, S.与Forina, M.，Wine，UCI Machine Learning Repository。
   - 官方数据页：https://archive.ics.uci.edu/dataset/109/wine
   - DOI：https://doi.org/10.24432/C5PC7J
   - 许可：https://creativecommons.org/licenses/by/4.0/
   - 已核对178行、13特征、三个品种以及CC BY 4.0许可。官方页显示捐赠日期1991，推荐引文年份1992；本课程沿用官方推荐引文，不将两者混写。
   - 冻结数据来自已安装scikit-learn的load_wine随包副本。只取每类原始行号前10条、保留13特征、将原类别1–3表示为0–2。详见data/README.md。

## 证据边界

论文说明推导与既有研究，不能替本课的代码正确性背书。代码通过独立有限差分、手算值、边界输入、双解释器unittest及真实小样本训练检验；完整记录在测试与实验JSON文件中。三种子范围不是置信区间；记忆成功不是泛化验证；初始化随机探针不是训练损失的全部雅可比谱。
