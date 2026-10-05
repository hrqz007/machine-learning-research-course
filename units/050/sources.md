# 第050讲来源与实现约定

核对日期2026-10-05。推导、设备机制、四行手算、两设备权重例和十幅图均为本讲原创。未复制论文图、书章或受版权保护的大段正文。

## 主要原始资料

1. Roberts, D. R. 等（2017），Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure，Ecography 40:913–929，DOI 10.1111/ecog.02881。用于支持依赖结构与预测目标共同决定留出设计的论述，尤其原文关于先确定预测目标、再按结构划块的步骤。作者机构原论文副本已读取：https://www.biom.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf/at_download/file
2. Kapoor, S. 与 Narayanan, A.，Leakage and the Reproducibility Crisis in ML-based Science，作者预印本 arXiv:2207.07048（2022首版）。用于泄漏风险和明确信息流程的报告动机，不复用跨领域数量或原论文案例作本课结果：https://arxiv.org/abs/2207.07048

## 官方API资料

3. scikit-learn1.8 GroupShuffleSplit。test_size按组，重复测试集合未必互斥。主课固定索引自己保存，独立真实API调用只核对组隔离语义：https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
4. scikit-learn1.8 TimeSeriesSplit。gap按输入样本数，先对日索引切分，避免将两条设备记录误作两天；明确有序和等间隔假设：https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
5. scikit-learn1.8 OneHotEncoder。handle_unknown='ignore'使未知设备指示列全零，类别集合仅fit训练数据：https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html
6. scikit-learn1.8 Ridge。残差平方和加alpha系数平方和；使用fit_intercept=True、solver='svd'作独立科学参照，截距不放入惩罚列：https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.Ridge.html

主运行锁定sklearn1.8.0。在线stable页面可能指向不同版本，因此这里列实际检查的1.8链接。独立参照还以Python Fraction作精确有理数Gauss-Jordan，以SciPy1.17的lstsq(lapack_driver='gelsy')作带列主元QR，与NumPy2.3.5增广最小二乘比较。

## 证据边界

原论文提供方法论背景；本课程的有限合成实验不构成原论文所有结论的独立现实复现。固定种子只保证复现，80份面板不是80个真实工厂，测试设备Bootstrap不等于全部算法不确定性。图中趋势必须与本课生成机制和可用信息一起解读。
