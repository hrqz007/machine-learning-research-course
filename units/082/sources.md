# 一手来源与核验范围

核验日期2026-10-10。在线stable文档当时指向scikit-learn 1.9.1；实际冻结运行1.8.0。只使用公开且本地实测的共用接口，不把网页版本当作运行版本。

1. Aeberhard, S. & Forina, M. (1992). Wine. UCI Machine Learning Repository：https://archive.ics.uci.edu/dataset/109/wine 。DOI https://doi.org/10.24432/C5PC7J 。核对178行、13项化学特征、三个品种来源与CC BY 4.0许可；课程只取酒精、黄酮两列，重新分割并保留原始行号。来源范围有限，不作为高斯潜类真值。
2. scikit-learn，load_wine：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html 。核对随包加载接口、特征与标签。实验不下载网络数据。
3. scikit-learn，Gaussian mixture models：https://scikit-learn.org/stable/modules/mixture.html 。核对混合密度、EM应用与模型选择背景。课程的匹配、真值恢复、硬近似缺口、稳定性和复制检查均为原创研究流程。
4. scikit-learn，GaussianMixture：https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html 。核对full协方差、reg_covar、n_init、converged_、score_samples、predict_proba与BIC公开接口；不依赖库私有API进行独立责任度核验。

数学范围：ELBO恒等式、标签交换不变性和已知方差正态共轭后验均在lecture.md给出可检查推导与数值测试。主GMM仅固定参数复制；独立标量共轭基准才执行参数后验预测。两者不得混称。

外部资料只支持定义、接口及真实数据来源；课程的合成机制、实验、图形、中文叙述与实际数值全部为原创制作。数据衍生与许可说明见data/README.md。
