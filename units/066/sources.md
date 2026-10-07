# 第066讲 一级资料与核查范围

核查日期：2026-10-07。公式使用标准推导，数据、代码、图表及教材文字重新制作。未把网页的特定性能结果当成本实验结果。

1. [KernelRidge 1.8.0 API](https://scikit-learn.org/1.8/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
   - 核查范围：alpha、kernel、gamma接口与实际库对照
   - 状态：retrieved
2. [SVC 1.8.0 API](https://scikit-learn.org/1.8/modules/generated/sklearn.svm.SVC.html)
   - 核查范围：支持向量属性、核参数与缓存设置
   - 状态：retrieved
3. [Kernel ridge regression guide](https://scikit-learn.org/stable/modules/kernel_ridge.html)
   - 核查范围：KRR和SVR损失及稀疏表示区别；不引用来源性能倍数
   - 状态：retrieved
4. [Kernel approximation guide](https://scikit-learn.org/stable/modules/kernel_approximation.html)
   - 核查范围：Nystroem、RBFSampler、核内积与近似特征
   - 状态：retrieved
5. [Williams and Seeger Nyström paper record](https://papers.nips.cc/paper_files/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html)
   - 核查范围：NIPS 2000论文身份；官方PDF文本编码异常，算法说明以官方库指南交叉核对
   - 状态：retrieved_metadata
6. [Rahimi and Recht original RFF paper](https://papers.nips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf)
   - 核查范围：8页原论文；随机特征内积近似机制
   - 状态：retrieved

稳定版网页可能更新；运行环境固定scikit-learn 1.8.0。文档内容与实际接口、数值实验交叉验证。
