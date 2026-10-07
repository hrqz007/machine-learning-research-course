# 第067讲 一级资料与核查范围

核查日期：2026-10-07。公式使用标准推导，数据、代码、图表及教材文字重新制作。未把网页的特定性能结果当成本实验结果。

1. [Rasmussen and Williams GPML chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf)
   - 核查范围：26页原书章节；式2.22至2.24条件后验与Cholesky算法
   - 状态：retrieved
2. [Rasmussen and Williams GPML chapter 5](https://gaussianprocess.org/gpml/chapters/RW5.pdf)
   - 核查范围：24页原书章节；式5.8对数边际似然与局部最优限制
   - 状态：retrieved
3. [GaussianProcessRegressor 1.8.0 API](https://scikit-learn.org/1.8/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
   - 核查范围：alpha训练对角、optimizer、normalize_y与协方差返回
   - 状态：retrieved
4. [Gaussian processes official guide](https://scikit-learn.org/stable/modules/gaussian_process.html)
   - 核查范围：核类型、噪声参数和WhiteKernel；stable文档当前为1.9.1，实验冻结1.8.0
   - 状态：retrieved

稳定版网页可能更新；运行环境固定scikit-learn 1.8.0。文档内容与实际接口、数值实验交叉验证。
