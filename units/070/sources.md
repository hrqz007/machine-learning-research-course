# 一级资料与核对范围

2026-10-08重新读取。讲解、数据、手算和邻域实现为原创；以下为作者论文或官方项目文档。

1. [van der Maaten与Hinton 2008 t-SNE论文](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)：条件相似度、perplexity、对称化与Student t低维分布、KL目标。JMLR原站PDF首次超时，改用作者主页同篇全文成功读取。
2. [McInnes等 UMAP论文](https://arxiv.org/abs/1802.03426)：UMAP理论框架与作者参考实现。
3. [UMAP工作原理](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)：局部尺度、加权图与低维图布局的直观。
4. [UMAP参数说明](https://umap-learn.readthedocs.io/en/latest/parameters.html)：n_neighbors与min_dist作用。
5. [UMAP复现说明](https://umap-learn.readthedocs.io/en/latest/reproducibility.html)：随机种子、并行与首次编译时间。
6. [scikit-learn 1.8 TSNE](https://scikit-learn.org/1.8/modules/generated/sklearn.manifold.TSNE.html)：perplexity、exact路线、初始化、迭代与随机状态。
7. [scikit-learn 1.8 trustworthiness](https://scikit-learn.org/1.8/modules/generated/sklearn.manifold.trustworthiness.html)：排名惩罚公式与k的输入范围。

在线UMAP文档页显示0.5.8文档标识，本实验实际包为0.5.9.post2；固定版本和真实执行结果见requirements.txt与experiment-result.json。所有比较只适用于本次固定合成数据与预设配置，不作通用算法排名。
