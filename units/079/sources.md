# 一手来源与核验范围

本单元模型、数据、推导例子、代码和图均为原创。参考材料仅核对术语、定义与方法背景，没有移植长段原文或图。核验日期：2026-10-10。

- [Blei, Kucukelbir, McAuliffe (2017): Variational Inference: A Review for Statisticians](https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf)
  - 核验范围：KL(q||p), ELBO decomposition, mean-field coordinate update; original course Gaussian model not copied

- [Stan: ADVI for Variational Inference](https://mc-stan.org/docs/cmdstan-guide/variational_config.html)
  - 核验范围：meanfield vs fullrank option terminology only; no Stan execution

实验不需要访问这些站点；离线数据和数值结果由本目录脚本重建。网页核验不等于已复现原论文的全部实验或生产实现。
