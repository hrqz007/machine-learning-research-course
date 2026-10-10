# 一手来源与核验范围

本单元模型、数据、推导例子、代码和图均为原创。参考材料仅核对术语、定义与方法背景，没有移植长段原文或图。核验日期：2026-10-10。

- [Sohn et al. (2020): FixMatch](https://papers.nips.cc/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f733-Paper.pdf)
  - 核验范围：confidence screening and consistency background; this unit does not implement FixMatch

- [Ratner et al. (2016): Data Programming](https://proceedings.neurips.cc/paper_files/paper/2016/hash/6709e8d64a5f47269ed5cea9f625f7ab-Abstract.html)
  - 核验范围：labeling functions, noisy/conflicting sources; course aggregation is a given-accuracy teaching formula

- [Snorkel: Intro to Labeling Functions](https://snorkelproject.org/use-cases/01-spam-tutorial/)
  - 核验范围：labeling function outputs, abstention, aggregation workflow

实验不需要访问这些站点；离线数据和数值结果由本目录脚本重建。网页核验不等于已复现原论文的全部实验或生产实现。
