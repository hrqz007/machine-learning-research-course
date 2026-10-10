# 一手来源与核验说明

核验日期：2026-10-10 UTC。正文为独立教学撰写，未复制来源长段原文。

1. PyTorch Linear官方文档：https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html 。已打开正文，核验仿射定义、输入/输出形状与权重(out_features,in_features)存储约定。网页版本不等于本课运行环境；本课不依赖PyTorch。
2. PyTorch ReLU官方文档：https://docs.pytorch.org/docs/2.14/generated/torch.nn.ReLU.html 。已打开正文并核验max(0,x)定义。本文对XOR构造的所有权重、证明和实验为独立推导。
3. Hornik K., Stinchcombe M., White H. (1989), Multilayer feedforward networks are universal approximators, Neural Networks 2(5), 359–366. DOI:10.1016/0893-6080(89)90020-8。出版记录：https://www.sciencedirect.com/science/article/pii/0893608089900208 。已核验搜索服务返回的出版方摘要、作者、题目、年份与DOI；直接打开出版页返回403，因此未声称核验全文证明。正文仅用于有条件存在性结果的背景，不把摘要扩大成所有激活的统一定理。

曾尝试Deep Learning第6章官方网页与课件，阅读工具均因页面/文件过大未成功提取，因此不将其列为已读正文证据。无需访问上述网站即可重跑本课所有实验。源站更新可能改变URL布局，数学案例与结果以随课版本为准。
