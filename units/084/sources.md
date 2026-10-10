# 一手来源与核验范围

核验日期：2026-10-10 UTC。全部链接经实际读取；正文案例、代码与图为本课独立编写，没有复制来源实现。

1. PyTorch官方Automatic Differentiation with torch.autograd：https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html 。已读正文的计算图、链式法则、叶子梯度累积、非标量输出种子章节。本课每次清可达梯度的行为特意不同，不能把本课API当成PyTorch语义。
2. JAX官方Forward- and reverse-mode autodiff：https://docs.jax.dev/en/latest/jacobian-vector-products.html 。已读JVP、VJP数学定义及逐列/逐行构造Jacobian说明；用于术语与方向检查。不引入JAX作为依赖，未声称运行过JAX代码。
3. Baydin A.G., Pearlmutter B.A., Radul A.A., Siskind J.M., Automatic differentiation in machine learning: a survey，arXiv:1502.05767：https://arxiv.org/abs/1502.05767 。已读取论文记录与摘要核验标题、作者和自动微分综述范围；未冒称逐页审读全文。

曾尝试JAX旧连字符路径advanced-autodiff.html，返回404；已通过官方检索找到当前jacobian-vector-products.html并读取。资料中的实现细节可随框架版本变化，数学推导和本课的明确API约定以随课代码为准。实验离线，不需要读取这些网页。
