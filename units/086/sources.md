# 一手资料、版本与核验范围

核验日期：2026-10-10（UTC）。使用官方Python与PyTorch文档实际页面正文；不是仅检查链接可打开。代码实测版本为Python 3.12.14、PyTorch 2.14.1+cpu。Python 3.12文档当前展示小版本3.12.15，所引用的是类、继承、迭代器与生成器等语言语义，不声称运行了该小版本。

1. Python 3.12 Classes： https://docs.python.org/3.12/tutorial/classes.html
   核验类/实例、方法、继承、迭代器与生成器。
2. PyTorch 2.14 Module： https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html
   核验基类初始化、子模块登记、调用入口、train/eval、设备迁移与状态接口。
3. PyTorch 2.14 Linear： https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html
   核验权重存储为(out_features,in_features)，前向使用转置。
4. PyTorch 2.14 CrossEntropyLoss： https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html
   核验输入为未归一化logits、类别索引范围、目标形状/类型与归约。教材只使用无权重普通硬标签，未把其简化公式推广到所有选项。
5. PyTorch自动微分入门： https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
   核验requires_grad、backward及禁用梯度记录的基本语义。
6. PyTorch 2.14 Optimizer.zero_grad： https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html
   核验set_to_none=True与零张量的行为区别，尤其无梯度参数的处理。
7. PyTorch Dataset/DataLoader入门： https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
   核验Dataset接口与批量迭代。教材不下载教程中的FashionMNIST，使用本地原创小数据。
8. PyTorch优化循环入门： https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
   核验训练、清梯度、反向、更新以及评价基本顺序。教程的展示性批均值统计未直接照搬；本课为不等大尾批实现按实际样本数加权，并独立测试。

所有实验数字来自本地真实CPU执行，不来自文档示例。图形、数据与教学例原创，可由随课脚本重建。文档说明设备支持不等于我们已在GPU执行；本课仅验收CPU float64。
