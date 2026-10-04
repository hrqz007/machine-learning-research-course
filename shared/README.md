# 运行课程实验

前几讲的计算仅使用 Python 标准库。已有 Anaconda 且能启动 Jupyter 时，可直接使用，不必新装 GPU 或深度学习框架。之后需要的数值与学习库会在相应单元明确列出。

## 可选独立环境

从仓库根目录运行：

```bash
conda env create -f shared/environment.yml
conda activate ml-research-course
jupyter lab
```

environment.yml 是建议环境约束，不是逐平台安装成功的承诺。每讲 verification.json 单独列出实际运行版本、测试方式和未检查的部分。安装失败时先记录完整错误，不要随意修改系统安全配置。

## 运行顺序

1. 阅读该讲 lecture.pdf 与 lab.pdf。
2. 运行对应 experiment.py，核对指南的已知答案。
3. 打开 experiment.ipynb，先预测每格输出，再顺序执行。
4. 最后重启内核并运行全部，核对没有依赖残留变量。

课程制作环境使用新进程中的真实 IPython InProcessKernel 顺序执行，因该环境不支持 socket，未声称检验过浏览器界面及所有外进程内核传输；这一限制在各讲记录中保留。

公开的教学测试标签适合练习流程，不是未见盲测数据。修改过的测试反馈参与开发后，不再将同一数据称为完全独立的最终评价。
