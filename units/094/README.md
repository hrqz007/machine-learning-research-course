# 第094讲 表示学习与对比目标

接续课程大纲第094讲，完整讲解、CPU实验、逐题答案与实际输出。

- [正文 Markdown](lecture.md) · [正文 PDF（6页）](lecture.pdf)
- [实验指南](lab.md) · [实验 PDF（2页）](lab.pdf)
- [逐题答案](answers.md) · [答案 PDF（1页）](answers.pdf)
- [已执行 Notebook](experiment.ipynb) · [实验代码](experiment.py) · [测试](test_experiment.py)
- [结果 JSON](outputs/results.json) · [原始运行日志](outputs/run.log) · [数据说明](data/README.md)
- [验证记录](verification.json) · [文件哈希清单](file-manifest.json)

## 运行

在本讲目录，使用独立Python环境安装requirements.txt后运行：

```bash
python experiment.py
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python execute_notebook.py
```

机制图可由python draw_mechanism.py重建。PDF重建还需安装build-requirements.txt及npm install，再运行python build_pdf.py，输出在outputs/pdf。构建过程只处理本讲可信Markdown。

## 验证范围

本讲CPU脚本、6项普通模式和优化模式单元测试、全5个Notebook代码单元已实际运行。Notebook采用新进程内的IPython顺序执行，未测试Jupyter浏览器界面、socket内核通信或跨平台安装。数据为本地合成，结果不证明真实部署能力。计时随机器负载变化，随机性边界见正文。

## 数据与来源

无外部预训练权重、无下载数据、无付费模型调用。参考原始论文和官方接口的链接在lecture.md；本讲合成数据与图解由随附程序生成。目录映射核对远端catalog/curriculum.json，参考版本d99211f0a806045538f370ba41fce4e835bacd03。
