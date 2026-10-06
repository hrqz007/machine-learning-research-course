# 第058讲 决策树的分裂机制

从一条条件判断出发，手算回归方差减少、Gini与熵，再从零建立能逐节点审计的浅树。

## 学习材料

- [教材正文](lecture.md) · [正文PDF，10页](lecture.pdf)
- [独立实验指南](lab.md) · [实验PDF，4页](lab.pdf)
- [练习完整答案](answers.md) · [答案PDF，4页](answers.pdf)
- [已执行Notebook](experiment.ipynb)
- [从零浅树](tree.py) · [实验入口](experiment.py) · [独立检查](test_experiment.py)
- [实验结果](experiment-result.json) · [284项检查记录](test-result.json) · [验证范围](verification.json)
- [合成数据说明](data/README.md) · [资料来源](sources.md) · [来源访问记录](source-checks.json)

直接先修：019、028、043、046、050。建议先完成五个阈值的纸笔候选表，再运行程序。正文解释节点加权、均值推导、概率语义、XOR反例、边界数值与因果限制；独立实验含12题和全部详解。

## 数值实验

在本讲目录运行；全部CSV随包提供，运行无须联网。

```bash
conda env create -f environment.yml
conda activate ml058
python experiment.py --out outputs/result.json
python test_experiment.py --report outputs/result.json --out outputs/test-result.json
python -O test_experiment.py --report outputs/result.json --out outputs/test-optimized.json
python plots.py --report outputs/result.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

已有Python环境也可用python -m pip install -r requirements.txt。environment.yml指定Python 3.12。作者验证的是已有隔离运行环境，并未另外创建全新Conda环境；相关边界见verification.json。

深度固定2、最小叶样本固定12。Gini和熵分类测试准确率均0.768750，训练多数类基线0.643750；回归测试MSE约0.146755379，训练均值基线约1.831889079。分类与scikit-learn预测一致，回归最大预测差约1.11e-15。所有数字仅来自本讲原创合成总体。

## 重建PDF和完整教学包

数值依赖与排版依赖分开。PDF需要Python的Markdown与WeasyPrint、系统Pango与Noto中文字体，以及Node.js/MathJax。按build-requirements.txt准备系统组件后：

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
bash build_all.sh
```

build_all.sh在outputs/rebuild下重新执行、审计、绘图、执行Notebook与排版，不覆盖发布版资料。PDF构建器针对受信任的课程源文件，不是处理任意不可信HTML或TeX的安全沙箱。

## 实现范围与复现边界

- 仅数值、有限、非空二维输入；二分类0/1或单输出实值回归
- float64穷举候选，确定性平局，等于阈值走左；叶概率平局取类别0
- 不支持缺失值、样本权重、多输出、类别特征和剪枝
- 每个节点保存训练行号、统计量与路径，适合教学审计而非大规模部署
- 使用显式异常，普通与-O模式各通过284项检查
- Notebook真实执行并嵌入6幅图；未验证Jupyter浏览器界面与外进程socket传输
- 来源均为官方文档；本讲代码、图和数据为原创教学材料

生成器使用固定种子58058。需要重建数据时执行python generate_data.py --directory outputs/new-data；生成器拒绝直接覆盖data目录。改变种子、深度或生成分布应另存为探索性实验，不覆盖本讲确认结果。
