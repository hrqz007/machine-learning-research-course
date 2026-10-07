# 第059讲 树的复杂度与剪枝

从“把训练样本切成记忆碎片”的问题出发，手推最弱环节剪枝，比较预剪枝与后剪枝，并在25次固定删行扰动中观察预测稳定性。

## 学习材料

- [教材正文](lecture.md) · [正文PDF，9页](lecture.pdf)
- [独立实验指南](lab.md) · [实验PDF，4页](lab.pdf)
- [12题完整详解](answers.md) · [答案PDF，4页](answers.pdf)
- [已执行Notebook，7个代码单元](experiment.ipynb)
- [从数组实现最弱环节剪枝](pruning.py) · [实验入口](experiment.py)
- [独立检查](test_experiment.py) · [2751项检查记录](test-result.json)
- [实验结果](experiment-result.json) · [验证范围](verification.json)
- [数据说明](data/README.md) · [资料来源](sources.md) · [来源访问记录](source-checks.json)

直接先修：025、048、051、058。先手算四叶树的0.09、0.18两个临界α，再检查真实训练树37个路径状态。正文解释根归一化风险、最弱环节重算、训练/验证/测试隔离、小叶子的概率限制与结构不稳定性。

## 数值复现

在本讲目录运行，所有CSV随包提供，实验无需联网。

```bash
conda env create -f environment.yml
conda activate ml059
python experiment.py --out outputs/result.json
python test_experiment.py --report outputs/result.json --out outputs/test-result.json
python -O test_experiment.py --report outputs/result.json --out outputs/test-optimized.json
python plots.py --report outputs/result.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

已有Python3.12环境也可执行python -m pip install -r requirements.txt。作者验证的是隔离Python3.12.14环境，没有另建全新Conda环境。

预剪枝选择深度2、最小叶样本1，得到4叶；后剪枝选择α约0.010590696，得到3叶。两者测试准确率均0.746667，Brier分别0.163508、0.164178；95叶完整树测试Brier为0.32。全部为本讲原创合成样本上的结果，不代表现实任务性能。

## 完整重建

数值与排版依赖分开。PDF需要Markdown、WeasyPrint、Pango、中文Noto字体及Node.js/MathJax，详见build-requirements.txt。

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
bash build_all.sh
```

默认重建写入outputs，不覆盖发布文件。PDF构建器只处理受信任的课程源，不是任意HTML/TeX安全沙箱。

## 实现与验收边界

- 树生长使用scikit-learn；最弱环节路径与导出树预测由本讲实现
- 二分类0/1，有限数值输入；导出预测明确匹配库的float32输入语义
- 路径只用训练统计；预剪枝参数与后剪枝状态只用验证Brier选择；不合并重训
- 对路径内部α与库比较，所有测试概率最大差0；精确平局另用可手算树验证
- 普通模式与-O模式各通过2751项显式检查，Notebook真实执行并嵌入6图
- 25次删行共享原训练样本；标准差是扰动敏感性，不是总体置信区间
- Notebook未验证Jupyter浏览器界面和外进程socket传输
