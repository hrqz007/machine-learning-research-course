# 046 分类指标与排序评价

本单元属于166单元机器学习教程，直接先修7、18、45。核心问题是“同一预测为什么在不同指标下得出相反结论”。固定12行、3阳性9阴性的合成零件复核案例，贯穿每个指标；完整保留0.80同分组、全部阈值、空警报端点和平均方式。

## 阅读顺序

1. `lecture.pdf` / `lecture.md`：逐步定义、完整推导、9幅实际图与10道练习
2. `lab.pdf` / `lab.md`：纸笔到Python、真实库核验、先验敏感性、Notebook与重建
3. `answers.pdf` / `answers.md`：全部10道练习、9道读图题、10项实验的完整答案
4. `experiment.ipynb`：带真实执行输出的逐步Notebook；请在新内核从头重跑

## 安装与运行

```bash
conda env create -f environment.yml
conda activate ml046-metrics
python experiment.py
python audit.py
python -O audit.py --out ./outputs/optimized
python plots.py
python execute_notebook.py --out ./outputs/fresh-kernel.ipynb
```

也可用Python3.12虚拟环境安装 `requirements.txt`；浏览器Notebook另需JupyterLab。`execute_notebook.py`在新Python进程启动真实的进程内IPython内核，依次执行代码单元并收集输出。未测试浏览器JupyterLab与网络socket内核传输。

默认输出位于 `outputs/`，数据路径根据脚本自身目录定位，从其他cwd调用也可复现。`--out`可指定独立输出目录；不要把路径指向课程源文件目录。Notebook交互启动时应在046目录；需要时设置 `ML046_ROOT` 为本机解压路径。

## 结果锚点

- 阈值0.80，规则 `score >= threshold`：TP=2、FP=2、FN=1、TN=7
- 阳性precision=1/2、recall=2/3、F1=4/7、accuracy=3/4
- ROC AUC=43/54；非插值AP=9/14；梯形PR面积=79/126
- macro F1=83/119，micro F1=3/4，support-weighted F1=181/238
- 目标阳性率0.10/0.25/0.50且类条件评分分布不变：AP依次29/60、9/14、127/156，AUC均43/54
- 评分立方保持AUC/AP；部署阈值需同时由0.80变为0.512才能保持预测

实际数值结果在 `experiment-result.json`，完整阈值CSV在 `threshold-sweep.csv`。这些是可复现的课程输出，不是私有运行日志。

## 文件与重建

- `data/review_scores.csv`、`data_integrity.json`：固定合成记录与数据哈希
- `experiment.py`：输入检查、阈值计数、曲线、均值、先验变化和保序反例
- `audit.py`：独立Fraction参照、27配对和1134个小规模穷举案例
- `plots.py`、`figures/`：全部图像的真实数值来源与PNG
- `build_notebook.py`：重建无输出Notebook源；新内核执行由 `execute_notebook.py` 完成
- `build_pdf.py`、`mathjax_render.cjs`、`pdf.css`：全部PDF构建源码
- `requirements.txt`、`environment.yml`、`build-requirements.txt`、`package.json`：数值、交互与可选排版环境
- `sources.md`、`input_contract.json`：核对来源、参数、输入数值域与限制
- `build_all.sh`：运行主实验、正常和优化模式自检、图像、Notebook以及可选PDF构建
- `SHA256SUMS.txt`：交付文件的逐文件哈希；可用 `sha256sum -c SHA256SUMS.txt` 检查

PDF重建另需Node.js、MathJax、WeasyPrint、Markdown及系统Pango/中文字体。已交付PDF可直接阅读，数值实验不需要这些排版依赖。

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory ./outputs/pdf
```

完整重新计算可运行 `bash build_all.sh`；包含PDF则运行 `BUILD_PDF=1 bash build_all.sh`。将 `PYTHON` 设置为目标环境的Python路径即可从其他cwd调用脚本。构建默认使用随包的已核对正文图；可通过 `--figure-directory` 指向刚重建的图像目录。

## 支持范围与限制

所有指标均基于冻结评分，不含训练、随机种子或测试集阈值选择。数据不是总体随机样本，不能宣称生产性能。类不平衡在此为3比9，只作计数机制演示。

教学API支持两类均存在、2至100000行的0/1标签、[0,1]有限评分、处于[1e-6,1e6]的可选权重；阈值为[0,1]或正负无穷，先验比例参数为[0.01,0.99]。一般sklearn可支持更广输入，本包装函数主动缩窄以便明确教学契约。

空警报精确率按0报告并标未定义，PR绘图另有(Recall,Precision)=(0,1)约定端点。API显式保留全部阈值，不通过标签打破并列；主例没有为了更漂亮结果改数据或协议。正常与`-O`使用相同非assert测试逻辑。

作图、公式和PDF构建只处理可信课程源。源码不包含机器专属绝对依赖路径、凭据或用户个人数据。课程自检仅证明计算和展示的一致性，独立发布验收另行进行。
