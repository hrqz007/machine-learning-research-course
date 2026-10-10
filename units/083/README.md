# 第083课 多层感知机与表示能力

严格承接既定166单元大纲。正文为详尽中文自学教程，配纸笔实验、10题逐步答案、4幅原创PNG/SVG、离线可执行代码与冻结实测证据。

## 建议顺序

1. lecture.md：概念、逐步推导、具体反例、四图与边界。
2. lab.md：先写预测再手算、运行、审计，含10道习题。
3. answers.md：完整解题步骤与冻结实验数值。
4. experiment.py、test_experiment.py：检查代码与独立测试。

## 离线复现

核心实验仅需Python标准库，无网络、GPU、在线模型或付费API依赖。推荐Python 3.10以上；交付实跑为Python 3.12.14。进入本目录执行：

```bash
python generate_data.py --directory outputs/generated-data
python experiment.py --out outputs/result.json
python test_experiment.py --out outputs/test-result.json
python -O test_experiment.py --out outputs/test-result-optimized.json
```

普通与优化解释器各16项unittest。核心没有使用可被-O移除的裸assert做验收。experiment.run()无参数，返回JSON可序列化字典；数据从脚本所在目录读取，不依赖调用者当前目录。

图像重建先安装requirements.txt，离线已有依赖则无需安装：

```bash
python plots.py --directory outputs/figures
```

输出4幅PNG与4幅SVG，脚本使用Agg无界面后端。中文优先加载NotoSansCJK-Regular.ttc，读者机器若无该字体，应安装合法中文字体或修改字体选择。当前交付中文与下标均已视觉核验。核心运行不需要字体或绘图库。默认输出不会覆盖根目录冻结结果。

## 证据

- data/README.md：数据定义、字段、生成职责与可重复性
- experiment-result.json：真实执行的数值与Python版本、数据哈希
- test_experiment.py：读者可运行的测试源码；命令会在outputs下生成测试统计
- figures/README.md：逐图来源与阅读边界
- sources.md：已核验一手来源和核验限制
- notebook_cells.json：供统一Notebook构建的可执行教学单元

PDF与执行Notebook由本批次统一构建；源文本和脚本可单独使用。教材正文用MathJax兼容公式，不依赖网络在实验时求值。

## 本课边界

参数固定以隔离表示问题，不报告训练成功率；四点是整个二进制定义域，不虚构独立测试集。连续延拓反例说明四点一致不识别区间内行为。列向量权重采用(out,in)，批量行向量写XA+b时A=W的转置。

## PDF与可执行Notebook

最终教学文件：[正文PDF](lecture.pdf)、[实验指南PDF](lab.pdf)、[习题详解PDF](answers.pdf)、[已执行Notebook](experiment.ipynb)。每份PDF的Markdown源文件与图源都在本目录。

安装notebook-requirements.txt后，运行`python execute_notebook.py --out outputs/experiment.ipynb`，即可在新Python进程的IPython中顺序执行全部代码格并保存输出；这是实际代码执行，但未验证独立Jupyter套接字内核或浏览器界面。若要交互使用Notebook，另按Jupyter官方说明安装JupyterLab及ipykernel并选择本课Python环境。

PDF重建：安装build-requirements.txt，安装Node.js并在本目录运行`npm install`以安装package.json中的本地MathJax；系统还需要Pango与Noto CJK字体。随后运行`python build_pdf.py --directory outputs/pdf`。构建不请求在线字体或数学服务。环境文件是复现说明，未据此声称跨平台Conda安装已验证。
