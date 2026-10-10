# 第085课 反向传播的完整推导

按166单元既定大纲承接083至084。重点是两层MLP完整前向/反向、张量形状、bias广播归约、batch因子与逐个参数有限差分，最后用纯NumPy从零训练。

## 学习顺序

1. lecture.md：约6300中文字符，完整推导、手算、误区、4幅原创图与边界。
2. lab.md：预测、纸笔、数值检查、负对照、训练与8道练习。
3. answers.md：逐步答案、实测数字和结论限制。
4. experiment.ipynb：构建后可读的逐单元执行版；JSON、代码与图是独立验收证据。

## 环境与离线复现

Python 3.12。先安装核心requirements.txt；只运行计算不需要PyTorch。绘图另装plot-requirements.txt；Notebook另装notebook-requirements.txt；PDF另装build-requirements.txt和package.json对应Node依赖。environment.yml提供核心环境入口。

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python test_experiment.py --out outputs/test-result.json
python -O test_experiment.py --out outputs/test-result-optimized.json
python generate_data.py --directory outputs/generated-data
python plots.py --directory outputs/figures --result outputs/result.json
```

首次安装后实验完全离线。默认输出不会覆盖冻结文件。若需要重建Notebook/PDF，在对应可选依赖准备好后运行execute_notebook.py与build_pdf.py，详见各脚本--help。图中文字优先使用系统Noto Sans CJK；缺少字体时请安装同类CJK字体后重建。

## 验证范围

审计网络2→3→2：17个参数全部中心差分，外加输入梯度差分、sum/mean关系、批量复制、稳定softmax和错误输入。主训练网络2→8→2：42个参数、1200次固定全批量更新。普通/-O各10项unittest通过；结果为真实执行，非预填模板。

四图支持PNG/SVG与脚本重建。sources.md记录一手资料核验。数据原创、离线、可重建。没有现实效果、全局最优或跨硬件逐位一致保证。

## PDF与可执行Notebook

最终教学文件：[正文PDF](lecture.pdf)、[实验指南PDF](lab.pdf)、[习题详解PDF](answers.pdf)、[已执行Notebook](experiment.ipynb)。每份PDF的Markdown源文件与图源都在本目录。

安装notebook-requirements.txt后，运行`python execute_notebook.py --out outputs/experiment.ipynb`，即可在新Python进程的IPython中顺序执行全部代码格并保存输出；这是实际代码执行，但未验证独立Jupyter套接字内核或浏览器界面。若要交互使用Notebook，另按Jupyter官方说明安装JupyterLab及ipykernel并选择本课Python环境。

PDF重建：安装build-requirements.txt，安装Node.js并在本目录运行`npm install`以安装package.json中的本地MathJax；系统还需要Pango与Noto CJK字体。随后运行`python build_pdf.py --directory outputs/pdf`。构建不请求在线字体或数学服务。环境文件是复现说明，未据此声称跨平台Conda安装已验证。
