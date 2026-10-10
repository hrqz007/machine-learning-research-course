# 第087课 激活初始化与梯度传播

核心问题：网络为什么会一开始就梯度消失或爆炸？按既定大纲，连接激活导数、二阶矩传播、Xavier/He、饱和、残差路径与梯度范数。直接先修：015、021、034、083、085至086。

## 阅读顺序

1. lecture.md：逐变量推导、23节中文讲义与4幅原创图
2. lab.md：离线实验步骤、控制变量、10道练习与验收清单
3. answers.md：逐步答案、冻结实测结果及结论边界
4. experiment.py与test_experiment.py：NumPy实现和真正unittest

## 运行

Python 3.11以上。首次安装依赖后，所有实验均离线，不需GPU、在线模型或scikit-learn。数据已随包提供。建议固定BLAS单线程，便于低资源复现。

```bash
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/outputs/mpl-cache"
export XDG_CACHE_HOME="$PWD/outputs/cache"
python experiment.py --out outputs/result.json
python test_experiment.py --out outputs/test-result.json
python -O test_experiment.py --out outputs/test-result-optimized.json
python plots.py --result outputs/result.json --directory outputs/figures
```

Python接口：experiment.run()无参返回可JSON序列化的完整结果。notebook_cells.json提供供课程统一执行器使用的教学单元定义；Notebook/PDF的构建状态以最终交付包的对应执行记录为准，本目录文本不提前声明构建验收通过。

## 冻结实测与关键边界

- 随机传播：固定宽度64、批量128、深度2/8/24、权重种子8701–8703；ReLU四种初始化及tanh/sigmoid对照
- 真实训练：30条Wine、13特征、六层48宽ReLU；1200步全批量SGD，学习率0.05；输出层始终Xavier
- 记忆成功条件：100%训练准确率且平均交叉熵小于0.02
- He与Xavier三个种子均通过；small虽然硬准确率可达93.3%，交叉熵仍约ln3，未通过
- 无测试集、无泛化结论；随机上游探针与训练损失梯度分开记录；残差例子不冒称完整ResNet

## 文件与证据

- experiment-result.json：完整72组传播、18组残差和9组真实训练结果
- test_experiment.py：14项可运行测试，含全参数有限差分；运行后在outputs中生成测试统计
- data/wine_tiny.json、data/README.md、data/manifest.json：真实数据、来源、许可、抽样规则与哈希
- figures：4幅原创PNG和4幅SVG；plots.py可重建
- sources.md：原论文及UCI官方页的核验范围

默认重建命令输出到outputs，不覆盖冻结文件。

## PDF与可执行Notebook

最终教学文件：[正文PDF](lecture.pdf)、[实验指南PDF](lab.pdf)、[习题详解PDF](answers.pdf)、[已执行Notebook](experiment.ipynb)。每份PDF的Markdown源文件与图源都在本目录。

安装notebook-requirements.txt后，运行`python execute_notebook.py --out outputs/experiment.ipynb`，即可在新Python进程的IPython中顺序执行全部代码格并保存输出；这是实际代码执行，但未验证独立Jupyter套接字内核或浏览器界面。若要交互使用Notebook，另按Jupyter官方说明安装JupyterLab及ipykernel并选择本课Python环境。

PDF重建：安装build-requirements.txt，安装Node.js并在本目录运行`npm install`以安装package.json中的本地MathJax；系统还需要Pango与Noto CJK字体。随后运行`python build_pdf.py --directory outputs/pdf`。构建不请求在线字体或数学服务。环境文件是复现说明，未据此声称跨平台Conda安装已验证。
