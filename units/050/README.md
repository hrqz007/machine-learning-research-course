# 第050讲 数据划分与评价对象

核心问题：随机切分是否真的模拟了未来要面对的数据？

本讲用同一份60设备48天的合成面板，比较随机记录、整组、按时间和组别加时间的不同评价对象。显式处理两天标签延迟、已知与未知设备、时间漂移、原始窗口重叠和评价权重，不把分数最低的划分宣布为最正确。

## 阅读与实验

- [正文](lecture.pdf) 与 [可编辑源](lecture.md)
- [独立实验指南](lab.pdf) 与 [源](lab.md)
- [完整答案](answers.pdf) 与 [源](answers.md)
- [已真实执行Notebook](experiment.ipynb)
- [实验脚本](experiment.py)、[独立审计](audit.py)、[十图重建](plots.py)
- [固定协议](data/protocol.json)、[数据字典](data/README.md)、[划分归属](data/split_membership.csv)
- [完整结果](experiment-result.json)、[作者自查](test-result.json)、[执行与视觉说明](verification.json)
- [主要原始来源](sources.md) 与 [核对记录](source-checks.json)

直接先修：002、007、022、025。正文自足解释四行线性模型的一步同步梯度更新，再以同一目标的增广最小二乘完成主实验。进一步矩阵求解可回看029与040；051接交叉验证与模型选择，052接训练内处理流水线。

## 最短运行

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py --report outputs/result.json --out outputs/audit.json
python plots.py --report outputs/result.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

Python3.12推荐。Conda环境定义见environment.yml。数值实验离线使用随包数据，不需要下载真实个人数据。支持从不同当前目录调用脚本；源Notebook的首格要求在本单元目录运行，以便明确校验固定输入。

## 重建

准备build-requirements.txt、Node.js、MathJax3.2.2、Pango与中文字体后运行 `bash build_all.sh`。它重建数值、普通与-O审计、十图、Notebook和三份PDF，验证数据和图像复现。结果在outputs/rebuild，交付数据与PDF不被覆盖；build_notebook.py会重建源Notebook并清空原输出，先保留原包。

所有临时路径可用TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME、IPYTHONDIR配置。共享解释器无需修改，完整过程不依赖本机绝对路径或相邻课程单元。公式在本地渲染，无在线公式服务或XeLaTeX依赖。

## 结果边界

第0份Personalized随机/整组/同设备未来/新设备未来MSE约0.636795、1.847585、1.496320、2.938758，四者评价目标不同。80个新面板960次拟合全部保留，包含Personalized在整组34次、联合38次差于Pooled，以及Trend在联合3次差于Personalized。

作者用Fraction、SciPy带主元QR及真实sklearn独立编码与Ridge核验。Notebook在新的真实进程内IPython内核运行，未测试浏览器Jupyter界面或外进程网络传输。Bootstrap区间未验证覆盖率。合成机制用于理解评价，不构成任何现实设备部署保证。
