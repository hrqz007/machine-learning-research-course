# 第十四讲 多元导数与链式法则

核心问题是“多个参数一起变化时，怎样沿计算路径传播变化率”。直接先修第9至10讲与第13讲，不需要Hessian、SVD或自动微分。

阅读顺序：`lecture.pdf`、`answers.pdf`中的练习、独立 `lab.pdf`、`experiment.ipynb`。Notebook调用同目录 `experiment.py`，请保留 `data`。正文引用12幅原创教学图。

脚本运行：`python experiment.py`。实际验证环境为Python 3.12.14、NumPy 2.3.5。Notebook另需Matplotlib与Jupyter/IPython。小CPU、离线、无随机过程、无外部数据下载、无付费接口。输出保存在相对目录 `outputs`。

主点损失19/48，列梯度(-1/6,-5/6)，预测Jacobian形状3×2。独立展开、求和、链式与有限差分在25个确定性点核对；34组边界通过。来源及详细范围见 `source-checks.json` 与 `test-result.json`。

Notebook已由新进程真实进程内内核顺序执行并保留输出。浏览器UI、外进程socket传输、本机Anaconda安装及其他操作系统未测试。数值检查不能代替一般证明；方向导数和梯度内积关系明确以可微条件为前提。
