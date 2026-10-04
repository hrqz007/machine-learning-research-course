# 第十二讲 特征值与奇异值分解

本讲回答“怎样找出矩阵最重要的方向，并理解低秩近似”。直接先修第十一讲；不要求微积分。材料以实对称手算矩阵及增加一个差异测量通道的长方形矩阵贯穿。

建议先读 `lecture.pdf`（对应源稿 `lecture.md`），完成 `answers.pdf` 中的题目，再按独立的 `lab.pdf` 运行 `experiment.ipynb`。Notebook 逐步调用同目录 `experiment.py`，需要保留 `data` 文件夹。12幅原创教学图在 `figures` 中，均由正文引用。

脚本运行方式为 `python experiment.py`，仅需 NumPy；Notebook 额外使用 Matplotlib、Jupyter/IPython。实际核验为 Python 3.12.14、NumPy 2.3.5，小CPU、离线、无随机过程、无付费服务。结果写入相对目录 `outputs`。

数学检查覆盖特征方程、正交谱分解、已知 SVD、重构、完整与薄形状、符号和重复值不唯一、正定类别、低秩误差、加权反例及条件数。测试包含27组输入与形状边界，全部断言及重复运行记录见 `test-result.json`。参考教材和官方接口的核验日期与范围见 `source-checks.json`。

Notebook 已在新 Python 进程的真实进程内内核中顺序执行并保留输出；浏览器UI、外进程socket传输及学习者本机Anaconda安装没有测试。有限案例检查不能替代讲义中的一般证明。实对称谱定理作为明确前提给出，未伪装成已完整建立的基础定理。
