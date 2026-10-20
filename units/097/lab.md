# 第097讲 实验指南

本实验把《循环网络与时间反传》的机制变成可运行CPU程序。先手算，再运行，再看完整输出；不要仅截图最终分数。预计首次安装环境需要网络，实验本身完全离线。

## 1 准备与文件地图

建议Python 3.11或更新版本，创建独立虚拟环境。requirements.txt是实验依赖；build-requirements.txt和package.json仅用于重建PDF。无GPU要求、无API密钥、无付费模型调用。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python experiment.py
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python execute_notebook.py
```

应在本讲目录执行。若pip自动选择带CUDA的大型轮子，可按PyTorch官方安装页选择CPU版本；本讲实际运行版本写在verification.json。不要把本环境的路径当作你电脑上的固定目录。

experiment.py负责生成输入、训练、评价和绘图；data/保存本次真实输入；outputs/results.json保存结构化结果；outputs/run.log保留运行文本；figures/是正文中的图；test_experiment.py包含独立断言；experiment.ipynb可以逐单元运行。

重复运行会重写本讲data、figures与outputs中同名实验结果。若要保留自己的改动实验，请先复制整个讲次到新目录。不要把学习笔记放进这些自动输出文件。

## 2 读取序列任务

train形状128×6×1，第一个值决定标签，其余小噪声。每条独立序列h0为0。确认测试长度6、12、24、48各有256条。

动手检查：验证每条序列首位正负与标签一致，不用未来噪声决定标签。

## 3 手算与梯度检查

scalar_bptt保存h0到hT，倒序累加共享w的贡献。有限差分步长1e-6，float64。检查手写、差分、自动微分误差小于1e-6。

动手检查：打印每一步delta与dw累计值，确认共享参数贡献没有被覆盖。

## 4 运行显式循环

Memory网络98个参数，180步Adam，lr=0.015。逐步看输入、循环矩阵、tanh与末端分类头；没有LSTM或GRU门控。

动手检查：数参数应为98，长度由6改为12不应改变参数数目。

## 5 故意传错状态

比较零初始状态与从前一批传来的状态，输出差异应非零。该副本仅做故障注入，不用于正式长度准确率。

动手检查：将测试batch顺序打乱后恢复顺序，零状态独立预测应保持一致。

## 6 检查截断与外推

truncate=2时前向最大差为0，但循环权重梯度不同。长度准确率实测非单调，完整记录0%与100%的翻转，不用理论增益示意图替代实际机制诊断。

动手检查：分别检查forward_max_error与gradient_difference，不能只验证其中一项。

## 7 Notebook怎样验证

execute_notebook.py在一个新的Python进程内创建IPython执行环境，顺序运行每个代码单元，收集真实stdout、测试输出和图像。它不是手工填写Notebook输出。执行记录写入outputs/notebook-execution.json。已验证的是顺序计算单元，不包含Jupyter浏览器界面、远程内核连接或各平台安装。

笔记本先检查工作目录，再运行完整实验、运行单元测试、读取JSON、显示图像，最后要求你解释结果。已保存输出可能与当前机器运行秒数不同；精确数值容差、随机性和软件版本应分别看待。

## 8 故障定位与提交

ImportError通常表示在错误解释器中安装依赖，先比较python路径与pip目标。文件找不到时先检查当前目录。损失NaN时检查输入有限性、学习率及归一化。数值结果不同先核对软件版本、种子与线程数，再判断是否是算法差异。时间不同不自动构成复现失败。

提交一段问题、方法、完整结果、一个失败或局限、下一步检验。附上修改前后配置和所有运行，不只交最好一次。测试通过只能证明已写断言覆盖的性质，不能证明模型在现实中可靠。
