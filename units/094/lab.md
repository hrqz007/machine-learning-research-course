# 第094讲 实验指南

本实验把《表示学习与对比目标》的机制变成可运行CPU程序。先手算，再运行，再看完整输出；不要仅截图最终分数。预计首次安装环境需要网络，实验本身完全离线。

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

## 2 区分无标签与标签阶段

data/images.npz中的unlabeled是128张训练图。probe_y只用于之后的线性头，test_y只用于最终评价。阅读main中对比循环，确认y不进入contrastive。

动手检查：在对比训练函数接口中查找标签参数，解释其为何不需要方向标签。

## 3 核对视图排列与正例

拼接顺序是[a0…a127,b0…b127]，正例偏移128。相似度矩阵256×256，对角线为负无穷。正例在分母里，自身不在。

动手检查：B=3时手写完整正例索引，并与程序生成的索引比较。

## 4 理解训练循环

180次完整batch更新，噪声标准差0.08，温度0.2。每次两份噪声独立采样。记录损失而不按测试集选择训练步数。投影器参与训练，探测使用编码器输出。

动手检查：检查每步loss是否有限；不要通过修改已保存数字掩盖NaN。

## 5 冻结探测

no_grad提取训练和测试特征。150步训练线性头，随机编码器与训练编码器使用相同头初始化。比较95.5%与100%时明确这是单个对比种子、固定200张测试图。

动手检查：打印特征requires_grad，确认probe损失不会连回编码器图。

## 6 构造坍塌对照

16个完全相同的向量应得到log15；这验证候选数量和对角线屏蔽。不能据此声称所有低方差表示一定坍塌。

动手检查：将视图数改成8，先预测坍塌损失为log7再运行。

## 7 Notebook怎样验证

execute_notebook.py在一个新的Python进程内创建IPython执行环境，顺序运行每个代码单元，收集真实stdout、测试输出和图像。它不是手工填写Notebook输出。执行记录写入outputs/notebook-execution.json。已验证的是顺序计算单元，不包含Jupyter浏览器界面、远程内核连接或各平台安装。

笔记本先检查工作目录，再运行完整实验、运行单元测试、读取JSON、显示图像，最后要求你解释结果。已保存输出可能与当前机器运行秒数不同；精确数值容差、随机性和软件版本应分别看待。

## 8 故障定位与提交

ImportError通常表示在错误解释器中安装依赖，先比较python路径与pip目标。文件找不到时先检查当前目录。损失NaN时检查输入有限性、学习率及归一化。数值结果不同先核对软件版本、种子与线程数，再判断是否是算法差异。时间不同不自动构成复现失败。

提交一段问题、方法、完整结果、一个失败或局限、下一步检验。附上修改前后配置和所有运行，不只交最好一次。测试通过只能证明已写断言覆盖的性质，不能证明模型在现实中可靠。
