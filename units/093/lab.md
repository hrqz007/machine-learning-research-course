# 第093讲 实验指南

本实验把《视觉任务与迁移学习》的机制变成可运行CPU程序。先手算，再运行，再看完整输出；不要仅截图最终分数。预计首次安装环境需要网络，实验本身完全离线。

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

## 2 核对源模型与目标数据

打开data/images.npz，核对source有256张、train有40张、test有200张，每张1×10×10。源图噪声0.12，目标0.32。分别核对标签平衡；不要把测试集传给train函数。

动手检查：打印三个拆分的shape与各标签计数，确认没有把source误当目标测试。

## 3 阅读训练方案

找到三个mode分支，画出哪组参数requires_grad。目标head按seed重新初始化，所有目标方案100次更新。源预训练120步额外计费于成本。微调lr=0.005，其余0.02，所以报告的是方案比较。

动手检查：列出三组优化器持有的参数名称，冻结组只能包含head。

## 4 执行冻结不变性检查

读取records，所有frozen行的encoder_max_change必须为0。scratch和finetune通常不为0，但不能把变化大当成效果好。检查每个seed三组结果，保留迁移没有获益的情况。

动手检查：取seed=0的一行结果，与训练前后参数最大差逐项核对。

## 5 审计重复

audit使用精确SHA256与最近像素RMSE。注入一张完全重复和一张加0.001近重复后，exact应增加1，near至少增加2。原始评价集没有被修改。源预训练与目标测试也需单独审计。

动手检查：把近重复阈值改为0.0001，预测注入加噪副本还会不会被标出，再验证。

## 6 手工评价框与掩码

使用iou函数计算1/3。正文两预测示例TP=1、FP=1、FN=1。4×4分割示例前景IoU=0.6，Dice=0.75，accuracy=0.875。注意这些是手算单元例，不是检测模型训练结果。

动手检查：改动预测框右边界，手算新的交集与并集后调用iou验证。

## 7 Notebook怎样验证

execute_notebook.py在一个新的Python进程内创建IPython执行环境，顺序运行每个代码单元，收集真实stdout、测试输出和图像。它不是手工填写Notebook输出。执行记录写入outputs/notebook-execution.json。已验证的是顺序计算单元，不包含Jupyter浏览器界面、远程内核连接或各平台安装。

笔记本先检查工作目录，再运行完整实验、运行单元测试、读取JSON、显示图像，最后要求你解释结果。已保存输出可能与当前机器运行秒数不同；精确数值容差、随机性和软件版本应分别看待。

## 8 故障定位与提交

ImportError通常表示在错误解释器中安装依赖，先比较python路径与pip目标。文件找不到时先检查当前目录。损失NaN时检查输入有限性、学习率及归一化。数值结果不同先核对软件版本、种子与线程数，再判断是否是算法差异。时间不同不自动构成复现失败。

提交一段问题、方法、完整结果、一个失败或局限、下一步检验。附上修改前后配置和所有运行，不只交最好一次。测试通过只能证明已写断言覆盖的性质，不能证明模型在现实中可靠。
