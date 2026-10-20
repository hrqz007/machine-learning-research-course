# 第096讲 实验指南

本实验把《离散表示与词嵌入》的机制变成可运行CPU程序。先手算，再运行，再看完整输出；不要仅截图最终分数。预计首次安装环境需要网络，实验本身完全离线。

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

## 2 先打开语料

data/corpus.json可逐句检查；vocabulary.json确定词ID。词表只来自训练语料，不能任意调整后仍使用旧权重。语料242句，含240条重复模板和2条罕见句。

动手检查：数一数comet出现次数，并核对frequency结果。

## 3 核对上下文窗口

窗口半径2，不跨句。pairs函数输出中心词与上下文词的有向对，每个词可多次作为中心。手写一条短句与程序逐项核对。

动手检查：手写一个三词句的全部六个有向词对，与pairs测试核对。

## 4 训练完整softmax

Embedding(V,6)加无偏置Linear(6,V)，180次全批Adam，lr=0.04。输入和输出权重独立。这里没有负采样、子词或位置信息。

动手检查：打印emb.weight与output.weight的形状及参数总数，确认未绑定。

## 5 验证查表与累加

one_hot(ids,V) @ E应与E[ids]一致；[cat,cat,bus]的总和损失使cat行每维梯度2、bus行1，其余0。两项误差均应0。

动手检查：将重复索引改成cat出现3次，先写预期梯度再运行。

## 6 解释近邻与频率

读取neighbors并对照frequency。近邻只反映本语料共现与训练约束，不是世界知识。loss不为0可能来自一个中心词对应多个上下文，不能直接叫欠拟合。

动手检查：交换词表的cat和bus但不交换权重，在副本中说明为什么结果意义改变。

## 7 Notebook怎样验证

execute_notebook.py在一个新的Python进程内创建IPython执行环境，顺序运行每个代码单元，收集真实stdout、测试输出和图像。它不是手工填写Notebook输出。执行记录写入outputs/notebook-execution.json。已验证的是顺序计算单元，不包含Jupyter浏览器界面、远程内核连接或各平台安装。

笔记本先检查工作目录，再运行完整实验、运行单元测试、读取JSON、显示图像，最后要求你解释结果。已保存输出可能与当前机器运行秒数不同；精确数值容差、随机性和软件版本应分别看待。

## 8 故障定位与提交

ImportError通常表示在错误解释器中安装依赖，先比较python路径与pip目标。文件找不到时先检查当前目录。损失NaN时检查输入有限性、学习率及归一化。数值结果不同先核对软件版本、种子与线程数，再判断是否是算法差异。时间不同不自动构成复现失败。

提交一段问题、方法、完整结果、一个失败或局限、下一步检验。附上修改前后配置和所有运行，不只交最好一次。测试通过只能证明已写断言覆盖的性质，不能证明模型在现实中可靠。
