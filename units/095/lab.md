# 第095讲 实验指南

本实验把《深度视觉阶段项目》的机制变成可运行CPU程序。先手算，再运行，再看完整输出；不要仅截图最终分数。预计首次安装环境需要网络，实验本身完全离线。

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

## 2 先写分析计划

在查看results之前写模型、训练步数、配对种子、准确率差及误差分组规则。不要事后更换主要指标。当前计划固定128训练、300测试、5种子、3模型、100步。

动手检查：把主要指标和固定预算写在运行记录最上方，注明这是运行前计划还是事后分析。

## 3 核验手写小网络

gradcheck同时计算显式链式法则、中心有限差分与PyTorch梯度，最大误差应远低于1e-6。失败先查mean、矩阵转置、tanh导数和float64。

动手检查：把手写梯度中的1/n临时移除，在副本中观察有限差分测试是否失败。

## 4 复跑15次比较

linear、mlp、cnn均全批Adam，lr=0.02。参数量不同、计算量不同，不能称完全同算力。train_seconds只用于当前CPU的成本描述。

动手检查：统计records应有15行，每种模型各5个不同seed。

## 5 检查配对区间

用同一seed配对CNN与MLP，重抽5个差值4000次。区间仅反映训练种子，条件于固定测试集。不要把300张图×5次当1500个完全独立样本。

动手检查：手算五个差值均值，对照mean_difference，避免把准确率比例与百分点混淆。

## 6 检查失败案例

outputs/predictions.npz保存每次预测，results中的error_indices可回到图像检查。按预定顺序挑图；零错误只说明当前测试集，不证明总体错误率为0。

动手检查：取一条error_indices，回看真实标签与保存预测，确认索引对齐。

## 7 Notebook怎样验证

execute_notebook.py在一个新的Python进程内创建IPython执行环境，顺序运行每个代码单元，收集真实stdout、测试输出和图像。它不是手工填写Notebook输出。执行记录写入outputs/notebook-execution.json。已验证的是顺序计算单元，不包含Jupyter浏览器界面、远程内核连接或各平台安装。

笔记本先检查工作目录，再运行完整实验、运行单元测试、读取JSON、显示图像，最后要求你解释结果。已保存输出可能与当前机器运行秒数不同；精确数值容差、随机性和软件版本应分别看待。

## 8 故障定位与提交

ImportError通常表示在错误解释器中安装依赖，先比较python路径与pip目标。文件找不到时先检查当前目录。损失NaN时检查输入有限性、学习率及归一化。数值结果不同先核对软件版本、种子与线程数，再判断是否是算法差异。时间不同不自动构成复现失败。

提交一段问题、方法、完整结果、一个失败或局限、下一步检验。附上修改前后配置和所有运行，不只交最好一次。测试通过只能证明已写断言覆盖的性质，不能证明模型在现实中可靠。
