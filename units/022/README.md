# 第022讲 抽样与极限定理

核心问题：重复收集数据时，平均值和估计误差为什么会变化？直接先修第019至021讲。

## 学习顺序

先阅读lecture.pdf，再按lab.pdf运行脚本与Notebook。独立完成正文16题后，阅读answers.pdf的逐步解答。先写抽样对象、重复机制和假设，再画均值分布；不要用有限图像代替定理证明。

## 文件

- lecture.pdf、lecture.md：14页正文，22节，12幅原创彩色图
- lab.pdf、lab.md：4页独立实验指南
- answers.pdf、answers.md：4页，16题完整解答
- experiment.py：精确二项、样本矩、有限总体、等相关理论与重复采样接口
- experiment.ipynb：13个实际执行代码单元，6幅保留PNG
- data/model_spec.json：完整合成模型与随机种子设定
- data/finite_population.csv：四个带身份的合成总体单位
- data/README.md：数据字典、随机流和输出解释
- requirements.txt、environment.yml：运行依赖
- source-checks.json、test-result.json：一手来源与可复核证据

## 快速运行

```text
conda env create -f environment.yml
conda activate ml-course-022
python experiment.py
jupyter lab experiment.ipynb
```

脚本默认创建outputs/report.json和sample_means.csv；后者包含64000个批均值。可用--output指定独立结果目录、--spec指定另存的JSON、--population指定另存的总体CSV。所有输入和计算在写结果之前完成；错误输入不会覆盖旧输出。Notebook的标题和手算对应固定默认设定，首格明确验证；其他配置通过脚本运行并重算理论。

核心需要Python3.12和NumPy2.3.5，Notebook绘图使用Matplotlib3.10.8。所有数据合成，小CPU离线运行，无账号、GPU或付费API。

## 关键结果

- Bernoulli(1/4)单项方差3/16，iid均值方差3/(16n)
- n=4的偏差至少0.1概率37/64；n=256约0.000241382，Chebyshev界约0.0732422
- 样本(0,0,0,1)有s²=1/4，估计SE=1/4；真实SE为√3/8
- n=30、p=0.01时零事件质量约0.739700，没有统一的n=30准则
- 8组各复制8次的64行均值方差3/128，为错误iid理论值的8倍
- 四单位(0,0,1,1)不放回抽2单位，均值方差1/12
- Pareto I(3/2)均值3、方差无限；Cauchy无通常期望
- 偏选模型稳定到1/2而目标为1/4，MSE为1/(4n)+1/16

默认4000批中，Bernoulli均值样本标准差随n=4、16、64、256分别约0.219221、0.107629、0.053990、0.026518。群组均值标准差约0.155665。Cauchy均值IQR约2。这些是有限模拟结果，不是总体定理证明或现实系统性能。

## 验证与限制

268项核心参考检查、69组边界检查（55拒绝、14接受）通过；另有1757项独立算术与失败检查，其中包括204个Fraction样本矩、348个精确协方差求和、35个有序不放回枚举、1125个70位Decimal二项递推、39个创建随机流前的拒绝哨兵和6个错误CSV旧输出保护用例。两个无关工作目录及python -O的三次新进程结果JSON、CSV和stdout一致，检查不依赖assert。所有值完成验证后，严格常数样本使用零离差分支，避免fsum/n舍入制造伪方差；小数、极小/极大正常范围常数及非法末项均有回归检查。

所有22页PDF逐页渲染并查看，12幅正文图与6幅Notebook PNG核验。Notebook从新Python进程的真实InProcessKernel顺序执行。浏览器Jupyter界面、常规socket传输、读者Anaconda安装与跨NumPy版本逐位复现未测试。

有限方差弱大数定律在正文完整证明；一般可积iid大数定律与经典CLT明确作为接受定理，未把有限方差证明或模拟冒充完整证明。数值实现保守拒绝极端范围；double正态CDF在极尾可舍入成0或1，精确二项分数另行保留。重尾模拟不删极端值，IQR线性插值规则已定义。
