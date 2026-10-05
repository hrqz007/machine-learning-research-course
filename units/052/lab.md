# 第052讲实验指南 无泄漏的数据处理流水线

## 1 实验目标和交付物

本实验让你对一条流水线逐步回答“这个状态用了哪些行”。你将完成四行从原值到梯度更新的手算、三种分开机制的正确与错误对照、真实TargetEncoder内部cross-fit核验，以及独立代码与新内核执行。目标不是做最高分，而是让统计参数、标签来源和评价对象可审计。

建议纸笔45至60分钟，运行与读图60至90分钟，研究报告30分钟。阅读三份PDF不需要安装。讲义正文解释原理，本文给独立操作路线，answers.pdf完整覆盖E1至E16、G1至G10和下文L1至L8。

交作业至少包含：数据角色卡、四行完整链、一个实际折的处理状态、八行目标编码来源表、实际命令结果、负面结果和适用边界。只交一个交叉验证分数或一张Pipeline截图不通过。

## 2 文件地图和固定协议

|文件|用途|
|---|---|
|lecture.pdf 与 lecture.md|完整原理、十图、16题|
|lab.pdf 与 lab.md|独立操作路线|
|answers.pdf 与 answers.md|题目、读图与实验详解|
|experiment.py 与 experiment.ipynb|计算核心与真实执行Notebook|
|audit.py 与 test-result.json|独立科学参照和失败案例|
|plots.py 与 figures|全部十图及重建|
|data/protocol.json|首次分数前固定的机制与种子|
|data/draws.npz|12份三机制全部原始抽样与处理输入|
|data/splits.npz 与 split_membership.csv|实际外层行索引和可读归属|
|data/hand.csv|四行训练与两行验证小例|
|experiment-result.json|全部折结果、状态、编码、预测|
|build_all.sh 与 verify_rebuild.py|整包独立重建与结果核验|

三机制不能混成一个因果实验。A是48设备×6行的新设备任务，设备内依赖，外层整组。B是200行×300噪声特征，外层随机行。C是360个独立类别抽样及独立y，外层随机行；同类别不代表共享随机效应。A不声称时间外推，C不声称适用于依赖或时序标签。

固定数据seed52017、外层随机seed52031、内部编码seed52043；每机制12份独立数据、四外折。A只改变标准化fit范围，B只改变监督选列范围，C分cross-fit、自编码和全标签三路线。不要改种子寻找漂亮结果；新探索另立目录与协议，保留本包原结果。

## 3 环境准备和最短运行

进入解压后的052目录，用Python3.12隔离环境：

```bash
python -m venv .venv
# Linux 或 macOS
source .venv/bin/activate
# Windows PowerShell 用 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py --report outputs/result.json --out outputs/audit.json
```

也可以 `conda env create -f environment.yml`，再激活其中写明的环境。作者实际验证Python3.12、NumPy2.3.5、SciPy1.17.0、scikit-learn1.8.0、Matplotlib3.10.8、nbformat5.11.1、ipykernel7.4.0。环境文件不等于已经验证所有平台的Conda求解。

数值实验离线读取随包数据，不需付费API、真实个人记录或GPU。PDF重建另需build-requirements.txt、Node.js、MathJax3.2.2、Pango与中文字体。先跑数值部分即可学习，不必为了阅读PDF安装排版依赖。

所有脚本默认按自身所在目录找输入；输出相对路径按调用者当前目录解释。缓存目录可用TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME、IPYTHONDIR指向可写空间，先创建再运行。build_all.sh会设置单元自己的这些目录，不修改共享解释器。

## 4 L1 先写角色与参数来源卡

打开protocol.json和split_membership.csv，不先读MSE。为三机制各写一句评价目标，再列训练、验证、每折样本数、是否有组效应、每个fit读什么。

A：目标同一制度下未见设备。36台216行训练、12台72行验证。设备ID是分组元数据，两个数值与站点类别是输入。真实latent和effects仅在模拟器中用于说明，不得当作训练特征。

B：150行训练、50行验证。所有特征与y无关；正确选列只读150行X,y。C：270训练、90验证；内部三折每个映射180个标签，验证映射读完整270个训练标签。缺失机制只属于A，不要把它写进B或C的原因说明。

```python
import experiment as e
c, d, splits = e.load_data()
for name in ['group', 'selection', 'encoding']:
    for fold in range(4):
        tr = splits[f'{name}_{fold}_train']
        va = splits[f'{name}_{fold}_valid']
        print(name, fold, len(tr), len(va))
```

L1验收：解释为何A设备不能交叉、站点类别可以交叉；说明C类别重复不等于A设备重复；指出时间任务需要另设标签可用性，而非默认套当前KFold。

## 5 L2 四行原值到下一前向

只拿hand.csv H1至H4训练。原值1、缺失、3、5；非缺中位数3；插补后1、3、3、5；训练最小最大界1、5；均值3；总体型方差2。H5的7必须截到5，未知C不扩列，H6缺失只能用训练3。

设计顺序为截距、z、A、B、缺失。初始参数全0，局部导数四行分别为(−1,√2,−1,0,0)、(−2,0,−2,0,−2)、(−2,0,0,−2,0)、(−5,−5√2,0,−5,0)。先相加再除4，不再重复平均。

η=0.2同步更新成(0.5,√2/5,0.15,0.35,0.1)。下一训练预测0.25、0.75、0.85、1.25，平均半平方损失2.18875；仅验证预测0.9、0.95。用代码检查：

```python
import audit
h = e.hand()
print(h['statistics'])
for row in h['rows']:
    print(row)
print(h['gradient_mean'], h['theta_next'])
print(h['next_half_loss'], h['valid_next_prediction'])
print(audit.exact_hand())
```

L2验收：提交原值、插补、截断、缩放、独热、指示、初始预测、逐行损失导数、聚合、新参数和下一预测。这个无正则单步GD不是主实验Ridge求解器；只写主实验调用Ridge不能替代手算。

## 6 L3 复算一个实际折的状态

以下从原始数据拟合第0份第0折，不使用保存的模型：

```python
X = e.group_X(d, 0)
y = d['group_y'][0]
tr = splits['group_0_train']
va = splits['group_0_valid']
pipe = e.make_pipeline().fit(X[tr], y[tr])
print(e.state(pipe['pre'], tr))
print(pipe['pre'].get_feature_names_out())
print(pipe['pre'].transform(X[va[:3]]))
```

手工核对第一数值列：只从tr找非缺值排序求中位数；将tr缺值填好；排序后在位置(216−1)×0.05和×0.95线性插值；截断全部训练值，再求均值和分母216的方差。这与“先对所有非缺值缩放，再填缺失”不是同一顺序。

对照audit.design_reference，它用排序插值和求和重建矩阵，不调用被测的imputer、clipper、scaler或one-hot。再由SciPy QR核验预测，避免只检查“函数没报错”。

L3验收：提供本折中位数、上下界、均值、方差、词表与216个fit行号的来源。把验证X改大以后，训练状态应不变；验证预测可以变，不能误要求全部输出都不变。

## 7 L4 三机制输出和负结果

```bash
python experiment.py --out outputs/normal.json
python -O experiment.py --out outputs/optimized.json
python audit.py --report outputs/normal.json --out outputs/audit.json
python -O audit.py --report outputs/optimized.json --out outputs/audit-O.json
```

Python -O会移除普通assert。本包保护和验收用显式异常，不依赖assert。正常与-O结果应相同。再从另一当前目录用脚本绝对路径运行，确认默认输入不依赖cwd。

```python
import json
from pathlib import Path
r = json.loads(Path('outputs/normal.json').read_text())
for mechanism, block in r['summary'].items():
    print(mechanism)
    for key, value in block.items():
        print(key, value)
```

A正确均值2.037497，全局缩放2.037060；12份里错误版本4次更差、8次稍好。B正确1.205764、全局监督选择0.874681、均值基线0.983907。C正确cross-fit1.018423、自编码1.263601、全标签0.733944、均值基线1.008078。

A差很小也不能证明没有越界；C正确路线不必胜过均值基线。跨机制样本、目标与信息不同，不能用三个均值差作一种“泄漏程度排名”。先在每份数据内平均等大四折，再对12份平均；四折相互依赖，12份重新生成的数据才是这里的独立重复单位。

L4验收：完整保留全部12×4折，不过滤高MSE，分别解释协议越界、自编码过拟合和实际分数变化。预测器拟合总数为12×4×(2+2+3)=336，另有基线与独立核查，不能把336当成独立样本量。

## 8 L5 八行编码与真实API

先纸笔计算讲义八行表。前四行编码只读后四标签，均值6.5；后四只读前四，均值2.5；验证A、C、Z用全部训练均值4.5。运行：

```python
print(json.dumps(e.encoder_hand(), ensure_ascii=False, indent=2))
from sklearn.preprocessing import TargetEncoder
import numpy as np
Xh = np.array(['A','B','A','C','A','B','D','C'])[:, None]
yh = np.arange(1., 9.)
enc = TargetEncoder(target_type='continuous', smooth=2,
                    cv=2, shuffle=False)
z_oof = enc.fit_transform(Xh, yh)
z_full = enc.transform(Xh)
print(np.column_stack([z_oof, z_full]))
print(enc.transform(np.array(['A','C','Z'])[:, None]))
```

fit_transform与fit后transform不同，是实际API的训练语义，不是误差。未来transform用完整训练映射；不要随机挑一份内部映射，也不要在验证上调用fit_transform。

打开r['replications'][0]['encoding'][0]，查看inner_provenance、crossfit_train_encoding、full_train_encoding和valid_encoding_from_full_train。选一个训练行，追溯它所属内部留出折，再从该折fit行重新数类别次数、标签和、总体均值；再选一个验证行，从全部270训练行重新算。

L5验收：训练行的自身y不参与它的折外编码，验证y不参与任何映射；总体均值也要来自相应内部训练子集。若只把类别和设为折外，而全局均值读了内层留出标签，仍不是本实现。

## 9 L6 干预与独立科学参照

audit.py核验全部三机制的12份×4折，非仅核对一个漂亮例。A用独立手工插补、分位点、标准化、词表及SciPy QR；B用手工Pearson绝对相关排序与QR；C用计数求和、手工重建随机KFold、QR，并额外调用真实TargetEncoder Pipeline。

关键干预如下：验证y加50，正确预测不变而全标签编码改变；某训练行自己的y加10，其自身折外编码不变；验证X加100，本折拟合状态不变。正负对照同时存在，才知道测试确实能发现错误路径。

独立参照当前误差量级：A最大预测差约5.33e−15；B最大MSE差约6.66e−16；C最大编码差约2.22e−16、预测差约5.27e−16；手算中心差分约1.05e−10。不同平台末位可能不同，采用代码中合理容差。

```python
checks = audit.run(r)
print(checks['status'])
print(checks['exact_hand'])
print(checks['global_encoding_mutation_max_change'])
print(checks['failures_preserved'])
```

L6验收：说明每条参照避开了哪个被测函数，以及每个不变性固定了什么。所有路线一致也不证明评价总体正确；若大家共同使用错误时间信息，数值仍可一致。

## 10 L7 画图和新核执行

```bash
python plots.py --report outputs/normal.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

十图依次为外层边界、数值处理、纸笔处理、纸笔梯度、缩放配对差、OLS不变对照、监督选列、八行编码、编码成绩、内部来源。读图先看轴、随机对象、分母与信息源，不只看曲线谁更低。

交付Notebook保留执行输出。首格只用标准库验证源码与数据哈希，后面才导入数值库。它顺序运行手算、机制、审计与十幅图。execute_notebook.py在新的Python进程启动真实ipykernel InProcessKernel，保存每个代码格结果，失败不静默跳过；不声称测试了浏览器Jupyter界面或外进程网络传输。

若在JupyterLab打开，进入052目录、重启内核、运行全部。首格文件校验不通过时检查是否打开错目录或改过源，不应删除校验继续使用旧输出。需要重建空白Notebook时运行build_notebook.py，默认写outputs，不覆盖交付已执行版。

L7验收：全部代码格有execution_count且无error输出，十图由实际重建显示；图5四个正差不被裁掉，图8两种训练编码不被错误合并，图10不把外层验证标签画进允许集合。

## 11 输入范围和失败输出保全

数值管道支持至少一行、两列数值含NaN与一列站点字符串；支持全缺训练列、常数列、未知站点和单行训练。它不是任意表结构清洗库。必须核对列顺序与语义，同形状错序不一定被程序发现；无穷值或无行数据会明确失败；分类缺失请先按声明的固定标记规则处理。本主实验站点本来无缺值。

QuantileClipper在fit前transform、上下分位倒置、特征数改变时抛显式异常。训练全缺列插补0，截断界0，验证首次非缺值也截成0；常数列标准化scale=1。未知类别独热为全零，不扩列。了解这些行为比笼统称“支持缺失”更重要。

JSON输出拒绝NaN/Infinity，先完成序列化才替换旧文件。源码、数据、PDF不得作为实验输出目标；符号链接和多硬链接被拒绝。审计实测序列化或路径失败时原输出KEEP保持不变。这是防误操作，不是恶意并发文件系统的安全保证。

## 12 整包重建

数值与Notebook依赖齐备后，若还需重建PDF：

```bash
python -m pip install -r build-requirements.txt
npm install
bash build_all.sh
```

MathJax3.2.2将可信课程公式离线转SVG，WeasyPrint70排版；需要系统Pango、Noto Sans/Serif CJK及DejaVu字体。无在线公式服务或XeLaTeX依赖。脚本不会默默安装依赖，安装在自己的隔离环境。

build_all.sh将普通、-O、审计、十图、新Notebook、新内核、三PDF串联，输出到outputs/rebuild，不覆盖交付原版。verify_rebuild.py比较数值、数据再生成、图像字节、Notebook格与PDF可用性。PDF字节不要求跨平台一致；仍须人工逐页检查公式、表格与分页。

## 13 L8 写研究报告并验收

写300至500字报告，包含三机制各自目标、所有fit边界、一个配对结果、一个负结果、TargetEncoder两种训练编码、真实库参照、重复单位、没有做什么现实验证。不要把“结果更差”直接称为泄漏，也不要以“分数没变”排除泄漏。

验收清单：

- 原值到下一前向的四行链及两验证预测完整
- 一个实际折的处理状态可从训练行独立算回
- 三机制分开，336次主要预测器拟合全部保留
- 目标编码逐行追到正确的内部或完整训练标签集合
- 正确路线的干预不变性与错误路线敏感性均通过
- 普通、-O、不同cwd与新核有实际记录
- 全部十图与三PDF已阅读，全部题目有完整解答
- 全缺列、未知类、时间与组别适用限制明确
- 未将合成结果包装成现实部署、未按分数换seed

完整参考见答案册L1至L8。自查数值通过不是独立课程审阅，更不是现实科学主张的最终验证。
