# 045 从概率到决策

给定类别概率，怎样按错误成本选择行动，并区分总体风险、一次观察和拒绝负担。此讲属于完整166单元课程，直接先修018至019；概率模型训练再接042/043，后续046讨论分类指标。

## 材料

- lecture.pdf / lecture.md：20节完整正文，9幅原创计算图，18道练习
- lab.pdf / lab.md：独立实验指南，逐组风险链、先验修正、真实模拟和故障核验
- answers.pdf / answers.md：18题完整计算与证明，包括全部八组账本和支持条件
- experiment.ipynb：13个实际执行代码格、九张真实PNG、三阶段完整逐状态损失贡献
- experiment.py：通用行动×状态代价矩阵、Bayes行动、拒绝、先验修正与配对模拟
- audit.py：Fraction逐组账本与256/6561种政策枚举、独立Bayes联合质量、真实库和接口核验
- plots.py、build_notebook.py、execute_notebook.py：完整图与Notebook重建入口
- build_pdf.py、mathjax_render.cjs、pdf.css、package.json：自包含PDF重建源码
- data/、requirements.txt、environment.yml：原创固定输入与运行环境
- experiment-result.json、test-result.json、verification.json、source-checks.json：结果与证据范围

PDF页数：正文12、实验5、详解5。作者完成与独立验收、远端发布分开记录。

## 八组主线和保留的结果

主CSV一行是一个条件组，已知P(Y=1|组)为(0,1/8,1/4,3/8,1/2,5/8,7/8,1)，每组人口权重1/8。模型不训练，概率在各政策间不变。损失矩阵行是行动、列是真类；每行先保存各真实状态贡献，再加总成条件风险、比较行动、乘人口权重。

- FP1/FN3阈值1/4，相等选0；成本最优行动(0,0,0,1,1,1,1,1)，精确成本11/32=.34375，准确率.75
- 等错误成本的准确率最优规则准确率25/32=.78125，但按FP1/FN3评价成本17/32=.53125，保留这次排名反转
- 拒绝成本3/8时行动(0,0,2,2,2,1,1,1)，总成本1/4，拒绝质量3/8；拒绝费没有从分母删除
- 三类成本例选择概率并非最大的乙行动，逐状态贡献与矩阵方向明确
- 类条件分布不变、先验1/4变3/4时，来源后验(1/13,1/4,4/7)修正为(3/7,3/4,12/13)。目标权重随之改变；旧行动目标成本37/32，修正后1/4
- 固定种子4505、每组256标签的三政策观察成本为.5234375、.337890625、.2451171875，不要求与精确期望逐位相等

本单元没有参数梯度下降。风险对给定概率的斜率、硬行动的跳变和概率误差到超额风险的界均单独推导；不把代价变化后的重新决策叫作模型更新。

## 数值运行

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python -O experiment.py --out outputs/result-O.json
python audit.py --out outputs/audit.json
python plots.py --report outputs/result.json --directory outputs/figures
```

可从其他工作目录以绝对脚本路径运行，默认输入相对脚本定位。改实验用复制的CSV/JSON再传--data、--shift、--config。九图布局只接受固定默认数据与配置报告；自定义实验用数值接口另画图，避免沿用主例固定标题。Notebook在单元目录开启，重启内核并运行全部；首物理格只用标准库验证九份输入/源码/环境契约。

## 接口和边界

- risk_table接受n=1..2000、K=2..20的概率矩阵和A=1..20个行动；成本矩阵为A×K，可非方阵
- 每个概率行和、总体权重和必须在1±1e-12内，只修正此容差内的舍入，不自动接受任意计数。成本非负且≤10000，非零原始数绝对值≥1e-100
- 拒绝NaN/Inf、bool伪数、复数、数值字符串、错形状、错误概率和、非法行动、重复键/ID；完整配置在随机生成前校验
- 并列采用精确计算风险相等后的首行动。主例二进分数边界有精确核验；任意十进输入在边界附近可能受浮点舍入影响，未暗加isclose容差
- correct_prior要求两先验位于[1e-6,1-1e-6]，但后验可以是0或1；直接Bayes入口在边缘质量正时可处理0/1先验。零边缘条件概率不定义，明确拒绝
- 先验修正只在类条件分布不变、来源后验适用及先验已知时有这里的解释；不提供未知漂移估计器
- 结果先完成严格JSON序列化再原子替换，拒绝覆盖输入/教材以及符号或多硬链接别名；不声称抵御恶意并发文件系统竞争

## 重建与实际验证

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
python build_notebook.py
python execute_notebook.py experiment.ipynb
```

可选PDF构建还需要Node.js、Pango和Noto CJK字体。构建只处理可信课程源，不是陌生HTML/TeX安全沙箱。现成PDF和数值实验不依赖排版软件。

作者数值核验含全部逐状态成本、精确最优政策枚举、零成本、并列、矩形成本矩阵、先验端点与目标联合质量，以及由模拟计数独立重算成本和分层方差。真实sklearn加权混淆矩阵已与同一观察成本对齐；没有使用测试标签挑阈值。

实际Python3.12.14、NumPy2.3.5、sklearn1.8.0、Matplotlib3.10.8。普通/-O、新工作目录、真新核记录见verification.json。新核逐格全文与PNG比较一致；stdout可能因刷新时机分成不同传输片段，不将这种分块差异伪装成科学差异或声称所有Notebook文件必然字节相同。Notebook通过新进程中的InProcessKernel实际顺序执行；浏览器Jupyter、外进程传输、Windows和Anaconda安装未测试。

这些是已知合成分布的决策机制，不是现实模型已校准或行动已部署的证据。总体成本、拒绝质量、一次观察成本和准确率分别报告；不将一种指标的改善当作所有目标的改进。
