# 034 随机与小批量梯度下降

同一四行常数回归问题，逐样本展开两次随机梯度更新，推导条件无偏、小批量方差、固定步长噪声平台与递减步长二阶矩，再按相同样本梯度预算比较full-batch、SGD、小批量与顺序机制。先修为第022至023、032至033讲。

全部标签和机制图为原创合成教学内容，不说明真实任务泛化性能、实际硬件吞吐或一般神经网络收敛保证。

## 文件

- lecture.pdf / lecture.md：18节正文、18题、9张解释图
- lab.pdf / lab.md：独立实验指南与完整复现验收
- answers.pdf / answers.md：18题逐项详解，含同一数据的三张完整前向表
- experiment.ipynb：12个真实内核顺序执行代码格与6张重新计算PNG图
- experiment.py：实际随机批更新、精确手算参照与可复现报告
- author_audit.py：独立Fraction/Decimal、全部短路径、逐步浮点与错误输入审计
- data/：四行标签、配置与数据说明；figures/：9张原创正文图
- environment.yml、requirements.txt：运行环境
- source-checks.json、test-result.json、verification.json：来源和有限验证范围

## 运行

```bash
conda env create -f environment.yml
conda activate ml-course-034
python experiment.py --output-dir outputs
python author_audit.py
jupyter lab experiment.ipynb
```

也可在Python3.12虚拟环境安装requirements.txt。CLI默认输入相对脚本目录定位；显式--data、--config、--output-dir相对当前工作目录定位。输出目录中生成report.json，完整计算成功后用原子替换更新；坏输入不会新建目录或覆盖旧结果。

Notebook须从本讲目录启动，第一格在任何数值或图形输出之前验证默认两个输入文件完整SHA256。改数据做研究请使用脚本，或复制并同步修改Notebook说明与基线，不能把旧手算答案冒充新数据结论。

实际运行环境为Python3.12.14、NumPy2.3.5、Matplotlib3.10.8、nbformat5.11.1、ipykernel7.4.0。生产验证用了新Python进程中的真实ipykernel InProcessKernel顺序执行。未测试实际Anaconda安装、浏览器Jupyter界面或socket内核传输。实验本身不需要网络、GPU、账号、外部API或真实个人数据。

## 数学与成本合同

标签y=(−2,0,1,5)，单行ℓ=(w−y)²/2，F=平均ℓ=13/4+(w−1)²/2。最优1，最低目标13/4。单样本梯度噪声方差13/2。学习率1/4、索引4再1给参数0→5/4→7/16，完整目标15/4→105/32→1745/512。

full每次恰好使用4行；replace_b4是4次独立放回，仍可能重复。subset_b2每步从全部4行重新抽不放回子集；reshuffle是每轮生成排列后依次用完。固定顺序与reshuffle没有套用独立新抽样二阶矩公式。

默认400个重复、每条160次样本梯度预算。整批不足时停止并记录unused_budget；不同方法更新次数不同。curve保留自己的实际网格，Notebook共同成本图只取真实4倍数状态。诊断完整损失重算4(T+1)样本项/重复，首条完整更新链额外重算4T项，单列而不算训练梯度。没有用这些操作数推断墙钟速度。

mean_excess使用非负等价式(w−mean(y))²/2；mean_full_loss实际逐行重算。有限精度下直接F−F*可能归零，非负式仍能展示小误差。自定义数据的均值与方差按binary64计算，不承诺任意病态输入的精确算术。hand和enumeration则明确用存储浮点值的Fraction参照。

## 随机与统计合同

位生成器为PCG64，每种随机机制和每次重复使用SeedSequence([root_seed, method_group, repetition])；组号放回100+B、子集202、重排300。固定和递减步长复用同一批索引数组，形成有意配对。full与固定顺序的400行是确定性重复，不能当作400份额外随机证据。

分位带是400条运行的10%至90%分位，不是均值置信区间。mean_excess_standard_error用重复之间样本标准差/√400；配对比较直接用每次重复的差值标准误。theory_mean_excess_z仅是点态Monte Carlo诊断，未作多重比较修正，接近零标准误时返回null。理论与实际逐步舍入可能在末期有微小区别。

## 支持与拒绝范围

标签需2至64个普通有限实数，绝对值不超过10000，非零值不小于10⁻⁸；字符串、布尔、复数以及更极端值拒绝，避免教学实验静默强制转换。初值同样有界，固定步长严格在(0,2)且不小于10⁻⁸，decay_offset在2至100，重复2至2000，预算4至2000，重复×预算不超过500000。批列表不超过8个、互异、含1、每项不超过样本数。种子是0至2³²−1的Python整数。每个概率不小于10⁻⁸、和在10⁻¹²内等于1，仅用于有限枚举，不用于主采样。

所有输入字段包括末尾非法值在第一个RNG创建之前校验。run_batches还验证完整索引张量，确保无越界、布尔索引或预算超支。固定步长理论仅用于相应新独立抽样，非理论机制报告不填理论曲线。

## 复核事实与限制

完整短路径枚举直接汇总终点矩，不复用生产矩递推；随机二进制可精确数据核查实际原始浮点操作；Decimal独立高精度参照、24个排列、新进程normal/-O、坏输出保护及Notebook完整JSON一致性均在verification.json记录。所有PDF最终页与真实Notebook图片人工目视；独立复核与远端发布是后续独立状态，文件存在不表示已发表。
