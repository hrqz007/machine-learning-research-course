# 035 动量与加速方法实验指南

## 1 实验目的与阅读顺序

本实验用同一四行双特征回归，验证heavy-ball与lookahead Nesterov的完整更新链，比较狭长谷底的轨迹与超调，再与真实PyTorch SGD核对缓冲区约定。建议先读正文第2至8节，独立手算两轮，再运行Notebook；最后读状态谱和库映射部分。

完成后应能给出三类证据：每一行如何贡献梯度；状态根如何约束步长；库参数怎样转换为论文主点。只看到一条下降曲线不算完成。所用数据全部为人工合成、无量纲、无私人信息，没有训练集外预测任务，因此没有测试集精度结论。

## 2 文件与环境

lecture.pdf、lab.pdf、answers.pdf分别是正文、本指南与20题详解。experiment.py实现算法，library_check.py实际对照PyTorch，author_audit.py独立审计；experiment.ipynb可顺序执行，data/提供全部输入与数据字典。

核心脚本只依赖Python3.12；Notebook与审计额外需要NumPy、Matplotlib、nbformat、ipykernel，库对照需要torch。实际测试版本为Python3.12.14、NumPy2.3.5、Matplotlib3.10.8、torch2.7.1+cpu、nbformat5.11.1、ipykernel7.4.0。PyTorch使用CPU float64、单线程，并关闭foreach与fused。没有GPU测试。

Linux/Windows CPU环境可使用requirements.txt从官方CPU索引安装torch，也可使用environment.yml。实际验证的是已有Linux环境；未测试新Conda安装、学生机器或macOS。

```bash
python -m venv .venv
# 激活环境后运行
python -m pip install -r requirements.txt
python experiment.py
python library_check.py
python author_audit.py
python -O author_audit.py
```

不要用torch.py、numpy.py或matplotlib.py命名自己的文件，以免遮蔽依赖包。脚本从自身目录定位输入；Notebook须从本讲目录打开。

## 3 先检查数据与目标

CSV恰有四行，列为id,x1,x2,y。默认两列特征是(1,1,−1,−1)和(4,−4,4,−4)，标签为(1,1,−1,−1)，没有截距项。目标为$F(a,b)=\frac18\sum_i(x_{i1}a+x_{i2}b-y_i)^2$，不是MSE。由数据导出$F=\frac12(a-1)^2+8b^2$、H=diag(1,16)、最优点(1,0)。

不要只使用二次闭式写优化器。训练循环必须逐行预测、求残差、形成r×特征/4并累积两个梯度，最后同步更新。闭式只用作独立诊断与谱分析。报告同时保留直接前向loss和稳定计算的geometric_gap；接近最优时两者可能因相消有微小差异。

Notebook第一代码格比较data/regression.csv与data/model_spec.json的完整SHA256字节摘要，随后验证全部字段，最后才输出数字。如果只改空格也会失败，这是有意设计：已执行Notebook中的固定解释只对应完整默认输入。不要跳过保护格后声称自定义数据仍支持固定结论；自定义实验使用CLI并重写自己的解释。

## 4 手算两轮 再看屏幕答案

固定θ₀=(0,1)、d₀=(0,0)、η=1/16、β=1/2。先列四行预测(4,−4,4,−4)，残差(3,−5,5,−3)，损失贡献(9,25,25,9)/8。局部导数r/4乘特征后，梯度贡献分别为(3/4,3)、(−5/4,5)、(−5/4,5)、(3/4,3)，合计(−1,16)。

HB和NAG第一步相同，因为z₀=θ₀。求出d₁=(1/16,−1)，θ₁=(1/16,0)，然后重新算四行残差(−15/16,−15/16,15/16,15/16)。各行损失225/2048，梯度贡献a列均为−15/64，b列依次−15/16、15/16、15/16、−15/16。合计F=225/512、g=(−15/16,0)。

第二步分两条路线。HB使用刚才旧点的梯度，得到d₂=(23/256,−1/2)、θ₂=(39/256,−1/2)。重新前向残差为(−729/256,295/256,−295/256,729/256)，损失为309233/131072。必须由四行平方加总核对，不能只抄闭式。

NAG先记录同一个旧点前向，再令z₁=(3/32,−1/2)。预看残差为(−93/32,35/32,−35/32,93/32)，梯度为(−29/32,−8)，预看目标4937/2048。然后d₂=(45/512,0)、θ₂=(77/512,0)。新残差为(−435/512,−435/512,435/512,435/512)，F=189225/524288。新点与预看点不能混写。

Notebook用独立Fraction函数逐行打印这两条完整链，并核对从零实现。读者应能指出每一次梯度是在哪一个参数点计算的，以及平均因子出现在哪一步。

## 5 运行完整报告 看预算而不是只看终点

```bash
python experiment.py --output results/report.json
python -O experiment.py --output results/report-optimized.json
```

默认各方法申请80次更新，最多320次训练样本梯度。每条trace保存old、old_velocity、evaluation、gradient_point、new_velocity和new；old与new均含四行预测、残差、局部导数与梯度贡献。states保存所有完成的有效状态，t从0起算。

training_sample_gradients等于4乘实际训练梯度前向次数；training_full_forwards为实际梯度评估点的前向次数。若本轮梯度已计算，随后因数值范围或前向相消而停止，该次计算仍计入成本，即使本轮没有完成更新。为了完整教学记录，初始点与每个新点另外做诊断前向，diagnostic_full_forwards=1+更新数。all_forward_calls是两者之和。诊断函数内部也计算梯度字段，因此本实现的总算术开销高于只做优化所必需的开销，不能把训练梯度预算当成实际wall-clock基准。

比较gd_balanced、hb_quadratic_tuned、nag_strongconvex的首次目标≤10⁻⁶更新数，应为64、23、31。gd_safe在80步内未达到。各方法的η和β不同且有解析选择依据，报告完整保存。固定手算设置下HB与NAG的首次达标步数分别42和47；不要用这一小差距推广算法优劣。

## 6 检查谱与边界

令h=ηλ，HB用A=1+β−h、D=β，NAG用A=(1+β)(1−h)、D=β(1−h)。state matrix为[[A,−D],[1,0]]。自己用NumPy eigvals求两个根，与报告roots和spectral_radius比较。重根处平方根病态，独立数值求根允许较大的相对差；这不是放宽参数轨迹的比较精度。

验证严格区间：HB为0<h<2(1+β)，NAG为0<h<2(1+β)/(1+2β)，0≤β<1。报告的strict_jury直接检查三个不等式，网格图只展示它们，没有用“网格没出问题”代替证明。

在probes中找hb_boundary与nag_boundary。β=1/2、λ=16、只激活高曲率方向，分别用η=3/16与3/32；根含−1，所以尽管有限值未爆炸也不趋于0。再检查η=0.12、β=0.9的压力组：h=1.92使HB稳定而NAG发散。抄录完整假设之后，才可以解释这一对照。

## 7 实际核对PyTorch的约定

```bash
python library_check.py --output results/library.json
python -O library_check.py --output results/library-optimized.json
```

脚本显式设定CPU、float64、torch单线程、foreach=False、fused=False、weight_decay=0、dampening=0、maximize=False，并在每一轮清零梯度。损失仍为`(r*r).sum()/(2*4)`，因此库autograd与手写梯度的平均因子相同。

HB固定学习率时应满足d=−ηbuffer。NAG时库保存的是z=θ+βd，需恢复θ=z+ηβbuffer后才与paper主点比较。默认第一步库参数是(3/32,−1/2)，主点却是(1/16,0)；这是应当出现的差异。

再看dampening_first_buffer记录。τ=1/4只衰减第二步起的新梯度，第一buffer直接等于g₀。另一个测试确实尝试创建非零dampening的Nesterov优化器并确认拒绝。最后看schedule_rates计划：lr在buffer外与在velocity内的两种写法在学习率改变后不再相同。normalized_ema_rate给出归一化EMA匹配未归一化HB所需的α=η/(1−β)。

默认库比较的最大绝对误差约1.11×10⁻¹⁶；比较阈值是5×10⁻¹²乘max(1,参考参数最大绝对值)。这个容差只用于本讲有界短程对照，不是通用优化成功标准。

## 8 输入契约和失败保护

所有公开计算入口检查原始类型。只接受内置int或float，拒绝bool、字符串、complex、Fraction、Decimal和NumPy标量。向量仅接受长度2的list或tuple；steps必须是0至400的内置整数，不能为2.0。初始参数、位移分量在[−100,100]；非零原始数的绝对值不得小于10⁻¹⁰⁰。

CSV仅支持明确的四角家族：(1,s,1)、(1,−s,1)、(−1,s,−1)、(−1,−s,−1)，1≤s≤10，行id固定S1至S4。默认s=4。它不是任意回归数据加载器。JSON必须只有默认文件列出的字段，kind精确匹配；手算/压力步长在[10⁻⁸,0.3]、动量在[0,0.99]、dampening在[0,0.99]、target_gap在[10⁻³⁰,100]、library_steps在[1,40]；schedule_rates必须含2至20个全部合法的正步长。

JSON拒绝重复键和非标准NaN、Infinity；十进制浮点token先用Decimal检查，再转float，避免1e−400悄悄变0。所有尾字段都验证完之后才调用更新函数，python -O下也一样。库对照额外要求正hand_momentum与零初始速度，明确不替用户猜测非零buffer初始化。

在包含旧report.json的临时目录中制作一个坏JSON副本，把schedule_rates最后一项改为true。让--output分别指向旧文件与尚不存在的子目录。两次执行都必须非零退出，旧字节摘要不变，新目录没有出现。再用最后一行CSV额外多一列、最后一项NaN和JSON重复键重复测试。author_audit.py自动执行这些案例与输入/源码别名保护，包括符号链接和同文件系统硬链接。

CLI只允许包内results/或明确指定的外部报告目标，不能覆盖输入或源码；计算和序列化成功后才创建目标父目录，使用同目录临时文件与原子替换。这里没有对恶意并发路径替换实现操作系统级锁定；测试不是安全隔离证明。

## 9 理解停止原因与极小损失

budget_exhausted表示完成有限预算。floating_state_stagnation要求θ和d同时舍入不动；一轮参数不动不足以停止。represented_optimum_zero_velocity表示存储点等于本例已知可表示最优点且速度为0；forward_cancellation_limit表示逐行前向丢掉了非零数学误差，numeric_range_limit表示下一状态超过保守的数值范围。它们都与目标阈值是否达到单独记录。

检查parameter_pause_not_state_stop探针：a=nextafter(1,2)、b=0、初速度(10⁻³⁰,0)、η=10⁻⁸、β=1/2。第一轮位置不动但速度改变，代码继续执行。nearby_floating_stagnation使用零初速度、β=0，第二次相同位移后整个状态不再变化。不能把这类现象写成精确梯度为零。

近最优点的直接残差损失和几何目标之间，以预测、残差和乘法的ULP传播界交叉检查，不用一个固定10⁻⁸容差掩盖极小量。发散压力路径的低曲率分量会受到大数相消污染；独立Decimal对照对该路径按完整前向量级计误差，不能据此声称小坐标具有高相对精度。

## 10 验收清单与提交建议

先保存两轮手算表、状态矩阵与稳定条件推导，再附上自己实际运行的report.json和library.json。记录Python/torch版本、dtype、线程、buffer初始化、loss归一化、梯度预算与停止原因。至少解释一处HB目标上升、一处NAG错误步长发散、一处库坐标映射以及一处浮点停滞。

作者普通与-O审计均通过42,348项检查，含独立Fraction/Decimal、完整轨迹、谱、坏输入及输出保护；明细见test-result.json。Notebook在新进程中的真实InProcessKernel顺序执行并保存PNG，未测试浏览器或socket传输。逐页检查与摘要见verification.json。
