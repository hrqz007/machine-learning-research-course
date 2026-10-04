# 第035讲 动量与加速方法

核心问题：利用历史更新方向能解决哪些震荡，又会引入什么？直接先修为第015、032、034讲。

同一四行双特征回归贯穿所有核心手算和算法，残差定义为预测减标签，F=Σr²/8=½(a−1)²+8b²。正文完整展开HB与lookahead NAG两轮逐样本前向、局部导数、梯度累积、同步更新与新前向，进一步推导二维状态矩阵、根、判别式及严格稳定区间。

## 文件与阅读顺序

1. lecture.pdf / lecture.md：22节独立正文，12幅原创图，20题自测
2. lab.pdf / lab.md：独立实验指南，环境、输入契约、完整手算、真实库对照与失败测试
3. answers.pdf / answers.md：20题独立详解
4. experiment.ipynb：已顺序执行的14代码格教学Notebook，图由实际状态重算
5. experiment.py：标准库从零实现；library_check.py：真实CPU PyTorch对照；author_audit.py：独立精确算术与CLI审计
6. data/：全部默认输入及数据字典；figures/：12幅正文图
7. requirements.txt、environment.yml：学习环境；source-checks.json：一手来源实际核查范围
8. test-result.json、verification.json：作者已完成检查与限制；results/为自行运行生成的报告，不属于冻结公开包

## 运行

核心脚本仅需Python3.12，默认输入相对于脚本本身定位，可从任何当前目录调用：

```bash
python experiment.py --output results/report.json
python -O experiment.py --output results/report-optimized.json
```

Linux/Windows CPU学习环境可通过python -m pip install -r requirements.txt安装依赖，或conda env create -f environment.yml。PyTorch来自官方CPU wheel索引；未验证macOS安装。随后运行：

```bash
python library_check.py
python -O library_check.py --output results/library-optimized.json
python author_audit.py
python -O author_audit.py
```

打开Notebook时把当前目录设为035，选择Restart Kernel and Run All。第一代码格在任何数字和图之前验证data/regression.csv及data/model_spec.json的完整字节；默认数据变化时立即拒绝，不能跳过后仍使用原固定说明。

实际执行环境：Python3.12.14、NumPy2.3.5、Matplotlib3.10.8、nbformat5.11.1、ipykernel7.4.0、torch2.7.1+cpu。Torch使用CPU、float64、单线程，foreach=False、fused=False。作者使用新Python进程中的真实InProcessKernel顺序执行Notebook；没有声称测试Jupyter浏览器、socket内核传输、新Conda安装或GPU。

## 关键数学与实测结果

默认数据X行依次(1,4)、(1,−4)、(−1,4)、(−1,−4)，y=(1,1,−1,−1)，模型不含截距，最优点(1,0)，Hessian=diag(1,16)。初值(0,1)、零位移，η=1/16、β=1/2：

- 共用第一步θ₁=(1/16,0)，F₁=225/512
- HB第二步θ₂=(39/256,−1/2)，F₂=309233/131072，目标上升但状态仍严格稳定
- NAG第二轮先看z₁=(3/32,−1/2)，g(z₁)=(−29/32,−8)，再到θ₂=(77/512,0)，F₂=189225/524288

固定正曲率二次方向λ、0≤β<1下，HB严格稳定要求0<ηλ<2(1+β)，本讲NAG要求0<ηλ<2(1+β)/(1+2β)。边界排除；整个目标要对所有方向成立。不是一般非二次/非凸/随机优化的保证。

各自理论设置、相同80次全梯度预算下，GD平衡、HB二次谱调参、NAG标准强凸设置首次F≤10⁻⁶分别在64、23、31步。保守GD80步内未达标。额外诊断前向单独计数，不能据此判断硬件速度。实际已计算但因数值范围或前向相消而未提交更新的梯度前向，仍计入training_full_forwards和training_sample_gradients；更新次数不能代替计算调用次数。

真实PyTorch检查不仅比较参数：HB核对d=−ηbuffer；NAG核对库参数z=θ+βd并恢复θ=z+ηβbuffer。默认各12步最大绝对差1.11×10⁻¹⁶。还实际核对dampening首buffer不衰减、Nesterov拒绝非零dampening、学习率变化导致两种velocity约定不同、归一化EMA的步长换算。

## 输入与输出保护

脚本只支持数据字典定义的四角尺度家族，不能装入任意回归数据后套用这些常数。所有公开入口验证原始内置数值类型，拒绝bool、字符串、complex、Fraction、Decimal和NumPy标量；向量维度、整数步数、全部配置字段及末项在计算前一次性检查。JSON重复键、非标准NaN/Infinity、十进制非零数过小均拒绝。

初始坐标与速度每项绝对值≤100，steps为0..400整数；手算/压力η在[10⁻⁸,0.3]，β在[0,0.99]；其余完整范围见lab.pdf。输入非零绝对值须≥10⁻¹⁰⁰。谱函数接受η=0以展示边界，训练接口要求正步长。库对照额外要求hand_momentum>0、初始velocity=0。

CLI在验证、计算、序列化全部成功后才创建输出目录与同目录临时文件，原子替换旧报告。包内仅results/允许写入，源码、输入及其符号/硬链接别名不能被覆盖；外部指定报告可写。不防御主动并发恶意路径替换。

每种方法最多执行steps次，不把预算耗尽当成收敛。whole-state停滞要求θ和d同时不动；forward_cancellation_limit与numeric_range_limit表示数值限制。逐行loss与稳定几何目标分开保存，绘图使用后者但训练梯度仍来自真实逐行累积。

## 验证范围

普通与-O独立审计每轮42,348项检查；含120组Fraction路径/1,440个状态、1,440个独立浮点原语次序状态、100位Decimal的320个长路径状态、87组谱检查、34类计算前错误输入、76个坏CLI输出保护及5个保护目标。正常与-O完整报告字节一致。

精确Fraction用于小型有限案例；Decimal长轨迹比较对发散压力组按完整前向尺度设误差界，不宣称被大数相消淹没的小坐标有高相对精度。直接残差与几何目标的差用ULP传播界检查。所有这些是实现与有限实例证据；稳定区域的无限时间结论来自正文解析推导。

最终PDF逐页、Notebook实际PNG逐张检查和冻结摘要记录见verification.json。公开包不含构建脚本、依赖缓存、临时页面或运行结果目录。来源与本讲原创内容的版权彼此独立。
