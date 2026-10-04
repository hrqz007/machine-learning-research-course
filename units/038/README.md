# 038 Newton与拟Newton方法

直接先修011、015、032–033。同一四行回归先用线性模型a+bx，再明确换成a+b²x；从每行前向、局部导数、Hessian两项到方程求解、线搜索与下一轮前向。进一步独立推导BFGS割线/正定与L-BFGS同窗口双循环，实际比较时间和内存。

## 文件与运行

- lecture.pdf / lecture.md：20节讲义、10幅原创机制图、20题
- lab.pdf / lab.md：可独立使用的运行指南、完整实验与读图问题
- answers.pdf / answers.md：20题详解，含两轮逐行表、局部收敛和BFGS证明
- experiment.py / experiment.ipynb：从零核心代码；15个真实代码格与6幅重算图
- benchmark.py / benchmark-result.json：实际墙钟和独立内存测量及全部原始结果
- library_check.py / library-result.json：实际SciPy求解器、BFGS策略和Wolfe检查
- audit.py：独立Fraction、Decimal、秩二式重建与文件保护检查
- data/：原创四行CSV、完整配置和字典；source-checks.json记录一手来源范围
- requirements.txt / environment.yml：可重建环境；test-result.json / verification.json：执行证据与精确摘要

```bash
python -m pip install -r requirements.txt
python experiment.py --output-dir outputs
python audit.py
python -O audit.py
python library_check.py --output-dir outputs
python benchmark.py --output-dir outputs
```

实际使用Python3.12.14、NumPy2.3.5、SciPy1.17.0、Matplotlib3.10.8、threadpoolctl3.6.0。Notebook从单元目录开启，新内核按序执行；可自行安装JupyterLab4.x。构建用新进程中的真实IPython InProcessKernel，保留数值和PNG输出，未测试浏览器界面、socket传输或全新Conda安装。

核心CLI生成outputs/report.json，Notebook生成notebook_results/report.json，完整确定性JSON需要逐字节相同。约6.7MB的运行报告不随教材发布；所有步骤由脚本重建。Notebook确实重算核心、Fraction表和6幅图，并真实调用SciPy对照。计时/内存图则读取随教材附带的已执行benchmark-result.json，清楚标为记录读取；运行benchmark.py才是重新测量。

## 数学与数值范围

n=4，x=(−2,−2,2,2)，y=(−1.5,−0.5,2.5,3.5)，F=Σr²/(2n)。线性Newton一步到(1,1)、F=1/8。平方参数模型F=1/8+(a−1)²/2+2(b²−1)²，主初值(0,3/2)，两步到(1,27/23)、(1,19683/19067)。Hessian始终从JJᵀ和残差二阶项逐行汇总；没有显式求逆。默认数据的稳定闭式目标差只作trace诊断，与实际row F分列；其他数据为null。

默认回溯c₁=0.0001、缩减因子0.5、最多40次试探；有限正定移位表写在config。初值(1,1/4)原Newton是上升方向，λ=8改正方向后仍须拒绝完整步、接受半步。梯度为0的(1,0)报告stationary_indefinite。所有梯度阈值都只是驻点阈值，不声称全局最小。

BFGS的C表示逆Hessian近似；sᵀy必须超过tol‖s‖‖y‖，否则显式跳过。L-BFGS只保留实际最近m对，γ由最新对计算；独立重建使用相同γ与窗口。教学Armijo+skip与SciPy Wolfe搜索不同，不断言逐步相同。默认平方模型GD在100次预算内未达到1e−10梯度阈值，原样保存max_updates，而不将小目标差改写为成功。

高层load_inputs、validate_config、optimize、run_experiment完整验证原始字段/维数/末元素后再进入回调或线性代数。2–64行、2参数，数据及初值绝对值≤100；非零原始数值至少1e−100。只接受内置int/float和常见NumPy32/64位实数标量；拒绝布尔、字符串、复数、Decimal/Fraction与扩展精度隐式转换，外部已转换数组的原始历史无法恢复。配置字段必须精确匹配，无重复JSON键；非有限、下溢文本和重复CSV ID拒绝。

主预算1–300，回溯1–60，历史1–10，初始C和历史每个元素完整检查。移位有限有序且不重复；实验维度2–64且最多3个、重复1–9、预算≤2000。参数内部演化不重新套用户初值≤100限制。非有限、超过1e140或非零平方下溢触发明确范围保护；低层数值函数接收已验证状态。线搜索失败不提交候选点，只有新损失与所需导数有效才同步更新。

## 时间与内存实验到底测了什么

基准是单独标明的满列秩合成稠密线性设计，seed3817、d8/32/64、n=2d、H谱1到20。统一初值和相对初始梯度10⁻⁶阈值、最多2000更新。输入生成单列时间且排除；每个方法预热一次，5轮轮换顺序，60个原始纳秒区间全部保存。

计时包含optimize完整调用：配置/状态校验、方法设置、每次行前向/梯度/Hessian、正定检查、solve、回溯和最后的曲率诊断。没有给任何方法缓存H或分解；Newton正定检验和solve是两次独立库调用，其开销均包含。详细trace关闭且不保存完整参数历史。实际f/g/H、solve、Cholesky和试探计数分列；教学trace打开时，为显示逐行量还会额外计算梯度/Hessian，计数也包含它们。

本教学BFGS为直观保留V C Vᵀ稠密乘法，并每次Cholesky检验正定；这些额外O(d³)操作都进入实测时间。它不是优化过的O(d²)秩二更新实现，不能把本次BFGS时长当成这类方法的最佳性能。同一严格二次控制中Newton一步的结果不代表一般非线性任务的速度。小尺寸受Python验证/调用开销影响，BLAS限制为单线程，机器信息只记录系统/架构/版本。每次时间允许变化，失败目标不会归入成功速度。专门缓存H或直接最小二乘是另一个可研究基线，本次未测。

内存分两类：指定保留状态数组nbytes与单独tracemalloc运行的Python跟踪峰值；均不等同原生进程RSS。数组不含数据、模型、输出trace、临时矩阵或求解器工作区；代表性临时数组尺度也不是完整峰值界。本次没有测RSS，不对大模型或其他硬件作吞吐/存储结论。

## 复现与失败保护

先计算并完成JSON序列化，再检查输出类型、父目录链接、与输入/教材同路径或硬链接，最后用同目录临时文件原子替换。坏输入不能修改旧输出或建立新目录。此检查不是对抗并发恶意文件系统变动的安全证明。

首格在输出任何数值或图之前核对默认CSV/JSON完整字节。合法数据修改也会拒绝，以免固定说明误导；自定义研究用CLI并重新解释结果。Fraction、Decimal120默认Newton轨迹、随机dyadic导数、中央差分、实际SciPy和有限窗口独立重建各测试不同性质，不以检查数量替代推导。所有结论限定于所列算法、有限样本、浮点范围与实测环境。
