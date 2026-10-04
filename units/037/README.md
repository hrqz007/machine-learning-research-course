# 037 约束与非光滑优化

核心问题：目标有L1尖点或参数约束时，普通梯度为什么不够？直接先修16、32至33。

本单元从次梯度、投影变分条件和近端子问题推到L1 soft-threshold，再用同一4×2原创回归数据逐行核对两次同步更新。循环坐标下降与光滑盒约束作为明确区分的问题/算法对照，相关设计压力例另列，不暗换主数据。

## 文件与运行

- lecture.pdf / lecture.md：22节讲义、9幅原创机制图和20道练习
- lab.pdf / lab.md：独立运行指南、完整两轮纸笔账本、失败实验及输入契约
- answers.pdf / answers.md：20题详解，含引理证明、逐行表与相关设计顺序反例
- experiment.ipynb：实际顺序执行的12个代码格、8幅重新计算PNG输出
- experiment.py：从零实现、完整轨迹、原始输入验证及原子输出保护
- build_assets.py：重算9幅教学机制图；author_audit.py：可重复作者检查
- data/：CSV、完整配置及数据字典
- library_check.py / library-result.json：真实scikit-learn1.8.0核对
- source-checks.json / test-result.json / verification.json：来源范围、测试证据和内容摘要

```bash
python -m pip install -r requirements.txt
python experiment.py --output-dir outputs
python author_audit.py
```

自定义CSV/JSON用--data和--config，输出使用独立目录。Notebook从本单元目录打开，重启内核，第一格先校验CSV、JSON、experiment.py和build_assets.py的SHA，之后才执行任何数值计算或出图。合法修改默认文件也会拒绝，这是防止固定图注与数据不一致的保护；自定义实验请用CLI。

可选库对照：安装requirements-library.txt，运行python library_check.py --output-dir library_outputs。核心实验实际Python3.12.14、NumPy2.3.5、Matplotlib3.10.8。Notebook记录nbformat5.11.1、ipykernel7.4.0。environment.yml供自行创建环境，未宣称测试所有平台Anaconda或浏览器Jupyter UI。

## 数学与数值契约

n=4，X两列为(−1,−1,1,1)与(−1,1,−1,1)，y=(−7,−9,5,11)/4。无截距。f=Σr²/(2n)，J=f+λΣ|θj|，默认λ=η=1/2、θ⁰=(0,1/2)。两轮得到θ¹=(3/4,1/8)、θ²=(9/8,0)，J₀=77/32、J₁=173/128、J₂=141/128。所有样本局部导数与新旧前向都在完整轨迹中保留。

默认Lasso最优θ*=(1.5,0)、J*=33/32。光滑梯度不为零，停止以Lasso KKT无穷范数为准，默认容差1e-10。近端gradient mapping与该KKT向量分列，不能互换名字；盒实验只优化f，停止以投影映射和可行性为准。坐标下降一次更新是完整顺序扫描，不是同步向量步。

默认主路径：近端34步、投影32步、坐标1次扫描达到容差；固定次梯度80步预算耗尽。所有状态与实际参数都保留。相关设计rho=7/8明确单列并保存新矩阵；其步长取min(输入η,1/L)，派生输入若超公共范围则跳过压力例而不拒绝合法主实验。默认闭式目标差仅在数据逐值匹配时返回，其他数据为null。

## 输入与文件保护

高层load_inputs、validate、run_path、run_experiment先验证全部原始类型、形状和末字段，再调用数值求解。低层forward、soft、diagnostics、one_step接收已验证数组，供教学检查，不是任意对象的数据导入器。

范围：2至64行、固定2特征；原始数据/参数/盒边界绝对值≤100；非零原始标量绝对值至少1e-100；2至200步；η∈[1e-6,10]，λ∈[0,10]，容差∈[1e-12,1e-3]，rho∈[0,0.99]。普通与NumPy布尔、字符串、非有限数、扩展精度、Decimal/Fraction输入不隐式转换；常见NumPy有限实数标量允许。数组在外部已发生的类型转换无法逆推。

JSON重复键、非零字面量下溢、未知字段、CSV错误尾行和重复id均拒绝。运行中超过1e120、非有限或非零乘法下溢触发明确arithmetic_range_stop，保存最后完整有效状态。算术范围停止是报告中的真实状态，不等于达到容差。

输入验证、求解和序列化完成后才创建输出目录并原子替换。现有symlink/hardlink、输入文件或教材资产不能被覆盖；常规写入失败清理本次新建空目录且保留旧报告。此防护不声称抵抗并发恶意文件系统竞态。运行输出目录和第三方源材料不在公开包中。

## 核验与限制

作者检查包括两轮精确Fraction、120位Decimal、32组dyadic随机设计、6658个原语链相关标量检查、光滑有限差分、阈值边界、轨迹连续性、19类前置验证拒绝和CLI文件保护。真实scikit-learn核验主/相关两设计：alpha=0.5，fit_intercept=False，selection=cyclic，tol=1e-12，max_iter=10000；两者返回(1.5,0)，dual_gap_=0。初始化不同，所以只比较最终解与诊断，不宣称全部库轨迹逐步一致。

普通与−O新进程在本单元外两种工作目录运行；Notebook用新进程中新建真实IPython InProcessKernel逐格执行，两次实际重跑并检查报告。四种默认输入篡改都在首格失败，后续无数值/图片输出。单独查看全部最终PDF页与8张真实NotebookPNG。作者自检不等于独立外部验收；公开结论不超出这些有限证据。
