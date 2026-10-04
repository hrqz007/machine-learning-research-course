# 固定合成数据字典

`observations.csv` 是第014讲相同的三条原创、确定性、无量纲教学记录。没有随机生成过程、没有现实个人或机构资料，也不来自外部数据集。

- `record_id`：非空且唯一的记录编号，文本解析后仅作标识，不是输入特征
- `x`：固定实数输入，依次为0、1、2，无量纲
- `y`：固定实数目标，依次为1、2、2，无量纲

任务是研究同一个损失函数的局部导数和近似误差。模型为 p_i(a,b)=a*x_i+b²，损失为预测减目标的残差平方平均。数据在求导、方向扫描和驻点求解时始终固定。a 与 b 是自由实数参数，b² 作为共同非负偏移。

这里没有训练/验证/测试拆分，也不作预测泛化或真实任务性能声明。公开手算数据只用于机制核验。加载器明确检查这三条记录，避免换数据后仍使用旧的展开式答案。

## 再生成输出字典

`outputs/approximation_errors.csv`：每行为一个正位移长度。`step` 是单位方向(3/5,4/5)的位移长度；`error1_subtraction`、`error2_subtraction` 为直接浮点相减的一阶二阶绝对误差；`error1_polynomial`、`error2_polynomial` 用精确余项的浮点求值避免近值相减；`r1_over_t2`、`r2_over_t3` 为对应归一化误差；`cancel_error_polynomial`、`cancel_error_subtraction` 为方向(-1,1)/sqrt(2)的二阶余项两种计算方式。直接相减结果为零时不表示数学误差真的为零。

`outputs/hessian_audit.csv`：每行为一个固定参数点。`a`、`b` 为中心，`hessian_max_abs_error` 为梯度中心差分与解析 Hessian 的最大元素绝对误差，`raw_symmetry_error` 为未对称化差分矩阵与其转置的最大元素差。差分步长为1e-5。

`outputs/experiment_report.json`：记录环境、分数基准、驻点分类、边界检查和两种扫描。该文件由实验生成，不作为预先硬编码的测试答案。
