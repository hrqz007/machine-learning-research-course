# 第031讲 一维梯度下降与学习率

为什么沿负导数走，一步到底该走多远？本讲从五点回归的剖面构造已知答案的一维二次目标，逐行展开前向、残差、局部导数、共享梯度累积、更新和再次前向，再严格分类收敛、交替、循环与发散。直接先修是第013、028讲；用到第008讲的幂、对数和序列记号。

## 文件与阅读顺序

- lecture.pdf / lecture.md：20节完整正文，12张原创解释图，18道自测题
- lab.pdf / lab.md：独立实验指南，可从环境搭建一直做到失败测试
- answers.pdf / answers.md：18题逐步完整解答
- experiment.ipynb：实际执行的Notebook，保留分数计算、真实轨迹和重新计算的PNG图
- experiment.py：仅依赖Python标准库的脚本和明确区分的浮点/精确API
- data/regression.csv：五行原创合成回归数据
- data/learning_rates.csv：整数分子/分母表示的七种精确学习率
- data/model_spec.json：全部实验参数；data/README.md：字典与来源
- figures/：讲义用解释图；Notebook独立计算图，不读取这些预制图片
- requirements.txt / environment.yml：依赖与环境说明
- source-checks.json：实际核对的一手来源与核查边界
- test-result.json / verification.json：执行、数学、失败与页面检查记录

不要把完整单位材料分拆成只剩一个公式或一个终点的速记表。建议先读正文1–8节并逐行手算两轮，再做实验4–6节，随后读边界、缩放、停止与浮点解释。

## 一分钟运行

从本讲目录：

```bash
conda env create -f environment.yml
conda activate ml-course-031
python experiment.py
jupyter lab
```

或在Python3.12环境中使用：

```bash
python -m pip install -r requirements.txt
python experiment.py --output outputs/my_result.json
python -O experiment.py --output outputs/optimized_result.json
```

CLI默认输入与默认输出以脚本目录为准，可以从其他目录执行脚本绝对路径；显式相对参数以当前目录为准。`--data`、`--rates`、`--spec`可指向复制后的自定义输入。输出为完整JSON，包含每轮状态而非只有最后一个数。outputs/是在运行时产生的目录，不是原始数据。

## 三个不可混淆的模型

1. 原始有理数理论：f(b)=4/25+(b−6/5)²，q=2，初值0；CSV中的整数/分数给出精确学习率
2. 存储系数精确算术：先把q、中心和常数转成实际binary64，再用Fraction.from_float记录这些存储数的精确值，之后用精确算术递推
3. 实际浮点迭代：每轮真实执行梯度乘法与参数减法，每一轮都可能舍入

JSON分别保存rational_theory、stored_model_exact_arithmetic和fixed.history。求解器从不以闭式公式替换更新。默认原始回归输入是可精确表示的整数，因此原始profile的精确结果是q=2、center=6/5、constant=4/25、intercept=3/5。

## 关键验收锚点

- 主例每个b配最佳截距a=3−2b；不同于固定截距或二维同时更新
- η=1/4的参数0、3/5、9/10、21/20、9/8；前3个完整损失8/5、13/25、1/4
- η=3/4参数交替，但与η=1/4的精确超额逐轮相同
- 非最优初值的稳定区间0<ηq<2；初始已最优是例外
- η=1/2精确一步到位；η=1在0和12/5循环；η=5/4交替发散
- 参数误差10⁻⁶首次21轮，梯度10⁻¹⁰首次35轮，并检查前一轮不满足
- 目标乘10需要η除10；坐标v=100w需要η_v=10000η_w，本例匹配η为2500
- w⁴从1以η=1走到−3，损失从1变81；二次函数定理有明确适用条件

## 停止与固定预算

固定模式默认保留20次更新及初始状态，共21行。达到存储最优点后仍可重复记录固定余下轮数，用于对照；只有显式growth_guard可以提前结束。停止模式分别标记exact_stored_optimum、gradient_tolerance、max_steps、stagnation、two_cycle和growth_guard。fixed_budget_completed只表示指定轮数完成。

循环判断把返回的端点记录后停止；停滞或增长保护不执行被拒绝的下一步。初始状态总计为t=0，因此更新次数等于history行数减1。非法预算在任何提前成功判断前拒绝。数值运算超范围、非有限或不支持的下溢直接失败，不生成部分成功报告。

## 输入与数值范围

外部float接口接受Python int/float和普通NumPy整数、float16/32/64标量；先检查原始类型，拒绝bool、复数、文本、Fraction/Decimal外部实数、扩展精度浮点和会丢失整数信息的转换。精确接口exact_trace和first_threshold明确接受Fraction，角色与solver不同。profile的x/y应是一维普通list或tuple，禁止隐式二维广播。

一般输入绝对值≤1e8，非零值≥1e-100。计算中非零次正规结果以及正量或非零乘积舍入成0明确拒绝；状态参数也受同一输入合同约束。默认增长界为1e7；派生缩放模型与匹配学习率必须先通过范围检查。以上保守范围可能拒绝数学上合法的问题，不是通用高精度优化库。

数据2–1000行，学习率1–30条；固定展示预算0–1000，停止预算0–10000。读完并验证所有字段、所有rate与派生缩放后才开始第一条轨迹。JSON拒绝重复键和非有限token；原始非零1e-400不会静默转0，实际0e-400作为0保留，再依具体字段检查符号。

Notebook完整SHA256守卫覆盖三份默认输入，在首次数值输出和画图前执行。它不会自动把固定推导改成自定义实验。输入失败不得改旧报告，也不得提前创建新输出目录；正常与-O模式均检查这一点。直接超额与f−c并列保存，用于观察消去误差，并不承诺能恢复已经丢失的信息。

## 验证环境与解释边界

实际运行Python3.12.14、NumPy2.3.5、Matplotlib3.10.8、nbformat5.11.1、ipykernel7.4.0。脚本从两个无关工作目录各以新进程运行，正常与-O输出一致；Notebook以新Python进程中的真实InProcessKernel顺序执行，保存所有计算图。

本次没有测试实际Anaconda安装、JupyterLab浏览器界面或外部进程socket传输。JupyterLab给出兼容范围而非伪造的实测版本。Notebook图中英文轴标签便于未安装中文字体的环境，推导与说明为中文。

验证包括独立Fraction数据剖面、完整dyadic轨迹、高精度Decimal系数参照、逐原始运算舍入核对、阈值邻轮、缩放补偿、区间端点、浮点停滞和坏输入保护。有限检查不是所有输入的形式化证明。CPU小模型只解释机制，不代表真实任务泛化或大模型训练速度。来源记录中的公开材料只用于核查，正文、数据、图和解答独立编写，不复制教材图文。
