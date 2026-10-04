<style>p{text-align:left} table{break-inside:avoid} figure{break-inside:avoid} h2{margin-top:14pt;margin-bottom:5pt} p{margin:5pt 0;line-height:1.55}</style>

# 第040讲 独立实验指南

## 1 实验目标与数据边界

先完整手算同一四行的两轮Ridge更新，再用真实成熟求解器核验Ridge/Lasso/Elastic Net，最后仅由训练/验证选择正则强度。你要能解释：F和J为何不同、何时出现精确零、库alpha为何不能直接互抄、标准化怎样改变惩罚。

data/regression.csv是4×2共线手算例；data/selection.csv另含80训练、60验证、200测试，共340行20特征。两套数据都为原创合成数据，不能把测试MSE用于真实应用效益结论。data/synthetic_truth.json仅为数据来源与再生审计，不是选择函数的输入。

## 2 环境与第一次运行

在本单元目录创建Python3.12环境，安装requirements.txt后顺序运行Notebook，或执行下列完整脚本。可以从其他工作目录用脚本绝对路径启动；输出路径由你指定，教材文件禁止覆盖。

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py
python -O audit.py
python plots.py
```

实际验证版本为Python3.12.14、NumPy2.3.5、SciPy1.17.0、scikit-learn1.8.0、Matplotlib3.10.8。提供environment.yml供Anaconda建立环境，但没有声称已经测试你的本地安装或浏览器Notebook界面。

Notebook第一格先以标准库核对data_integrity.json中的五个SHA256：核心脚本和四个数据/配置文件。若你希望探索修改输入，请复制一份并使用脚本显式--data、--selection、--config参数；不要删除检查后仍把修改结果当作官方基准。真实篡改测试确认任一个契约文件改变都会在数值输出和作图之前停止。

## 3 纸笔与代码的逐样本对账

先抄出讲义四行与β⁰=(0,0)，在纸上写每个预测、残差、r²/8和r x/4。然后在Notebook读取hand_trace的第0、1、2个状态。每个状态保留行局部导数、Jacobian、梯度分项、数据梯度、L2梯度、总梯度与三项损失。

第一轮β¹=(1,57/64)，第二轮β²=(1009/1024,3623/4096)。不要用ridge_solve直接替代这两轮；闭式解只作另一条核验线。每轮更新是同步的，后一坐标不能看到刚更新的前一坐标。

|状态|F|惩罚P|J|
|---|---|---|---|
|β⁰|69/32|0|69/32|
|β¹|41673/262144|7345/32768|100433/262144|
|β²|175757993/1073741824|29415425/134217728|411081393/1073741824|

读图4，指出第二步F略升、J略降的位置。完整分数逐行答案在answers.pdf第6至7题。audit的Fraction链独立用原始分数逐项构造，不读取结果里的浮点梯度来充当参照。

## 4 固定目标后核验真实库

四行无截距、无标准化。Ridge λ=.25得到(129/134,61/67)，库alpha=1。Lasso λ=.5得到(1.5,0)，Elastic Net λ=.5、ρ=.5得到(119/134,49/67)。比较系数还不够，需要查data_loss、l1_penalty、l2_penalty、objective、kkt_inf、dual_gap、n_iter和warnings。

L1独立参照是二维9个符号/支持分支枚举；一般20维主实验仍由成熟坐标求解器负责。代码不把对OLS的直接soft-threshold冒充相关设计的Lasso。把所有四行复制一次，重新拟合三个库：Ridge alpha加倍，另两者不变，系数应保持一致。

λ=0端点走lstsq最小范数；Elastic Net的ρ=0转Ridge等价目标，ρ=1转Lasso。svd的tol不控制迭代、n_iter为null，并不说明它不花时间。原Elastic Net论文的事后重缩放不在这里额外执行。

## 5 路径、共线与偏差方差

图3比较谱方向收缩和原坐标路径，解释为什么第一原系数反而变大。图6用独立约束半径展示L1角点和L2圆；半径不是λ。图7实际制造完全重复列，检查正L2下相同系数与纯Lasso可能不同的分配。

Monte Carlo固定同一4×2设计、真系数(1,1)、σ=.5、种子4018，生成1000组独立噪声；每个λ使用同一组噪声，完整noise和所有拟合系数留在运行JSON。图5分别显示系数MSE和固定X上预测MSE，不把两者混成一条曲线。理论新标签误差比无噪声均值预测误差多.25；经验均值/方差和理论不要求精确相等。

这一块已知真系数只作机制诊断。禁止用它为后面340行数据选λ，亦不从最漂亮的模拟曲线推出真实任务泛化保证。

## 6 只用训练/验证的完整选择

先声明λ网格[.001,.003,.01,.03,.1,.3,1,3]；Elastic Net的ρ网格[.2,.5,.8]。Ridge、Lasso各8个候选，Elastic Net24个，共40个。仅训练集拟合StandardScaler的均值、ddof=0方差与标签均值。每个候选复用同一训练scaler，并保存标准化γ与原单位β、原截距。

select_candidate只接收trainX、trainy、valX、valy、cfg，没有test或真系数参数。它以验证MSE最小选择；完全相等时先更大λ、固定族顺序Ridge/Lasso/Elastic Net、再更大ρ。任何ConvergenceWarning或绝对KKT残差>1e−7的候选均保留记录而被排除；这个独立门槛不等同库的tol。所有候选失败就明确返回selection_failed。

默认运行实际选Lasso λ=.03，验证MSE约.807542、训练MSE约.751308，存储零系数5个。标准化与原单位预测的逐行最大误差应在浮点舍入量级。测试先不看。

## 7 冻结之后才做测试

保存选择对象完整字节与choice_fit_sha256，再由final_evaluate读取测试标签。默认锁定模型测试MSE约.937357；预先声明的OLS基线约1.075616。没有按每个模型族测试分数重新挑选，也没有训练+验证重拟合。

把testy改成testy+100，再评价同一个模型。测试MSE应改变，锁定参数/预处理/选择摘要必须不变。这是直接反事实检查，而不是“代码里似乎没用test”的口头保证。更改验证特征均值也不得改变训练scaler和训练候选系数，但允许验证排序变化。

不得据测试结果新增一个网格值，再把新结果称为一次性最终测试。本次Lasso优于OLS只属于这一合成划分，选中非零变量也不是因果发现。

## 8 刻意失败与恢复

audit.py会真实运行以下失败案例，并用明确条件而非可被-O删除的assert判定：

- Elastic Net预算改为1次，保留真实收敛警告，状态not_converged。
- 对成熟求解器受控抛出异常，检查solver_error；全候选不合格时检查selection_failed。
- 坏值放在最后一个配置或测试标签字段，验证第一次solve/lstsq/RNG前就拒绝；含布尔、字符串、复数、NaN/Inf及过小非零数。
- 重复JSON键、重复跨split样本ID、坏CSV末字段和错形状拒绝。
- 输出指向教材输入、符号链接或多硬链接时拒绝；原子替换受控失败后旧文件字节不变，新空目录和临时文件清理。

公开数据有明确范围限制，这不是任意条件数或浮点精度的保证。若换成极病态问题，应重新设计精度、残差缩放与求解方法。

## 9 交付与可复核记录

提交两轮逐行账本、三种库目标映射、40候选表和锁定摘要、一次测试与OLS基线、偏差方差图、至少一个真实失败解释。保存完整输出JSON，比较正常脚本、-O脚本和新内核Notebook核心输出的SHA256，应完全相同。

作者实际复核包含Fraction、110位Decimal、光滑方向有限差分、增广解、二维所有符号分支、样本复制与原始CSV字节再生。PDF、图和Notebook输出的核验范围记录在verification.json；独立审核和远端发布状态由课程总目录另行记录。数量不是教学质量的替代品，能够解释差异和失败才是目标。
