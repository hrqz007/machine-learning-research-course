# 040 正则化与收缩估计

为什么主动引入一点偏差，有时反而能改善预测？本讲沿同一四行共线回归例解释Ridge、Lasso、Elastic Net、MAP和正则路径，再以独立340行合成数据实践训练/验证选择、锁定后测试。

直接先修：024、027、029。037的L1从零求解是延伸。本讲采用成熟库求解高维L1，并用二维精确分支作独立参照。

## 教材文件

- lecture.md / lecture.pdf：20节连贯讲义、9幅原创计算图、20题练习
- lab.md / lab.pdf：独立实验与验收指南
- answers.md / answers.pdf：20题完整推导与逐样本手算
- experiment.ipynb：真实顺序执行的Notebook，保留全部文字与实际PNG
- experiment.py：完整行账本、目标/库对照、路径、40候选选择、最终测试、Monte Carlo
- audit.py：独立Fraction/Decimal/增广解、KKT、随机有理设计、输入/文件守卫和泄漏检查
- plots.py / figures/：真实数值驱动的原创图，可重建
- data/：两套合成CSV、明确配置、合成来源及字段说明
- data_integrity.json：核心脚本与四个数据/配置文件的首格契约
- requirements.txt / environment.yml：实测依赖与可选conda配置
- test-result.json / source-checks.json / verification.json：实际验证、来源范围和精确文件哈希

运行输出outputs/result.json、outputs/notebook_result.json包含完整行状态、40个候选、锁定参数、测试逐行误差、1000组模拟噪声和每个λ的全部拟合系数；它们由运行再生，不是下载遗漏的静态文件。构建缓存、临时渲染页和检查工作目录不属于教材。

## 运行

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py
python -O audit.py
python plots.py
```

Notebook需Jupyter/IPython环境；从本单元目录开新内核按顺序执行。首格在数值导入前核对五个契约文件，有变化先停止。自定义合法输入应复制并显式传入--data、--selection、--config；输出禁止覆盖教材输入/资产，或通过符号链接/多硬链接写入。脚本可从任何工作目录使用自身路径读取默认输入。

实测Python3.12.14、NumPy2.3.5、SciPy1.17.0、scikit-learn1.8.0、Matplotlib3.10.8。Notebook实际由新Python进程的ipykernel InProcessKernel执行；未测试浏览器Jupyter界面、socket传输、Windows或本地Anaconda安装。

## 必须读懂的结果

主手算X=[[-1,-1],[-1,-.75],[1,.75],[1,1]]，y=[-1.75,-2.25,1.25,2.75]，无截距/无标准化。F=‖Xβ−y‖²/(2n)。Ridge λ=.25、η=.5，从(0,0)到(1,57/64)，再到(1009/1024,3623/4096)。第二步F略增，但J略降；每一步保留四行局部链和同步更新。

Ridge解析解为(129/134,61/67)，第一原坐标比OLS的1/4更大。Lasso λ=.5解(1.5,0)，相关设计第二梯度−15/32。现代Elastic Net λ=.5、ρ=.5解(119/134,49/67)，不额外采用原论文事后重缩放。

Ridge库alpha=nλ；Lasso/Elastic Net alpha=λ。复制所有训练行时只有Ridge库alpha翻倍。svd的tol被忽略、n_iter=null表示不适用而非无成本；λ=0走lstsq最小范数端点。rho=0和1分别显式走Ridge/Lasso等价目标。

340行实验训练80、验证60、测试200，20特征。训练单独拟合ddof=0尺度与标签中心；40候选只能由验证MSE排序。真实默认赢家Lasso λ=.03，验证MSE约.807542；锁定后测试约.937357，预声明OLS基线约1.075616。该单次合成结果不证明普遍泛化改善，也未恢复真实所有零系数，更非因果结论。

## 数值诊断与失败

每个候选保留F、L1、L2、J、KKT、dual_gap、n_iter、警告、尺度及原单位参数。独立准入为绝对KKT≤1e−7且无ConvergenceWarning，与库tol不同；未收敛/库错误不静默删除，若没有合格候选则selection_failed。精确并列时偏好更大λ、Ridge/Lasso/Elastic Net顺序、再更大ρ。不用不同λ下各自J直接选泛化。

Monte Carlo用固定设计、真系数(1,1)、σ=.5、种子4018、1000组配对噪声，区分系数MSE、无噪声预测和新噪声标签误差。理论OLS系数MSE7.125与实测约6.688040有有限模拟差异，这些真值诊断不参与另一数据集选择。

作者审计包含三状态Fraction逐行链、120个随机二进制有理Ridge设计、110位Decimal病态参照、80个光滑方向差分、所有主路径二维分支、实际CSV逐字节再生、测试标签反事实、训练scaler隔离、端点/秩亏/重复列/常数列、受控库异常与原子失败。测试不会被Python -O删除。公开validation范围仅为有限教学输入，不声称适用于任意条件数、缺失数据或无限量级。

所有PDF页数与真实Notebook格/PNG数从最终文件读取，最终全页目视及完整运行证据见verification.json。独立审核与远端发布状态以课程总目录为准。
