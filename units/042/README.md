# 042 二分类与逻辑回归

这一讲把线性得分、Bernoulli似然、稳定数值实现、优化停止和有限MLE是否存在分别连起来。它属于完整166单元机器学习科研课程。直接先修014、020、023、029、032、039；后接043多分类。

## 学习材料

- lecture.pdf / lecture.md：20节正文，9幅原创计算图，20道练习
- lab.pdf / lab.md：独立实验指南，包含环境、纸笔链、代码对照和失败验收
- answers.pdf / answers.md：全部20题详细答案
- experiment.ipynb：真实执行的16代码格、9张真实PNG与完整逐行输出
- experiment.py：稳定softplus/sigmoid，行级loss/梯度/Hessian，GD与保护Newton，分离方向、阈值、真实库与差分
- audit.py：独立Fraction、110/1100位Decimal、数值与接口/文件失败检查
- plots.py：从完整实际结果再生成9幅图；指定新目录或outputs/，不会覆盖原教材图
- data/：两个分开的四行合成CSV、预声明配置与数据说明
- requirements.txt / environment.yml：依赖入口
- source-checks.json / test-result.json / verification.json：来源范围、实际测试与精确文件摘要

最终PDF实际页数：正文11、实验4、详解6，共21页。所有最终页与9张Notebook输出PNG已逐一查看。文件存在和作者完成不表示已通过独立验收，更不表示已发布远端。

## 一条同例贯穿的主线

参数顺序[b,w]，x=(-2,-1,1,2)，y=(0,1,0,1)，不额外缩放；F为平均Bernoulli负对数似然。η=.5、初始零、无正则：

- θ⁰=(0,0)，F=log2，g=(0,−1/4)，H=diag(1/4,5/8)
- θ¹=(0,1/8)，F≈.6667692275980063，g_w≈−.17221881242732377
- θ²≈(0,.21110940621366188)，F≈.6542101267359844，g_w≈−.11969482505055372

每一轮均保留全部四行的新logit、概率、loss、平均loss、局部导数、Jacobian、平均梯度和Hessian外积贡献。先整批汇总再同步更新，第二轮后重新前向。初始链是精确分数；后续超越数不能把打印小数当有理真值。

独立令t=eʷ推得t³−t−2=0，110位二分给有限MLE斜率≈.4196176249910979。默认GD真实68次更新，保护Newton4次，满足梯度L2≤1e−10。训练准确率仍为1/2，这个负面比较完整保留。

另一CSV的标签(0,0,1,1)完全分离。沿w=s损失与梯度趋0但参数无穷增加，无有限MLE。该方向证书仅适用于这个已知例子；未实现通用高维/准分离检测。斜率L2不惩罚截距，单类训练仍可截距发散，所以fit入口要求两类。

## 运行

在本目录中使用Python3.12环境，安装依赖后：

```bash
python experiment.py --out outputs/result.json
python -O experiment.py --out outputs/result-O.json
python audit.py --out outputs/audit.json
python plots.py --report outputs/result.json --directory outputs/figures
```

默认CLI核验固定教学文件SHA256。需要新数据时复制CSV/config，显式传--data、--separable、--config；自定义主例不冒用默认三次方程证书。原始数值范围由入口明确限制；不是所有任意大/小实数都支持。保护逻辑不依赖assert。

从新内核顺序运行Notebook，首物理代码格仅用标准库检查9个精确契约，先于数值导入、输出与图。实际逐文件篡改测试全部先失败。Notebook在outputs/notebook-result.json写出与脚本相同的完整科学结果；该运行时输出不随公开包保存。

## 数值约定与实际验收

- 损失直接算signed softplus；正类导数用−σ(−z)，Hessian权重用σ(z)σ(−z)，保留±40的可表示尾项。
- ±1000的真实微小尾项由1100位参照计算，转binary64可能下溢为0；不把0解释成数学上的有限零损失。
- 未正则GD整条69状态逐行与110位Decimal比较，另外80个随机二进有理设计检查目标、梯度、Hessian和PSD恒等式。
- 3个非驻点×5种步长实际核验梯度与Hv；实际SciPy expit/log_expit交叉核验。
- 真实scikit-learn1.8 lbfgs：无正则C=np.inf；λ=.25时C=1/(nλ)=1，复制全部行后C=.5，目标一致。
- 未正则库路径实际UserWarning如实保存；与ConvergenceWarning区分，另行计算完整梯度。classes_和控制零得分的并列规则实际核对。
- 3804项作者数值/输入/JSON/原子输出检查，普通/-O、原/新工作目录完整core及audit字节相同。
- 真实短预算、初始驻点、首步拒绝CLI均正常结束，第二轮不存在时为null；坏最后字段普通/-O不覆盖旧输出。
- 每次完整行状态尝试和成功返回分别计数，拒绝/失败也保留。sample_*_attempts是尝试覆盖行数，不等于已完成FLOPs；没有墙钟/内存优劣声明。
- JSON全序列化后再原子提交；禁止覆盖输入、源码、教材与通过符号/硬链接绕过。绘图输出独立保护原教材位置，逐图完整序列化后原子替换。

## 解释边界

四行原创合成数据仅用于机制推导，不能评估现实泛化、校准、因果效应或高影响个人决策。固定阈值.2/.5/.8是预声明示例，未用测试标签调阈值。梯度停止只是数值标准，不能替代有限估计存在性证明。

真实执行环境为Python3.12.14、NumPy2.3.5、SciPy1.17.0、sklearn1.8.0、Matplotlib3.10.8。Notebook通过新Python进程中的真实InProcessKernel顺序执行；不声称验证浏览器Jupyter、socket传输、Windows或Anaconda安装。跨平台末位可能不同，判断应依据明示容差和统计目标，不能把本机字节一致承诺为所有机器的性质。

来源仅使用Stanford原课程与sklearn/SciPy/statsmodels官方资料，范围详见source-checks.json。图、数据与教学文字原创；statsmodels仅检查文档，未执行该库。
