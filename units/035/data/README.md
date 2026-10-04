# 第035讲数据字典

regression.csv是原创无量纲合成数据，没有私人或真实观测信息。

- id：固定有序S1、S2、S3、S4
- x1：第一个特征，依次1、1、−1、−1
- x2：第二个特征，默认4、−4、4、−4
- y：标签，等于x1

模型ŷ=x1*a+x2*b，不含截距。残差为ŷ−y；单行对总目标的贡献为r²/8，总目标为四行贡献相加。最优参数(1,0)，最小损失0，默认Hessian=diag(1,16)。

CLI允许同结构的尺度家族x2=(s,−s,s,−s)，1≤s≤10，曲率为(1,s²)。不支持任意数据集；标签与行顺序固定。Notebook的固定叙述只支持完整默认CSV和JSON字节。

model_spec.json字段：

- kind：精确匹配four_corner_momentum_v1
- initial、initial_velocity：初始二维参数与旧位移，默认(0,1)、(0,0)
- steps：各主方法最多更新次数，默认80
- target_gap：记录首次达到的几何目标阈值，默认10⁻⁶；不提前终止常规轨迹
- hand_rate、hand_momentum：HB/NAG同参数手算组，默认1/16、1/2
- stress_rate、stress_momentum：稳定区间反例组，默认0.12、0.9
- library_steps：真实PyTorch常步长对照步数，默认12
- schedule_rates：显式学习率计划，默认1/16、1/32、1/32、1/64
- dampening：仅用于库特殊首步测试，默认1/4

GD平衡参数、HB二次谱参数、NAG标准强凸参数从已验证的s推导，未在JSON重复存储可能互相矛盾的L或μ。probes是源代码中固定且明确标记的边界、初速度和浮点诊断状态。输入范围、拒绝规则与输出保护见README和实验指南。
