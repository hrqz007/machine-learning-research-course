# 原创模型数据与字段字典

chain.json和star.json都是课程人工设计，无抽样、无真实业务数据、无下载。generate_data.py重建同样字节；generation.json列出字节数与SHA256。运行experiment.py时会核对这两份冻结参数文件与生成器一致。

| 字段 | 含义 |
|---|---|
| prior | [P(X0=0),P(X0=1)]；和为1 |
| transition | 行是前一状态0/1，列是后一状态0/1；每行和为1 |
| length | 隐状态节点数；本课4，对应X0至X3 |
| emission | [P(E=1\|X3=0),P(E=1\|X3=1)]；不要求这两个数之和为1 |
| leaves | 星图叶节点数；本课6，对应L0至L5 |
| center_prior | [P(C=0),P(C=1)]；和为1 |
| same_weight | 中心与叶同态时的边势，3 |
| different_weight | 中心与叶异态时的边势，1 |

X表示某台人工设备的低/高负荷状态，0=低、1=高；E表示末端指示器是否亮，0=灭、1=亮。时间步无物理单位，不能当作实际秒数。星图C和Li仅为抽象二元变量，没有被附会为真实人群。势权重无单位，不是条件概率。star.json完整声明星图生成约定，exact_inference.star_factors与其一致。
