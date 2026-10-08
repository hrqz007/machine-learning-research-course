# 数据说明

原创独立合成质检数据，共900行。固定种子68021；train450、validation225、test225，ID互不重叠。六个标准正态无量纲输入x0至x5；异常概率sigmoid(1.6*x0*x1+0.9*x2-0.6)，Bernoulli抽样得到y。p_true仅用于事后解释，严禁作为模型输入。完整生成公式与拆分见generate_data.py。

运行python generate_data.py --directory outputs/generated-data可重建原字节。generation.json保存SHA256；common.load_data还与源生成器逐字节比较，防止改数据后改摘要。数据不是实际工厂调查，不能据此声称现实部署效果。
