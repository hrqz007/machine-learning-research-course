# 第052讲数据字典

全部数据为课程原创合成数据，无真实个人、设备或商业记录。protocol.json在第一次观察分数前固定；之后未改种子挑结果。generate_data.py只向新空目录写，默认拒绝改交付data。

## 三个机制

A为48设备×6行，group为长度288的设备ID；numeric为12×288×2的实际输入，含NaN；site为12×288的0至3站点类别，模型转换成S0至S3字符串；group_y为12×288标签。latent为污染前数值，effects为12×48设备偏置，eps为标签噪声，missing和contamination为布尔开关。latent与effects不能当模型特征。

B的selection_X为12×200×300的独立标准正态特征，selection_y为12×200独立标准正态标签；无真实预测信号。

C的category为12×360独立均匀类别ID，范围0至119；category_y为12×360独立标准正态标签。类别出现次数随机，不是每类固定三行，也没有同类共享随机效应。外层和内层行折的合理性仅属于该声明机制。

第一轴rep为0至11，各自独立重新抽样；不同方法在同rep共享观察便于配对。同rep外层四折共享训练观察，不能当四个独立数据集。

## 文件

- draws.npz：上述全部数组，allow_pickle=False，无对象序列化
- splits.npz：键为机制_折号_train或valid，整型原始行号
- split_membership.csv：每行明确机制、折、角色和原始row_id；同原始行在不同折会重复出现，这是索引说明，不是新增样本
- group_example.csv：rep0的A输入与y，空白表示数值NaN；不含oracle输入
- encoding_example.csv：rep0的C类别与y，用C前缀显示类别ID
- hand.csv：独立纸笔例，H1至H4训练、H5至H6验证
- protocol.json：三个机制、种子、样本数、常数与报告约定

## 生成与保护

所有抽样使用固定NumPy随机生成器与seed52017。外层B/C使用KFold随机状态52031；C内层随机状态52043。A使用确定GroupKFold。原始抽样顺序由generate_data.py固定，NPZ内部日期固定以支持同版本字节复现。

数值/数据字节哈希保存在上级data_integrity.json。改数据时应另建协议与目录，不可仅重写哈希来消除警报。CSV可只读查看；应用软件自动改换行或数值格式也会触发字节校验。不同库版本可能影响浮点或归档，先核对数组与协议，不能只看文件名相同。
