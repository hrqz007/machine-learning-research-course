# 数据字典

全部为本课程生成的合成数据，不含真实个人信息。生成种子5300至5305；每份600行独立样本，10个标准正态输入。y=2x0+1.5(x1²−1)+x0x2+ε，ε独立标准正态。x3至x9为纯噪声。无缺失值或类别列。

- draws.npz：X0至X5，形状600×10；y0至y5，形状600。行0至239开发，240至599测试。
- protocol.json：预先固定的候选域、搜索政策、资源与划分规则。
- 折归属由KFold(3, shuffle=True, random_state=531)作用于240个开发行生成；每次拟合完整保存train_ids、valid_ids，足以逐行复核。

generate_data.py默认输出到outputs/regenerated-data，不覆盖交付数据。比较再生数组使用np.array_equal，而不是默认把ZIP封装字节是否相同当唯一科学标准。experiment-result.json记录实际使用的draws.npz SHA256。
