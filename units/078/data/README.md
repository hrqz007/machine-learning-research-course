# 数据说明

model.json是原创固定教学读数，不是真实观测或随机抽取的研究样本。八个无单位数：0.8、1.2、0.9、1.4、1.1、0.7、1.3、1.0。observations为一维数组；noise_sd=1是观测模型已知标准差；prior_sd=2是零均值正态先验标准差。没有人员、隐私信息或外部下载。

运行python generate_data.py --directory outputs/generated-data可重建相同字节。generation.json记录model.json的字节数和SHA256。采样随机种子写在experiment.py中，不参与数据生成。若修改noise_sd或prior_sd，须同步给实验函数传入这些参数，不要仅改文字元数据。本次固定实验使用默认1与2。
