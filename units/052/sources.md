# 第052讲来源与核对范围

核对日期：2026-10-05。全部课程机制、数据、表、手算、图和实验数字为原创。以下原始资料用于定义与API核对，无逐段翻译或复制外部实验结果。

1. Rubin, D. B. (1976). Inference and Missing Data. Biometrika 63(3), 581–592. DOI：https://doi.org/10.1093/biomet/63.3.581 。作者机构收录的摘要与书目信息：https://dash.harvard.edu/entities/publication/73120378-8764-6bd4-e053-0100007fdf3b 。本次核对摘要与书目，不宣称逐页复核论文完整证明；本讲没有把MAR等同于任意插补法有效。
2. Stef van Buuren, Flexible Imputation of Missing Data，作者公开源第1章“Concepts of MCAR, MAR and MNAR”：https://github.com/stefvanbuuren/fimdbook/blob/master/Rmd/01-introduction.Rmd 。用于核对缺失机制相对于已观测信息的含义、MCAR/MAR/MNAR区别与插补局限。作者静态网页本次抽取不完整，转而读取作者公开书源，不据空页面声称完成核查。
3. scikit-learn 1.8 Common pitfalls：https://scikit-learn.org/1.8/common_pitfalls.html 。用于检查处理作用域与监督选列泄漏；本讲B使用自己生成的数据和回归目标。
4. SimpleImputer：https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html 。median、keep_empty_features行为另由真实1.8.0库全缺列测试确认。
5. StandardScaler：https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html 。训练均值、ddof=0方差与零方差scale约定；另手工求和核验。
6. OneHotEncoder：https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html 。训练词表、handle_unknown='ignore'与保留全部类别；真实未知类别测试确认。
7. Pipeline：https://scikit-learn.org/1.8/modules/generated/sklearn.pipeline.Pipeline.html 。中间步骤训练与变换；真实TargetEncoder Pipeline预测与显式路线对齐。
8. TargetEncoder：https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.TargetEncoder.html 及用户指南 https://scikit-learn.org/1.8/modules/preprocessing.html#target-encoder 。核对fit_transform内层cross-fit、fit后transform差异、完整训练映射用于验证、cv整数及连续目标KFold。每个编码由独立计数与手工折重建核验，不只引用文档。
9. Target Encoder's Internal Cross fitting：https://scikit-learn.org/1.8/auto_examples/preprocessing/plot_target_encoder_cross_val.html 。用于理解训练自身标签过拟合与外部评价的关系；本讲C数据与数字独立生成。

版本化URL优先于随时间改变的stable。来源说明与代码参照各有用途：文档说明语义，独立计算检查实现；二者都不证明当前合成机制代表某个现实部署总体。
