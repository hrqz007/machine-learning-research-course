# 来源与使用范围

本讲的解释、手算、数据与代码为原创教学材料。下列一手来源用于核对定义与研究出处；没有下载任何外部实验数据。source-checks.json记录真实访问结果。

- [PPCA author publication page](https://www.microsoft.com/en-us/research/publication/probabilistic-principal-component-analysis/)：original publication and PCA relation；实际访问状态 readable。
- [FactorAnalysis official API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html)：diagonal noise covariance and loading shape；实际访问状态 readable。

官方stable文档可能随后更新，运行依赖以requirements.txt与verification.json实际版本为准。原论文若只核验出版元数据，不声称已通读或下载全文。

- [PPCA原论文PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ppca-jrss.pdf)：已实际读取并视觉核对第3至5页的生成模型、观测协方差、后验、载荷正交旋转和残余特征值噪声估计。原文使用t表示观测、x表示隐变量，本课对应改记为x、z，含义保持。未将论文副本打包到课程。
