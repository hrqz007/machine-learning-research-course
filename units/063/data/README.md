# 教学数据说明

原创非个人合成数据。生成种子63021；生成规则在generate_data.py。1160行，五个独立角色：train480、stop120、tune160、calibration160、test240。

- id：唯一标识，只核验不预测
- x1：标准正态，连续特征
- x2：[-2,2]均匀数，约18%为NaN
- category：0/1/2无序类别，显式类别掩码
- y：Bernoulli二分类标签

真实生成logit = 1.3 sin(1.8 x1) + 1[x2>0] + category_effect + 0.7 missing - 0.6；category_effect = [-0.9,0.1,0.8]。先生成完整特征与缺失标志，再生成标签，最后遮盖x2。数据不是现实业务效果证据。

生成器重新生成字节及SHA256，load_data同时核验摘要与生成器字节。各数据角色ID不重叠，NaN是已规定的缺失，不是无穷大。
