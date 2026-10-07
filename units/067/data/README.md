# 原创合成数据

种子67021；train 12、interpolation 121、extrapolation 121。id全局唯一，x为输入，f_true为仅评估用潜在真值，y为叠加独立标准差0.18高斯噪声的观测。拟合只使用train的x和y。

生成器generate_data.py保留12位小数。load_data核对摘要及固定生成器字节，避免同时篡改CSV与摘要绕过固定实验验证。所有数据无个人信息，全部本地合成。
