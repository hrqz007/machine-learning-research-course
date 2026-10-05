# 数据字典与生成权限

全部为合成数据，每机制六份。signal种子5400至5405，null种子5450至5455。每份360行80列。除x2外输入独立标准正态；x2=x0+0.08η，η独立标准正态。signal的y=2x0+1.5x1+ε，null的y=ε，ε独立标准正态。没有真实个人信息。

- draws.npz键为signal_X0、signal_y0等，另有null前缀；X形状360×80，y形状360。
- 行0至159开发；160至359独立确认。四折外层种子540；三折内层种子541。
- 每个内层记录的train_local与valid_local是外层训练子表局部索引，要通过外层train_ids映射回原表。
- 错误对照全局排名只用160开发标签，未使用200确认标签。它对外层评价构成故意泄漏。
- 距离实验使用单独种子5490、4000对独立端点；interaction是四行完整手工真值表。

固定协议在protocol.json。generate_data.py默认输出outputs/regenerated-data，避免覆盖原件。experiment-result.json保存实际输入文件SHA和全部排名、选列、预测。
