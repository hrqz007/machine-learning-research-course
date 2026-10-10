# 数据字典与来源

本课原创确定性数学核验数据，不含外部样本或个人信息。CSV以CC0提供；代码许可遵循仓库统一规则。

gradient-cases.csv含case、x、y三列与5个预声明核验点：(0.7,-1.2)、(1.1,0.3)、(-0.9,0.8)、(0,-0.5)、(2,-2)。case只是可读标识，x、y是复合函数(x*y+sin(x))*exp(y)的输入，不是待训练标签。它们覆盖正负与零等情况，但不代表穷尽数学定义域。

运行python generate_data.py --directory outputs/generated-data重建；无随机种子需求，无外部下载。固定点用于微分实现核验，不需要训练/验证/测试拆分，也不报告泛化指标。字节SHA256保存于experiment-result.json。
