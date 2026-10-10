# 来源与核验范围

核验日期：2026-10-10（UTC）。以下均为课程作者维护的一手教学资料。通过联网读取实际页面正文核验，不把搜索摘要当已读正文；本课数据、图、完整推导、示例与代码均独立制作。

1. Stanford CS231n，Backpropagation, Intuitions：
   https://cs231n.github.io/optimization-2/
   已核验局部梯度、链式法则、反向累积与向量化梯度的讲解。用于确认方法语境；本课两层矩阵推导和数值例独立推导。
2. Stanford CS231n，Neural Networks Part 3 / Gradient Checks：
   https://cs231n.github.io/neural-networks-3/
   已核验中心差分、双精度、相对/绝对误差尺度、差分步长、不可微折点与确定性要求。采用自己的17参数逐项审计和独立结果，未转载示意图或文字。

本课没有使用外部训练数据。数据生成过程、种子、划分与哈希均在data中。资料支持概念，不提供本次运行数值；数值来源是experiment.py真实执行。检查某个页面不等于核验页面引用的全部二手资料。
