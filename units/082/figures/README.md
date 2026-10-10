# 五幅原创机制与证据图

全部由plots.py从确定性数据与实际拟合结果重建，不复制外部图表。PNG用于PDF和Notebook，SVG为可缩放源图。

- 01_truth_recovery.png / 01_truth_recovery.svg：合成真值Z与估计标签；颜色编号不要求一致。
- 02_model_selection.png / 02_model_selection.svg：全部候选训练/验证目标与预声明选择。
- 03_responsibility_gap.png / 03_responsibility_gap.svg：真实Wine软责任度与硬q证据缺口。
- 04_stability.png / 04_stability.svg：固定测试点上重启与bootstrap一致性。
- 05_predictive_checks.png / 05_predictive_checks.svg：主GMM固定参数复制与保留观测；非参数后验预测。

命令：python plots.py --directory outputs/figures。中文字体使用本机Noto CJK；若缺失请安装官方字体再渲染。
