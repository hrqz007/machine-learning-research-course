# 五幅原创机制与证据图

全部由plots.py从确定性数据与实际拟合结果重建，不复制外部图表。PNG用于PDF和Notebook，SVG为可缩放源图。

- 01_mechanisms.png / 01_mechanisms.svg：原创正常混合与桥区/远端异常；原始二维坐标。
- 02_boundaries.png / 02_boundaries.svg：相同干净秩校准下四模型真实边界；红=报警。
- 03_calibration.png / 03_calibration.svg：实际LOF干净校准排序与第190个阈值。
- 04_budget.png / 04_budget.svg：正常子群实测误报及预声明基率情景公式。
- 05_real_negatives.png / 05_real_negatives.svg：真实保留0高分负例与真实任务外6图像。

命令：python plots.py --directory outputs/figures。中文字体使用本机Noto CJK；若缺失请安装官方字体再渲染。
