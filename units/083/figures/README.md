# 四幅原创科学图

均由plots.py从自有公式和真实run()结果绘制，没有使用外部图片或AI生成图。每图含PNG和SVG；SVG将字体转为路径，便于跨机器显示。重建命令：python plots.py --directory outputs/figures。

1. 01_affine_collapse：仿射复合的函数等价示意，不代表训练轨迹等价。
2. 02_xor_representation：四点输入与固定隐藏表示，标明两个正例重合；不代表恢复真实潜在因素。
3. 03_extension_ambiguity：两种同顶点、不同连续延拓的分数热图，白线为0.5阈值；分数不视作校准概率。
4. 04_piecewise_approximation：固定平方插值及误差，网格实测与已证明区间误差比较；不声称任意目标有同速率。

本次已检查中文、数学下标、坐标标签与图例。图像数据没有手动替换成预期值。
