# 四幅原创教学图

所有图由plots.py生成，均提供PNG和SVG。PNG适合直接阅读，SVG便于放大。图中中文使用可用的Noto CJK字体；SVG文本保留为文本，查看端需有相应字体或替代字体。

1. 01_activation_moments：解析激活和导数曲线、标准正态输入示意、ReLU理论均值/二阶矩/方差。
2. 02_depth_propagation：初始化随机探针的真实逐层与末端统计；宽度64，三种子中位数，阴影为最小–最大范围，非置信区间。
3. 03_distributions：固定种子8701，深度2/8/24下激活和输入梯度非零部分的log10绝对值直方图；零激活点质量另标，密度以非零条件归一化。
4. 04_training_residual：上半为30条真实Wine的实际SGD训练，下半为独立简化残差探针。两个模型不混为一个实验。

重建：python plots.py --result experiment-result.json --directory outputs/figures

四幅PNG均在制作时打开逐像素视觉检查：中文可读、坐标完整、未发现标签裁切。曲线来源与统计定义见lecture.md；图1是理论图，其余包含实测。未使用论文图片或模型生成位图。
