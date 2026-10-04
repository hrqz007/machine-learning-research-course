# 第010讲 矩阵与线性变换

核心问题：矩阵乘法究竟对数据和空间做了什么？先从非方阵逐格手算，再理解列向量、几何变换、复合、转置、基、旋转与批量预测。先修第009讲；不要求已经学过矩阵、一般方程求解、行列式或秩。

## 学习材料

- lecture.pdf / lecture.md：约1.2万汉字正文，12幅原创教学图，16道练习。
- lab.pdf / lab.md：独立Anaconda与Notebook指南，逐步检查点与排错。
- answers.pdf / answers.md：全部练习详细解答。
- example_report.pdf / example_report.md：核验报告示例。
- experiment.py：离线CPU确定性实验，68项断言与边界检查。
- experiment.ipynb：49个单元，其中23个代码单元，已保留真实执行输出与2个网格图PNG输出。
- data/samples.csv：4条无量纲合成样本，特征顺序x1、x2、x3。
- data/matrices.json：固定矩阵、权重与偏置。
- data/data_dictionary.md：轴、字段、单位和参数说明。
- figures：12幅中文教学图；make_figures.py为可选重建程序。
- outputs（运行时生成）：样本逐行结果、摘要、输入摘要值与测试日志。
- source-checks.json：官方文档与一手教材的核对记录。
- test-result.json：脚本、Notebook、图形与确定性重跑的汇总。
- environment.yml：建议本地环境配置，未实际测试conda创建。

正文、实验指南、详解和报告示例的PDF均已提供；实际核验范围见 verification.json。

## 运行方式

在本讲文件夹打开experiment.ipynb，按顺序运行。最后重启内核并运行全部单元，保存输出。

完整脚本：

```text
python experiment.py
```

可指定另一衍生目录：

```text
python experiment.py --output-dir my_outputs
```

脚本按自身文件位置寻找data，因此从其他工作目录调用也可运行；Notebook默认在本讲目录运行。原始数据只读，重复运行重写指定衍生输出，不改写data输入。

## 已知答案

- A为2×3、B为3×2、X为4×3。
- AB=[[5,-1],[-4,6]]。
- XA转置=[[5,-2],[-1,8],[-1,4],[1,6]]。
- Xw=[0,8,8,12]，加偏置0.5得到[0.5,8.5,8.5,12.5]。
- 先D后H把(1,1)送到(3,1)，先H后D送到(4,1)。
- (3,1)在基(1,1)、(1,-1)下的坐标为(2,1)。
- 45度=π/4弧度；旋转前后(2,1)与(1,3)的内积为5。

## 实际核验与限制

实际运行Python 3.12.14、NumPy 2.3.5、Matplotlib 3.10.8。脚本68项检查通过；Notebook23个代码单元在新Python进程中的真实ipykernel InProcessKernel顺序执行，无错误输出，并保留图形。两次独立脚本进程的衍生文件逐字节一致，原始两个输入摘要值保持不变。

没有实际测试学习者Anaconda安装、浏览器Jupyter界面或外进程套接字内核传输。有限测试不替代正文中的一般代数证明。所有坐标、特征、权重与输出分数为课程合成，不拟合模型、不计算预测性能。

## 可选中文绘图

核心脚本不需要中文字体，Notebook网格图用英文标签。讲义中文PNG已随包提供。make_figures.py优先使用Noto Sans CJK字体，Linux常见探测位置为/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc。其他系统可调整为本机可用中文字体；缺少该字体不会影响核心数值实验。
