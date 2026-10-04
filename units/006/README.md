# 第006讲 读表清洗与可视化

围绕“怎样从原始表看清分布与记录问题，而不偷偷改变研究对象”完成一个可审计的表格探索。适合已经读过第002、004、005讲的初学者。所有记录与图均为课程合成；不需要网络、API、账号、GPU或付费服务。

## 学习顺序

1. 阅读lecture.pdf（另附lecture.md），先手算重复、合并行数、分母和分位数。
2. 按lab.pdf（另附lab.md）打开experiment.ipynb，从顶部运行23个代码单元。
3. 完成15道练习，再对照answers.pdf（另附answers.md）。
4. 运行experiment.py检查完整流程，按报告要求写自己的探索结论。

## 文件说明

- lecture.pdf / lecture.md：正文、10幅图和15道练习。
- lab.pdf / lab.md：独立Anaconda与Notebook操作指南、检查点及排错。
- answers.pdf / answers.md：全部练习的详细解答。
- example_report.pdf / example_report.md：完整探索报告示例，便于对照自己的报告。
- experiment.py：离线清洗、合并、汇总、训练填补演示与43项断言。
- experiment.ipynb：逐步教学，已保留23个代码单元的真实执行输出与2个PNG图形输出。
- data/raw：4个原始CSV，运行脚本不会覆盖。
- data/data_dictionary.md：单位、身份、缺失约定与合成构造说明。
- outputs（运行时生成）：衍生表、清洗日志、冲突、决策、摘要与测试结果。
- figures：10幅已生成的中文教学PNG。
- make_figures.py：可选的讲义图片重建程序。
- source-checks.json：官方文档核对日期、版本与使用范围。
- test-result.json：脚本与Notebook核验汇总。
- environment.yml：建议的本地Anaconda环境配置，未在构建环境实际执行conda创建。

PDF讲义、实验指南、详解与报告示例均已随包提供；实际检查范围见 verification.json。

## 运行

在本讲文件夹中打开终端：

```text
python experiment.py
```

可用另一衍生目录保留不同实验：

```text
python experiment.py --output-dir my_outputs
```

脚本从自身所在位置寻找原始数据，所以从其他当前目录调用也可运行。Notebook则应在本讲文件夹打开并以该文件夹为工作目录。重复运行会重写指定的衍生输出目录，不会改写data/raw。

## 已核验结果

- 原始16条记录，移出2条完整复制后14次事件，来自12套房屋。
- 面积12个有效观测、2个缺失；价格14个有效观测。
- 正确多对一左合并保持14行，13行匹配、1行未匹配。
- 故意错误合并产生19行，均价从2888/14变成3576/19。
- 训练均值只用9个已观测面积，670/9≈74.4444；全表均值1220/12仅用作泄漏反例。
- 43项脚本断言通过，逐步Notebook23个代码单元全部通过。
- 原始CSV的SHA-256摘要值运行前后相同。

## 实际测试环境与边界

实际运行Python 3.12.14、Pandas 2.2.3、NumPy 2.3.5、Matplotlib 3.10.8。Notebook使用新Python进程中的真实ipykernel InProcessKernel顺序执行，保留表格、文本与PNG图形输出。没有实际测试Anaconda安装、浏览器Jupyter界面或外进程套接字内核传输。这里不报告任何模型性能。

关键表与JSON输出在两次独立脚本进程中逐文件比较一致；脚本不使用随机数，因此不需要用随机种子掩盖数据来源。全部输入由固定文字明确构造。

核心脚本与Notebook图形使用常规英文字体即可运行。可选make_figures.py为讲义中文图优先加载Linux常见的Noto Sans CJK字体，其探测路径为/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc。其他操作系统可以安装已有可用的中文字体并调整该字体设置，或直接使用随包PNG；缺少中文字体不影响核心表格计算。该系统字体路径仅说明可选绘图依赖，不是学习者的数据目录。
