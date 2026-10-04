# 第008讲 函数代数与数学表达

这是一套零基础数学自学单元。直接先修只有第003讲 Python执行与基本运算。核心问题是：公式怎样准确描述输入输出，怎样判断一句数学结论成立？

## 阅读顺序

1. 读 lecture.pdf（另附lecture.md），完成其中手算和15道练习。
2. 按 lab.pdf（另附lab.md） 独立完成纸笔部分，再运行 experiment.ipynb。
3. 用 answers.pdf（另附answers.md） 对照推理步骤与成立条件。
4. 保留自己的数学解释及执行报告，不把有限测试等同普遍证明。

## 内容

正文以受限预测模型 f(x)=2x+1 串联函数、定义域、陪域、值域、代数条件、复合、幂与对数、求和乘积、集合与序列、命题量词、反例和归纳证明。11幅彩色示意图均为原创。

数据只有5条合成观测。程序只使用 Python 标准库；不联网、不训练模型、不需要 GPU 或任何付费接口。建议 Python 3.10 或更新版本；制作阶段实际使用 Python 3.12.14。

## 运行

在本文件夹运行 python experiment.py。或者使用 Anaconda 中的 Jupyter 打开 experiment.ipynb，重启 Python 3 内核并 Run All。保持 experiment.py 与 data 文件夹在同层。

生成结果位于 outputs/experiment_report.json 和 outputs/four_representations.csv。原始 data/observations.csv 不会被实验覆盖。再次运行只更新生成输出。

预期：平方差之和 3，精确平均 3/5；复合在输入1分别为9与3；两个量词顺序分别为真、假；五点一致反例在1/2处差105/32；360项对数有限核验、22项定义域边界核验和n从1到100的奇数和精确核验通过。

## 数值与证明边界

小分数使用 Fraction 精确计算。对数用相对容差和绝对容差各1e-12。有限网格通过不是普遍数学证明；实数指数的完整构造和极端浮点范围不在本讲实现保证内。合成设备不代表任何真实系统。

Notebook 已在新 Python 进程中用真实 ipykernel 进程内内核顺序执行，保存输出。构建环境限制套接字，因此未测试浏览器 Jupyter UI、外进程内核传输或学习者本机 Anaconda 安装。source-checks.json 与 test-result.json 提供资料和执行核验记录。
