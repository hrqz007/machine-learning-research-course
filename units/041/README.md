# 041 回归与优化阶段项目

本项目回答：怎样证明优化实现和回归结论可信？新四行中心化数据贯穿完整两轮更新，接着用独立解析/库参照、统一梯度残差、真实成本和失败记录检查数值结论，再用训练/验证锁与区间压力试验检查预测结论的边界。

直接先修：007、025、029、032、034、036、039、040。全课程范围166单元，本单元是阶段项目。全部教学数据原创合成，不包含私人记录或第三方论文复制。

## 完整文件

- lecture.md / lecture.pdf：完整推导、两轮逐行账本、12幅原创解释图和16题
- lab.md / lab.pdf：独立环境、运行、实验步骤和验收指南
- answers.md / answers.pdf：全部16题完整答案
- experiment.ipynb：新内核顺序运行的Notebook，含实际文本与PNG
- experiment.py：确定性实验、输入契约、选择隔离、成本与原子输出
- audit.py / test-result.json：Fraction、库、随机梯度、泄漏、拒绝路径与多进程测试
- benchmark.py / benchmark-result.json：独立真实计时，含暖机、全部重复和失败
- build_assets.py / figures/：由实际运行对象构造的12幅原创图
- data/：四行、压力与选择数据，完整配置、字典及生成机制
- requirements.txt / environment.yml：固定依赖与可选conda配置
- experiment-result.json：完整确定性参考报告，保留全部轨迹和600次覆盖重复
- source-checks.json：实际检查一手来源及使用范围
- verification.json：公开文件白名单、字节、SHA-256与实际验收范围

## 快速开始

Python3.12。先建虚拟环境并安装requirements.txt，建议将结果存到单元目录外的新文件夹。输出父目录必须预先存在。

```text
python -m pip install -r requirements.txt
mkdir ../results041
python experiment.py --output ../results041/report.json
python audit.py --output ../results041/audit.json
python benchmark.py --output ../results041/timing.json
python -O audit.py --output ../results041/audit-optimized.json
```

默认输入相对脚本位置定位，因此换工作目录可重跑。自定义数据使用 `--data-dir`，自定义配置使用 `--config`，不与隐藏默认值合并。保护范围包括输入、源码、PDF、Notebook、图及交付报告；拒绝符号链接、符号链接父目录和多硬链接输出。Notebook的notebook_results是专用可替换运行目录，不在公开白名单内。

## 当前数据的实际结果

参数顺序[w1,w2]。两轮Ridge更新为[3/8,5/8]和[29/64,49/64]，总目标依次7/4、29/64、1597/4096。OLS参照[1/2,1]，Ridge λ=1/2参照[5/11,9/11]。Fraction独立逐项核验。

五条求解路线×三个病例×两个λ共有30条标准路径，18条达到完整梯度相对残差1e−6，12条预算耗尽。另有unsafe步长首候选被拒绝，0次提交仍保留实际成本。不能删掉未收敛结果再比较平均耗时。

验证选λ=0.1，锁定后测试MSE约0.775054；预声明OLS基线约0.767754。本次测试并未支持验证所选Ridge胜过OLS，原结果保留。

600次独立OLS覆盖模拟：均值区间580次、同分布新观测预测区间562次、错用均值区间预测247次、未来噪声SD偏移至1.5时321次。该t区间不用于选择后的Ridge，不提供因果结论。

## 确定性与实际计时

experiment-result.json不含墙钟计时；普通、-O和不同cwd完整JSON须相同。benchmark-result.json使用真实perf_counter_ns，每配置一次记录暖机和三次正式测量，BLAS单线程；时间应随机器与运行波动，不要求字节一致。31个配置包括失败运行。

计时含诊断求解内部Gram、两个谱求解、参照求解、监测、轨迹存储及失败尝试；不含输入加载/中心化、模型选择、覆盖模拟、画图、文件写入或解释器启动。它不是生产实现或端到端基准。

## Notebook与验证范围

Notebook首格先校验数据及执行模块全部字节摘要，之后才导入数值/绘图库。真正运行全部科学实验、快速审计、实际计时与12幅PNG。输出PNG来自字节缓冲区，未以Figure对象文本冒充图形。

实际执行使用Linux CPU、Python3.12.14及requirements.txt所列版本。Notebook使用新进程中的真实ipykernel InProcessKernel，从空状态顺序运行；未声称Jupyter浏览器UI、内核网络传输或Anaconda安装已测。作者检查包含所有PDF页面及Notebook实际PNG，真实页数、测试数量及哈希见verification.json。

输入守卫覆盖尾字段、重复JSON键与ID、错维度、bool/string/complex、NaN/Inf和过小非零原值，包括NumPy longdouble转换成0的情形。合法零、范围内NumPy实数、单轮预算和手算驻点初值分别测试。输出原子失败保留旧字节；本地单用户保护不等同抵御恶意并发文件系统攻击。

这是本地完整交付与作者验证记录，未执行远端发布。作者自检不取代课程协调者的独立验收。
