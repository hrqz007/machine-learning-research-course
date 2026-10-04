# 039 优化实现与故障诊断

本单元用同一四行原创合成数据定位“损失不下降”的不同原因。先完整手算仿射平方损失的两轮更新，再注入符号、广播和数值稳定性错误。差分检查、实际更新尺度、停止理由与真实计算调用数一起组成故障报告。

直接先修：005、014–015、032。动量或Adam状态故障只在学完034–036后作为选做；本单元未实测这些优化器。

## 文件

- lecture.md / lecture.pdf：连贯推导、完整两轮逐行账本、8幅机制图和16题练习
- lab.md / lab.pdf：可单独执行的实验与验收指南
- answers.md / answers.pdf：全部16题的完整解答
- experiment.ipynb：真实内核顺序执行后的Notebook，保留文本和PNG输出
- experiment.py：主实验、显式输入守卫、故障注入、成本与原子报告写入
- audit.py：独立Fraction、Decimal、逐运算舍入、差分、输入破坏与失败计数核验
- library_check.py / library-result.json：真实固定NumPy/SciPy语义对照与结果
- build_assets.py / figures/：从验证后的输入重新生成原创图
- data/：CSV、完整配置与字段字典
- experiment-result.json：默认输入的完整可解释故障报告，保留每一行和失败候选
- test-result.json：作者数值、拒绝路径与多进程核验结果
- source-checks.json：实际检查的一手来源、相关结论及范围
- verification.json：公开文件白名单、bytes与SHA-256以及本地验收范围

## 快速开始

使用Python 3.12，推荐新建虚拟环境后安装requirements.txt。普通脚本运行并不需要启动Notebook。

```text
python -m pip install -r requirements.txt
mkdir results
python experiment.py --output results/report.json
python audit.py --output results/audit.json
python library_check.py --output results/library.json
python -O audit.py --output results/audit-optimized.json
```

脚本按自身位置读取默认数据，与当前工作目录无关；输出父目录需预先存在。Windows激活环境方式与Linux不同，详见lab.md。environment.yml提供可选conda配置，但本次未实测Anaconda安装。

对于新数据，复制CSV与JSON并显式指定：

```text
python experiment.py --data my_rows.csv --config my_config.json --output results/custom.json
```

程序不合并隐藏默认配置。它会拒绝覆盖单元源码、教学输入和已存在交付文件；指定输出需为常规文件路径。输出失败保留旧文件并清理临时写入。

## 默认应该看到什么

参数顺序为[b,w]。首轮更新到[0.25,0.375]，第二轮到[0.28125,0.40625]，二轮损失为0.025146484375。初始损失0.25，第一轮0.02734375。正确最优点为[0.3,0.4]，目标1/40。

默认正确路径25次提交后达到梯度阈值；广播错误路径30次提交后达到它自己的梯度阈值。符号和大步长路径第一候选被拒绝；极小步长路径被标为“梯度仍大时停滞”。这些数值属于本目录默认输入，不能套到自定义配置。

广播残差形状为(4,4)，每次16个损失元素；正确残差为(4,)，每次4个。广播函数与自身导数的差分检查会通过；因此还要核对预定目标和行配对。每条路径的最终参数另用原配对目标重新评价。

二元logit稳定性实验沿用同一四行，明确更换损失。它比较朴素指数、先算概率再取对数、稳定softplus后相减和带符号softplus。正负极端有限logits由NumPy2.3.5、SciPy1.17.0与1100位Decimal独立对照。库版本不匹配时library_check.py明确拒绝。

## Notebook契约

第一格先校验data/observations.csv、data/experiment_config.json、experiment.py、audit.py、library_check.py、build_assets.py的完整字节摘要，随后才导入数值/绘图库和读取输入。在新内核中顺序运行全部代码格，不跳格，不沿用旧变量。

若第一格提示文件改变，停止展示默认结论；恢复原文件或在另存的脚本试验中使用新配置。不要只修改hash绕过保护。摘要一致性不等于数字签名或科学正确性。

Notebook每幅图都从当前运行结果重新绘制为PNG，不只显示Figure对象文本。最后一格保存严格JSON报告到notebook_results/report.json；此目录是运行产物，不在发布白名单内。输出报告中的非有限故障值用“NaN”“Infinity”等字符串明确标记，稳定结果仍为普通有限数。

## 实际验收与边界

作者在Linux CPU、Python3.12.14下实际运行普通与-O模式，并从单元外的新工作目录重跑。Notebook在新Python进程中的真实ipykernel InProcessKernel内从清空状态顺序运行；这不是静态检查，也不声称测试了Jupyter浏览器UI、套接字传输或Anaconda安装。

审计包括：两个完整Fraction更新；32组固定种子二进制有理数据；120位Decimal多步参照；7168个逐运算标量检查；中央差分扫描与非光滑反例；43种原始输入拒绝；6种严格JSON拒绝；6个-O新进程最后字段破坏测试；写入失败保护；失败后已发生计算仍计费；合法内部状态不受窄初值范围误拒绝。Notebook六个契约文件分别进行了真实首格篡改测试。

三份PDF按程序读取的真实页数逐页渲染并逐页查看；Notebook所有实际PNG单独查看。verification.json列出最后冻结时的数量、文件大小与哈希。作者自检不代替课程协调者的独立验收；本单元没有执行远端发布。

有限合成试验不能证明任意输入、复杂非凸模型或真实泛化表现。稳定函数的入口约定为有限实数与二元标签，不声称覆盖软标签、带权BCE、非有限log-sum-exp或GPU混合精度。
