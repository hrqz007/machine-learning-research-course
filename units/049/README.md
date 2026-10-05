# 049 学习理论的第一组保证

阅读顺序：**lecture.pdf → lab.pdf → answers.pdf**。正文详细展开固定规则、有限类同时控制与ERM保证；实验手册可独立运行；22题答案包含逐行计算及前提诊断。

## 交付内容

- 八状态精确总体、五个固定规则；六行样本与追加整批后的八行完整前向/损失/选择/下一前向。
- 固定模型Hoeffding、union bound、一致收敛到2ε ERM推导，PAC与VC维概念入口。
- 小n有理数精确计数组合枚举，与4096个有序序列独立核对。
- 七个n共14000次IID重复，完整类别计数和结果保留。
- 6000次纯噪声重复、嵌套候选、解析最小值分布、错误使用单模型界与正确同时界。
- 独立留出评价和复制依赖反例，负面结果、零观测事件与不确定性明确保留。
- 十张原创真实图，`learning_guarantees.ipynb`含真实输出；源码、数据、环境和离线PDF构建工具齐全。

`reference_results.json`为完整默认科学结果，`reference_audit.json`为可复跑作者数值参照，`verification.json`为作者重建范围说明；它们不是独立QA签名。外部原文只提供链接，不打包复制论文。

## 运行

使用Python 3.12隔离环境安装`requirements.txt`；固定数值版本为NumPy 2.3.5、SciPy 1.17.0、scikit-learn 1.8.0、Matplotlib 3.10.8。

```bash
python experiment.py --out outputs/result.json
python audit.py --out outputs/audit.json
python -O experiment.py --out outputs/result-optimized.json
python -O audit.py --out outputs/audit-optimized.json
python plots.py --report outputs/result.json --directory outputs/figures
```

脚本以自身位置读取数据，可用绝对路径从不同cwd执行。默认四个数据文件均有字节契约。`experiment.py`可接受`--data-directory`及`--config`研究新协议；`audit.py`和十图绑定默认教学例子，不把旧解析结论套在任意自定义数据上。

## Notebook与完整重建

首个物理单元先用标准库校验15项数据/源码/环境摘要，再导入科学代码。Notebook须保留完整单元，可在根或其`outputs/`子目录运行。修改源码后需重新审视科学依据，不能仅更新摘要就宣称旧验证继续成立。

```bash
python build_notebook.py
python execute_notebook.py learning_guarantees.ipynb
```

请在课程副本中重建以保留原输出。验证使用新Python进程里的InProcessKernel，逐格执行并保存输出；未声称验证浏览器或进程外Jupyter传输。相邻同通道stdout有时可能分块不同；完整文本连接后比较，PNG逐字节比较，原Notebook不为凑相同而改写。

直接读PDF不需要额外工具。重建PDF时安装`build-requirements.txt`、官方Node.js、`package.json`固定的MathJax3.2.2、Pango及Noto Sans/Serif CJK字体。只处理可信教学Markdown，不是通用不可信HTML服务。

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
PYTHON=python bash build_all.sh
```

`build_all.sh`将临时文件和缓存限定在单元的`outputs/`，重新生成科学结果、普通/-O审计、图、Notebook与三PDF，不覆盖交付原PDF/CSV/图。固定CSV本身完整提供，随机实验由固定协议重新生成。PDF元数据可导致文件字节不同，应同时检查每页像素。

## 合理输入域与失败行为

- 总体为声明的四个输入、两个标签的八行，顺序固定，概率分子为0至10000整数、共同分母1至10000，质量和必须为1。
- 假设为1至16个不重复的四位二元向量；标签/错误数要求整数而不是bool。完整ID与所有数据列在主计算前验证。
- 手算样本含追加批次总计最多1000行；输入仅−2、−1、1、2，标签仅0/1，并列取第一个。
- 边界函数支持n=1至1000000、M=1至1048576、δ在[1e−12,1−1e−12]；零一偏差ε在[0,1]。样本量充分公式目标ε在[1e−6,1]，计算出的充分样本数可大于模拟允许规模。
- 精确枚举n=1至8。主模拟每个n至10000、每种2至10000次重复，最多12个n；噪声示例n=2至200、候选至2048，每个计数矩阵最多500万项；留出n至2000。
- 数值拒绝非有限值、复数、布尔、字符串及非零绝对值小于1e−100的原始标量。不宣称处理任意超大计算或任意精度实数。
- JSON先完整计算/序列化，再原子替换；拒绝覆盖教学/给定输入文件、符号链接或多硬链接输出。输入错误或写入替换失败应保留旧JSON。图与PDF工具不声明多文件事务。

本讲100位Decimal用于解析最小值公式求值；小n枚举则是精确整数/有理数。两者与2000次Monte Carlo频率分开命名。极小指数概率可能在浮点中下溢，尾界同时保留对数；不得把数值零当成理论零。

## 已知范围

保证假设的是IID观测与有界损失，不是任意相关记录、时间漂移或无界对数损失。有限候选一致界容许在事前固定类内数据依赖选择；不容许事后把已筛选类改写为单元素类。VC模式展示只是概念入口，不是无限类一般定理的完整证明。全部资料与核对范围见`sources.md`。
