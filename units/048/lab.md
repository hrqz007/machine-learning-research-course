# 第048讲实验指南 泛化误差与复杂度

## 1 实验目标和路线

本实验让你亲手区分训练拟合、训练数据引起的预测变化、独立测试评估波动和数值求解。先完整重算三行数据，再穷举八种标签，最后读取固定协议生成的400次训练数据，画复杂度曲线和学习曲线。代码直接解最小二乘；本讲不需要梯度更新轮数。

建议先读lecture的第2至10节，纸笔工作约45至75分钟；数值实验和读图约60至90分钟；研究型解释约30分钟。耗时因熟练程度不同，不是机器运行承诺。只阅读三个PDF不需要任何软件；要重新计算，需要Python和本目录中的数据文件。

交付文件相互独立：lecture.pdf解释原理，lab.pdf给运行和验收路径，answers.pdf包含20题完整解答。experiment.ipynb是真Notebook，保留顺序执行输出；experiment.py是同一计算核心。图由plots.py实际生成，PDF不是手画曲线截图。

## 2 文件地图和不可改的比较协议

|文件或目录|用途|
|---|---|
|data/config.json|预声明种子、重复数、n、p、噪声及真系数|
|data/draws.npz|主实验全部原始输入、噪声和独立测试抽样|
|data/hand.csv|三行纸笔数据|
|data/enumeration.csv|八种等概率标签噪声|
|data_integrity.json|数据字节SHA256|
|experiment.py|风险、拟合、全枚举和主实验|
|audit.py|分数精确参照、独立求解器和不变量检查|
|plots.py|十幅图的全部绘图代码|
|experiment-result.json|所有10800次拟合的系数、风险和诊断|
|test-result.json|作者科学自查证据|
|build_notebook.py 与 execute_notebook.py|重建与真实新核执行Notebook|
|build_pdf.py 与配套文件|可信Markdown的离线公式及PDF构建|

种子48019在本次教学协议内固定。p是参数个数，最高多项式次数为p−1。不得为了得到平滑U形图反复换种子；不得从结果中删去大风险重复。想研究另一种协议，应复制到新目录、明确新问题并保存两组结果，不覆盖本讲数据。

## 3 环境准备

建议创建隔离环境。下列命令在本单元目录执行，Python版本建议3.12。数值环境和可选排版环境分开，运行实验不需要WeasyPrint或MathJax。

```bash
python -m venv .venv
# Linux 或 macOS
source .venv/bin/activate
# Windows PowerShell 对应 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py --report outputs/result.json --out outputs/audit.json
```

本次实际执行为Python3.12.14、NumPy2.3.5、SciPy1.17.0、scikit-learn1.8.0、Matplotlib3.10.8、nbformat5.11.1、ipykernel7.4.0。requirements.txt锁定数值版本；JupyterLab范围只用于可选交互界面，本次未验证浏览器Jupyter界面或网络传输。平台BLAS和字体变化可能造成末位浮点数或排版差异。

受限环境可以将TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME、IPYTHONDIR设为自己可写的临时目录。本次所有缓存和运行临时文件使用单元私有目录，没有修改共用依赖。不要把机器特定的绝对路径写进你的作业。

## 4 数据字典和随机性地图

主实验的数组x与eps均为400×256：第一维是独立训练重复，第二维是该次训练样本。输入均匀分布在[−1,1]，eps是标准差0.35的独立正态噪声。标签不单独存储，而是按固定真函数加eps生成，以便追踪每项来源。

xt与et也是400×256，但属于独立测试集；每个训练重复对应一份测试集。eps_fixed为400×40，只用于固定训练网格。测试输入、测试噪声、随机训练输入、随机训练噪声和固定设计噪声按同一个公开种子的连续独立伪随机抽样生成；伪随机种子保证复现，不是数学独立性的实验检验。

- 复杂度曲线：每次取训练前40行，p=1至9
- 学习曲线：p=2、4、8，n=16、32、64、128、256，使用同次前缀
- 固定设计：40个等距点从−0.95到0.95，重抽标签；评价分布仍为均匀[−1,1]
- 纸笔例：训练输入固定在−1、0、1，真函数0，噪声为±1；它与主实验的噪声水平不同

同一次不同p和n共享数据，允许配对比较，但不同横轴点不能被当作独立抽样点。主实验不是从一张现有表bootstrap；它在已知机制下真正重新生成训练样本。现实中的有限数据重采样通常回答条件于现有数据的另一个问题。

## 5 阶段一 逐样本链不跳步

打开data/hand.csv，先不运行Python。用常数预测1/3逐行写出：输入、标签、预测、预测减标签、平方损失、除以n后的贡献。应得到三行贡献4/27、16/27、4/27，总和8/9。

接着写线性模型的设计矩阵A，计算 $A^\top A$ 和 $A^\top y$，但不要在代码里显式求逆。纸笔上得到截距1/3和斜率0。二次模型先解三条插值方程，得到系数(−1,0,2)，列顺序为常数、x、x²。

最后固定这一次拟合结果，以新输入的均匀密度1/2积分。常数信号误差1/9，二次信号误差7/15。只有预测新含噪标签时才再加1，得到10/9和22/15。向同学解释这两个1来自不同表达式中的哪一步，避免把二次多项式平方常数项1与噪声方差1混为同一项。

运行experiment.py后，查看hand.rows：每一行都保存上述计算链；浮点二次训练MSE约 $2.63\times10^{-31}$，应解释为精确0的数值残差，不是新的统计噪声。

## 6 阶段二 八种标签精确枚举

data/enumeration.csv包含全部(−1或1)的三维组合，每种概率1/8。对每种标签重新拟合p=1、2、3，计算新输入上的信号误差，再加一次新噪声1。平均风险必须接近4/3、3/2、9/5。

这里ddof=0是准确概率权重：八个数就是完整等概率样本空间，而不是从某个未知方差总体随机抽来的八个数。若把这八种状态的预测方差改成ddof=1，就会人为放大8/7倍，失去精确期望含义。

audit.py用Fraction做独立分数计算。它不用主实验的Legendre系数平方公式，而是对单项式系数做精确积分：偶次幂 $x^k$ 的均匀均值为1/(k+1)，奇次幂为0。检查test-result.json中的exact_reference，不应只有“程序未报错”四个字。

![实验图A 八种数据下训练误差下降、总体风险上升，且偏差始终为0。请先手算方差后再看代码结果。](figures/02_exact_enumeration.png)

## 7 阶段三 理解一次随机设计拟合

打开主实验协议，明确真系数长度9，拟合p列后补零到9维。A的第i行第j列是 $\phi_j(x_i)$。np.linalg.lstsq返回系数、残差摘要、数值秩、奇异值。代码另外重算残差向量，不依赖库残差摘要在所有秩情形下都有相同形状。

一次结果要同时读四类信息：

1. train：训练标签上的平均平方残差
2. risk：系数差平方和加0.1225，已知机制下该次函数的真实含噪总体风险
3. test：独立256行含噪标签上的平均平方残差
4. condition、normal_residual、rank：训练设计矩阵条件数、平均正规方程残差与数值秩

训练损失小不推出risk小；test与risk不必完全相等；normal_residual小不推出risk小；condition大提示对扰动敏感，但不能直接判定本次求解错误。必须用独立积分和求解器对照。

## 8 阶段四 重复400次再看复杂度曲线

```bash
python plots.py --report outputs/result.json --directory outputs/figures
```

十张PNG由实际结果生成，输出目录不要指定为随包figures。绘图程序拒绝覆盖随包图和符号链接。代码不挑最漂亮的训练曲线：第0次用于单次对照，前20次用于函数束，其选择与表现无关。

先看01_hand_fit与02_exact_enumeration，再看03_complexity。记录p=4和p=9的平均risk、risk_sd_ddof1、risk_mcse、risk_q10、risk_q90，并从runs求最大值。p=9平均risk约0.661216，最大约76.177279。为什么平均比常见的单次风险高？结合07_risk_distribution解释尾部。

所有400个p=9结果都有效、数值秩为9。若自己的重建出现失败，先保留错误日志和输入；不得只对成功结果取均值后宣称重复400次。当前代码遇到秩亏或非有限输入会明确报错，不静默跳过。

## 9 阶段五 ddof和噪声计数验收

查看一个block的summary。确认

$$\text{risk\_mean}=\text{bias2\_mc}+\text{variance\_ddof0}+\text{noise}.$$

这是一条有限400次的代数恒等式，浮点容差内应成立。再检查variance_ddof0 = variance_ddof1 ×399/400。最后把bias2_unbiased_estimate和variance_ddof1相加，应仍配上同一个noise得到risk_mean。

名称中的unbiased只在独立重复及所需矩存在时具有统计无偏解释；这次有限数组上的减法无论如何都可算。随机设计的近奇异事件可能造成很重的尾部，本讲没有证明每个高阶小样本配置的总体风险均值或方差矩条件。因此，不把有限400次平均当已知无限重复期望，不把SD/√400当无条件有效的置信保证。

不要把test再加noise。test已经与新含噪标签比较，噪声已在损失里。risk的系数平方部分与真均值比较，才需加noise。若图里已经显示bias²+variance+noise，也不能再追加“不可约误差”这一相同项。

## 10 阶段六 学习曲线和负面结果

看05_learning_curves。右侧训练曲线随n增加可能上升，这不矛盾：少量数据更容易被匹配；增添更多不同样本后，最小平均残差也可能变大。嵌套类训练风险不增的证明要求同一个D，不能拿来证明改变n后的单调性。

p=8、n=16的平均risk约805.824574、最大约118455.527851。先复核原始系数、rank、condition和独立积分，再讨论采样稀疏性。图使用对数轴，没有截尾或Winsorize；负面结果是实验结论的一部分。

请回答：如果想进一步研究正则化，应该直接把原结果替换为一组更好看的岭回归结果吗？不应。应另立新协议，明确惩罚强度的选择方式、是否惩罚截距、使用的训练验证划分，然后与保留的原结果比较。

## 11 阶段七 Notebook新核执行

交付Notebook已经执行。阅读时按顺序运行全部，不手动跳过报错格。第一格使用标准库验证数据SHA256，第二格才导入计算依赖；后续依次做手算检查、主实验、全量审计、结果表、十张实际图和解释。

```bash
python build_notebook.py
python execute_notebook.py experiment.ipynb
```

execute_notebook.py在新的Python进程启动真实ipykernel InProcessKernel，顺序执行并保存代码输出。它不是把代码文本复制到普通exec后宣称Notebook运行，也不测试Jupyter浏览器界面或独立内核传输。在支持完整Jupyter的环境也可用界面“重启内核并运行全部”，但要保留输出和原始协议。

若只想阅读已交付版本，勿运行build_notebook.py，它会重建Notebook源码并清空先前输出。需要重建时先保留原包；所有执行报错须修复后从新核重跑，不能留下一部分旧输出与一部分新输出混合。

## 12 新进程与不同工作目录检查

```bash
python experiment.py --out outputs/ordinary.json
python -O experiment.py --out outputs/optimized.json
python audit.py --report outputs/optimized.json --out outputs/optimized-audit.json
# 换到其他目录，使用本单元的绝对路径
python /absolute/path/048/experiment.py --out /writable/path/result.json
```

默认输入按脚本所在目录寻找，不依赖当前工作目录。-O会关闭Python的assert，因此输入保护使用显式异常。本次普通、-O与不同cwd结果的规范JSON内容一致；随机数据来自固定随包数组，而非每次临时换种子。

程序防止误写随包教学文件、符号链接和多硬链接，并在严格JSON序列化成功后原子替换输出。它是防误操作约定，不宣称抵御恶意并发文件系统攻击。generate_data.py是重新生成原创数据的构建入口，会改写data与对应hash；日常实验不要运行它。

## 13 可选重新构建PDF

阅读现有PDF无需安装排版依赖。要修改可信课程Markdown并重建，需本机Node.js、MathJax3.2.2、Pango、Noto Sans/Serif CJK字体及以下依赖：

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
```

公式先由MathJax转换为本地SVG，再由WeasyPrint排版，不调用网络公式服务。构建脚本只针对可信课程源码，不接受不可信HTML或TeX。当前环境的XeLaTeX缺少可用格式文件，因此本次没有采用XeLaTeX路线。

交付的build_all.sh串起数值重算、审计、绘图、Notebook构建执行与PDF构建。请先准备好依赖，再运行；脚本不会静默安装软件。若要验证图文一致，PDF重建前应将新图以明确操作替换figures中的对应文件，并保留旧包；默认数值重算不修改交付图。

## 14 验收清单

完成作业应能提供以下证据，而不只是十张图：

- 三行完整逐样本损失表与10/9、22/15的总体积分
- 八种标签的精确风险4/3、3/2、9/5以及对应方差来源
- 主实验10800次结果全部存在，无过滤异常尾部
- 15组NumPy、SciPy枢轴QR、scikit-learn同目标对照及函数积分对照
- 方差ddof约定、新标签噪声只计一次、负修正偏差平方原值保留
- 复杂度和学习曲线标清p、n、重复数及阴影含义
- 普通、-O、不同cwd与Notebook新核的真实执行记录
- 一段边界说明：合成机制、有限重复、矩条件未全面证明、没有现实泛化保证

## 15 读图作业

A. 图1与图2分别固定了什么、平均了什么？为什么柱高不同并不冲突？

B. 图3右侧分位带不是置信区间；它为什么可能掩盖尾部损失？图7如何补充？

C. 图4中p=4的偏差平方很小，是否可据此宣布算法“在所有问题上无偏”？

D. 图5为何使用对数风险轴？在报告文字中漏掉大风险数值会造成什么误导？

E. 图6是点态预测变化，图9是测试评估变化，请分别写出随机变量和条件。

F. 图8两类训练设计不同，能否把差异全归因于“样本量不一样”？两者n都为40。

这些作业的完整解释见answers.pdf的读图补充，20道正文练习也全部在其中。没有标准答案的研究延伸必须说明自己的假设、证据与未确定部分，不应为了与教材趋势一致篡改结果。
