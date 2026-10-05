# 044 广义线性模型

本单元回答：响应不是连续正态变量时，线性建模怎样扩展？直接先修020、023、029、042至043，保持既定166单元大纲。

## 学习材料

- lecture.md / lecture.pdf：20节正文，8幅原创计算图与逐幅读图问题，20道练习
- lab.md / lab.pdf：独立实验指南，从手算、运行、核验到曝光量与离散度解释
- answers.md / answers.pdf：20题逐步详解与固定实验数值
- experiment.ipynb：17个真正执行的代码格、8张实际PNG输出、完整三状态行级账本
- experiment.py：Poisson目标、梯度、Hessian、保护Newton、三模型两情景完整评分
- audit.py：80位Decimal标量概率参照、解析有限MLE、差分、实际SciPy/sklearn核验与针对性输入检查
- plots.py：从真实科学结果重建8图，默认写outputs，不覆盖交付图
- generate_data.py / data/：固定模拟协议、三行手算与560行合成CSV、数据字典
- data_integrity.json / input_contract.json：固定输入与Notebook核心代码完整性契约
- requirements.txt / environment.yml：科学计算与Notebook依赖
- build_pdf.py / mathjax_render.cjs / pdf.css / build-requirements.txt / package.json：自包含PDF构建入口
- execute_notebook.py：受限环境中的真实IPython新核执行入口
- source-checks.json：官方来源、使用范围与实际执行版本的区别

PDF最终页数为正文14、实验5、详解7，共26页。作者已逐页查看实际渲染图。作者制作、数值核验和独立课程验收是不同阶段；这里不声称已完成独立验收或远端公开发布。

## 一条可逐行核对的主线

参数顺序[b,w]，x=(-1,0,1)，y=(0,1,3)，曝光量全部为1。F为完整平均Poisson负对数似然，保留log(y!)。学习率0.2，零参数，无正则。

- 状态0：beta=(0,0)，F=1.5972531564093515，g=(-1/3,-1)
- 状态1：beta=(1/15,1/5)，F=1.3916034664721781
- 状态2：beta≈(0.116685493543,0.371304543132)，F=1.246373395442681

每一状态均有三行eta、mu、loss、局部导数、Jacobian、平均梯度贡献及Hessian贡献。两次更新都先整批汇总再同步更新。第二次更新后重新前向；它还没有收敛。

独立令t=exp(w)、a=exp(b)，两条得分方程推出t²−3t−7=0，正根给唯一有限MLE约(-0.364917136586,1.513231209206)。保护Newton实际6次更新与该代数解一致。完整MLE存在性证书只适用于这个手算例，未实现任意设计的一般存在性判别。

## 固定模拟与不掩盖的失败

种子4402026；320训练、240测试；x均匀于[-2,2)，曝光量e均匀于[.5,2)，真率exp(.2+1.1x)。Poisson情景与Gamma-Poisson情景均保持该条件均值；后者条件方差为μ+μ²/2。

三模型为Poisson加log(e) offset、Poisson遗漏曝光量、曝光校正线性均值e(b+wx)。全部无正则；协议未根据测试表现修改。线性模型在两个情景分别产生53、56个非正测试预测，完整保留；全测试Poisson deviance记为未定义，未删行或夹断。

Poisson情景：带offset测试MSE约3.758404，线性约6.857005。混合情景：带offset约17.081238，线性约17.490457，差距较小也保留。训练Pearson离散度分别约1.013523和2.355146。原始样本方差均值比还包含不同条件均值的变动，不把它当条件过度离散证据。

曝光加权率拟合用真实sklearn PoissonRegressor(alpha=0, solver='newton-cholesky')，与原始计数目标只差常数与正尺度；核验参数和代回计数目标的梯度，不直接把两个接口显示的目标值当成相等。

## 运行

在本目录的Python3.12环境中：

```bash
python experiment.py --out outputs/result.json
python -O experiment.py --out outputs/result-O.json
python audit.py --out outputs/audit.json
python -O audit.py --out outputs/audit-O.json
python plots.py --report outputs/result.json --directory outputs/figures
python generate_data.py --out outputs/regenerated.csv
```

Notebook在Jupyter中使用Restart Kernel and Run All。若环境不支持socket：

```bash
python execute_notebook.py --out outputs/executed.ipynb
python -O execute_notebook.py --out outputs/executed-O.ipynb
```

每次应从新Python进程启动；执行器建立真实InProcessKernel，17格按顺序执行。首物理代码格只用标准库验证8份输入/实现文件，之后才导入数值包。该工具不验证浏览器Jupyter或网络传输。

PDF重建另需Node.js、系统Pango/中文字体，按build-requirements.txt安装Python构建依赖，在本目录npm install安装MathJax后：

```bash
python build_pdf.py --directory outputs/pdfs
```

构建只读本单元Markdown与PNG，不依赖其他讲的文件。只做科学实验不需要安装PDF依赖。

## 支持域与核验范围

数学Poisson模型的支持域与本教学实现的binary64工作范围不同。state入口接受实数设计、参数与曝光，拒绝布尔、字符串、复数或非有限值；n为1至100000、列数1至20、abs(X)≤10、整数y在[0,10^6]、e在[10^-6,10^6]、完整eta在[-30,30]。fit入口还要求第一列全为1作为显式截距、满列秩、至少一个正计数。候选超界会拒绝，不夹断eta。必需检查不依赖assert。

作者实际核验包括三状态80位Decimal参照、解析有限MLE、3个非驻点×3种差分步长、Hessian二次型、SciPy logpmf、sklearn deviance与两情景实际拟合、曝光单位不变性、零与饱和值、固定数据重生及代表性错误输入。没有用极端用例数量充当教学深度。

实际运行版本为Python3.12.14、NumPy2.3.5、SciPy1.17.0、sklearn1.8.0、Matplotlib3.10.8、nbformat5.11.1、ipykernel7.4.0。普通/-O科学JSON与作者核验JSON一致；新进程普通/-O Notebook逐格输出一致且与科学脚本结果一致。启动时环境提示无法发现网络接口，不影响InProcessKernel执行；未把它描述为浏览器或socket验证成功。跨平台末位与图文件元数据可能变化，判断以科学定义和明示容差为准。

合成数据只验证机制，不证明现实泛化、因果效应、推断区间或高影响应用安全。均值关系正确不能补救Poisson条件方差错误。来源仅用于概念与接口核验，具体文字、数据、图和例子重新制作。
