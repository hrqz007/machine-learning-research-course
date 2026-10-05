# 048 泛化误差与复杂度

完整166单元课程的第048讲，直接先修019、022、027、029。核心问题：训练误差下降时，测试误差为什么可能上升？

## 材料

- lecture.pdf / lecture.md：20节，三行纸笔链、偏差方差推导、误差概念、10张实际计算图与20题练习
- lab.pdf / lab.md：独立实验指南、数据字典、环境、分阶段操作和验收
- answers.pdf / answers.md：20题完整解答、6组读图解释及数值核查
- experiment.ipynb：12个真实顺序执行代码格，包含全部图的实际输出
- experiment.py：固定设计全枚举、随机设计400次重复、10800次拟合的完整结果
- audit.py：Fraction精确参照、15组求解器及积分对照、27组各配置最坏风险核查
- plots.py：十图全部绘图源码，图例置于坐标区域外
- generate_data.py、data/、data_integrity.json：原创数据生成器、固定原始抽样和字节校验
- requirements.txt / environment.yml：数值环境入口
- build_notebook.py / execute_notebook.py：Notebook构建与新进程真实内核执行
- build_pdf.py、mathjax_render.cjs、pdf.css、package.json、build-requirements.txt：PDF构建源码和依赖入口
- build_all.sh：串起完整重建；不自动安装软件
- experiment-result.json、test-result.json、verification.json、source-checks.json：科学输出、作者自查、执行范围和来源

正文15页，实验指南7页，详解7页。作者自查不等于独立课程验收或远端发布。

## 同一三行例子的全链

固定训练输入x=(-1,0,1)，真条件均值0，独立标签噪声等概率取±1。本次标签(1,−1,1)。新输入均匀分布在[−1,1]，新标签噪声独立、方差1。

常数与线性模型预测均为1/3，三行损失贡献4/27、16/27、4/27，训练MSE=8/9，真实含噪风险10/9。二次插值2x²−1训练MSE=0，真实含噪风险22/15。没有欠收敛的梯度过程，训练变好但泛化变差仍可发生。

穷举八种等概率标签后，p=1、2、3的平均风险为4/3、3/2、9/5；偏差平方都为0，预测方差1/3、1/2、4/5。总体风险单调增加，直接说明U形并非普遍规律。p始终为参数数，次数是p−1。

## 主实验和保留的长尾

固定seed=48019，400个独立重复，随机输入均匀[−1,1]，标签噪声标准差0.35。归一化Legendre前9项中真系数(1,.6,.4,−.25,0,0,0,0,0)。同次不同p、n共享训练前缀和独立256行测试数据，不独立对待横轴点，不使用测试选模型。

- 复杂度曲线n=40、p=1..9；p=4平均风险0.136885，p=9为0.661216，后者最大76.177279
- 学习曲线p=2、4、8；n=16、32、64、128、256
- p=8、n=16平均风险805.824574，最大118455.527851，所有大风险原样保留
- 固定40点设计另列，评价分布与随机设计相同，但训练随机性不同
- 风险由正交系数差平方加0.1225精确计算；独立含噪测试MSE不再加噪声
- 有限重复分解使用ddof=0；ddof=1配对修正偏差平方。负修正估计不裁零

本讲没有证明每个高阶随机设计配置的无限重复风险均值或方差矩条件。400次有限均值不等于已知总体期望，样本SD/√400不构成无条件有效的CLT置信保证。中心80%分位带也不是均值置信区间。结果为合成机制研究，不用于现实任务性能承诺。

## 数值复现

在本目录准备Python3.12隔离环境，然后安装requirements.txt。实际锁定版本见verification.json。

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python -O experiment.py --out outputs/result-O.json
python audit.py --report outputs/result.json --out outputs/audit.json
python plots.py --report outputs/result.json --directory outputs/figures
```

程序从脚本目录定位输入，支持从其他cwd调用绝对脚本路径。普通、-O、不同cwd与Notebook的数值结果在本次环境中逐字节相同。输入和输出保护使用显式异常，不依赖assert。

实验使用np.linalg.lstsq，比较sklearn时使用同一归一化基底、fit_intercept=False；额外SciPy gelsy为枢轴QR路线。最大平均正规方程残差约6.80e−14。最大风险样本的独立积分绝对差约1.03e−9，但尺度化差仅8.72e−15，不能混淆大风险与求解失败。

所有拟合保留rank、condition、normal_residual、train、risk、test及系数；不丢失败、不静默nanmean、不显式矩阵求逆。无效输入测试包括秩亏、样本少于参数、非有限值、域外输入和非法参数数。遇到新的失败会明确退出，需保留输入和错误后分析，不允许只统计成功子集。

## Notebook

直接打开experiment.ipynb即可阅读已有输出。重建会清除再生成该Notebook，故先保留原交付包。

```bash
python build_notebook.py
python execute_notebook.py experiment.ipynb
```

第一代码格仅用标准库核对8个源码/数据字节契约。执行器在新的Python进程中启动真实ipykernel InProcessKernel，顺序执行并保存输出。本次未检验浏览器Jupyter界面或外部内核传输；受限环境的网络接口发现警告不影响in-process执行。

## PDF和全量构建

现有PDF阅读不需要排版依赖。重建可信Markdown需Node.js、MathJax3.2.2、Pango和Noto CJK字体。当前实际采用MathJax SVG加WeasyPrint70.0；XeLaTeX因环境缺格式文件而未使用。

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
# 在依赖已准备好后，完整重建
bash build_all.sh
```

可设置PYTHON指定解释器。已有MathJax模块时可用NODE_PATH指定其node_modules；正常npm install会建立本目录模块。TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME、IPYTHONDIR可指向可写私有目录。

build_all.sh已在独立复制的单元目录真实跑完；结果输出于outputs/rebuild，并刷新experiment.ipynb。PDF引用随包figures以避免不经确认改交付图；新生成图在outputs/rebuild/figures。若主动修改协议或图，须保留旧包并明确替换图后再重建PDF。

generate_data.py会重写本目录data和data_integrity.json，日常复现实验不需要调用。独立复制目录中重新生成的数据已核验与随包数据字节一致。不要把改过的数据配上旧结果图。

## 已知范围

支持范围是本讲固定协议和basis/fit函数所声明的输入域，不是通用生产训练框架。PDF构建器面向可信课程Markdown，未经不可信HTML/TeX安全沙箱测试。输出保护用于防误操作，不抵御恶意并发文件系统攻击。跨平台BLAS末位、字体和PDF字节可能不同，应比较语义数值和合理容差。
