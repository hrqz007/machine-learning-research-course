# 043 多分类与信息论损失

从互斥三类概率、Categorical似然和逐样本链式导数，走到两轮同步更新、有限最优点以及熵、交叉熵与KL。完整166单元课程中的第043讲，直接先修018至019、023、039、042，后接044广义线性模型。

## 学习材料

- lecture.pdf / lecture.md：20节正文，9幅原创计算图，18道练习
- lab.pdf / lab.md：独立实验指南，数据、环境、手算、常见错误与验收
- answers.pdf / answers.md：18题完整详解，逐行梯度表和证明中间步
- experiment.ipynb：11个真实执行代码格，24行完整局部链，9张实际PNG
- experiment.py：稳定softmax/log概率、行级导数、批量GD、信息论和独立优化路线
- audit.py：Fraction初始链、Decimal完整路径与尾项、形状/控制流/输出核验
- plots.py：从实际计算结果重建九图
- build_notebook.py / execute_notebook.py：重建Notebook与新进程真实InProcessKernel执行
- build_pdf.py / mathjax_render.cjs / pdf.css / package.json：完整可选PDF构建源码
- data/、requirements.txt、environment.yml：固定原创数据与数值依赖入口
- experiment-result.json、test-result.json、source-checks.json、verification.json：科学输出、作者核验、来源与版本范围

正文13页，实验指南5页，详解6页。作者自查完成不表示独立验收或远端发布；课程发布状态由仓库总清单记录。

## 同一例子的三个状态

X为八行一维特征：前四行-1、后四行1。标签为(0,0,1,2,0,1,2,2)，类列顺序甲乙丙。Theta第一行为截距b，第二行为斜率w；所有行先用同一旧Theta求平均梯度，再同步更新。

- 初始F=log3≈1.098612288668110，gb=(-1/24,1/12,-1/24)，gw=(1/8,0,-1/8)
- 第一次新b=(.025,-.05,.025)，w=(-.075,0,.075)，F≈1.076152315758658
- 第二次新b≈(.044881634194,-.089763268388,.044881634194)，w≈(-.134644902581,0,.134644902581)，F≈1.062011537613334

每行保存得分、概率、log概率、loss、loss/n、局部loss导数、softmax Jacobian、稳定得分导数与全部平均参数梯度。后两次状态含超越函数，不把打印小数当精确分数。

由两组经验频率独立构造有限最优点：x=-1的概率(.5,.25,.25)，x=1为(.25,.25,.5)，最小F=1.5log2≈1.039720770839918。零和参数表示b=(log2/6,-log2/3,log2/6)，w=(-log2/2,0,log2/2)。原参数仍存在所有类共同平移的不识别，不能宣称无约束Theta唯一。

默认GD85次更新满足1e-10梯度阈值。SciPy BFGS16次迭代返回success=False和precision loss；重算完整梯度约7.52e-11，与解析点接近，但原状态完整保留。最终训练准确率1/2，乙类从不成为最大概率，不能省略这些不漂亮的结果。

## 运行数值实验

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python -O experiment.py --out outputs/result-O.json
python audit.py --out outputs/audit.json
python plots.py --report outputs/result.json --directory outputs/figures
```

默认输入相对脚本自身定位。要改数据，复制CSV/config后传--data与--config，不改固定教学输入。Notebook在本目录打开，先重启内核再运行全部。首物理代码格只用标准库检查八个精确字节契约，然后才导入数值库和生成图。

## 稳定实现与支持范围

- softmax沿每行类别轴归一化；先减行最大值，再把一个最大指数1与其余尾项分开，用log1p保留赢者的小log概率
- 不通过log(舍入概率)计算损失，不隐式裁剪；真实类得分导数用负的其他类概率和，避免p_y-1相消
- 原始X、Theta有限、绝对值≤100；非零原始数绝对值≥1e-100；n=1..2000、d=1..20、K=2..20。独立softmax入口得分范围±10000。内部有限迭代不套原始初值盒
- e.information输入是同序概率向量，长度2..20、和在1±1e-12内，只修正此范围的浮点求和误差；不会把任意计数自动归一化。近相等KL的机器精度级误差不能作统计结论
- q正而p零的交叉熵/KL标记为无穷；JSON数值字段null加明确标记。q零项按极限贡献0，不用nanmean删异常
- 短/零预算、初始驻点、首步拒绝均合法报告；失败候选计算计入尝试成本。记录尝试不代表实际FLOPs或速度
- 主结果写入先完成严格JSON序列化，再原子替换；保护教学/输入文件以及符号链接和多硬链接。该约定不声称抵御恶意并发文件系统竞争

## 可选重建PDF与Notebook

阅读现成PDF和运行数值实验不需要排版依赖。重建可信课程源码时，在本单元目录执行：

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
python build_notebook.py
python execute_notebook.py experiment.ipynb
```

还需要Node.js、Pango及Noto Sans/Serif CJK字体。MathJax由本地npm包转成SVG，再由WeasyPrint排版，不下载教材内容。构建器仅针对可信课程源，不当作陌生HTML/TeX的安全沙箱。生成后仍须逐页目视检查。先前尝试XeLaTeX发现本制作环境缺格式文件，最终交付采用上述已真实运行的WeasyPrint路线。

## 证据与限制

作者用Fraction核验初始链，用100位Decimal核验全部86状态及每行梯度/Jacobian；1000位参照检验可表示的微小概率与log概率。另有非驻点差分、单行/非方阵/多类别/类别排列、SciPy ordinary softmax/log_softmax/entropy对照，以及预计算坏字段拒绝和原子输出失败检查。

实际Python3.12.14、NumPy2.3.5、SciPy1.17.0、Matplotlib3.10.8。Notebook经新Python进程中的真实InProcessKernel顺序执行，保存实际文本和九PNG；不声称测试Jupyter浏览器UI、外进程传输、Windows或Anaconda安装。不同平台末位与PDF排版可能不同。

只有固定八行数据的有限MLE证书；没有一般分离检测器、现实泛化估计、校准证据或新研究贡献声明。正则化、标签平滑与多标签建模在正文作边界说明，不冒充已经做过的比较实验。
