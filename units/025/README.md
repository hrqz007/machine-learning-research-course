# 025 区间估计与Bootstrap

本单元回答：一个估计值有多不稳定，区间里的95%如何解释？先修第21至24讲。正文从已知方差Gaussian的精确覆盖证明开始，接上Beta后验可信区间，再讲Wald边界失败、精确Bootstrap枚举、配对和分组重采样、两层Monte Carlo误差。

## 文件与阅读顺序

- lecture.pdf / lecture.md：完整正文，十幅原创机制图。
- lab.pdf / lab.md：独立实验路线、操作命令、验收要求。
- answers.pdf / answers.md：18题完整解答。
- experiment.ipynb：已逐格执行的教学Notebook，7幅内存生成PNG保留在输出中。
- experiment.py：同一核心方法、输入校验、模拟与命令行入口。
- data/：四份小型合成数据/配置和字典。
- figures/：正文使用的十幅原创PNG。
- requirements.txt / environment.yml：学习者依赖声明。
- source-checks.json：一手来源链接、访问情况和用途。
- verification.json：已完成验证的范围与限制摘要。

所有观察数值合成，不含真实人群、业务或私人数据。既有文献只给链接和用途，不随包复制论文内容。无需联网、密钥、GPU或付费API。不凭这组小实验声称真实系统效果。

## 运行

在本单元目录启动：

```bash
python experiment.py --self-test
python experiment.py --output-dir outputs
jupyter lab experiment.ipynb
```

Notebook请Restart Kernel并Run All。它自己检查默认CSV/config并重新模拟，不读取历史输出报告。脚本按自身位置找data目录，因此也可以从别处用绝对路径运行：

```bash
python /your/path/025/experiment.py --output-dir /your/path/results-025
python -O /your/path/025/experiment.py --self-test
```

输出为指定目录中的report.json，含模拟规格、各方法覆盖摘要和所有外层区间。脚本不改输入；report.json会在全量输入验证、计算成功后以临时文件替换。输出目录是运行产物，不属于发布包。既有同名报告会被本次成功运行替换，请为不同实验选择不同目录。

修改模型先复制data/model_spec.json，再运行：

```bash
python experiment.py --spec my_spec.json --output-dir my_results
python experiment.py --input my_tiny_sample.csv --output-dir my_results
```

--input只接受id,value两列的CSV，至少2条、至多6条用于精确Bootstrap枚举；不可把paired.csv或grouped.csv直接传给这个选项。后两份文件供Notebook核验与配对/分组练习读取。Notebook固定讲解默认数据，修改数据时应使用脚本或新Notebook并重算理论，不能沿用旧图题。

## 环境与诚实边界

已执行环境：Python 3.12.14、NumPy 2.3.5、Matplotlib 3.10.8、nbformat 5.11.1、ipykernel 7.4.0。核心统计代码只依赖NumPy和标准库。SciPy 1.17.0仅用于作者的独立数值参考，不是学习者运行依赖。若环境缺少学习者依赖，可用：

```bash
python -m pip install -r requirements.txt
```

或自行用Anaconda：

```bash
conda env create -f environment.yml
conda activate ml-course-025
```

上述Anaconda安装、浏览器Jupyter交互、socket内核传输未在作者环境实际测试。Notebook在独立新Python进程中使用真实IPython InProcessKernel逐格运行，保存了execution_count、文本和PNG；它是执行证据，不等于浏览器或跨进程传输证据。PDF由共享MathJax/WeasyPrint制作，不需要学习者安装这些构建依赖。

## 基线结果

PCG64种子25025；Gaussian外层3000次、组实验外层1000次；每次Bootstrap B=399。Gaussian n=20、mu=3、sigma=.5。组实验G=24、m=6、tau=2、sigma=.5。

- 已知sigma Gaussian公式：2867/3000，覆盖.9556667，MC标准误.0037580，平均宽度.4382613。
- 非参数百分位：2797/3000，覆盖.9323333，MC标准误.0045858，平均宽度.4237951。
- 已知sigma参数百分位：2866/3000，覆盖.9553333，MC标准误.0037715，平均宽度.4420730。
- 组模型按行：590/1000，覆盖.590，MC标准误.0155531，平均宽度.6588882。
- 组模型按组：936/1000，覆盖.936，MC标准误.0077398，平均宽度1.5666202。
- Wald n=20、p=.02：精确二项求和覆盖.3317923493，没有MC误差。
- Beta(5,7)等尾可信端点：[.1674880941,.6920952850]。
- 样本(0,2)精确Bootstrap：均值0、1、2的质量分别1/4、1/2、1/4，条件方差1/2。

这些确切命中数用于相同算法与版本的复现核对，不能作为统计定理。理论95%的已知sigma公式也不要求有限模拟恰好命中95%；其数学证明在正文。其他名义95%方法没有被宣称为有限样本精确95%。

## 公共函数与输入契约

所有检查用显式require/ValueError，不依赖可被-O去掉的assert。数值数组只能是列表、元组或NumPy数组，标量支持实数数值类型；拒绝bool、文本、复数、NaN、无穷及不支持类型。先逐元素验证，再转float64；不靠dtype自动把True变1。原数组已被上游类型转换所丢失的信息无法恢复。

- 一般实数值：绝对值不超过1e6；非零绝对值至少1e-100。普通向量长度1..10000；标准差至少2项。非恒定样本中心化后的非零离差小于1e-100时明确拒绝，请先选择合适单位。恒定float值全量验证后mean原值返回、sample_sd精确返回0。
- empirical_quantile和weighted_quantile：逆CDF、无插值；q在[0,1]。按q最短十进制表示做有理数秩计算；权重累加按实际float的精确Fraction值。零权重值仍验证；q=0取最小正质量支持点；全零/负权重拒绝。与直接float乘法在秩边界可能有细小不同，教学接口刻意避免0.14×50略大于7的问题。
- percentile_interval：至少2个复制值；level在[1e-6,1-1e-6]，按level十进制表示计算两尾概率。
- known_sigma_interval：sigma在[1e-6,1000]；Gaussian且sigma真已知时的公式。wald_interval允许n=1..10000、k=0..n；exact_wald_coverage为枚举限制n<=100。
- beta_cdf_integer / beta_quantile_integer：整数a、b各1..40。CDF输入x在[0,1]且服从一般数值域；总和下溢为0而x>0时拒绝。PPF q在[1e-6,1-1e-6]，固定60次二分，不等于60位精度。任一数学上正的二项因子或质量若数值下溢即拒绝，包括所求尾部以外的项；真正端点0/1的零质量仍合法。靠近1的CDF可舍入为1；不保证任意尾部的相对精度，不替代通用log-CDF实现。
- exact_bootstrap：n<=6，枚举n^n个有序索引；Fraction代表已验证输入的实际float值，绝非先验或后验抽样。
- bootstrap_mean：输入至少2项；method为nonparametric或parametric_known_sigma。即使当前选择非参数分支也先检查sigma。参数分支直接抽Gaussian均值，数学上等价于抽n个Gaussian再平均。
- paired_bootstrap：等长的before/after，目标after-before均值。paired保持个体索引，independent仅用来演示破坏配对；差值也必须在支持数值域。
- group_bootstrap：矩阵(G,m)，G=2..100、m=1..100，所有组等长。group抽完整组，row抽展平行。不等组长明确拒绝，先决定新估计目标。
- 单次重采样B=2..10000，seed为0..2^32-1的整数，B乘数据单元数不得超过200万。公共API不接受外部可变RNG或任意统计回调，避免隐式状态与未验证回调输入。
- 模拟JSON键严格匹配，任何重复键都拒绝（即使两个值相同）；十进制数在原始token阶段检查，非零的1e-400不能被当成合法0；NaN、Infinity等非标准常量拒绝。外层Gaussian次数20..5000、组次数20..2000、B=99..1999、n=2..100、G=3..100、m=1..30、mu∈[-100,100]、sigma∈[.01,10]、tau∈[0,10]。总单元运算预算不超过1.2亿。

数值域是本实现的限制，不是统计概念的定义域。普通均值小量相消等仍服从float64舍入规则。原始非零数转成0、非恒定输入整体塌缩成float64常数、非零平方离差下溢都会拒绝；接近1的高精度概率若转换使正上尾塌缩为0也拒绝。代码不声称精确表示所有有限实数。CSV的专用解析器保留原文本并拒绝解析溢出、非零文本下溢为0和额外字段；数值API若只收到已经下溢的0，无法猜回原文。

## 如何解释验收

自测涵盖9个核心手算恒等式。作者另用Fraction、65位Decimal、SciPy Beta PPF、独立积分加求根、Normal CDF/PPF和NumPy分位数交叉核验；对错误输入使用RNG替身确认先拒绝、后副作用；检查损坏CSV不改既有输出。脚本以两个无关工作目录在新进程运行并与-O结果对齐。所有最终PDF页面及7个Notebook PNG逐张查看，未把联系表当成逐页验收。

没有分发构建脚本、作者审计脚本、QA截图、日志、缓存、运行outputs或私人冻结清单。source-checks.json说明实际检查过的来源；出版方仅返回入口框架的Efron论文只列为延伸书目，未伪称看过全文。没有为外部文献添加未经授权的版权许可。
