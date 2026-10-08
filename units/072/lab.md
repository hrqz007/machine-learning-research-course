# 第072课实验 从树和密度解释分组

## 实验目标

本实验要求你在环形、不同密度和桥接噪声三种数据上比较层次方法与DBSCAN，解释它们依赖的假设。只交聚类图不合格；你还需要四点linkage手算、六点核心与边界手算、参数扫描记录、噪声统计和复杂度预算。

预计150至210分钟。先完成第071课，能够说明为什么两个均值原型不适合表达完整同心环。所有数据人工合成、固定种子、可离线复现。generated_group只是人工构造标记，不参与距离或参数拟合。请先预测各方法的结果，再运行代码检验，避免看到答案后倒推一套理由。

## 1 安装与运行

Python 3.12环境中执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py --out outputs/rebuild/result.json
python test_experiment.py --report outputs/rebuild/result.json \
  --out outputs/rebuild/test-result.json
python -O test_experiment.py --report outputs/rebuild/result.json \
  --out outputs/rebuild/test-result-optimized.json
python plots.py --report outputs/rebuild/result.json \
  --directory outputs/rebuild/figures
```

Windows可直接调用 .venv\Scripts\python.exe，无需为了激活环境修改系统安全策略。build_all.sh还执行Notebook与PDF构建；PDF额外依赖见build-requirements.txt和package.json。纯数值部分不需要浏览器或外部数据服务。输出写在outputs，随课冻结报告不会被默认重建命令覆盖。

## 2 数据说明与公平对照

| 数据 | 行数 | 组成 | 主要挑战 |
|---|---:|---|---|
| rings | 300 | 内外环各150点 | 非凸结构与不同沿环密度 |
| unequal | 265 | 紧密100 松散130 背景35 | 一个全局密度阈值的矛盾 |
| bridge | 197 | 左右各90 桥13 远处4 | 短边链与非核心边界 |

种子7201控制数据，种子72控制K-means对照。每个算法只看到x1和x2，同一场景使用相同坐标、相同欧氏距离。层次方法统一切成两组以观察指定分辨率下的机制，DBSCAN的簇数由eps和m决定。不要把这理解为双方已经使用同样的模型选择预算：本实验是解释性比较，不是排行榜。

对照包括K-means、single、complete、average、Ward，以及m=6下多个eps。plots.py主比较图选取其中六个设置，完整结果含average和全部半径，均在experiment-result.json中。画图中的灰叉表示被拒绝为噪声的观测。

## 3 实验A 手算层次合并

使用0、1、4、7四个样本，依次计算single、complete和average。

1. 找出第一对合并及其高度。
2. 计算新簇到剩余两个单点的距离。
3. 遇到平局时按当前簇ID较小的候选对优先，记录第二次合并。
4. 写出最终跨簇的四个样本距离，并求三种最终高度。
5. 在高度3.2切三棵树，说明留下几组；再解释为何按精确K=2切single树不完全等同于一条水平高度线。

```python
import numpy as np
from density_hierarchy import agglomerative, cut_k
X = np.array([0, 1, 4, 7], dtype=float)[:, None]
for method in ['single', 'complete', 'average']:
    Z = agglomerative(X, method)
    print(method, Z, cut_k(Z, 2))
```

Z每行含两个被合并的簇ID、高度和新簇大小。原始点ID为0到n-1，第t次新簇ID为n+t，其中t从0开始。把样本值4与样本ID2混淆，会导致树记录完全错误。

![实验图1 相同数据 不同linkage的纵轴含义不同。](figures/04_dendrograms.png)

扩展题：推导Ward的合并平方误差增量。对 $A=\{0,1\},B=\{4,7\}$ 求 $\Delta$，再求SciPy式Ward高度。不要将其与complete的最大距离混同。

## 4 实验B 手算密度可达

使用 $0,0.1,0.2,0.3,0.55,1.5$，eps=0.3、min_samples=4。逐点列邻域，注意小于等于边界且包含自身。对每个点填写核心、边界或噪声，并画出只包含核心点的邻接图。

回答：0.55为什么能与0同簇，虽然两点距离大于eps？0.3能否从0.55直接密度可达？把m从4提高到5后哪些点仍为核心？这些问题必须依据定义逐项判断。

运行dbscan并检查neighbor_counts、core、labels与手算。针对浮点边界，另用恰好可表示的一维点0和1、eps=1、m=2检查闭邻域条件。不要为让例子通过随意把所有距离比较加一个很大的容差。

## 5 实验C 共享边界不能充当桥

构造一维点 $-1.3,-1.2,-1.1,-1,1,1.1,1.2,1.3,0$，eps=1.05、m=4。左右各四点形成核心分量；中间0只邻接自己、-1和1，计数3，因而不是核心。

运行并检查：得到两个核心分量，0是歧义边界，程序按编号最小分量归属。改变输入顺序后，边界归属可以变，但左右核心分量仍应分离。将程序改成“所有邻居都继续扩展”，观察这个错误规则怎样把两个分量串起来，再恢复正确实现。

验收需要一段解释：DBSCAN的核心连接部分与普通全样本eps邻接图的连通分量有什么差别。只记住“DBSCAN能处理噪声”无法回答本题。

## 6 实验D 三种结构对照

先读未着色数据，再填写预测表：哪些方法可能恢复完整环，哪些可能沿稀疏桥合并，哪些必须把远处点分给某簇。运行所有场景，记录簇数、簇大小和噪声比例。

环形数据固定m=6，比较eps=0.10、0.13、0.24。解释内外环同样各150点，为何外环更容易被小半径拒绝。不得仅说“参数不合适”，需要联系周长、相邻间距和局部计数。

异密度数据固定m=6，比较eps=0.13、0.24、0.40。检查小半径遗漏多少松散团点，大半径是否合并两团及背景。不要把“噪声最少”或“恰好两簇”当作充分验收条件。

桥接数据比较single与eps=0.24的DBSCAN。用桥点的非核心身份解释为何一个方法串接而另一个没有。记录被DBSCAN接纳的桥点，避免把实验写成“全部噪声均被准确剔除”。

![实验图2 全局半径扫描展示漏掉稀疏结构与合并不同团之间的取舍。](figures/06_unequal_density_sweep.png)

## 7 实验E 代码与库一致性

density_hierarchy.py的DBSCAN直接计算邻域，先找核心连通分量，再分配边界。对每组参数与scikit-learn对照核心索引、噪声集合及分区。簇编号可能置换，所以比较成对“是否同组”关系；共享边界时还要结合声明的归属规则解释差别。不要把所有噪声点的负1当作一个真实相似群体。

自写凝聚算法只支持single、complete、average。Ward实验使用SciPy，并在讲义中给出其增量公式。独立小数据对照使用无距离平局的二维点，逐步比较Z矩阵；四点手算存在平局，不应把不同合法平局顺序误报为错误。

测试还应覆盖全部噪声、单点、重复点、闭半径边界、非法eps、非法m、非有限数据、无效链接记录和距离溢出。ordinary与-O报告应检查数一致，且-O中python_optimized字段为true。

## 8 实验F 资源预算与报告

计算n=1000、10000、100000时一个float64完整距离矩阵及压缩上三角的字节数。说明教学广播代码还有n乘n乘d临时量。对一个十万行高维项目，提出先抽样审查、评估邻域稀疏性或采用合适索引等策略，而不是直接承诺内存一定够。

报告结构：问题与度量、两组手算、实现校验、三类数据对照、eps扫描、至少两个明确失败机制、资源预算和结论限制。参考评分为手算25分、定义与边界实现25分、实验解释30分、复现与复杂度20分。

一份合格结论应像这样明确条件：“在本次环形间距与半径参数下，核心连接恢复两环；同一方法在异密度数据中面临碎裂与合并取舍。”不要写“DBSCAN适合所有非球状数据”。树状图的切法也应给出研究或操作依据，不能只说“看起来这里最合适”。

## 9 重新执行Notebook

experiment.ipynb已经执行并包含图像与中间输出。重新执行全部单元可复算手算、场景结果、曲线和图像。命令行为：

```bash
python execute_notebook.py --out outputs/rebuild/experiment.ipynb
```

执行器在新Python进程中使用真实IPython内核，逐单元捕获输出并拒绝错误。它验证计算顺序和图像嵌入，不验证外部Jupyter服务、浏览器界面或网络内核传输。若改变数据，应该重建并重新验证整个报告，不能把旧图与新结果混排。
