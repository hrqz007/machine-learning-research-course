# 第076课 实验：让图与八行概率表相互检查

## 1 实验问题与独立验收标准

本实验不采集现实数据，也不下载天气表。数据是generate_data.py中明确写出的原创概率参数；全部结果来自枚举，不含抽样误差。目标一是验证链的条件独立；目标二是观察碰撞条件化；目标三是用不同方向的条件表重建同一联合分布；目标四是把无向势函数归一化。

开始前先记录四个预测：A与C边缘相关；给定B后独立；R与S边缘独立；给定W后通常相关。每个预测都要同时写图理由和数表验收方法。若计算结果与预判不一致，应检查条件集合、变量顺序和归一化，不能修改预判来伪装预先假设。

| 对象 | 操作 | 预期判据 |
|---|---|---|
| 链 | 枚举8行联合表 | 非负且总和距1小于10的负12次方 |
| 链的A,C | 边缘与给定B分别计算残差 | 前者大于0.01，后者小于10的负12次方 |
| 碰撞结构 | 给定W前后比较R,S | 先接近0，后明显非零 |
| 等价链 | 反向重算条件表 | 每行联合概率与原表差小于10的负12次方 |
| 无向链 | 权重求和再除Z | Z=24，总和为1 |

## 2 准备本地环境

进入本单元目录；Python 3.12或3.11均可运行教学算法。requirements.txt列出绘图与Notebook依赖。数值算法本身只使用Python标准库。终端中`python`表示你选定环境的解释器；路径中的`outputs`是重建结果目录，避免覆盖随课程提供的冻结参考材料。

```bash
# 安装版本记录中的运行依赖；只需准备环境时执行。
python -m pip install -r requirements.txt
# 独立重建原创参数表与完整联合表，不改data中的冻结副本。
python generate_data.py --directory outputs/generated-data
# 运行实验，程序先检查冻结数据是否与生成器逐字节一致。
python experiment.py --out outputs/result.json
```

预期终端输出包含chain_total约1、mrf_partition为24。浮点总和可能显示1.0000000000000002；这不是额外的概率质量，而是二进制浮点无法精确表示部分十进制数。不要用`== 1`检验这样的浮点总和。

<div class="page-break"></div>

## 3 从一行代码追踪一次条件概率

打开graph_models.py。下面四行可以单独放进Python交互环境。每行注释解释目的。

```python
from generate_data import model  # 读取有名字、可重建的教学参数。
from graph_models import joint_from_bn, conditional  # 导入枚举与条件化函数。
joint = joint_from_bn(**model()['collider'])  # **将字典展开为具名参数。
print(conditional(joint, {'R': 1}, {'W': 1}))  # 事件是下雨，证据是地面湿。
```

字典`{'R': 1}`表示把R固定为1。`joint`是八个字典构成的列表；每行有state和probability两项。函数先把W=1的四行相加得到0.383，再把其中R=1的两行相加得到0.1854，最后相除。把第二个证据改成`{'W':1,'S':1}`，观察输出从约0.484073降到0.236277。

手工对照时按(R,S)顺序阅读W条件表：索引0、1、2、3分别对应00、01、10、11。父节点顺序由nodes列表决定，绝不是由edges偶然列出的先后决定。把两条入边交换书写顺序，数值必须保持不变；专门测试会检查这一点。

## 4 图检验与数值检验分工

先运行`d_separated`检查链、叉、碰撞与碰撞后代；再运行`ci_residual`检查具体概率表。尝试把有边A→B的两行条件概率都改成0.3。图仍不保证独立，但新数表里的A和B独立。请在实验笔记写明这不违背d分离的正确方向。

不要直接改冻结data文件再期待原实验继续通过。复制参数到交互变量进行探索，或者把新方案写入outputs。原实验会将data字节与生成器对比；它阻止不知不觉替换参考模型。若要设计新数据，需一同更新参数来源和预注册判断，而不是只改结果数字。

```bash
# 普通模式运行6组定义与反例测试。
python test_experiment.py --out outputs/tests.json
# 优化模式关闭语言级assert，但unittest检查仍应执行。
python -O test_experiment.py --out outputs/tests-optimized.json
# 重画本课五幅图，用于对照数值变化。
python plots.py --directory outputs/figures
```

两次测试都应显示6组通过，tests-optimized.json中的optimized为true。测试不等同于真实数据外部有效性，也不声称外部CI已经运行。

<div class="page-break"></div>

## 5 Notebook与PDF的复现路线

```bash
# 从课程单元模板创建并在新Python内核中依次执行Notebook。
python execute_notebook.py --out outputs/experiment.ipynb
# 任选：准备PDF构建依赖和本地公式渲染器。
python -m pip install -r build-requirements.txt
npm install
# 用本地MathJax及中文字体生成三份PDF，不访问在线渲染服务。
python build_pdf.py --directory outputs/pdf
```

Notebook要求每个代码单元都有执行序号，并且不存在error输出；嵌入图应直接显示在文件中。execute_notebook.py使用当前Python注册的临时内核规格，防止一个名叫python3的旧内核误用别的环境。提供的experiment.ipynb已实际执行；仍建议学习者在自己环境重跑以理解中间量。

PDF还需要Pango与Noto CJK中文字体。PDF构建是可选学习工具，不是数值实验前提。若缺字体或渲染器，应报告明确依赖错误，不能把空白页当成成功。渲染后的页脚页码、条件表、箭头方向、图注和代码换行均需逐页检查。

## 6 失败注入与最小交付

将一张条件表概率改为1.1，预期抛出ValueError；加入A→B和B→A，预期拒绝有向环。建立P(A=1)=0后，查询给定A=1的条件概率，预期拒绝零概率证据。这里的失败是正确行为：返回某个“看起来合理”的0或0.5反而会掩盖模型没有定义的问题。

提交自己的学习记录时包括：模型变量与状态字典、八行联合表、两套独立残差、一个d分离路径解释、反向条件表、Z手算、普通与优化模式报告、Notebook及一个失败注入截图或文本。不要包含本机凭据或私有路径。

最后回答：数值残差很小是在检验哪张表？图分离是在约束哪一类分布？如果将这两句话混成一句“图证明了现实因果”，则还没有完成本课目标。完整参考数值和逐题解释见answers.md；读答案前先保留自己的计算过程。
