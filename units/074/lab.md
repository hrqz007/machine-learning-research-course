# 第074课实验 看见EM的一次次更新

## 1 问题、数据和固定方案

实验比较八个初始化对同一一维三成分模型的影响。训练360点，独立审计240点，原始来源均值−3、0、3，标准差0.45、0.8、0.45，权重0.35、0.4、0.25。source记录生成来源，只用于理解合成数据；EM只读取x。

id是对象标识，x是人工连续读数，source是0、1、2之一。方差是标准差的平方，代码生成时乘的是标准差。不要把0.45当作方差写进Gaussian密度。各CSV与generation.json的SHA-256由本地生成器记录，并在加载时重新核对。

固定设置：K=3，方差下限0.0001，最多300次更新，对数似然绝对增量小于10⁻⁹则停止。八个初始化中包含合理分离、集中在左侧、完全对称以及五个随机抽样初值。选择训练似然最大的重启，然后报告审计平均对数密度；审计集不选择重启。

## 2 运行与预期产物

在本讲目录的Python 3.12独立CPU环境运行：

```bash
# 仅从正常包源安装依赖；正式计算全部离线。
python -m pip install -r requirements.txt
# 产生完整参数轨迹和八条似然曲线的数据。
python experiment.py --out outputs/rebuild/result.json
# 检查手算、似然不降、数值稳定性和独立库对照。
python test_experiment.py \
  --report outputs/rebuild/result.json \
  --out outputs/rebuild/test.json
# 确认优化模式下显式检验未消失。
python -O test_experiment.py \
  --report outputs/rebuild/result.json \
  --out outputs/rebuild/test-optimized.json
```

控制台应显示best_run、八个最终似然与hand三点结果。完整报告还保存每轮权重、均值、方差、converged与iterations。若达到上限，必须写“未满足停止阈值”，不能只见到曲线平就宣称已收敛。

## 3 实验A 手算与数组形状

读mixture.py，确认x形状为(n,)，责任矩阵为(n,K)，权重、均值、方差各为(K,)。`x[:,None]`把样本列改为(n,1)，与(1,K)参数广播，得到每点对每成分的值。若省略新增轴，在n恰好等于K时可能产生静默错算。

```python
# 导入数组库，np是惯用短名。
import numpy as np
# 导入E步与M步，分别返回责任与参数。
from mixture import expectation, maximization
# 三个对象的观测数值，无需任何来源标签。
x = np.array([-1., 0., 1.])
# 固定初始参数，先算条件来源概率。
r, before = expectation(x, [.5, .5], [-1., 1.], [1., 1.])
# 用相同责任度加权更新，再算新观测似然。
w, m, v = maximization(x, r)
_, after = expectation(x, w, m, v)
print(r, w, m, v, before, after)  # 应与纸笔结果对应。
```

交付每点两条责任、每成分有效计数、更新均值和方差。预期中点两边各0.5，新均值约±0.507729，新方差约0.408877，似然从−4.389254增到−3.549215。请说明为什么这不是“误差变大”。

## 4 实验B 重启与对称性

对每条history计算相邻差。允许浮点误差下限−10⁻⁸；若出现明显下降，优先检查是否用旧均值算新方差、责任是否每行归一、参数是否错位。不要先扩大容差掩盖错误。

找到initial_means=[0,0,0]的那条记录。预期终值约−819.551261，比其他约−687.091545低。沿parameters观察：三个成分的参数一直相同。用E步与M步公式各写一句解释，证明这是对称性保持，而不是代码没调用循环。

比较其他重启的最终均值时先按均值排序；否则标签置换会制造假差异。记录converged以及迭代次数。即使所有正常重启抵达相近目标，也只能说“本批重启结果一致”，不能证明全局最优。

## 5 实验C 下溢与方差塌缩

将读数改成−1000和1000，均值保持−2和2、方差1。普通密度可能下溢为0，但expectation应返回有限对数似然，每行责任和为1。这说明计算技巧保持了原数学含义，而不是给概率随便加常数。

读取collapse列表，横轴是一个成分的方差，从1递减到10⁻⁸，均值固定在观测0；其余点2和4由宽成分承接。观察似然上升。这里没有运行EM，所以这条曲线证明的是目标存在病态方向，不是在冒称算法必然走到该处。

再运行两点完全重合的单成分M步，指定floor=0.01。原始加权方差为0，返回0.01。解释这是满足v≥0.01约束的边界最优解。若你尝试其他正则方法，必须注明它是不同目标或不同更新，另核对单调性质。

## 6 图与Notebook验收

```bash
# 使用同一报告重建混合密度、责任、重启和塌缩四图。
python plots.py \
  --report outputs/rebuild/result.json \
  --directory outputs/rebuild/figures
# 新进程真实IPython内核顺序执行，不借用先前变量。
python execute_notebook.py --out outputs/rebuild/experiment.ipynb
# 原创数据可在新目录重建，避免覆盖冻结数据。
python generate_data.py --directory outputs/rebuild/data
```

Notebook会真实重新执行实验，并包含手算中间值与内嵌图。提交简短实验报告：生成假设、固定超参数、手算表、所有重启目标、退化机制、约束解释和边界。PDF可用build_all.sh连同数据校验一起重建，额外依赖见build-requirements.txt。

评分不以曲线是否漂亮为准。能够解释“目标不降但停在差解”和“目标越来越高但统计意义越来越差”这两个看似矛盾的现象，才说明理解了EM的保证范围。
