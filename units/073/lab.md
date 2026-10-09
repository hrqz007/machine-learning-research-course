# 第073课实验 聚类证据的四道检查

## 1 实验契约与运行前预测

研究问题：已知三个生成来源时，内部几何评价、来源一致性和扰动稳定性是否给出相容证据？把相同指标与候选K用于一个没有离散来源的高斯点云负对照诊断，会发生什么？这是一项预先规定的教学对照，不是现实聚类发现的显著性检验。

先在纸上写三条预测：K=3是否可能比K=2获得更好轮廓；无结构数据是否可能有正轮廓；使用连续坐标预测线性outcome是否会优于只用簇编号。预测可以错，但必须给出因果理由。运行后保留原预测，用结果修改理解。

数据分development、audit、null三张CSV，每张240行。字段id是身份标识；x0、x1是输入；source是合成来源编号；outcome是响应。null里的source仍是独立抽取的无意义编号，不是其真实簇标签，程序不以它报告负对照ARI。data/generation.json保存种子和字节摘要；数据全部由本地生成器原创构造，无下载。

## 2 从环境到第一份报告

在本讲目录打开终端，Python 3.12 CPU即可。requirements.txt列出本次环境版本。建议用独立环境，安装操作需要正常包源访问，后面的数值计算没有网络请求。

```bash
# 安装数值实验及Notebook所需依赖；不包含GPU框架。
python -m pip install -r requirements.txt
# 默认报告输出到outputs，不覆盖冻结报告。
python experiment.py --out outputs/rebuild/result.json
# 用明确的输入报告运行数学性质与独立实现对照。
python test_experiment.py \
  --report outputs/rebuild/result.json \
  --out outputs/rebuild/test.json
# 优化模式会删除assert；本测试用显式异常仍应工作。
python -O test_experiment.py \
  --report outputs/rebuild/result.json \
  --out outputs/rebuild/test-optimized.json
```

第一条计算命令会输出五个K的指标、20次稳定性、负对照和三个下游MSE。不要只截取chosen_k；保存完整JSON以便看到选择过程。正常测试和优化模式都应写入status为passed的报告。若数据摘要不一致，程序会停止；先重新生成到新目录比较，不要为了得到绿灯删除校验代码。

## 3 实验A 从定义算轮廓和ARI

打开validation.py，读每行注释。`X[:,None,:]`为每个点增加一个轴，和`X[None,:,:]`相减得到n×n×d的全部差；沿最后一轴平方求和并开方得到距离矩阵。这里内存随n平方增长，所以240行是教学尺度，不能直接照搬到百万行。

先用Notebook或交互Python运行：

```python
# 导入我们从定义写的指标，不调用现成指标替代手算。
from validation import silhouette, adjusted_rand
# 每个内层列表是一行一维样本，不能省掉二维形状。
s = silhouette([[0], [1], [5], [6]], [0, 0, 1, 1])
print(s, s.mean())  # 预期四个分数对称，均值约0.798。
print(adjusted_rand([0, 0, 1, 1], [0, 1, 0, 1]))  # 预期-0.5。
```

交付表格列出每点a、b、s。另做两个修改：把所有坐标乘8；将一个点独立成单点簇。预期前者分数不变，后者对应点分数按约定为0。只有一个簇时应抛ValueError；这是明确的定义域限制。

## 4 实验B 保留完整选择路径

读取selection的五行，画出K与轮廓系数。选择规则已经固定为开发集轮廓最大，不许看到ARI后改成“二者综合最好”。冻结数据选K=3，开发ARI为1。它证明当前样本中拟合分区与生成来源一致，不能证明任意新数据都会完全恢复来源。

检查audit_ari是1，20次重采样最低值约0.98755。逐次记录，不要只写“稳定”。解释：每次从开发集无放回抽192行，重新训练，再预测同一审计240行。测试的不是同一数组位置，而是同一id所代表的对象。

## 5 实验C 负对照为何也能很稳定

null的稳定性锚点是它自身的N对象，其中包含参与拟合的对象；主实验使用独立audit对象。两者不能直接比较外推稳定性，只用于演示稳定分区也可缺乏离散生成来源。

null的K=2平均稳定性约0.92127，轮廓约0.34013。回到图1观察：一团连续点云仍被切开了。请写出两个不矛盾的句子：“这个切分在本次子采样中比较稳定”和“生成器并没有两个混合来源”。用这两个句子解释指标不能单独承担科学主张的原因。

额外探索可改变null生成器为拉长高斯，并另存输出。先预测哪个切分更稳定，再试。不要把探索后的数据替换冻结data或参考报告；它属于你的派生实验。不要用一次负对照差值计算未经定义的p值。

## 6 实验D 下游用途与信息损失

冻结的outcome是两个坐标的线性函数加噪声。比较global_mean、cluster_mean、linear_features三种MSE。预期排序为线性特征最好、分组均值其次、全局均值最差。计算分组相对全局均值的绝对MSE降低，再说明为什么这种改善不充分证明分组最佳。

泄漏演示应在副本中进行：故意用audit的真实响应重算各簇均值，会得到一个使用了答案的模型。即便数值改善，也失去未见对象评价的含义。正式报告禁止将该结果和正确冻结评估混称为同一种测试。

## 7 图、Notebook与验收

```bash
# 根据同一份结果重画四幅有含义的对照图。
python plots.py \
  --report outputs/rebuild/result.json \
  --directory outputs/rebuild/figures
# 在新进程真实IPython内核中逐格运行，保留输出到新Notebook。
python execute_notebook.py --out outputs/rebuild/experiment.ipynb
# 重建原创数据到新目录，然后逐字节比较CSV和生成摘要。
python generate_data.py --directory outputs/rebuild/data
```

Notebook先呈现数据与手算，再真实执行run函数，最后内嵌图片。它不会只读冻结报告假装重新计算。验收需包含手算表、选择路径、稳定性机制、负对照解释、三个下游误差和一段边界声明。普通测试通过只说明列出的性质与这组冻结数据符合预期，不能代替对你的研究数据进行验证。

PDF重建另需build-requirements.txt、package.json中的MathJax与系统中文字体；运行build_all.sh可重复全部步骤。浏览器交互与外部socket内核未列为本实验检查项。
