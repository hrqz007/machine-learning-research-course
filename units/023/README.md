# 第023讲 似然与最大似然估计

核心问题：怎样从观测数据反推概率模型的参数？直接先修第13至14、19至20、22讲。

## 学习顺序

先读lecture.pdf，手算八条Bernoulli与四条Gaussian记录；按lab.pdf运行脚本与Notebook；独立完成18题，再看answers.pdf。先写观测规则、联合模型和参数空间，再求导、查全局与边界。不要把优化器停止当作最大值存在的证明。

## 文件

- lecture.pdf / lecture.md：完整正文，12幅原创彩色图，含逐步推导、手算、失败条件及18题
- lab.pdf / lab.md：单独可跟随的实验指南
- answers.pdf / answers.md：18题逐步详解
- experiment.py：Bernoulli/Gaussian似然、解析估计、导数、透明数值搜索、重复采样与严格输入校验
- experiment.ipynb：13格已执行代码、6幅保留PNG、逐步检查与解释
- data/：两份小型合成CSV、重复采样JSON、数据字典
- requirements.txt / environment.yml：运行依赖
- source-checks.json / test-result.json / verification.json：一手来源、数值测试与验证范围

## 运行

从本目录运行：

```text
conda env create -f environment.yml
conda activate ml-course-023
python experiment.py
jupyter lab experiment.ipynb
```

核心脚本只需Python与NumPy；Notebook还需Matplotlib、IPython和ipykernel。实际检查Python 3.12.14、NumPy 2.3.5、Matplotlib 3.10.8。无网络、GPU、账号或付费API要求。Anaconda环境文件是建议约束，未声称测试过读者的具体安装。

脚本默认读同目录data输入并写outputs/report.json与replicate_estimates.csv。默认后者有12000批估计。可使用--bernoulli、--gaussian、--spec指定另存的输入文件，以--output指定结果目录；显式相对路径相对于调用时的工作目录。默认路径相对于脚本定位，所以可从其他目录用绝对脚本路径运行。

Notebook首格要求默认输入，确保手算、图题与配置一致。另存的变化实验用脚本运行并重算理论。所有输入、计算和序列化在写结果之前完成，坏CSV不会覆盖旧结果；不承诺断电或磁盘满时跨文件的原子事务。

## 应得到的结果

- 序列(1,0,0,1,0,1,0,0)：n=8、k=3、MLE为3/8；计数似然多出常数56
- 二分导数依次检查1/2、1/4、3/8，在第三步恰好命中解析解
- Gaussian(0,1,2,5)：均值2、SSE14、方差MLE3.5、无偏方差14/3
- 全零/全一Bernoulli：闭区间边界MLE为0/1；有限logit参数中没有对应有限MLE
- 全同值Gaussian：在v>0模型下似然无界，无有限MLE；返回明确状态与variance_mle=None
- (0,1,2,21)：均值6、SSE302、方差MLE75.5
- 默认n=4的4000批模拟：方差MLE均值约2.644831，无偏版本约3.526442，对应理论期望2.625和3.5

模拟设定为Bernoulli(3/8)和Gaussian(2,3.5)，n取4、16、64，每个n独立初始化PCG64(seed+n)，seed=23023，先生成Bernoulli再生成Gaussian。有限模拟仅展示机制，不代表现实预测性能，也不是定理证明。

## 主要接口

- bernoulli_loglik(values,p,count_observation=False)：稳定log似然；严格支持与边界处理
- bernoulli_mle / bernoulli_derivatives / bisect_bernoulli：闭区间解析解、内部导数与专门求根
- gaussian_fit / gaussian_loglik / gaussian_derivatives：方差参数v，不是标准差
- golden_maximize(function,lower,upper,...)：透明黄金分割；调用者需给连续单峰目标及合法范围，converged不证明一般黑箱全局最优
- repeat_sampling(spec)：完整验证先于随机流创建；返回每个n的逐批估计和汇总
- reference_checks / boundary_checks：显式require检查，在python -O下仍然执行

## 数值契约

向量仅接收list、tuple或一维实数NumPy数组。逐元素拒绝bool、文本、复数、object dtype、扩展精度和非有限数；不先强制转浮点来隐藏混合类型。支持Python整数/浮点及不超过64位精度的NumPy实数。调用方若已把bool或文本转成普通浮点，原始类型信息无法恢复。

向量长度1..100000，绝对值不超过1e6，非零值绝对值不小于1e-12。Gaussian均值候选在[-1e6,1e6]内，评价方差及非退化拟合方差在[1e-12,1e12]内。数学模型仍为v>0；超出实现范围会报错，绝不裁剪成边界解。完整验证后先识别精确常量，避免[0.1]*3等因均值舍入形成伪残差。

Bernoulli p可取闭区间[0,1]；导数接口限定内部[1e-12,1-1e-12]。边界对数似然可能合法返回负无穷；Gaussian评价结果要求有限。平方、求和与模拟中间计算也核验，拒绝不支持的溢出/下溢情况。

模拟配置要求严格既定键，n为2..512，最多8个不同递增n，R为2..10000，每个分布的R乘n总和不超过200万；mu在[-100,100]，variance在[1e-4,10000]。完整配置先于任何随机流创建。黄金分割的范围、容差、迭代上限均先于任何目标回调检查。

## 验证与限制

474项核心参考检查、23组拒绝检查及4个接受边界通过。另有2445项独立检查，涵盖Fraction概率与Gaussian矩、65位Decimal log参考、偏导差分、28组精确常量、9组回调前验证、24组随机流前验证与11组坏CSV旧结果保护。两处无关工作目录下的三次新进程运行含python -O，stdout和两份结果文件逐字节一致；完整独立审计也在普通与-O模式重复一致。

所有正文、实验及详解PDF页面均渲染逐页检查；Notebook6幅实际PNG检查。Notebook从新Python进程的真实InProcessKernel顺序执行并保留输出。浏览器Jupyter界面、常规socket内核传输、读者Anaconda安装和跨版本逐位复现未测试。具体页数与最终核验状态见verification.json。
