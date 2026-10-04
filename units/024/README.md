# 024 Bayes估计与先验

用八个合成二元结果解释完整的Beta-Bernoulli更新：先验、似然、后验、证据与预测分别对什么归一化，先验强度怎样改变估计，为什么预测整批结果必须保留共同参数。

## 文件与学习路线

- lecture.pdf / lecture.md：19节完整教程，12幅原创彩图；从积分推导Beta递推、矩、共轭、MAP、后验决策与Beta-Binomial预测
- lab.pdf / lab.md：单独可跟随的实验指南，包含环境、数据、手算、程序、改动实验和故障定位
- answers.pdf / answers.md：正文18题逐题详解
- experiment.ipynb：12格实际顺序执行的代码及6幅保留图像
- experiment.py：可复用纯函数与命令行入口
- data/observations.csv：八条唯一合成观察，附两批标记
- data/priors.csv：五种明确的教学先验；data/README.md解释字段和来源
- environment.yml / requirements.txt：建议隔离环境和实际制作依赖版本
- source-checks.json / test-result.json / verification.json：来源、执行、科学与版面核查及限制

先修第十八至二十讲与第二十三讲。推荐先读正文至共轭更新，手算主例，再按实验指南打开Notebook，最后独立完成18题。第二十五讲继续讨论区间解释、Bootstrap与覆盖率。

## 一次运行

在本目录执行：

```bash
conda env create -f environment.yml
conda activate ml-course-024
python experiment.py
jupyter lab experiment.ipynb
```

也可使用现有、版本兼容的环境。脚本核心使用Python3.12和NumPy2.3.5；Notebook另用Matplotlib3.10.8、nbformat5.11.1、ipykernel7.4.0与IPython。PDF构建工具、渲染字体与临时缓存不属于学习者依赖。

脚本默认输入和输出相对于脚本目录，所以可从其他工作目录使用脚本的绝对路径。显式相对路径则相对于当前命令行目录：

```bash
python experiment.py --observations data/observations.csv --priors data/priors.csv --output my_results
python -O experiment.py --output optimized_results
```

输出目录运行时生成，报告为report.json。所有输入、计算和JSON序列化先完成，再以临时文件替换报告。坏CSV不会覆盖旧报告；没有声称测试了断电、满盘或操作系统级持久化事务。

## 主例精确答案

- 先验Beta(2,2)，三次成功五次失败后为Beta(5,7)
- B(2,2)=1/6，B(5,7)=1/2310
- 有序数据证据1/385，仅记录成功总数时证据8/55
- 后验均值5/12、二阶矩5/26、方差35/1872
- P(θ>1/2|D)=281/1024；下一次成功概率为5/12，二者是不同事件
- MLE3/8、θ坐标MAP2/5、η=logit(θ)坐标MAP映回θ为5/12
- 未来四次成功数质量：2/13、4/13、4/13、7/39、2/39；均值5/3、方差140/117
- 插件Binomial方差35/36；同批未来两次协方差35/1872、相关系数1/13

默认20000批、PCG64种子24024：共享θ的批总数均值1.67285、经验方差1.1856228775；逐次独立重抽θ的均值约1.66545、经验方差0.9756262975。经验方差使用除以R的经验分布约定。两条结果来自不同生成结构，只展示有限合成机制，不作为现实任务表现或数学证明。

## 函数契约

浮点数值API只接受普通Python整数/浮点或不超过64位的NumPy实数标量，拒绝bool、文本、复数、Fraction、扩展精度及非有限量；不能精确转成float64的整数也拒绝。Fraction仅用于名称明确的精确计算接口。计数必须是整数。完整列表在调用RNG或写报告前验证，包含最后一项与零权重支持点。对输入已先转成普通数值数组而丢失的原始类型，程序不能恢复历史。常量样本在完整验证后返回原常量均值和精确零方差。

数学Beta分布只要求a,b>0；本实现为有限精度演示限制a,b≥0.25且a+b≤10000。后验及预测中每个派生形状也必须符合这个范围。一般计数0..1000；精确阶乘接口的单个整数形状1..200；精确预测m≤100；模拟m≤100、1≤R≤100000且R×max(m,1)≤5000000。种子为0..2³²−1。中点网格1..100000。超界明确失败，不自动裁剪或修改先验。

beta_logpdf允许数学端点极限±∞；beta_pdf把负无穷映为零、正无穷保留为无界密度。内部密度必须有限且达到最小正规正浮点数；逐项正概率下溢/次正规数会被拒绝，包括插件质量、非零权重的似然乘积与后验质量；数学上确实为零的权重或端点不可能事件保留为零。非恒定样本的非零残差平方/均摊方差贡献若下溢或次正规，也明确拒绝，不返回伪零方差。predictive_pmf检查原质量和与1的差不超过1e-9，不静默重归一化。discrete_update的权重总和采用1e-12检查，再以实际证据除；证据为0明确报错。Numerical范围与容差是实现边界，不是更改数学定义。

posterior接受n=0,k=0并返回先验。predictive_pmf接受m=0，质量列表为[1]。beta_mode区分唯一内部峰、均匀不唯一、连续闭端点延拓以及单端或双端无界，不将无界密度伪装成有限内部众数。

CSV只在显式解析层把文本变成数值。观察文件表头固定为observation_id,batch,success；ID是唯一ASCII字母数字，批次first或second，success精确为0或1。先验文件表头固定为prior_id,alpha,beta,purpose，ID唯一且包含main，形状必须符合数值范围。更改Notebook默认数据会触发保护，因为图题与手算只对应既定基线；使用脚本参数另做实验并同步更新理论标签。

## 复核与限制

当前1405项核心参考检查、55项预期拒绝和17组接受边界通过；独立6061项检查包括Fraction多项式积分/预测/证据、80位Decimal对数、坐标众数、常量与坏末项、12项RNG前守卫、9项坏CSV旧输出保护。两种无关工作目录中的三个新进程（含-O）stdout与报告逐字节相同。显式require检查不会随-O消失。

Notebook已在新Python进程中的真实IPython InProcessKernel顺序运行，12格无错误，保留6PNG。浏览器Jupyter界面、标准socket传输、读者Anaconda安装和跨NumPy版本随机流逐位复现未测试。全部最终PDF页及Notebook图已逐页查看。独立验收最终状态见verification.json。

来源用于核对定义、条件与官方接口，正文、图像和例题为独立编写。没有复制教材正文，也不将公开可读误当成其他材料的版权许可。没有现实用户数据、秘密、联网执行或付费API调用。
