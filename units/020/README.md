# 第020讲 常见分布与建模假设

核心问题：二分类、计数和测量误差分别适合怎样的概率模型？直接先修为第019讲随机变量与期望。

## 学习顺序

1. 阅读lecture.pdf，先理解支持集和记录单位，再推导五种分布的矩与指数族表达。
2. 按lab.pdf配置环境并运行experiment.py。
3. 顺序执行experiment.ipynb，检查全部输出和四幅生成图，不挑选随机种子。
4. 独立完成answers.pdf中的16题后再核对详解。

完整生成数据在脚本结果目录，data中保留数据字典、参数说明和20行格式预览。

## 文件

- lecture.pdf与lecture.md：14页正文，18个主题小节、12幅原创彩色图
- lab.pdf与lab.md：4页独立实验指南
- answers.pdf与answers.md：4页、16题逐步详解
- experiment.py：可导入的概率、数值校验与生成接口，或独立运行的命令行入口
- experiment.ipynb：12个实际顺序执行的代码单元，4幅保留PNG
- figures：正文使用的12幅原创PNG
- data：全合成数据说明、设定副本与20行预览
- environment.yml、requirements.txt：教学运行依赖
- source-checks.json：实际阅读的一手来源与使用范围
- test-result.json：运行、精确交叉核验、边界和复现证据

## 快速运行

```text
conda env create -f environment.yml
conda activate ml-course-020
python experiment.py
jupyter lab experiment.ipynb
```

脚本默认在自身旁创建outputs目录，可用--out更改。没有网络请求、账号、付费API或GPU要求。核心计算需要NumPy；绘图和Notebook需要Matplotlib与Jupyter组件。所有样本均由明确种子生成，不含真实记录。

## 关键可复核结果

- Binomial(8,1/4)：均值2、方差3/2，恰好2次成功概率5103/16384
- 完全同步的8个Bernoulli(1/4)：同均值2，但总数方差12
- 等权Poisson(1)/Poisson(7)：均值4、方差13；Poisson(4)方差为4
- Categorical三类概率1/2、3/10、1/5；类别编号没有固定连续距离
- Gaussian(μ,v)：v严格为正，密度支持为整条实轴；自然参数η1=μ/v、η2=-1/(2v)
- 指数族归一化函数不能删除；边界概率与有限自然参数需要区分

实际12000条模拟中，Poisson均值3.99225、经验方差3.93252327，混合Poisson均值3.95858333、经验方差12.98136799。它们与理论量接近但不相等；不能用有限图形证明总体分布。

## 验证与范围

核心脚本通过1533项参考检查和74组边界/回调检查；另做813项独立公式路径的作者数值审计。两个无关工作目录的新进程与python -O产生字节相同的JSON与CSV，检查不依赖assert。Notebook由独立新Python进程启动真实ipykernel，12个代码单元顺序执行，保留四幅PNG。

浏览器Jupyter界面、socket传输、读者机器上的Anaconda安装和跨NumPy版本逐位复现未测试。Gaussian归一化积分、指数幂级数和无穷操作交换条件在正文明确区分接受定理、直接推导与数值核验。本单元不是现实业务分布适配保证。

数值接口保守拒绝bool、文本、复数、非有限数、错误形状、非法参数、溢出和有意义的下溢。概率列表要求math.fsum在float表示中恰为1，不静默重归一化；极小正质量可用logpmf保留，不能把下溢0当作支持外0。实现范围小于数学模型范围，详见lab.pdf和函数文档。
