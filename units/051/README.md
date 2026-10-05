# 051 交叉验证与模型选择

阅读顺序：**lecture.pdf → lab.pdf → answers.pdf**。主题是把内层选参、外层评价和最终全数据模型分清；22道完整解答与真实执行Notebook配套。

## 教学机制

- 四行纸笔内层CV，完整预测、验证损失、选参和重拟合。
- 同一四行固定λ1，三次逐样本前向、两次同步梯度更新，明确正则梯度只加一次。
- 80行IID开发数据、17候选、外5内4；保留每个内外折的局部/全局索引和全部候选分数。
- 80份独立开发数据，比较各自CV分数与同一批折训练模型的精确总体风险。
- 真正的sklearn分层、分组、时间划分索引演示；不把四种划分当同一评价对象比较。
- 最终80行模型锁定后才生成独立4000行测试，区分测试抽样误差与训练/选择不确定性。
- 十图、真实`nested_cross_validation.ipynb`、完整数据/代码/环境及离线PDF工具。

所有数据合成。Oracle真实系数不进入选择函数；固定Legendre特征无数据拟合步骤，052另讲需要fit的预处理。负的单次乐观差、反向结果和高误差候选全部保留。

## 运行

Python3.12隔离环境安装`requirements.txt`。NumPy2.3.5、SciPy1.17.0、scikit-learn1.8.0和Matplotlib3.10.8为验证数值版本。

```bash
python experiment.py --out outputs/result.json
python audit.py --report outputs/result.json --out outputs/audit.json
python -O experiment.py --out outputs/result-optimized.json
python -O audit.py --report outputs/result-optimized.json --out outputs/audit-optimized.json
python plots.py --report outputs/result.json --directory outputs/figures
python generate_data.py --directory outputs/regenerated-data
```

默认数据按源文件位置读取，允许从不同cwd用绝对路径调用。`experiment.py`支持`--data-directory`与`--config`；新数据必须满足完整协议和CSV字段/行数。`audit.py`与十图用于交付默认协议；自定义扩展需要自己的参照与图注。

`reference_results.json`为所有默认科学输出，含完整重复数据、全部折索引和候选分数；`reference_audit.json`记录可复跑的作者数值检查；`verification.json`说明本包实际验证范围，不是独立QA签名。

## Notebook

首个物理单元先用标准库检查17项数据、源码与环境SHA，再导入教学模块。Notebook放在完整单元内或其`outputs/`子目录时可定位源文件。仅复制一个ipynb不够。

```bash
python build_notebook.py
python execute_notebook.py nested_cross_validation.ipynb
```

重执行请使用课程副本，保留交付原输出。执行器每次在新Python进程中启动InProcessKernel顺序运行，保存所有输出，不声称测试浏览器或进程外Jupyter传输。stdout可能以不同相邻消息分块；完整同通道文本连接后比较，PNG逐字节比较，不删改输出以凑相同。

## PDF与完整构建

直接读PDF无须额外依赖。重建时准备`build-requirements.txt`、官方Node.js、`package.json`内MathJax3.2.2、Pango及Noto Sans/Serif CJK字体。

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
PYTHON=python bash build_all.sh
```

构建器仅处理可信教学Markdown，公式本地转SVG。显示公式采用块级排版，避免页底基线裁切。`build_all.sh`把TMPDIR和缓存放在`outputs/`，重建数据、正常/-O科学结果与审计、图、Notebook、三PDF，不覆盖交付图/CSV/PDF。PDF元数据可能使整文件字节变化，仍应检查每页像素。

## 数值与数据角色范围

- 主输入x在[-1,1]，有限y绝对值≤1000；每份开发30至300行，重复2至200份，最终测试20至10000行。
- 候选1至30个，不重复；次数0至9，λ在[0,10]，常数候选只允许λ0。最小内层训练行数须大于系数个数。
- 外/内折数2至10；各折训练/验证是互补集合，验证折恰好覆盖全部行一次。合并MSE按验证行数加权，不偷换为不等大小折的简单平均。
- 嵌套搜索最多250000个协议估计的内层拟合单位；不是任意规模通用搜索服务。
- 纸笔例子单独支持x绝对值≤5、y≤20；主实验的x范围不能拿纸笔例外放宽。
- 拒绝bool、字符串、complex、NaN、无穷及非零绝对值小于1e−100的原始数值。拟合秩不足、非有限系数或系数绝对值超过1e8会明确失败，不裁剪或过滤结果。
- JSON计算和序列化完成后再原子替换，拒绝覆盖源/给定输入、符号链接或多硬链接输出；图/PDF构建不声称有多文件事务语义。

主实验同样可能出现数值和统计不稳定，均须照实诊断。独立SciPy QR与实际sklearn对照不只核对最终平均分。Oracle风险用独立求积核验，但没有参与选择。

## 统计边界

嵌套CV评价外层训练规模下的完整选参/拟合程序；不是最终全数据模型的精确风险。五个外折训练集重叠，不是五个独立样本，未作简单独立折t检验。80份真正独立合成数据上的MCSE是模拟均值误差，不是现实部署保证。

主数据嵌套乐观差比非嵌套还大；80次平均方向不同，且有27/34次负差以及2次CV分数反向，均保留。具体来源见`sources.md`。
