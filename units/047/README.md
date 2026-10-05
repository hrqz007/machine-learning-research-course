# 047 校准与预测不确定性

先读 **lecture.pdf**，再完成 **lab.pdf**，最后对照 **answers.pdf**。PDF公式为本地MathJax SVG，中文字体嵌入，无需联网阅读。此单元延续总大纲，不更改课程编号或先修关系。

## 教学内容与文件

- 四行二分类实例：三次逐样本前向、两次同步梯度更新、局部导数链及解析温度最优。
- 正类可靠性与最高置信度可靠性分别计算；Brier、NLL与准确率并列。
- 固定合成分数，400行独立温度校准、4000行评价；不声称训练神经网络。
- 60/99/400训练、校准、评价回归；有限样本修正秩、无穷情况、并列语义。
- 120次独立重复保留整体/分组/噪声改变/错误复用结果，MCSE按完整重复计算。
- 十张原始实图、带真实输出的`calibration_uncertainty.ipynb`、所有代码/数据/环境及重建脚本。

`reference_results.json`保留完整默认科学输出，`reference_audit.json`保存作者可复跑数值参照结果，`verification.json`记录验证范围。它们不是第三方或独立QA签名。私有审核过程不在本包。

## 数值运行

在Python 3.12隔离环境安装`requirements.txt`。默认数据固定在`data/`，不依赖网络。一般命令：

```bash
python experiment.py --out outputs/result.json
python audit.py --out outputs/audit.json
python -O experiment.py --out outputs/result-optimized.json
python -O audit.py --out outputs/audit-optimized.json
python plots.py --report outputs/result.json --directory outputs/figures
python generate_data.py --directory outputs/regenerated-data
```

脚本以源码位置定位数据，因此可从任意工作目录调用绝对路径。`experiment.py`接受`--data-directory`和`--config`；自定义数据必须符合完整字段、行数和范围约束。`audit.py`及十图针对默认教学协议；自定义实验需要自己的验证与图注，不能把旧解析参照硬套到新结果。

## Notebook

首个物理单元仅用标准库验证16项输入/源码/环境SHA-256，之后才导入教学模块。Notebook位于完整单元根目录或其子目录时，可自动找到单元根。不要只复制一个ipynb而漏掉CSV与源码。

```bash
python build_notebook.py
python execute_notebook.py calibration_uncertainty.ipynb
```

重新执行请使用课程副本以保留原交付输出。`execute_notebook.py`在新的Python进程里启动InProcessKernel并逐格运行，输出写回Notebook。它测试真实Python内核执行，不声称测试浏览器Jupyter界面或进程外网络传输。不同执行可能把同一stdout拆成不同数量的相邻消息；比较时按同一输出通道连接相邻文本，并逐字比较内容与PNG，不删除输出或改变数值。

## PDF与全流程构建

阅读PDF不需要构建依赖。重建时安装Python依赖`build-requirements.txt`、官方Node.js和`package.json`中固定的MathJax 3.2.2；安装Noto Sans/Serif CJK字体。使用平台正常软件包渠道，例如`npm install`。PDF构建不请求外部URL。

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
PYTHON=python bash build_all.sh
```

`build_all.sh`将缓存与TMPDIR限制在单元的`outputs/`内，重建四份CSV、完整结果/审计、普通与-O结果、十图、Notebook、三份PDF。它不会覆盖原交付PDF、图或CSV。生成Notebook首格绑定当前源码字节；若修改算法，必须重新确认科学依据，不能只更新摘要来宣称原验证仍成立。

## 输入与失败契约

- 数值向量一维，1至20000行；标签仅0/1；拒绝bool、字符串、complex、NaN、无穷。
- 一般logit/响应绝对值≤1000；回归x绝对值≤100；非零原始数绝对值≥1e−100。具体协议还有更窄合理范围，见`protocol.py`。
- 逆温度a位于[0,4]，温度无限用独立标记；秩亏线性设计拒绝。协议与全部读入数据先验证，再进行第一次拟合。
- alpha在[1e−6,1−1e−6]；秩将`str(float(alpha))`解释为精确十进制分数；第k小值直接选取，k>n返回无限标记。
- 默认正态机制数学上无界，实现只承诺上述有限域；默认固定数据全部满足。无静默裁剪，也无任意病态矩阵高精度承诺。
- JSON输出先完整计算及序列化，再原子替换；拒绝覆盖教学文件、给定输入、符号链接或多硬链接输出。失败不破坏原JSON。图/PDF生成是独立的可信构建工具，不声明具有多文件事务语义。

## 证据与限制

见讲义第18节和`sources.md`。主要参照为Guo等温度校准论文、Angelopoulos/Bates conformal教程、NumPy及scikit-learn官方文档。默认数字来源于随包代码实际执行。

Conformal覆盖是可交换条件下的边际陈述，不能解读为每个x、每组或任意分布改变下90%。这里保留困难组欠覆盖、噪声改变失败及错误数据复用反例。有限正温度保留argmax；无限温度不沿用这一结论。
