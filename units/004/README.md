# 004 函数与可调试程序

主问题：怎样把实验拆成可检查的步骤，而不是一串难复用的代码？直接先修为第 3 讲；不假定已有软件工程经验。

lecture.pdf（另附可编辑 lecture.md）含 15 节连贯讲解、11 幅彩色机制图、12 道练习与详解。lab.pdf（另附可编辑 lab.md）提供亲手实现预测与 MAE 的练习、数据读取、故障定位和跨目录验收。experiment.ipynb 有 16 个代码单元，前半直接编写学习版函数，后半对照完整模块。test_experiment.py 用标准库核验 122 项明确检查。

## 运行

Python 3.10+，普通 CPU；计算和检查仅用标准库，无网络、付费 API、GPU 或数据下载。请在本讲目录运行：

```bash
python experiment.py
python test_experiment.py
```

已测输出：选中 B；训练候选 MAE 为 A 10、B 0、C 7.5；测试 MAE 5；训练均值基线为 150，测试 MAE 15，单位均为万元。真实读取 data/train.csv、test_inputs.csv、test_labels.csv；数据均为手工合成，不能支持现实房价部署。

主程序默认更新 outputs/result.json，含参数、逐样本误差、指标与输入文件内容摘要。重跑会覆盖同一结果文件。若要保留多个实验，请指定不同输出目录：

```bash
python experiment.py --output-dir outputs/my-run
python experiment.py --help
```

从课程根目录也可以运行 python units/004/experiment.py。默认输入随脚本位置查找；显式提供的相对 --data-dir 或 --output-dir 按当前工作目录解释。已用无关临时目录启动新进程核验。

## Notebook

需要 Jupyter 和 Python 3 内核；从本讲目录打开 experiment.ipynb，重启内核后全部顺序执行。它与 experiment.py 共用严格函数，另含自己编写函数的教学过程；并非把整个文件一次导入后宣称实验完成。

16 个单元已在新 Python 进程的真实 ipykernel InProcessKernel 中顺序执行，输出已保存。外部 socket 传输与 Jupyter 浏览器 UI 未测试，其他操作系统未实装验证。详情见 notebook-check.json。

## 输入与失败契约

- 数值核心仅接受普通 int/float，排除 bool，且须能转换成有限 float；数值预测核心允许零与负自变量，房屋数据读取另要求面积大于零
- MAE 要求非空等长 list/tuple、有限元素与有限累计；极端尺度可能被明确拒绝，不声称任意精度
- CSV 表头按名称与顺序精确匹配；不接受缺值、额外字段、坏引号、空/重复编号、非数值或非有限数
- evaluate 按 id 对齐，拒绝缺失、额外和重复 id；不依赖标签行顺序
- fit_candidates 仅接收训练记录；平局选列表中首个候选；训练均值基线也只从训练数据计算
- 不吞掉未预期异常，不悄悄跳过坏行或复用旧结果；导入 experiment 不自动训练或写文件

测试清单包含已知答案、单样本、不变性、输入拒绝、CSV 故障、非有限值、溢出、输入不变性、独立评价、JSON 读回、不同工作目录运行与 -O 下验证保留。Notebook 用 assert 调试验算，应按普通模式运行；关键输入验证与独立测试器使用 if + raise。

## 图与核验记录

已有 11 张 PNG，可以直接阅读，无需安装绘图库。重建图时 make_figures.py 需要 matplotlib 和中文字体；默认寻找 Linux Noto Sans CJK，其他系统可用 CJK_FONT 环境变量指定字体文件。图均为原创技术示意。

source-checks.json 记录 2026-10-04 核验的 Python 官方资料；test-result.json 记录标准库脚本的实跑清单；notebook-check.json 记录单元执行及局限。通过测试不等于证明所有输入都正确，也不等于证明合成数据代表真实应用。
