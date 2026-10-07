# 第066讲 核方法的选择与代价

面向零基础学习者的完整中文教材单元。直接先修：39、51、53至54、65。

## 阅读顺序

1. lecture.pdf：连续讲解、符号与维度、逐步推导、完整手算、6张彩色解释图及真实实验。
2. lab.pdf：可独立使用的操作手册、环境、数据字典、命令、验收与故障排查。
3. answers.pdf：逐题解析、常见错误与诚实的结论范例。
4. experiment.ipynb：新Python进程内实际顺序执行，13个代码单元与6张嵌入图，保留全部输出。

## 重现

```bash
conda env create -f environment.yml
conda activate ml066
python experiment.py --out outputs/my-run/result.json
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-O.json
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

普通与-O模式各执行175项显式检查，不靠assert。检查包括独立手算、库对照、非法输入、完整报告重算、CSV篡改、CSV与摘要同时篡改、报告篡改，以及符号链接/硬链接输出拒绝。

完整重建另需build-requirements.txt中的Python构建依赖、package.json中的MathJax 3.2.2，以及系统Pango、Noto Sans/Serif CJK与DejaVu Sans Mono字体。首次安装需联网，已安装环境的实验与构建不联网。运行bash build_all.sh，输出在outputs/rebuild；可通过PYTHON_BIN指定已有解释器，通过NODE_PATH指定外置MathJax依赖目录。脚本不自动安装软件，不覆盖冻结报告和PDF。

Notebook使用新的Python进程中的真实IPython InProcessKernel。Jupyter浏览器界面、外部socket内核传输及干净Conda重建未列为已验证。完整构建在已有、版本固定的独立Python环境中实际验证。

## 文件说明

- lecture.md、lab.md、answers.md及对应PDF：可编辑教材与阅读版
- data/、generate_data.py：原创非个人合成数据、来源字典与字节级复现
- common.py：输入检查、受保护输出与数据来源核验
- experiment.py、test_experiment.py：真实实验和普通/-O验收
- plots.py、figures/：所有图及可重建来源
- experiment-result.json：冻结本次实验数值与版本
- test-result.json、test-result-optimized.json：实际检查记录
- sources.md、source-checks.json：一级资料与重新检索记录
- verification.json：执行、Notebook、PDF与审核范围

核选择采用每族9次相同划分的验证拟合，再各重训一次。同等候选数不等于同等墙钟时间。线性基线与SVC损失实现不同，明确作为实用模型族比较。延迟含预热与15次原始观测；内存字段仅为选定数组和序列化大小，不是峰值RSS。计时重跑会有正常变化，因此资源图不保证逐字节一致。

探索不同数据或配置时另建实验，先写评估协议再看结果。
