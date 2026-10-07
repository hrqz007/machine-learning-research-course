# 第062讲 梯度提升的函数视角

面向零基础学习者的完整中文教材单元。直接先修：14、28、32、42、58、61。

## 阅读顺序

1. lecture.pdf：连贯推导、完整手算、六张解释图与实际结果。
2. lab.pdf：可独立使用的操作手册、环境、数据字典、命令与验收。
3. answers.pdf：十道练习及逐步解答。
4. experiment.ipynb：新进程中实际执行，保留全部输出和六张图。

## 重现

```bash
conda env create -f environment.yml
conda activate ml062
python experiment.py --out outputs/my-run/result.json
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-optimized.json
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

普通和-O测试均为120项显式检查。检查包括独立手算、边界输入、逐轮结果、CSV和摘要同时篡改拒绝、报告篡改拒绝及不安全输出链接拒绝。

完整重建需额外安装build-requirements.txt中的Python构建依赖、package.json中的MathJax 3.2.2，以及系统Pango和Noto Sans/Serif CJK字体。首次安装需联网；已安装环境中的实验和构建无网络请求。然后运行bash build_all.sh。可用PYTHON_BIN指定已有独立环境，NODE_PATH指定外置MathJax依赖目录。脚本不会安装软件或覆盖冻结PDF与报告。

Notebook使用新Python进程中的真实IPython InProcessKernel；浏览器UI、外部套接字内核传输及干净Conda安装未列为已验证。PDF使用本地MathJax公式SVG和WeasyPrint，源码、字体要求与依赖均随包说明。

## 文件说明

- data/：原创非个人合成数据、生成记录和数据字典
- generate_data.py：固定种子的字节级再现
- experiment.py：固定实验配置与库对照
- test_experiment.py：显式异常检查，支持-O
- figures/ 与 plots.py：六张图及其可重建来源
- experiment-result.json：实际冻结数值，不能只凭文件存在判定通过
- test-result.json、test-result-optimized.json：普通与优化模式的实际结果
- sources.md、source-checks.json：一级资料及实际检索记录
- verification.json：最终构建、页数、图、Notebook和审核范围

探索新配置请使用独立目录并记录变化；不要用测试集曲线选择超参数后仍称其为封存评估。
