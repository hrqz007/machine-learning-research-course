# 第054讲 特征设计与维度问题

核心问题：多加特征为何有时更差，筛选会制造什么偏差？

原166讲目录第054讲，直接先修009、012、048、052。从四行零相关交互表推导表达能力，逐步推导独立正态平方距离集中，再运行signal/null两机制的嵌套筛选、全数据标签筛选和固定80列对照。

- [讲义PDF](lecture.pdf) · [可编辑源](lecture.md)
- [独立实验PDF](lab.pdf) · [源](lab.md)
- [完整答案PDF](answers.pdf) · [源](answers.md)
- [真实执行Notebook](experiment.ipynb)：14代码格、6张实际重建图
- [实验](experiment.py) · [独立数学审计](audit.py) · [绘图](plots.py)
- [数据字典](data/README.md) · [固定协议](data/protocol.json) · [全结果](experiment-result.json)
- [原始来源](sources.md) · [执行和视觉范围](verification.json)

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py --report outputs/result.json --out outputs/audit.json
python plots.py --report outputs/result.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

Python3.12及Anaconda environment.yml。完整重建另需build-requirements.txt、Node.js、package.json的MathJax3.2.2、Pango与Noto CJK字体，再运行bash build_all.sh。重建输出outputs/rebuild，不覆盖交付文件。默认输入相对脚本目录；显式输出相对调用位置。

两机制各6份，360行80列；160开发/200独立确认，外层4折、内层3折选k。signal中nested和global_leaky最终确认MSE完全相同1.025244，但后者外层CV更乐观；null中泄漏CV较低而确认更差。全80列Jaccard稳定性1却在本固定alpha=1下更差。保留这些反例，不声称高维总失败或筛选具有因果含义。

独立审计按正确来源逐列复算Pearson，另用正规方程求解核对144个外层、1152个内层和36个确认预测。Jaccard是描述性指标，无机会校正；折间训练有重叠。Notebook在新Python进程真实IPython进程内内核执行，网页界面和外部socket内核未测试。PCA、wrapper、L1仅概念说明，未列入实测排名。
