# 第055讲 不平衡与有偏标签

核心问题：稀有类别和选择性标注怎样改变学习目标？

原166讲目录第055讲，直接先修002、018、045至046、050、052。逐步推导代价阈值、加权交叉熵输出、先验赔率修正、Wilson少数类区间、选择标注IPW与类别条件标签翻转；自然测试不平衡。明确区分概率含义、决策成本和标签机制。

- [讲义PDF](lecture.pdf) · [可编辑源](lecture.md)
- [独立实验PDF](lab.pdf) · [源](lab.md)
- [完整答案PDF](answers.pdf) · [源](answers.md)
- [真实执行Notebook](experiment.ipynb)：14代码格、7张实际重建图
- [实验代码](experiment.py) · [独立指标审计](audit.py) · [绘图](plots.py)
- [数据字典](data/README.md) · [固定协议](data/protocol.json) · [全结果](experiment-result.json)
- [原始来源与访问范围](sources.md) · [执行和视觉核验](verification.json)

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py --report outputs/result.json --out outputs/audit.json
python plots.py --report outputs/result.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

Python3.12，Anaconda environment.yml。完整重建另需build-requirements.txt、Node.js、package.json中的MathJax3.2.2、Pango与Noto CJK字体，再运行bash build_all.sh。输出outputs/rebuild，不覆盖交付数据和PDF。默认输入相对脚本目录，显式输出相对调用位置。

六份各7200行：训练2000、验证1200、自然测试4000。42个实际拟合模型支持八条主要路线与四条标签对照，每条均保存逐行概率与计数；oracle使用生成器真值，明确额外信息。自然模型1/9阈值平均成本0.324083，验证调阈值反而0.330958，保留负结果。原始平衡权重提高召回却损坏自然分布概率含义；修正改善Brier后还要按目标成本选阈值。

72组指标独立核对混淆计数、代价、Brier、Wilson区间、系数概率前向、验证阈值账本及训练复制来源，正常/-O通过。Notebook使用新Python进程真实IPython进程内内核，网页Jupyter与外部socket未测试。Wilson是固定分类器逐个区间，不含训练不确定性，不作多方法同时推断。oracle IPW依赖模拟器已知标注机制；未证明现实机制可识别，未实现EM先验估计或完整标签噪声修正。
