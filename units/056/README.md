# 第056讲 可信分类阶段项目

原166讲目录第056讲。直接先修007、025至026、042至043、045至048、050至053、055。整合评价总体、训练内预处理、嵌套选择、阈值、封存测试、配对区间和分组错误分析。

- [讲义PDF](lecture.pdf) · [可编辑源](lecture.md)
- [独立实验PDF](lab.pdf) · [完整答案PDF](answers.pdf)
- [真实执行Notebook](experiment.ipynb) · [带注释代码](experiment.py)
- [独立审计](test_experiment.py) · [全部运行结果](experiment-result.json)
- [数据字典](data/README.md) · [固定协议](data/protocol.json) · [原始来源](sources.md)

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python test_experiment.py --report outputs/result.json
python plots.py --report outputs/result.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

CPU离线实验，Python3.12。PDF重建另外需要build-requirements.txt、Node.js与npm install、Pango和Noto CJK字体；bash build_all.sh在outputs/rebuild生成独立重建物。

76次拟合和120条阈值评分完整保存，最终600行测试成本0.475，先验基线0.583333，差−0.108333。配对区间以已拟合模型为条件，不含重新训练/选择的不确定性。95.6%召回同时有241次误报，负结果如实保留。数据全为合成教学数据，不代表现实部署已经通过验证。
