# 第057讲 最近邻与局部预测

原166讲目录第057讲。由邻居手算进入单位与距离、分类投票、回归平均、逆距离及零距离、验证选k、偏差方差、噪声维度和复杂度。

- [中文讲义](lecture.pdf) · [讲义源](lecture.md)
- [独立实验](lab.pdf) · [完整答案](answers.pdf)
- [真实执行Notebook](experiment.ipynb) · [透明kNN实现](knn.py)
- [实验程序](experiment.py) · [独立审计](test_experiment.py)
- [完整数值](experiment-result.json) · [数据字典](data/README.md) · [原始来源](sources.md)

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python test_experiment.py --report outputs/result.json
python plots.py --report outputs/result.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

Python3.12 CPU离线运行。PDF重建另需build-requirements.txt、Node.js、npm install、Pango与Noto CJK字体；bash build_all.sh生成outputs/rebuild。实验与答案保留逆距离未全面胜出、方差不严格单调等负结果。

标准化等权路线验证选k=5，测试错误39/360；原始等权65/360。结果适用于当前固定合成实验，未作现实部署声明，也未按测试挑选最终路线。
