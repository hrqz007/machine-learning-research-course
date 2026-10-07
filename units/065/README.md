# 第065讲 对偶与核技巧

按拉格朗日乘子、弱/强对偶、Slater、KKT、支持向量、偏置区间、PSD核的顺序完整讲解。直接先修：10、12、16、64。

- [正文](lecture.pdf) · [独立实验指南](lab.pdf) · [详细答案](answers.pdf)
- [执行Notebook](experiment.ipynb) · [实验代码](experiment.py) · [小规模对偶](dual.py) · [独立原始求解](margin.py)
- [数据说明](data/README.md) · [真实结果](experiment-result.json) · [验证范围](verification.json) · [资料](sources.md)

```bash
conda env create -f environment.yml
conda activate ml065
python experiment.py --out outputs/my-run/result.json
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-optimized.json
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

固定齐次二次核与C=2，无超参数搜索。独立原始/对偶目标差约1.02e-13，显式/预计算核SVC测试分数差约1.78e-15；无噪声合成任务测试准确率1.0，不代表现实效果。包括无内部支持向量偏置区间的专项反例。

PDF重建需要build-requirements.txt、package.json的MathJax3.2.2、系统Pango和Noto CJK字体。首次安装需联网，已有环境bash build_all.sh离线重建到outputs，不覆盖冻结文件。可设置PYTHON_BIN和NODE_PATH。

Notebook在新Python进程内的真实IPython InProcessKernel执行，保留7代码格和6张嵌图。未验证浏览器UI、外部套接字内核传输或全新Conda创建。通用SLSQP仅用于小规模透明教学，不声称高效替代大规模SVM。
