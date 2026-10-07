# 第063讲 提升树的实际训练

连续讲解二阶叶值、分裂增益、缺失/类别、早停预算与独立校准。直接先修：39、51至53、55、62。

- [正文](lecture.pdf) · [独立实验指南](lab.pdf) · [详细答案](answers.pdf)
- [已执行Notebook](experiment.ipynb) · [实验代码](experiment.py) · [数据说明](data/README.md)
- [真实结果](experiment-result.json) · [普通测试](test-result.json) · [-O测试](test-result-optimized.json)
- [一级资料](sources.md) · [完整验证范围](verification.json)

## 运行

```bash
conda env create -f environment.yml
conda activate ml063
python experiment.py --out outputs/my-run/result.json
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-optimized.json
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

五份数据分工明确。四候选对四候选仅控制候选数，不声称等算力。实际校准略微变差，教材与图保留这一结果。测试重新执行实验而非只检查文件存在。

PDF重建需build-requirements.txt、package.json中的MathJax3.2.2、Pango和Noto Sans/Serif CJK字体；运行bash build_all.sh生成到outputs/rebuild，不安装软件或覆盖冻结产物。可用PYTHON_BIN和NODE_PATH指定已有依赖。首次环境安装需联网，实验与已装好环境中的构建离线运行。

Notebook使用新Python进程内的真实IPython InProcessKernel，保存输出和六张嵌入图。未验证浏览器Jupyter界面、外部套接字内核传输或干净Conda环境创建。时间测量是本机观察值，不要求跨机器精确一致。
