# 第064讲 最大间隔与软间隔

从零推导函数间隔、几何距离、硬间隔约束、松弛与hinge；独立原始优化对照线性SVC。直接先修：9、16、37、42、48。

- [正文](lecture.pdf) · [独立实验指南](lab.pdf) · [详细答案](answers.pdf)
- [执行Notebook](experiment.ipynb) · [实验代码](experiment.py) · [原始求解器](margin.py)
- [数据说明](data/README.md) · [实际结果](experiment-result.json) · [验证记录](verification.json) · [资料](sources.md)

```bash
conda env create -f environment.yml
conda activate ml064
python experiment.py --out outputs/my-run/result.json
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-optimized.json
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

真实实验选C=0.03，封存测试准确率0.95；原始求解与库目标差约1.67e-10。异常点和单位变换仅为预设诊断，不反馈选参。六张图保存于figures并嵌入实际执行Notebook。

完整PDF构建需要build-requirements.txt、package.json列出的MathJax3.2.2与系统Pango/Noto CJK字体。首次安装需联网；已有环境运行bash build_all.sh离线重建到outputs，不覆盖冻结文件。可设置PYTHON_BIN和NODE_PATH。

Notebook验证为新Python进程中真实IPython InProcessKernel；未验证浏览器UI、外部套接字内核或全新Conda安装。计时为本机观察值，重现检查不要求时间逐位相同。
