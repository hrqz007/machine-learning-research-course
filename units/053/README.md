# 第053讲 调参与实验预算

核心问题：怎样在有限预算内比较算法而不偏袒某一方？

原166讲目录的第053讲，直接先修007、026、051至052。三条搜索共享同一25配置域和数据边界，比较6点子网格、6次随机与12→4→1淘汰；每份每路线1080搜索树+60重拟合树。包含树模型最小先修、预算手算、选择噪声、慢热反例和Bayesian优化动机，不把概念介绍伪报实测。

- [讲义 PDF 10页](lecture.pdf) · [可编辑源](lecture.md)
- [独立实验 PDF 4页](lab.pdf) · [源](lab.md)
- [完整答案 PDF 5页](answers.pdf) · [源](answers.md)
- [真实执行Notebook](experiment.ipynb)：14代码格、7张实际重建图
- [数值代码](experiment.py) · [独立算术/协议审计](audit.py) · [图代码](plots.py)
- [数据字典](data/README.md) · [固定协议](data/protocol.json) · [全部结果](experiment-result.json)
- [来源与边界](sources.md) · [运行和视觉核验范围](verification.json)

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py --report outputs/result.json --out outputs/audit.json
python plots.py --report outputs/result.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

推荐Python3.12；Anaconda配置在environment.yml。所有默认输入相对脚本目录，显式输出相对调用位置。完整重建另需build-requirements.txt、Node.js与package.json的MathJax3.2.2、Pango和Noto CJK字体，然后运行bash build_all.sh；脚本输出outputs/rebuild，不覆盖交付数据和PDF。可设置TMPDIR、MPLCONFIGDIR、IPYTHONDIR、XDG_CACHE_HOME到已创建的可写目录。

六份独立数据平均测试MSE：grid 3.209982、random 3.290078、halving 3.219101。random有4/6份更差，halving有3/6份更差；负结果与平局完整保留。等树数不是等时间或等FLOPs，单一合成机制不能代表所有任务。

522次主要搜索拟合、18次最终重拟合及1次独立失败演示均记录。独立审计重算所有MSE和最终模型，并核对预算、幸存集合、划分和测试标签干预。Notebook在新Python进程的真实IPython InProcessKernel执行；网页Jupyter界面和外部socket传输未测试。计时字段与PDF二进制元数据可随重跑改变，科学内容单独核对。
