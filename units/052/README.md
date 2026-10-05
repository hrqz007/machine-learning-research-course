# 第052讲 无泄漏的数据处理流水线

核心问题：每一步清洗与变换应该在哪里学习参数？

先修006、050至051。用同一四行表逐项完成插补、截断、缩放、独热、前向、损失、导数、同步更新与下一前向；再用三个分开的合成机制检查预处理边界、监督选列和真实TargetEncoder内部cross-fit。保留缩放越界4次变差、OLS预测不变以及正确编码略差于均值基线的负结果。

## 阅读与运行

- [正文 PDF 15页](lecture.pdf) · [可编辑源](lecture.md)
- [独立实验指南 7页](lab.pdf) · [源](lab.md)
- [完整答案 8页](answers.pdf) · [源](answers.md)，含E1至E16、G1至G10、L1至L8
- [真实执行Notebook](experiment.ipynb)，13个代码格、10张实际重建图
- [实验脚本](experiment.py) · [独立审计](audit.py) · [绘图](plots.py)
- [固定协议](data/protocol.json) · [数据字典](data/README.md) · [外层逐行归属](data/split_membership.csv)
- [完整结果](experiment-result.json) · [作者自查](test-result.json) · [执行与版面记录](verification.json)
- [来源](sources.md) · [来源核查范围](source-checks.json)

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py --report outputs/result.json --out outputs/audit.json
python plots.py --report outputs/result.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

推荐Python3.12。Conda环境见environment.yml。离线数值实验无需API或GPU；Notebook在新进程真实IPython InProcessKernel顺序执行，未测试浏览器Jupyter界面或外部网络内核传输。

## 完整重建

准备数值依赖和build-requirements.txt，安装Node.js与本目录package.json的MathJax3.2.2，系统需Pango、Noto Sans/Serif CJK和DejaVu字体。然后 `bash build_all.sh`。脚本不自动安装依赖，输出到outputs/rebuild，不覆盖交付PDF、数据或已执行Notebook。

构建支持从不同cwd调用；所有默认输入相对本脚本目录。可自行设置TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME、IPYTHONDIR；先创建可写目录。未设置时build_all.sh使用单元自己的缓存。单独build_notebook.py默认写outputs/unexecuted.ipynb，不清空交付Notebook。

## 结果和边界

12份独立数据先各平均四折。A新设备MSE正确2.037497、仅全局缩放2.037060，错误路线4份变差。B纯噪声正确筛选1.205764、全标签筛选0.874681。C正确内部cross-fit1.018423、外层训练自编码1.263601、全标签编码0.733944；均值基线1.008078。

三机制样本、目标、路径不同，不作跨机制因果排名。A不是时间预测，C没有同类别共享随机效应。TargetEncoder.fit_transform的训练特征来自内部留出，外层验证transform来自完整训练映射；fit().transform()不等价。对有组或时间结构的任务，内部来源必须另按目标设计。

336次主要预测器拟合完整保留，独立手工预处理、Pearson选列、计数编码与SciPy QR核验全部12×4折；另有Fraction手算、有限差分、标签/X干预及常见输入/失败保护。作者自查与独立课程审阅不同，不构成真实缺失机制识别或现实部署证据。
