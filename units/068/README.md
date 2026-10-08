# 第068讲 经典方法比较项目

固定任务与搜索次数预算下，比较常数概率、逻辑回归、随机森林与K近邻。阅读lecture.pdf，再按lab.pdf重现，最后用answers.pdf核对推理。三份PDF共13页，7张彩色图；Notebook为15个真实执行代码单元，包含7张嵌入图。

## 运行

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/my-run/result.json
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-O.json
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

固定450/225/225拆分，每族三个配置。15近邻由验证集选中，森林测试损失更低；保留原选择，不根据测试集追改方案。概率阈值0.5，测试区间条件于已拟合模型，未覆盖全流程选择不确定性。计时为实测且随机器改变，序列化字节不是峰值内存。

63项普通/-O显式检查覆盖独立手算、指标库对照、完整结果重算、拆分、预算、非法输入与数据/报告篡改。所有数据为原创可重建合成数据。

完整重建另需build-requirements.txt、package.json及Pango/Noto CJK/DejaVu字体。运行bash build_all.sh；可设PYTHON_BIN和NODE_PATH使用已有环境。输出统一放outputs/rebuild，不自动安装软件，不覆盖冻结材料。实验运行离线；首次安装依赖需联网。Notebook真实内核为新Python进程中的IPython InProcessKernel；Jupyter网页与socket传输及干净Conda重建未验证。

lecture.md、lab.md、answers.md是可编辑教材源文；experiment-result.json与两份test-result保存本次真实输出；generate_data.py重建CSV；figures由plots.py产生；sources.md与source-checks.json列一级资料；verification.json说明验证范围。
