# 第069讲 主成分分析的推导

从四点手算推导中心化、协方差谱、最大方差与最小重构的等价性，使用SVD从零实现PCA。lecture.pdf8页、lab.pdf3页、answers.pdf3页，7张原创彩色解释图；Notebook14个真实执行代码单元、7张嵌入图。

## 重现

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/my-run/result.json
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-O.json
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

109项普通/-O显式检查覆盖手算、SSE谱恒等式、正交性、SVD/eigh/库对照、符号与平移不变性、输入错误及数据/报告篡改。原始数据为原创低秩合成数据与独立预测反例，生成器固定种子69021。训练集fit均值、主轴和缩放，新样本只transform。

关键反例：第一主成分解释99.8431%训练方差，但分类测试准确率0.48，保留两输入为0.99。PCA优化输入平方重构，不保证保留预测信号。讲义完整解释n与n-1分母、轴符号与重复特征值不唯一性、原始尺度和标准化的区别。

完整重建另需build-requirements.txt、package.json及Pango/Noto CJK/DejaVu字体。bash build_all.sh输出到outputs/rebuild；通过PYTHON_BIN与NODE_PATH选已有环境，不自动安装，不覆盖冻结材料。首次安装需联网，运行离线。实际验证为已有固定版本Python环境和真实IPython InProcessKernel；干净Conda安装、Jupyter网页及外部socket内核未验证。
