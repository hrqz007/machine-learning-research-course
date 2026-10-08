# 第070讲 流形与可视化的边界

固定连续Swiss roll和单Gaussian负对照，比较PCA、t-SNE、UMAP参数与种子。讲义完整解释流形、环境/内在距离、两个算法目标、手算邻域指标以及二维簇/距离的解释边界。所有14组卷曲表面配置与2组负对照坐标公开保存。

## 重现

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/my-run/result.json
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-O.json
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

212项普通/-O显式检查通过，包括完整种子实验重算、从零排名可信度与库核验、平移/缩放不变性、弧长导数、输入错误及数据/报告篡改。Notebook13个真实执行代码单元、7张嵌入图，另有7张完整彩色PNG。

核心边界：所有点参与探索性映射，指标不是新样本泛化成绩；高trustworthiness不等于相同比例的近邻重合，也不保证表面全局距离。颜色和真实弧长不参与算法拟合。两个种子仅为有限敏感性检查。UMAP首次编译计时不能直接用于速度排名。

完整PDF构建另需build-requirements.txt、package.json与Pango/Noto CJK/DejaVu字体。bash build_all.sh在outputs/rebuild重建；可设PYTHON_BIN和NODE_PATH，脚本不安装依赖、不覆盖冻结教材。固定单线程和种子有助于复现。首次安装联网，实验本地离线。真实Notebook内核为新Python进程中的IPython InProcessKernel；Jupyter网页、外部socket、干净Conda安装及样本外映射未验证。
