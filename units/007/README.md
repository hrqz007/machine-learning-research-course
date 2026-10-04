# 007 第一个可重现实验

固定60条合成房屋、36/12/12显式划分、小网格直线和训练均值基线，完整记录训练、验证、冻结与测试。先读 lecture.pdf 与 lab.pdf，练习与详解已在两份文档中。

## 运行

Python 3.11+、NumPy，CPU、离线，无付费接口。已有Anaconda可使用现有课程环境。必要时安装 requirements.txt 中依赖。

```bash
python experiment.py --output results_run1
python experiment.py --output results_run2
python experiment.py --self-test
```

不要使用 python -O。Notebook 需要Jupyter，重启内核后全部运行。数据与配置路径默认相对于脚本，输出目录可自行指定。不同输入或代码版本不能覆盖旧输出目录，应使用新目录。

预期选择 linear_grid，斜率2、截距10；教学测试 MAE=14/3≈4.666667，训练均值基线测试 MAE=97/3≈32.333333，单位万元。参考值由独立有理数算术复核；expected_result.json 保存数值与容差。

## 文件与边界

data/houses.csv 与 data/split.csv 是固定输入。config.json 保存候选及来源种子。generate_data.py 仅用于在新目录重建来源，正常实验不重新生成数据。结果包括版本摘要、配置、实际环境、逐条测试预测及完整分数。

公开合成标签并非真实盲测，单次固定种子不构成稳健性证明。实际执行和环境限制见 verification.json，来源核查见 source-checks.json。matplotlib 仅可选图源重建需要，计算依赖只有NumPy。

可选图源 make_figures.py 使用 Linux 的 Noto Sans CJK 字体路径 /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc；其他平台重建时需改为本机位置。全部教学PNG已提供，不重建也可完整学习。
