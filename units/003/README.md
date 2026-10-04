# 003 Python 执行与基本运算

主问题：怎样让程序可靠地重复手算而不改变问题？先修为第一讲，不要求完成第 2 讲或已有 Python 基础。

lecture.pdf 含连贯正文、12 幅机制图、逐步代码、10 道练习与详解。lab.pdf 提供逐格实验步骤、故障对照与加练答案。experiment.ipynb 的 13 个代码单元与 experiment.py 分段代码一致；数据均为合成，无付费 API 或外部下载。

## 运行

Python 3.10+，计算仅依赖标准库：

```bash
python experiment.py
```

末尾应显示 PASS: 15 explicit checks。训练候选 MAE 为 [10,0,7.5]，选中 B 的参数 2、10，教学测试 MAE 5，固定训练均值基线测试 MAE 15。Notebook 请重启内核并从头顺序执行。不要使用 python -O；优化模式会去掉 assert 调试检查。

为避免超前引入文件读写，本讲数值直接写在代码列表中，data/hand_calculation.csv 仅用于核对，修改 CSV 不会改变脚本。第 4 讲再学习数据读取函数。

## 数值与边界

包含括号错误、共享列表、错误累加器和浮点近似的显式反例。它们的错误输出是被标记的教学对象。练习无代表性现实数据，不支持房价部署。源代码存在并不等于统计结论已经成立。

## 可选重建图

已附全部 PNG，无需重建即可学习。make_figures.py 需要 matplotlib 与 Noto Sans CJK。当前字体位置按 Linux 设置为 /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc；其他系统需改为本机字体路径。计算实验本身没有这项依赖。

实际 Python 版本、运行方式、视觉和独立检查见 verification.json，参考网页核验见 source-checks.json。公开教材保留实际检查的限制，不把其他平台或浏览器界面当作已经测试。
