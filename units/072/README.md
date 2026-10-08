# 第072课 密度与层次聚类

本单元对应166单元课程目录的第072课。正文是可独立阅读的中文入门教材，实验包含从零实现、数值与反例检验、可重建人工数据和已执行Notebook。

## 阅读顺序

1. lecture.md 或 lecture.pdf：概念、推导、手算与失败机制
2. lab.md 或 lab.pdf：按步骤实验和验收要求
3. experiment.ipynb：已执行的中间结果与内嵌图
4. answers.md 或 answers.pdf：完整解释与冻结参考数值

## 复现

Python 3.12，CPU环境。安装 requirements.txt 后运行：

```bash
python experiment.py --out outputs/rebuild/result.json
python test_experiment.py --report outputs/rebuild/result.json --out outputs/rebuild/test-result.json
python -O test_experiment.py --report outputs/rebuild/result.json --out outputs/rebuild/test-result-optimized.json
python plots.py --report outputs/rebuild/result.json --directory outputs/rebuild/figures
python execute_notebook.py --out outputs/rebuild/experiment.ipynb
```

build_all.sh包括上述过程与PDF重建。PDF额外安装build-requirements.txt和package.json，系统需Pango及Noto CJK字体，Node.js用于本地MathJax SVG渲染。先运行 npm install，再运行 python build_pdf.py。数值实验不依赖这些PDF组件。

默认重建输出写入outputs，不覆盖冻结PDF、Notebook、数据或数值报告。生成数据可用 python generate_data.py，默认写outputs/generated-data。data/generation.json记录种子与SHA-256，加载时还与生成器字节核对。

## 文件说明

- density_hierarchy.py：中文注释的教学算法
- experiment.py 与 experiment-result.json：确定性实验和冻结结果
- generate_data.py 与data：合成数据及生成记录
- test_experiment.py 与两份test-result报告：显式检查，普通及-O模式
- plots.py 与figures：由报告和数据生成的概念图
- execute_notebook.py：新进程中的真实IPython单元执行
- sources.md 与source-checks.json：已核验的一手来源
- verification.json：本次执行与产物验收摘要

## 使用边界

这些是教学规模的直接实现，重点是可审计定义和数学性质，不能直接当作大规模生产算法。所有生成分组只用于解释人工构造，从不输入聚类。输出编号没有自然类别含义；测试通过也不代替真实研究的外部验证。

Notebook执行验证计算单元和嵌入输出；不声称测试Jupyter浏览器界面或外部网络内核传输。PDF采用本地公式SVG与中文字体，离线阅读无需MathJax服务。
