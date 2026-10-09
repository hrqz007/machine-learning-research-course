# 第073课 聚类结果的验证

本单元严格对应catalog/curriculum.json的073号，承接第072课后的既定课程，包含可独立自学的中文教材、实验和答案。

## 阅读与复现

1. [lecture.pdf](lecture.pdf)与[lecture.md](lecture.md)：从术语、手算到机制与边界。
2. [lab.pdf](lab.pdf)与[lab.md](lab.md)：固定问题、完整命令、预期结果与验收。
3. [experiment.ipynb](experiment.ipynb)：真实执行计算及内嵌图。
4. [answers.pdf](answers.pdf)与[answers.md](answers.md)：逐项推理、冻结数值和常见错误。

Python 3.12 CPU环境，安装requirements.txt后在本讲目录运行：

```bash
# 真实执行全部实验，写到新目录。
python experiment.py --out outputs/rebuild/result.json
# 普通模式验证数学性质、反例和独立实现。
python test_experiment.py --report outputs/rebuild/result.json --out outputs/rebuild/test.json
# 优化模式确认显式检查仍存在。
python -O test_experiment.py --report outputs/rebuild/result.json --out outputs/rebuild/test-optimized.json
# 绘图与新进程真实内核执行。
python plots.py --report outputs/rebuild/result.json --directory outputs/rebuild/figures
python execute_notebook.py --out outputs/rebuild/experiment.ipynb
```

build_all.sh重建完整输出到outputs/rebuild。PDF额外安装build-requirements.txt并运行npm install，系统需Pango与Noto CJK字体。公式由本地MathJax渲染，阅读PDF不访问外部服务。

## 材料与边界

- data/：原创合成CSV、字段解释、可重建种子与摘要。
- experiment.py、教学算法模块、test_experiment.py：逐行核心注释与显式边界检查。
- experiment-result.json、两种test-result：真实冻结参考结果。
- plots.py、figures/：四幅可重新生成的机制图。
- sources.md、source-checks.json：一手出处及实际核验范围。
- verification.json：真实运行环境、Notebook和PDF验收记录。

公开教学审计数据不等于未来未知盲测。Notebook验证真实IPython计算单元，不声称验证Jupyter浏览器UI或外部socket内核。无CI运行承诺；测试通过仅覆盖列出的性质与本批数据。
