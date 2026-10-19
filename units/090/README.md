# ML090 完整材料包

阅读顺序：lecture.pdf/lecture.md → lab.pdf/lab.md → experiment.ipynb → answers.pdf/answers.md。

## 包内内容

- 三份PDF与可编辑Markdown：主讲义、实验手册、练习答案
- experiment.py：带解释的CPU实验；test_experiment.py：可重复数值验收
- 已执行experiment.ipynb：实际IPython顺序执行输出、测试和图像
- outputs/results.json：逐项原始结果；run日志、测试日志与Notebook执行报告
- figures：原创图；verification.json：文件与PDF/执行检查记录
- build_pdf.py、pdf.css、mathjax_render.cjs：沿用087讲的离线公式排版工具

## 从零运行

推荐Python 3.12，在本目录执行：

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python experiment.py
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python execute_notebook.py
```

只用CPU，无数据下载、无GPU、无API密钥。训练数据/统计/预算详见讲义和results.json。结果在outputs与figures生成，重复执行会覆盖同名生成文件。修改前请自行备份。笔记本采用与087讲相同的独立进程内IPython逐单元执行，不依赖本地socket内核；这不验证Jupyter浏览器界面。

## 重建PDF（可选）

另安装Node.js与Noto CJK字体、系统Pango/相关WeasyPrint运行库，再执行：

```sh
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
```

公式在本地MathJax转成SVG后嵌入PDF；构建不需要远程公式服务。源码中图路径相对本目录。已交付PDF可直接阅读，无需安装构建环境。

## 复现边界

原始运行Python3.12、PyTorch2.14.1+cpu、CPU单线程；具体环境另见outputs/environment.json。跨版本/CPU结果可能略有差别，耗时不是性能承诺。教学数据只支持本讲机制结论，不能代替真实领域基准。故意错误分支用于诊断，不应复制到正式训练中。
