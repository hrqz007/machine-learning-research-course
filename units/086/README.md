# 第086课 PyTorch与训练循环

按既定大纲先补Python类、继承、迭代器与生成器，再讲Tensor/autograd、Module、Dataset/DataLoader、device/dtype、清梯度、训练/评价状态和检查点。真实CPU PyTorch与NumPy对齐前向、全部梯度和一次更新。

## 学习顺序

lecture.md → lab.md → experiment.ipynb（构建后）→ answers.md。正文约7100中文字符，实验9道练习、4幅原创可重建PNG/SVG。NumPy基准独立包含在本目录，不依赖085目录导入。

## 环境与执行

冻结验证环境：Python 3.12.14、NumPy 2.3.5、PyTorch 2.14.1+cpu，CPU float64。requirements.txt用官方CPU索引安装真实torch；无torch时实验与验收会失败，不会伪造或跳过。

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python test_experiment.py --out outputs/test-result.json
python -O test_experiment.py --out outputs/test-result-optimized.json
python generate_data.py --directory outputs/generated-data
python plots.py --directory outputs/figures --result outputs/result.json
```

首次安装后，上述计算完全离线。核心计算只需requirements.txt；绘图另装plot-requirements.txt，Notebook另装notebook-requirements.txt，PDF另装build-requirements.txt及package.json对应Node依赖。environment.yml提供核心环境入口。默认输出在outputs，不覆盖冻结交付。

## 验收证据与边界

17参数审计同时比较logits、损失、四块全部梯度与一次SGD；完整训练180个epoch，验证挑选深拷贝检查点，恢复后测试一次。单独演示30→60→30的梯度累积、Dropout状态差别、eval不关闭autograd、detach断链和不等大小尾批。固定模型训练评价使用独立无打乱加载器，不消耗训练随机序列。

普通与-O各11项unittest均通过。experiment-result.json提供冻结实验结果，test_experiment.py可生成本次运行的测试统计，sources.md列出参考来源。主任务是合成数据迁移与流程教学，不能据此比较框架优劣或保证实际业务效果；未声称GPU路径已验证。

## PDF与可执行Notebook

最终教学文件：[正文PDF](lecture.pdf)、[实验指南PDF](lab.pdf)、[习题详解PDF](answers.pdf)、[已执行Notebook](experiment.ipynb)。每份PDF的Markdown源文件与图源都在本目录。

安装notebook-requirements.txt后，运行`python execute_notebook.py --out outputs/experiment.ipynb`，即可在新Python进程的IPython中顺序执行全部代码格并保存输出；这是实际代码执行，但未验证独立Jupyter套接字内核或浏览器界面。若要交互使用Notebook，另按Jupyter官方说明安装JupyterLab及ipykernel并选择本课Python环境。

PDF重建：安装build-requirements.txt，安装Node.js并在本目录运行`npm install`以安装package.json中的本地MathJax；系统还需要Pango与Noto CJK字体。随后运行`python build_pdf.py --directory outputs/pdf`。构建不请求在线字体或数学服务。环境文件是复现说明，未据此声称跨平台Conda安装已验证。
