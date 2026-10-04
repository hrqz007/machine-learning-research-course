# 002 任务定义与数据生成

主问题：一行数据代表谁，预测发生在何时，答案怎样得到？先修仅第一讲，不要求掌握 Python 语法。

- lecture.pdf：完整正文、10 幅原创机制图、练习
- lab.pdf：独立纸笔与 Notebook 操作指南
- answers.pdf：详细答案、逐事件标签推理与完整数据字典
- 同名 .md：可编辑原文
- experiment.py、experiment.ipynb：标准库脚本与已执行交互实验
- data/：合成原始挂牌记录与纸笔预期标签
- source-checks.json、verification.json：来源与实际验证范围

## 运行

Python 3.10+，无第三方计算依赖、网络访问或 API：

```bash
python experiment.py
```

或使用已有 Anaconda 中的 Jupyter，打开 Notebook，重启内核后全部运行。不要用 python -O，教学核验使用 assert。outputs/ 由程序创建，重复运行会更新输出；原始数据不被覆盖。想做修改实验时先复制单元文件夹。

固定预期：11 导出行、10 次挂牌、9 套房屋、5 栋楼；5 个正例、3 个负例、2 个未知；去重后已知比例 5/8，带副本时 6/9；E06 面积晚到；坏划分共享 H01 和 B01/B02/B03。未知不等于负例，已知标签比例不是目标总体估计。

## 范围

全部数据、日期、价格与机构均为虚构。只审计样本、标签、时间与划分，不训练现实预测模型。日期精度为天，observed_through 的核实证据假定在该日期即可获取；真实场景需审查更精细的到达时间及核实延迟。程序无法自动找出所有近似重复或隐藏关联。

Notebook 在新进程中使用真实 IPython InProcessKernel 顺序执行，浏览器 Jupyter UI、外进程 socket 与各平台 Anaconda 安装未验证。具体证据见 verification.json。
