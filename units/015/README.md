# 第十五讲 曲率与局部近似

核心问题：斜率相同的地方，为什么优化难度仍可能不同？直接先修第12、14讲。沿用第14讲的三条固定合成记录和 p_i(a,b)=a*x_i+b²，逐步建立二阶 Taylor、Hessian、二次型与驻点判据，并测量近似余项。

## 阅读与运行

1. 阅读 `lecture.pdf`，包含12幅原创机制图与带前提的推导
2. 按 `lab.pdf` 完成手算、曲面、误差扫描和故意失败测试
3. 打开 `experiment.ipynb`，预测输出后从头顺序运行
4. 对照 `answers.pdf` 的12道完整练习与详细推导

PDF 由同名 Markdown 源文档构建。实验必须保留 `experiment.py` 和 `data` 的相对位置。

```bash
python experiment.py
jupyter lab experiment.ipynb
```

核心脚本仅依赖 NumPy；Notebook 另需 Matplotlib 与 Jupyter/IPython。实际验证环境为 Python 3.12.14、NumPy 2.3.5、Matplotlib 3.10.8。`requirements.txt` 和 `environment.yml` 是依赖与建议环境，不是逐平台安装保证。实验离线、小CPU、无随机过程、无数据下载、无GPU或付费接口。

## 可以核对的结果

主点 L=19/48、梯度(-1/6,-5/6)、Hessian=[[10/3,2],[2,1/3]]，特征值为(-2/3,13/3)，但主点非驻点。驻点(6/5,0)为鞍点；两处(1/2,±√(7/6))的全局最小损失为1/18，全局性由完成平方另证。

普通单位方向(3/5,4/5)上的一阶至四阶系数为 -23/30、5/3、224/125、256/625。抵消方向(-1,1)/√2的二阶余项为t⁴/4。半正定和零 Hessian 给出不确定结论，不强行分类。

脚本在新进程运行通过25个参数点的独立解析/差分核对和36组边界检查，结果写入 `outputs`。Notebook在新进程的真实 ipykernel InProcessKernel 中顺序运行12个代码单元并保留6个图像输出。未测试 Jupyter 浏览器界面、外进程 socket 传输、读者本机 Anaconda 与其他操作系统。详见 `test-result.json`。

## 文件说明

- `lecture.pdf`（可读源稿 `lecture.md`）、`lab.pdf`、`answers.pdf`：独立正文、实验指南与练习详解的源文件
- `experiment.py`、`experiment.ipynb`：对应脚本和已执行教学 Notebook
- `data/observations.csv`、`data/README.md`：固定合成数据及输入/输出字段字典
- `outputs`：可再生成实验报告与两份核验 CSV
- `figures`、`build_assets.py`：12幅原创 PNG 和可再生成绘图脚本
- `build_notebook.py`：Notebook 构建脚本，重建后须重新执行验证
- `source-checks.json`：实际检查过的一手来源、条件与原创性说明
- `test-result.json`：测试范围、证据、实际环境和未检验项

C² 保证二阶余项 o(||h||²)，不自动保证 O(||h||³)；局部极小不等于全局最优；数值梯度近零不等于精确驻点。实验只说明有限数学机制，不作真实预测或优化性能声明。

## 综合自检
合格答案需保留四个区分：梯度与 Hessian；精确驻点与数值近零；二阶局部模型与原函数；局部极小与全局最优。公式计算正确但丢掉这些前提，仍会导致错误决策。

一组实用复核值为：主点特征值 $-2/3,13/3$；驻点鞍点损失3/5；两处全局极小损失1/18；普通方向二阶余项三次系数224/125；抵消方向二阶余项四次系数1/4。实验脚本还会检查25个参数点与36组输入边界。
