# 第060讲 Bagging与随机森林

从五行bootstrap手算表出发，推导平均预测的方差，自己实现抽样成员、概率聚合与OOB，再用同组重复测量揭示袋外评价的边界。

## 学习材料

- [教材正文](lecture.md) · [正文PDF，10页](lecture.pdf)
- [独立实验指南](lab.md) · [实验PDF，4页](lab.pdf)
- [12题逐步详解](answers.md) · [答案PDF，4页](answers.pdf)
- [已执行Notebook，7个代码单元](experiment.ipynb)
- [透明bootstrap集成](bagging.py) · [实验入口](experiment.py)
- [独立审计](test_experiment.py) · [859项检查](test-result.json)
- [实验结果](experiment-result.json) · [验证边界](verification.json)
- [原始教学数据说明](data/README.md) · [一手来源](sources.md) · [来源核对](source-checks.json)

直接先修：021、025、027、051、058至059。重点是平均的协方差项、节点特征子采样、OOB逐行分母与群组独立单位。6幅原创图串起抽样、方差、树数曲线、覆盖率、特征权衡和群组反例。

## 离线复现

```bash
conda env create -f environment.yml
conda activate ml060
python experiment.py --out outputs/result.json
python test_experiment.py --report outputs/result.json --out outputs/test-result.json
python -O test_experiment.py --report outputs/result.json --out outputs/test-optimized.json
python plots.py --report outputs/result.json --directory outputs/figures
python execute_notebook.py --out outputs/fresh-kernel.ipynb
```

也可在已有Python3.12环境执行python -m pip install -r requirements.txt。作者使用隔离Python3.12.14环境验证，未额外验证全新Conda创建。全部CSV随包提供，实验不联网。

80树OOB在m=1、3、8中选择3，然后首次读取测试集。所选简化森林测试准确率0.766667，Brier0.169593；单树基线Brier0.266005。群组反例的行级OOB准确率1.0，新组测试却仅0.55。结果都是原创合成数据上的观察。

## PDF与完整重建

排版需要build-requirements.txt中的Markdown、WeasyPrint、系统Pango与中文Noto字体，以及Node.js/MathJax。

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory outputs/pdf
bash build_all.sh
```

输出写到outputs，不覆盖发布资料。构建器仅用于受信任课程Markdown/TeX，不是任意外部HTML安全沙箱。

## 范围与证据

- 自行实现bootstrap索引、概率聚合与OOB；基树生长由scikit-learn完成
- 二分类0/1、有限数值特征；单类bootstrap正确对齐概率列；OOB无票保留缺失
- m=8为纯Bagging条件，m=1/3增加节点特征随机化；同一80树前缀比较B
- OOB选择先于测试读入；相关系数诊断不冒充固定输入下的方差参数
- 库与自建森林不承诺相同抽样；对同一组库树独立重算聚合和OOB最大差0
- 859项显式检查在普通与-O模式通过，Notebook真实执行并嵌入6图
- Notebook未验证Jupyter浏览器界面或外进程socket传输；不含真实世界性能结论
