"""Rebuild the teaching notebook source. Execute separately in a fresh kernel."""
from pathlib import Path
import argparse
import nbformat as nbf
ROOT=Path(__file__).resolve().parent

def make_notebook():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# 046 分类指标与排序评价\n\n固定12行案例贯通混淆矩阵、阈值扫描、ROC/AP、平均方式、阳性率变化。先阅读lab.pdf并完成L1-L2纸笔题，再从上到下执行。无模型拟合、无随机数据、无测试阈值选择。')
    md('## 1 定位课程与版本\n在本单元目录打开Notebook；如果外部启动器使用其他目录，可先设置环境变量 ML046_ROOT 为本机解压后的046路径。')
    code("from pathlib import Path\nimport os, sys, json, csv\nroot = Path(os.environ.get('ML046_ROOT', Path.cwd())).resolve()\nif not (root / 'data/review_scores.csv').is_file():\n    raise RuntimeError('Open this notebook from the 046 directory, or set ML046_ROOT.')\nsys.path.insert(0, str(root))\nimport numpy as np\nimport sklearn\nfrom IPython.display import display, Image\nfrom experiment import load_case, run, counts_at, sweep, prior_weights, library_metrics, average_metrics\nprint(sys.version.split()[0], np.__version__, sklearn.__version__)")
    md('## 2 看每条记录\n标签1表示缺陷。评分大小是复核优先级，不保证已校准成概率。')
    code("ids, y, s = load_case()\nprint('id label score predict_at_080')\nfor i, yi, si in zip(ids, y, s):\n    print(i, yi, f'{si:.2f}', int(si >= .80))\nprint('positive, negative, prevalence:', int(y.sum()), int((1-y).sum()), y.mean())")
    md('## 3 不用指标库的逐行计数\n先预测，再同时检查真实标签。请先心算L1，再运行。')
    code("for t in [.80, .55]:\n    c = dict(TP=0, FP=0, FN=0, TN=0)\n    for yi, si in zip(y, s):\n        pred = si >= t\n        key = ('TP' if yi else 'FP') if pred else ('FN' if yi else 'TN')\n        c[key] += 1\n    tp, fp, fn, tn = (c[k] for k in ['TP','FP','FN','TN'])\n    print(t, c, 'P/R/F1/Accuracy:', tp/(tp+fp), tp/(tp+fn), 2*tp/(2*tp+fp+fn), (tp+tn)/len(y))")
    md('## 4 真实库核验单阈值\n矩阵轴顺序为真实行、预测列、labels=[0,1]。二分类F1明确average=binary。')
    code("from sklearn import metrics as m\nfor t in [.80, .55]:\n    pred = s >= t\n    print('threshold', t)\n    print(m.confusion_matrix(y, pred, labels=[0,1]))\n    print(m.precision_recall_fscore_support(y, pred, labels=[0,1], pos_label=1, average='binary', zero_division=0))")
    md('## 5 完整同分组扫描\n注意0.80只有一行决策，C和D同时入选。空警报精确率的0是报告约定，precision_defined会是False。')
    code("rows = sweep(y,s)\nprint('t TP FP FN TN P R F1 FPR defined')\nfor row in rows:\n    print(row['threshold'], *(round(row[k],6) for k in ['TP','FP','FN','TN','precision','recall','f1','fpr']), row['precision_defined'])")
    md('## 6 ROC与PR的端点不能混用\n请回答L4：两个阈值数组为什么方向不同？PR最后一个点对应哪个阈值？')
    code("fpr,tpr,rt = m.roc_curve(y,s,pos_label=1,drop_intermediate=False)\np,r,pt = m.precision_recall_curve(y,s,pos_label=1,drop_intermediate=False)\nprint('ROC lengths:', len(fpr),len(tpr),len(rt))\nprint('ROC thresholds:', rt)\nprint('ROC FPR:',fpr,'\\nROC TPR:',tpr)\nprint('PR lengths:',len(p),len(r),len(pt))\nprint('PR thresholds:',pt)\nprint('PR precision:',p,'\\nPR recall:',r)\nprint('AUC AP PR-trapezoid:',m.roc_auc_score(y,s),m.average_precision_score(y,s,pos_label=1),m.auc(r,p))")
    md('## 7 独立Fraction参照\n从CSV整数评分重新建立参照，不把sklearn输出当预期答案。完整1134个小例子在终端运行audit.py。')
    code("from audit import fraction_reference, run_audit\nwith (root/'data/review_scores.csv').open(newline='') as f:\n    raw = list(csv.DictReader(f))\nref = fraction_reference([int(a['label']) for a in raw],[int(a['score_percent']) for a in raw])\nprint({k:str(ref[k]) for k in ['auc','ap','pr_trapezoid']})\nprint(run_audit(exhaustive=False))")
    md('## 8 阳性率变化\n只改两类混合权重，保持每类内部评分分布。权重计数不是新增或删除真实零件。')
    code("for pi in [.10,.25,.50]:\n    w = prior_weights(y,s,pi)\n    curves = library_metrics(y,s,w)\n    print('pi=',pi,'weights=',w)\n    print('at .80:',counts_at(y,s,.8,w))\n    print('AUC/AP:',curves['auc'],curves['ap'])")
    md('## 9 同一预测的四种F1\n先看average=None的每类结果与支持度，再解释汇总。')
    code("for name, values in average_metrics(y,s).items():\n    print(name, values)")
    md('## 10 保序不等于同一概率\n立方是一个反例，不是拟合的校准器。将阈值也映射到0.512才能保留原决策。')
    code("result = run()\nprint(json.dumps(result['monotone'],indent=2))\nif counts_at(y,s,.8) != counts_at(y,s**3,.8**3):\n    raise RuntimeError('mapped decision changed')")
    md('## 11 从本次计算重建全部图\n图像输出到outputs/notebook-figures，不覆盖讲义原图。下方图像是本次执行产生的PNG。')
    code("from plots import create_figures\nfigures = create_figures(root/'outputs/notebook-figures')\nfor file in figures:\n    print(Path(file).name)\n    display(Image(filename=file, width=850))")
    md('## 12 写结果而非只看图\n完成L10报告。必须解释样本、阳性定义、阈值、平均方式、AUC/AP约定与外推限制。对照answers.pdf前保存自己的纸笔答案。')
    code("print('Frozen case: TP=2 FP=2 FN=1 TN=7 at threshold .80 (>=).')\nprint('AUC=43/54 AP=9/14; synthetic teaching data, no generalization claim.')\nprint('Executed all teaching cells without fitting a model or selecting a threshold.')")
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (ML046)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.14'}})
    nbf.validate(nb);return nb

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'experiment.ipynb'));a=p.parse_args();dest=Path(a.out).absolute()
    for part in (dest,*dest.parents):
        if part.is_symlink():raise ValueError('symlink notebook path')
    if dest.suffix!='.ipynb' or (dest.exists() and (not dest.is_file() or dest.stat().st_nlink>1)):raise ValueError('unsafe notebook target')
    dest.parent.mkdir(parents=True,exist_ok=True);nbf.write(make_notebook(),dest);print(dest)
if __name__=='__main__':main()
