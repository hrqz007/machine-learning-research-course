import argparse
import nbformat as nbf
from experiment import ROOT,safe_output

def build():
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s))
 def code(s):cells.append(nbf.v4.new_code_cell(s))
 md('# 第055讲 不平衡与有偏标签\n\n把概率、决策与标注机制分开。所有计算从当前数据重跑，不粘贴旧结果。完整推导、条件和参考见lecture.pdf、lab.pdf、sources.md。')
 md('## 1 数据与预测时点\n2000训练、1200验证、4000自然测试。观察开关和噪声标签只作用于训练。')
 code("import json,numpy as np\nfrom IPython.display import display,Image\nimport experiment as e\ndata=np.load(e.ROOT/'data/draws.npz')\nprint(data['X0'].shape,data['y0'].shape)\nprint(json.loads((e.ROOT/'data/protocol.json').read_text()))")
 md('## 2 纸笔公式的代码核验\n成本阈值1/9；原概率0.1加九倍正权重后变0.5；先验修正可反向恢复。')
 code("print('cost threshold',1/(1+8))\nprint('weighted q',(9*.1)/(9*.1+1*.9))\nprint('corrected p',e.prior_correct(np.array([.5]),.5,.1))\nprint('Wilson16/20',e.wilson(16,20))\nprint('noisy q at p=.1',.02+.73*.1)")
 md('## 3 真实运行所有路线\n新拟合42个LogisticRegression模型；多个阈值路线复用同一自然模型，oracle使用已知生成式。')
 code("result=e.run()\ne.dump(result,e.ROOT/'outputs/notebook-result.json')\nprint(json.dumps({'main':result['summary'],'labels':result['label_summary']},ensure_ascii=False,indent=2))")
 md('## 4 检查复制来源和计数\n复制不创造新独立样本。测试正例数自然波动。')
 code("s=result['runs'][0]\nids=np.array(s['oversample_source_ids'])\nprint('train/validation/test positives',s['training_positives'],s['validation_positives'],s['test_positives'])\nprint('resample length, unique, min, max',len(ids),len(np.unique(ids)),ids.min(),ids.max())\nfor m in ['natural_05','natural_cost','natural_tuned']:\n    print(m,s['methods'][m]['metrics'])")
 md('## 5 从验证预测重算阈值\n这里使用2000:3200的验证y，不传入测试y。')
 code("threshold,trials=e.choose_threshold(data['y0'][2000:3200],np.array(s['validation_probability']))\nprint('selected threshold',threshold)\nprint('best validation records',sorted(trials,key=lambda z:(z['cost'],z['threshold']))[:5])")
 md('## 6 独立指标审计\n另一套计数循环和Wilson公式核对72组记录、复制来源及概率前向。')
 code("import audit\nchecks=audit.audit(result)\ne.dump(checks,e.ROOT/'outputs/notebook-audit.json')\nprint(checks)")
 md('## 7 由本次结果重建图\n原始加权概率在自然分布中会有不同含义；图5区间不是重新训练的不确定性。')
 code("import plots\npaths=plots.draw(result,e.ROOT/'outputs/notebook-figures')\nprint([p.name for p in paths])")
 for i,t in enumerate(['自然测试的边界','召回与成本','验证阈值曲线','先验修正与Brier','少数类Wilson区间','有偏标签对照','分箱可靠性图']):
  md(t);code(f"display(Image(filename=str(paths[{i}])))")
 md('## 8 写出不能下的结论\n为什么验证阈值没胜出仍是有效结果？为什么oracle IPW不能证明现实选择机制已可识别？为什么修正概率后还要考虑决策阈值？先答再读answers.pdf。')
 return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/unexecuted.ipynb'));a=p.parse_args();nbf.write(build(),safe_output(a.out));print(a.out)
