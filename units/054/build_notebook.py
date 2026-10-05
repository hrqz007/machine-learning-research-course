import argparse
import nbformat as nbf
from experiment import ROOT,safe_output

def build():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# 第054讲 特征设计与维度问题\n\n按次序完成四行交互、距离推导、嵌套筛选和稳定性。所有输出为当前内核重新计算。来源与细节见lecture.pdf、lab.pdf、sources.md。')
    md('## 1 输入和角色\n数据完全合成。前160开发，后200确认；确认标签不得影响排名或k。')
    code("import json, numpy as np\nfrom IPython.display import display, Image\nimport experiment as e\ndata=np.load(e.ROOT/'data/draws.npz')\nprint(data['signal_X0'].shape, data['signal_y0'].shape)\nprint(json.loads((e.ROOT/'data/protocol.json').read_text()))")
    md('## 2 四行交互与二次列数\n先自己手算零相关，再看实际回归输出。')
    code("controls=e.controls()\nprint(controls['interaction'])\nfor p in [2,10,80]: print('p, quadratic columns excluding constant:',p,p+p*(p+1)//2)")
    md('## 3 距离变异系数\n精确推导针对平方距离。模拟端点成对独立，不共用锚点。')
    code("for z in controls['distance']:\n    print(z['d'], z['mean_squared_distance'], z['cv_squared_distance'], 'theory',z['theory_mean'],z['theory_cv'])")
    md('## 4 真实重新运行\n这格拟合所有内外层模型。故意错误对照只泄漏开发标签，确认集仍独立。')
    code("result=e.run()\ne.dump(result,e.ROOT/'outputs/notebook-result.json')\nprint(json.dumps(result['summary'],ensure_ascii=False,indent=2))")
    md('## 5 从局部索引回到原行\n内层80行不是原表前80行，必须映射外层train_ids。')
    code("s=next(z for z in result['runs'] if z['mode']=='signal' and z['rep']==0 and z['method']=='nested')\nf=s['folds'][0]\nz=f['inner_trials'][0]['folds'][0]\nprint('outer train/valid:',len(f['train_ids']),len(f['valid_ids']))\nprint('inner original first 10 IDs:',np.array(f['train_ids'])[z['train_local']][:10])\nprint('selected columns:',z['selected'])\nprint('four selected k:',[f['k'] for f in s['folds']])")
    md('## 6 Jaccard不是因果概率\n打印真实每折集合、频率与相似度，不只报告最好一列。')
    code("import itertools\nsets=[f['selected'] for f in s['folds']]\nprint('sets',sets)\nprint('six Jaccard',[e.jaccard(a,b) for a,b in itertools.combinations(sets,2)])\nprint('first 15 frequency',s['selection_frequency'][:15])")
    md('## 7 独立数学审计\n按训练边界复算Pearson，用另一种线性代数解法核对预测。')
    code("import audit\nchecks=audit.audit(result)\ne.dump(checks,e.ROOT/'outputs/notebook-audit.json')\nprint(checks)")
    md('## 8 从本次结果重建六图\n图4的两类误差对应不同训练行数，不能把差值全归因为泄漏。')
    code("import plots\npaths=plots.draw(result,e.ROOT/'outputs/notebook-figures')\nprint([p.name for p in paths])")
    for i,title in enumerate(['交互真值表','平方距离集中','嵌套数据边界','CV与独立确认','集合稳定性','选择频率']):
        md(title);code(f"display(Image(filename=str(paths[{i}])))")
    md('## 9 结论练习\n为什么全80列稳定性为1却误差较大？为什么signal的两种筛选最终预测相同仍不能为泄漏辩护？先回答，再读answers.pdf。')
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/unexecuted.ipynb'));a=p.parse_args();nbf.write(build(),safe_output(a.out));print(a.out)
