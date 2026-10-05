"""Build a pedagogical notebook. This does not execute it or forge outputs."""
import argparse
from pathlib import Path
import nbformat as nbf
from experiment import ROOT,safe_output

def build():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# 第053讲 调参与实验预算\n\n从纸笔账本走向真实CPU实验。先固定预算，再选择模型；测试标签不能进入搜索。下面所有输出都由当前内核实际生成。完整推导见lecture.pdf，逐项练习见lab.pdf。')
    md('## 1 确认输入\nX每份600×10，前240行为开发，后360行为测试。这里只检查形状，不用测试分数制定搜索规则。ROOT必须为本讲目录。')
    code("from pathlib import Path\nimport json, numpy as np\nimport experiment as e\nfrom IPython.display import display, Image\ndata = np.load(e.ROOT/'data/draws.npz')\nprint(e.ROOT.name, data['X0'].shape, data['y0'].shape)\nprint(json.loads((e.ROOT/'data/protocol.json').read_text()))")
    md('## 2 先算预算\n不是数fit调用就够了：每次拟合的树数不同。当前淘汰每轮从头拟合。')
    code("print('grid:', 6*60*3, 'halving:', (12*15+4*30+1*60)*3)\nprint('final refit per route:', 60)\nprint('counterfactual warm-start search:', (12*15+4*15+1*30)*3)")
    md('## 3 重新运行三条路线\n这格真正训练所有模型。负结果会完整保留，不读预制experiment-result.json。')
    code("result = e.run()\ne.dump(result, e.ROOT/'outputs/notebook-result.json')\nprint(json.dumps(result['summary'], ensure_ascii=False, indent=2))")
    md('## 4 手算一个候选的折损失\n按照valid_ids对齐标签，再求平方误差平均。')
    code("s = next(z for z in result['runs'] if z['rep']==0 and z['method']=='grid')\nvalues=[]\nfor row in s['fits']:\n    if row['candidate']==0:\n        values.append(float(np.mean((data['y0'][row['valid_ids']]-row['prediction'])**2)))\nprint('three folds', values, 'mean', sum(values)/3)")
    md('## 5 检查幸存者与失败记录\n候选编号是当前configurations的局部下标；失败演示单独记账。')
    code("s=next(z for z in result['runs'] if z['rep']==0 and z['method']=='halving')\nfor stage in s['stages']:\n    print(stage)\nprint(result['failure_demo'])")
    md('## 6 独立审计\n重算522次损失与18个最终重拟合，检查预算、生存集合和测试标签干预。')
    code("import audit\nchecks=audit.audit(result)\ne.dump(checks,e.ROOT/'outputs/notebook-audit.json')\nprint(checks)")
    md('## 7 重建图\n图从本次result产生。计时图允许随机器负载变化。后面每格显示一张实际重建图。')
    code("import plots\npaths=plots.draw(result,e.ROOT/'outputs/notebook-figures')\nprint([p.name for p in paths])")
    captions=['数据权限：没有测试到搜索的箭头。','同域不同候选政策，不等于实际点完全相同。','淘汰候选的未来曲线是未知的。','所有独立份数与配对差都保留。','同树数不等同时间。','人造选择噪声示意，非真实数据集胜率。','构造慢热反例，非本次森林实測。']
    for i,c in enumerate(captions):
        md(c);code(f"display(Image(filename=str(paths[{i}])))")
    md('## 8 自己写结论\n为什么random四份更差仍是有效结果？为什么本实验没有Bayesian优化的实测排名？请回答后再读answers.pdf。来源见sources.md；三方法只在固定同一合成机制下比较。')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    return nb
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/unexecuted.ipynb'));a=p.parse_args();nbf.write(build(),safe_output(a.out));print(a.out)
