"""Create ordered teaching notebook with deterministic cell IDs and integrity gate."""
from pathlib import Path
import hashlib,json
import nbformat as nbf
ROOT=Path(__file__).resolve().parent

def main():
    paths=['experiment.py','audit.py','plots.py','data_integrity.json',*['data/'+n for n in ['protocol.json','draws.npz','splits.npz','panel.csv','split_membership.csv','hand.csv']]]
    hashes={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in paths}
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# 第050讲 数据划分与评价对象\n\n先定义谁与何时，再看分数。60设备×48天、两天标签延迟、四份固定协议和三种简单模型。第0份作实例，另80份独立重抽；不选好seed、不删除负面结果。按顺序从新内核执行。')
    code('from pathlib import Path\nimport hashlib, json, os, sys\nROOT=Path.cwd().resolve()\nEXPECTED='+repr(hashes)+'\nfor name, expected in EXPECTED.items():\n    path=ROOT/name\n    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:\n        raise RuntimeError("Missing or modified teaching input: "+name)\nprint("Verified",len(EXPECTED),"source and data files")')
    md('## 1 环境和固定协议\n\n模型只读训练数据。模拟器a、μ仅用于解释，测试标签不用于选择。随机与整组在观察期结束后回顾拟合，时间与联合在32日截止。')
    code('import numpy as np\nimport scipy, sklearn, matplotlib, ipykernel\nimport experiment as e\nimport audit, plots\nfrom IPython.display import display, Image\nprint({"python":sys.version.split()[0],"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"matplotlib":matplotlib.__version__,"ipykernel":ipykernel.__version__})\nc,draws,splits=e.load_data()\np=e.panel(c,draws,0)\nprint(json.dumps(c,ensure_ascii=False,indent=2))')
    md('## 2 四行逐样本链\n\nθ顺序b,w,vA,vB。先做前向、残差、半平方损失，再聚合局部导数，η=0.2同步更新。H5、H6测试行不得进梯度。')
    code('h=e.hand()\nfor row in h["rows"]:\n    print(row)\nprint("Mean gradient",h["gradient_mean"])\nprint("Synchronized update",h["theta_next"])\nprint("J next",h["next_objective"],"test-only next predictions",h["test_next_predictions"])\nprint("Independent fractions:",audit.exact_hand())')
    md('## 3 验证分组与时间边界\n\n训练必须在实际拟合截止时拿到标签，不能仅比较事件日期。联合任务主动不用新设备早期数据，模拟冷启动。')
    code('for kind,(tr,te) in splits.items():\n    e.validate_split(p,tr,te,kind,c["fit_cutoff_day"])\n    print(kind,{"train":len(tr),"test":len(te),"unused":len(p["x"])-len(tr)-len(te),"overlap_groups":len(np.intersect1d(p["group"][tr],p["group"][te])),"last_label_available":int(p["available"][tr].max()),"first_test_day":int(p["day"][te].min())})')
    md('## 4 完整主实验与重复\n\n主例12次拟合，独立80面板再拟合960次。模型使用增广最小二乘，不用手算的一步GD作为最终拟合。')
    code('result=e.run()\ne.atomic_json(ROOT/"outputs/notebook-result.json",result)\nfor kind in e.SPLITS:\n    print(kind,{m:result["main"][kind]["models"][m]["mse"] for m in e.MODELS})\nprint("Retained repeated fits",len(result["replications"])*12)\nprint("All negative comparisons",result["negative_results"])')
    md('## 5 真实独立科学参照\n\n独立OneHotEncoder只fit训练类别，Ridge SVD与SciPy QR核验全部12个主例。测试标签变更、常见输入及输出保全也实际测试。')
    code('checks=audit.run(result)\ne.atomic_json(ROOT/"outputs/notebook-audit.json",checks)\nprint(json.dumps(checks,ensure_ascii=False,indent=2))')
    md('## 6 把未知设备编码看见\n\n以下不复用课程设计函数。整组测试设备未见，指示列全为0。相同数值alpha只有在目标尺度匹配时才能对照。')
    code('from sklearn.preprocessing import OneHotEncoder\nfrom sklearn.linear_model import Ridge\ntr,te=splits["group"]\nenc=OneHotEncoder(handle_unknown="ignore",sparse_output=False)\ngtr=enc.fit_transform(p["group"][tr,None]);gte=enc.transform(p["group"][te,None])\nX=np.column_stack([p["x"][tr],gtr]);Xt=np.column_stack([p["x"][te],gte])\nmodel=Ridge(alpha=1,fit_intercept=True,solver="svd").fit(X,p["y"][tr])\nprint("Unknown group feature sum",float(gte.sum()))\nprint("Actual sklearn MSE",float(np.mean((model.predict(Xt)-p["y"][te])**2)))')
    md('## 7 期望噪声与有限测试\n\nMSE已经与含噪y比较，不再加噪声。oracle_conditional_mse在当前固定测试设计下只对新的标签噪声取期望，不是所有未来的总体风险。')
    code('for kind in e.SPLITS:\n    q=result["main"][kind]["models"]["trend"]\n    print(kind,{"finite_test_MSE":q["mse"],"oracle_noise_average":q["oracle_conditional_mse"],"noise_variance":c["noise_sd"]**2})\nfor kind in e.SPLITS:\n    print(kind,result["summary"][kind]["personalized"])')
    md('## 8 权重与Bootstrap单位\n\n720行来自15个独立留出设备。固定模型不重训，只重采样测试；800次百分位区间未做覆盖率验证。')
    code('print(result["weight_example"])\nb=result["bootstrap"]\nprint({k:v for k,v in b.items() if "draws" not in k})\nprint("Random split row / device mean",{k:result["main"]["random"]["models"]["personalized"][k] for k in ["mse","group_mean_mse"]})')
    md('## 9 时间API的单位\n\n对48个日索引设置gap=2才表示两天，再映射为面板行。以下仅演示接口，不改变主实验四个协议、不作调参。')
    code('from sklearn.model_selection import TimeSeriesSplit\nfor trd,ted in TimeSeriesSplit(n_splits=3,test_size=8,gap=2).split(np.arange(48)):\n    tr=np.flatnonzero(np.isin(p["day"],trd));te=np.flatnonzero(np.isin(p["day"],ted))\n    print({"last_train_day":int(trd[-1]),"first_test_day":int(ted[0]),"last_test_day":int(ted[-1]),"train_rows":len(tr),"test_rows":len(te)})')
    md('## 10 重建并阅读全部十幅图\n\n图5粗线为10%至90%范围，不是均值置信区间；图10为单独窗口例，区分历史重用的可实施性和统计独立性。')
    code('figure_dir=ROOT/"outputs/notebook-figures"\nplots.render_all(result,figure_dir)\nprint("Generated figures:",len(plots.NAMES))')
    code('for name in plots.NAMES[:5]:\n    print(name)\n    display(Image(filename=str(figure_dir/(name+".png"))))')
    code('for name in plots.NAMES[5:]:\n    print(name)\n    display(Image(filename=str(figure_dir/(name+".png"))))')
    md('## 11 结束前写结论\n\n1. 随机评分是回顾性已知设备插值，不能重命名为新设备未来。\n2. 设备历史是否合法依赖部署目标与标签到达时刻。\n3. Trend对线性漂移的优势有机制前提；联合80次中仍有3次更差。\n4. 独立面板、测试设备Bootstrap、重复划分分别平均不同随机性。\n5. 指标、权重、信息边界和失败处理应在看测试分数前确定。\n\n完成讲义E1至E16、G1至G10和实验L1至L8，再对照答案。')
    for i,cell in enumerate(cells):cell['id']=f'ml050-cell-{i+1:02d}'
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.validate(nb);nbf.write(nb,ROOT/'experiment.ipynb');print('Created',sum(c.cell_type=='code' for c in cells),'code cells')
if __name__=='__main__':main()
