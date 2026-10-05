"""Build real notebook with standard-library integrity gate and ordered cells."""
from pathlib import Path
import hashlib,json
import nbformat as nbf
ROOT=Path(__file__).resolve().parent

def main():
    paths=['experiment.py','audit.py','plots.py','data_integrity.json','data/config.json','data/hand.csv','data/enumeration.csv','data/draws.npz']
    checks={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell;cells=[]
    cells.append(md('# 第048讲 泛化误差与复杂度\n\n本Notebook从三行手算和八种标签精确枚举，走到400次重采样。p是参数个数，最高次数p−1。先重启内核再运行全部。首格只使用标准库验证交付计算与数据字节，随后才导入数值依赖。所有大风险与负修正估计保留。'))
    cells.append(code('from pathlib import Path\nimport hashlib, json, os, sys\nROOT=Path.cwd().resolve()\nEXPECTED='+repr(checks)+'\nfor name, expected in EXPECTED.items():\n    path=ROOT/name\n    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:\n        raise RuntimeError("Missing or changed teaching input: "+name)\nprint("Verified",len(EXPECTED),"source/data files; cwd =",ROOT.name)'))
    cells.append(md('## 1 导入并解释随机对象\n\n训练D随机，拟合算法确定；新测试T独立。固定设计枚举只重抽标签，主实验同时重抽输入与标签。总体风险与有限测试MSE分开保存。'))
    cells.append(code('import numpy as np\nimport scipy, sklearn, matplotlib, nbformat, ipykernel\nimport experiment as e\nimport audit, plots\nfrom IPython.display import display, Image\nprint({"python":sys.version.split()[0],"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"matplotlib":matplotlib.__version__,"nbformat":nbformat.__version__,"ipykernel":ipykernel.__version__})'))
    cells.append(md('## 2 三行逐样本链\n\n常数预测1/3，残差−2/3、4/3、−2/3；损失贡献4/27、16/27、4/27。二次解2x²−1插值，训练损失0，但真实含噪风险22/15超过常数10/9。'))
    cells.append(code('hand=e.hand_reference()\nfor row in hand["rows"]:\n    print(row)\nprint("realized models:",json.dumps(hand["realized"],ensure_ascii=False,indent=2))'))
    cells.append(md('## 3 八种标签的独立精确参照\n\nFraction路线逐项积分单项式，不使用主实验的正交系数风险公式。三类平均风险4/3、3/2、9/5，偏差全为0。'))
    cells.append(code('print("exact fractions:",audit.exact())\nprint(json.dumps(hand["all_eight_equiprobable_designs"],ensure_ascii=False,indent=2))'))
    cells.append(md('## 4 完整协议与主实验\n\n400独立重复；复杂度n=40,p=1..9；学习曲线n=16,32,64,128,256,p=2,4,8；固定设计另列。每次不同p/n共享训练前缀以便配对。没有按结果选择种子或剔除尾部。'))
    cells.append(code('config,data=e.load_data()\nprint(json.dumps(config,ensure_ascii=False,indent=2))\nresult=e.run()\ne.atomic_json(ROOT/"outputs/notebook-result.json",result)\nprint("Retained fits:",sum(len(b["runs"]) for key in ["complexity","learning","fixed_design"] for b in result[key]))'))
    cells.append(md('## 5 真实库对照与全量不变量\n\n用SciPy枢轴QR、scikit-learn同目标，以及函数求积核验系数风险；检查10800次拟合，不只检查平均数。作者自查不等于独立课程验收。'))
    cells.append(code('verification=audit.run(result)\ne.atomic_json(ROOT/"outputs/notebook-audit.json",verification)\nprint(json.dumps(verification,ensure_ascii=False,indent=2))'))
    cells.append(md('## 6 复杂度曲线数值与分解\n\n有限B=400时ddof=0对应精确代数分解。ddof=1是另一种方差估计，必须配对修正偏差平方。risk已经含新噪声，test直接比含噪标签，因此二者均不能再加噪声。'))
    cells.append(code('for block in result["complexity"]:\n    s=block["summary"]\n    print({k:s[k] for k in ["p","train_mean","risk_mean","test_mean","bias2_mc","variance_ddof0","noise","risk_sd_ddof1","risk_mcse","bias2_unbiased_estimate"]})'))
    cells.append(md('## 7 学习曲线与大风险保留\n\np=8,n=16的巨大风险是此固定协议的结果，不应删掉。有限400次不证明无限训练分布矩存在，SD/√400不自动给可靠CLT置信区间。'))
    cells.append(code('for block in result["learning"]:\n    s=block["summary"]\n    print({"p":s["p"],"n":s["n"],"mean":s["risk_mean"],"max":max(v["risk"] for v in block["runs"]),"MCSE_descriptive":s["risk_mcse"],"max_condition":s["max_condition"]})\nprint("rep86 p9:",result["complexity"][8]["runs"][86])'))
    cells.append(md('## 8 重建十张实际图\n\n图例放在坐标轴外，中心80%分位带不是均值置信区间；长尾图明确使用对数轴。显示预先固定的第0次及前20次，不挑选结果。'))
    cells.append(code('figure_dir=ROOT/"outputs/notebook-figures"\nplots.render_all(result,figure_dir)\nprint("Generated",len(plots.NAMES),"figures")'))
    cells.append(code('for name in plots.NAMES[:5]:\n    print(name)\n    display(Image(filename=str(figure_dir/(name+".png"))))'))
    cells.append(code('for name in plots.NAMES[5:]:\n    print(name)\n    display(Image(filename=str(figure_dir/(name+".png"))))'))
    cells.append(md('## 9 读图解释与验收\n\n1. 图1是某个D的风险，图2对八种D精确平均。\n2. 随机复杂度曲线p=4均值较低，但该模拟不能决定现实模型的p。\n3. p=2近似误差0.2225，加噪声给下限0.345；增加数据不改变类的表达限制。\n4. p=8小样本风险大且数值残差小，说明统计误差与优化误差应分别诊断。\n5. 10%至90%带会忽略尾部幅度，须一起看全部重复与极值。\n\n在自己的解释中列出固定设计/随机设计、随机对象、评价目标、噪声次数、ddof与矩条件。详细20题解见answers.pdf。'))
    cells.append(code('v=np.array([-1.,0.,1.,2.]);B=len(v);bias2=float(v.mean()**2);var0=float(v.var(ddof=0));var1=float(v.var(ddof=1));corrected=bias2-var1/B\nprint({"mean_square":float(np.mean(v**2)),"bias2":bias2,"var_ddof0":var0,"var_ddof1":var1,"bias2_corrected":corrected,"raw_identity":bias2+var0,"corrected_identity":corrected+var1})\nprint("Negative tests retained:",result["negative_tests"])'))
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.validate(nb);nbf.write(nb,ROOT/'experiment.ipynb');print('Created notebook with',sum(c.cell_type=='code' for c in cells),'code cells')
if __name__=='__main__':main()
