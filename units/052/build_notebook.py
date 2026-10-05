"""Create an independent teaching notebook without overwriting shipped execution."""
from pathlib import Path
import argparse,hashlib,json
import nbformat as nbf
import experiment as e
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/unexecuted.ipynb'));args=p.parse_args();out=e.safe_output(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    paths=['experiment.py','audit.py','plots.py','data_integrity.json',*json.loads((ROOT/'data_integrity.json').read_text())];checks={v:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for v in paths};cells=[]
    def md(x):cells.append(nbf.v4.new_markdown_cell(x))
    def code(x):cells.append(nbf.v4.new_code_cell(x))
    md('# 第052讲 无泄漏的数据处理流水线\n\n先追踪参数和标签来源，再读分数。A新设备、B纯噪声选择、C独立类别编码是三种不同机制；不要把分数差混成一种因果影响。按顺序从新内核运行。')
    code('from pathlib import Path\nimport hashlib, json, sys\nROOT = Path.cwd().resolve()\nEXPECTED = '+repr(checks)+'\nfor name, digest in EXPECTED.items():\n    path = ROOT/name\n    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:\n        raise RuntimeError("Missing or modified teaching input: "+name)\nprint("Verified",len(EXPECTED),"source/data files")')
    md('## 1 固定协议与版本\n\n数值只从随包数据读取。A全缺列保留并填0，站点无缺失；未知类别回退明确。C连续目标显式声明，不让整数回归标签自动变成分类。')
    code('import numpy as np\nimport scipy, sklearn, matplotlib, nbformat, ipykernel\nimport experiment as e\nimport audit, plots\nfrom IPython.display import display, Image\nprint({"python":sys.version.split()[0],"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"matplotlib":matplotlib.__version__,"nbformat":nbformat.__version__,"ipykernel":ipykernel.__version__})\nc,d,splits = e.load_data()\nprint(json.dumps(c,ensure_ascii=False,indent=2))')
    md('## 2 从四行原值到同步更新\n\n中位数3，上下界1与5，均值3，方差2。设计顺序b,z,A,B,missing。检查每行损失和局部导数，再求和除4；η=1/5。H5/H6仅前向，不给训练标签。')
    code('h = e.hand()\nprint(h["statistics"])\nfor row in h["rows"]:\n    print(row)\nprint("Gradient",h["gradient_mean"],"updated theta",h["theta_next"])\nprint("Next half loss",h["next_half_loss"],"validation predictions",h["valid_next_prediction"])\nprint("Independent fractions",audit.exact_hand())')
    md('## 3 一个实际训练折的统计来源\n\n每折216行来自36设备；72行来自12个新设备。组ID只做划分，不做特征。类别站点共享合法，设备身份交叉不符合此目标。')
    code('X = e.group_X(d,0); y=d["group_y"][0]\ntr=splits["group_0_train"]; va=splits["group_0_valid"]\npipe=e.make_pipeline().fit(X[tr],y[tr])\nprint(e.state(pipe["pre"],tr))\nprint(pipe["pre"].get_feature_names_out())\nprint("First three validation designs",pipe["pre"].transform(X[va[:3]]))\nprint("Group overlap",np.intersect1d(d["group"][tr],d["group"][va]))')
    md('## 4 全部三机制执行\n\n12份×4折×(2+2+3)=336个主要预测器拟合，全部结果保留。所有常数在首次分数前固定，不依据成绩选择协议。')
    code('result=e.run()\ne.atomic_json(ROOT/"outputs/notebook-result.json",result)\nfor name,summary in result["summary"].items():\n    print(name, {k:v for k,v in summary.items() if k=="paired_differences" or k.endswith("mse")})\nprint("Retained datasets",len(result["replications"]))')
    md('## 5 独立科学参照与干预\n\n手工排序分位数、统计求和、独立Pearson选列、手工内部KFold、计数目标编码与SciPy QR。既要正确路线不变，也要泄漏负对照确实改变。')
    code('checks=audit.run(result)\ne.atomic_json(ROOT/"outputs/notebook-audit.json",checks)\nprint(json.dumps(checks,ensure_ascii=False,indent=2))')
    md('## 6 OLS不变负对照\n\n带截距无惩罚满秩OLS在可逆仿射坐标中优化同一函数。不同fit来源可产生相同预测，不能凭分数不变排除协议越界。')
    code('print(json.dumps(result["affine_control"],indent=2))')
    md('## 7 目标编码八行来源\n\n前四行只读后四标签，μ=6.5；后四只读前四，μ=2.5；验证查全8行，μ=4.5。fit_transform与fit后transform不等价。')
    code('eh=e.encoder_hand()\nfor i in range(8):\n    print(i+1,eh["categories"][i],eh["y"][i],eh["crossfit_training"][i],eh["full_training_transform"][i])\nprint("Validation",eh["valid_categories"],eh["valid_transform"])')
    md('## 8 真实TargetEncoder Pipeline\n\n中间变换器训练调用fit_transform，验证调用transform。当前内部行折适用于C的独立机制，不自动适用于组或时间任务。')
    code('from sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import Ridge\ntr=splits["encoding_0_train"];va=splits["encoding_0_valid"]\nCX=d["category"][0,:,None];CY=d["category_y"][0]\np=Pipeline([("encoder",e.make_encoder(c)),("model",Ridge(alpha=1,solver="svd"))]).fit(CX[tr],CY[tr])\nprint("Actual pipeline MSE",e.mse(CY[va],p.predict(CX[va])))\nblock=result["replications"][0]["encoding"][0]\nprint("First inner fit / encoded row IDs",block["inner_provenance"][0])\nprint("Train full mean",block["full_train_target_mean"])')
    md('## 9 从原始允许标签复算编码\n\n下面不调用TargetEncoder；比较全部270行折外值和90行验证值。自己的标签不参与自身折外值，其他合法训练标签仍可以影响它。')
    code('cfg=c["encoding_experiment"]\nz_oof=audit.manual_crossfit(CX[tr,0],CY[tr],3,5,c["encoder_seed"])\nz_valid=audit.manual_encode(CX[tr,0],CY[tr],CX[va,0],5)\nprint("OOF max difference",np.max(abs(z_oof-block["crossfit_train_encoding"])))\nprint("Validation max difference",np.max(abs(z_valid-block["valid_encoding_from_full_train"])))')
    md('## 10 十幅实图\n\n差值图保留正负，三机制各自解释。图9正确cross-fit不保证胜过均值基线。图10不包含任何外层验证标签。')
    code('figure_dir=ROOT/"outputs/notebook-figures"\nplots.render_all(result,figure_dir)\nprint("Rebuilt",len(plots.NAMES),"figures")')
    code('for name in plots.NAMES[:5]:\n    print(name)\n    display(Image(filename=str(figure_dir/(name+".png"))))')
    code('for name in plots.NAMES[5:]:\n    print(name)\n    display(Image(filename=str(figure_dir/(name+".png"))))')
    md('## 11 完成解释\n\n1. A仅改变缩放fit集合，4份变差保留。\n2. B选列先读全标签使验证不再独立。\n3. C训练自编码过拟合和外层标签泄漏不同。\n4. 每个内部均值和类别和都来自允许子集。\n5. 本包无现实部署、无时间外推、无新研究pilot结论。\n\n完成E1至E16、G1至G10、L1至L8后对照完整答案。')
    for i,cell in enumerate(cells):cell['id']=f'ml052-cell-{i+1:02d}'
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}});nbf.validate(nb);nbf.write(nb,out);print('Created',len([v for v in cells if v.cell_type=='code']),'code cells:',out)
if __name__=='__main__':main()
