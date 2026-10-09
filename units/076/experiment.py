"""冻结模型的枚举检查：概率独立与图分离分别计算再比较。"""
from pathlib import Path
import argparse,json,hashlib
from graph_models import joint_from_bn,ci_residual,conditional,d_separated,equivalence_signature,undirected_chain
from generate_data import payloads
ROOT=Path(__file__).resolve().parent

def run():
    for name, content in payloads().items():  # 加载前逐字节校验原创数据可重建。
        if (ROOT/'data'/name).read_bytes()!=content.encode():
            raise ValueError('frozen data differs from generator: '+name)
    specs=json.loads((ROOT/'data/models.json').read_text())
    chain=joint_from_bn(**specs['chain']);collider=joint_from_bn(**specs['collider'])
    reverse_cpds={'C':[sum(r['probability'] for r in chain if r['state']['C']==1)],
      'B':[conditional(chain,{'B':1},{'C':c}) for c in (0,1)],
      'A':[conditional(chain,{'A':1},{'B':b}) for b in (0,1)]}
    reverse=joint_from_bn(['A','B','C'],[['C','B'],['B','A']],reverse_cpds)
    mrf,z=undirected_chain()
    descendant_nodes=['R','S','W','D'];descendant_edges=[('R','W'),('S','W'),('W','D')]
    return {'chain_total':sum(r['probability'] for r in chain),
      'chain_ac_residual':ci_residual(chain,'A','C'),
      'chain_ac_given_b_residual':ci_residual(chain,'A','C',['B']),
      'collider_rs_residual':ci_residual(collider,'R','S'),
      'collider_rs_given_w_residual':ci_residual(collider,'R','S',['W']),
      'rain_given_wet':conditional(collider,{'R':1},{'W':1}),
      'rain_given_wet_sprinkler':conditional(collider,{'R':1},{'W':1,'S':1}),
      'reverse_max_error':max(abs(a['probability']-b['probability']) for a,b in zip(chain,reverse)),
      'markov_equivalent':equivalence_signature(['A','B','C'],[('A','B'),('B','C')])==equivalence_signature(['A','B','C'],[('C','B'),('B','A')]),
      'descendant_unobserved_d_separated':d_separated(descendant_nodes,descendant_edges,'R','S'),
      'descendant_observed_d_separated':d_separated(descendant_nodes,descendant_edges,'R','S',['D']),
      'mrf_partition':z,'mrf_total':sum(r['probability'] for r in mrf),
      'mrf_ac_given_b_residual':ci_residual(mrf,'A','C',['B'])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);result=run()
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
