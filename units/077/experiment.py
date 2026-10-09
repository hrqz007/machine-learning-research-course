"""同一目标的穷举、变量消元、sum-product与顺序代价对照。"""
from pathlib import Path
import argparse,json
from generate_data import payloads
from exact_inference import *
ROOT=Path(__file__).resolve().parent

def run():
    for name,content in payloads().items():
        if (ROOT/'data'/name).read_bytes()!=content.encode():raise ValueError('data mismatch: '+name)
    params=json.loads((ROOT/'data/chain.json').read_text());factors=chain_factors(**params)
    messages,mass,forward,backward=chain_sum_product(**params)
    marginal=[];errors=[];traces=[]
    for i in range(params['length']):
        query=f'X{i}';ve,z,trace=eliminate(factors,query,{'E':1});brute,bz=enumerate_query(factors,query,{'E':1})
        marginal.append(ve.table[(1,)]);errors.extend([abs(z-bz),abs(z-mass),abs(ve.table[(1,)]-brute.table[(1,)]),abs(ve.table[(1,)]-messages[i][1])]);traces.append(trace)
    star=star_factors();good,good_z,gt=eliminate(star,'L5',order=['L0','L1','L2','L3','L4','C']);bad,bad_z,bt=eliminate(star,'L5',order=['C','L0','L1','L2','L3','L4']);brute,bz=enumerate_query(star,'L5')
    return {'chain_p1_given_e1':marginal,'evidence_probability':mass,'max_method_error':max(errors),'forward':forward,'backward':backward,'chain_trace_x0':traces[0],
      'star_p_l5_1':good.table[(1,)],'star_partition':good_z,'star_order_difference':max(abs(good.table[(1,)]-bad.table[(1,)]),abs(good_z-bad_z),abs(good_z-bz),abs(good.table[(1,)]-brute.table[(1,)])),
      'star_good_peak_entries':max(t['entries'] for t in gt),'star_bad_peak_entries':max(t['entries'] for t in bt),'star_good_trace':gt,'star_bad_trace':bt}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);r=run();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
