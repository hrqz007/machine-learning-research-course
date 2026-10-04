"""Generate original synthetic course data into a NEW directory only.

Usage: python generate_data.py --output generated_copy
Normal experiment runs read the provided CSV files and do not regenerate data.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError('Choose a new output directory; original data will not be overwritten.')
args.output.mkdir(parents=True)
config = json.loads((Path(__file__).resolve().parent/'config.json').read_text())
rng = np.random.default_rng(config['data_generation_seed'])
areas = rng.integers(40, 101, size=60)
noise = rng.integers(-8, 9, size=60)
prices = 2*areas+10+noise
with (args.output/'houses.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['house_id','area_m2','price_wan'])
    for i,(area,price) in enumerate(zip(areas,prices),1):
        w.writerow([f'H{i:03d}',int(area),int(price)])
order=np.random.default_rng(config['split_generation_seed']).permutation(60)
roles=['']*60
for rank,index in enumerate(order):
    roles[int(index)]='train' if rank<36 else 'validation' if rank<48 else 'test'
with (args.output/'split.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f);w.writerow(['house_id','role'])
    for i,role in enumerate(roles,1):w.writerow([f'H{i:03d}',role])
print('Generated 60 synthetic rows and an explicit 36/12/12 split.')
