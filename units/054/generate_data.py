from pathlib import Path
import argparse
from experiment import generate_data,ROOT
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/regenerated-data'));a=p.parse_args();generate_data(Path(a.directory));print(a.directory)
