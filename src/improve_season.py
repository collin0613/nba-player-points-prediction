"""Reproduce expanded ingestion, historical features, evaluation, and forecasts."""
import argparse
import subprocess
import sys
from config import ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true',help='Skip API downloads and reuse all cached inputs')
    args=parser.parse_args()
    stages=[] if args.offline else [('fetch_population.py',[])]
    stages += [('population_features.py',[]),('evaluate_models.py',[]),('predict_population.py',['--offline'] if args.offline else [])]
    for script,options in stages:
        subprocess.run([sys.executable,str(ROOT/'src'/script),*options],cwd=ROOT,check=True)

if __name__=='__main__':main()
