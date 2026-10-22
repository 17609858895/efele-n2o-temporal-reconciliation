"""Render current manuscript/SI figures from frozen scientific results."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
from verify_package import main as verify_package

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=ROOT/'_runs/figures')
    ap.add_argument('--only',nargs='+',choices=[f'Fig{i}' for i in range(1,7)]+[f'FigS{i}' for i in range(1,7)])
    ap.add_argument('--preview',action='store_true')
    args=ap.parse_args();verify_package()
    out=args.output.resolve()
    if out.exists():raise RuntimeError(f'Choose a fresh output directory: {out}')
    out.mkdir(parents=True)
    env=os.environ.copy();env['EFELE_FIGURE_OUTPUT']=str(out);env['PYTHONIOENCODING']='utf-8'
    cmd=[sys.executable,str(ROOT/'17_图件精修_20261021/scripts/make_figures.py')]
    if args.preview:cmd.append('--preview')
    if args.only:cmd.extend(['--only',*args.only])
    subprocess.run(cmd,check=True,cwd=ROOT,env=env)

if __name__=='__main__':main()
