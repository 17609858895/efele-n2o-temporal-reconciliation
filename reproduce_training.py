"""Run the paper's complete training chain inside a newly created workspace.

This is substantially slower than saved-prediction recomputation. A compatible
CUDA GPU and separately downloaded TabPFN v2 checkpoint are required.
"""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from verify_package import main as verify_package

ROOT=Path(__file__).resolve().parent
R03='03_首轮实测_20261018';R05='05_框架再评估_20261018';R06='06_周期目标与Agent方法_20261018'
R07='07_周期损失与TabPFN_20261018';R08='08_形状水平解耦与施肥约束_20261018'
R11='11_图件诊断补强_20261018';R17='17_图件精修_20261021'
EXPECTED='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True,help='A new directory, separate from frozen outputs')
    ap.add_argument('--tabpfn-python',default=sys.executable,help='Python with TabPFN 2.1.2 / CUDA PyTorch')
    ap.add_argument('--figures',action='store_true')
    args=ap.parse_args();verify_package()
    ckpt=Path(os.environ.get('TABPFN_CHECKPOINT',str(Path.home()/'.cache/tabpfn/tabpfn-v2-regressor.ckpt'))).expanduser().resolve()
    if not ckpt.is_file():raise FileNotFoundError('Set TABPFN_CHECKPOINT to the separately downloaded v2 regressor checkpoint.')
    if hashlib.sha256(ckpt.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Checkpoint SHA-256 differs from the paper.')
    out=args.output.resolve()
    if out.exists():raise RuntimeError('Full training requires a new output directory.')
    out.mkdir(parents=True)
    # Copy only sources, protocols, geographic context and raw EFELE inputs.
    for sub in ['scripts','portable_helpers','00_原始数据']:
        shutil.copytree(ROOT/sub,out/sub,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'reproduce_baseline.py',out/'reproduce_baseline.py')
    for stage in [R03,R05,R06,R07,R08,R11,R17]:
        (out/stage).mkdir(exist_ok=True)
        if (ROOT/stage/'scripts').exists():
            shutil.copytree(ROOT/stage/'scripts',out/stage/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
        for p in (ROOT/stage).glob('*protocol.md'):shutil.copy2(p,out/stage/p.name)
    shutil.copytree(ROOT/R11/'evidence/geography',out/R11/'evidence/geography')
    (out/R11/'tables').mkdir(exist_ok=True)
    env=os.environ.copy();env['TABPFN_CHECKPOINT']=str(ckpt);env['PYTHONIOENCODING']='utf-8'
    env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1';env['TABPFN_DISABLE_TELEMETRY']='1'
    cpu=sys.executable;gpu=args.tabpfn_python
    commands=[
        [cpu,'reproduce_baseline.py'],
        [cpu,f'{R07}/scripts/joint_cycle_training.py','--prepare'],
        [cpu,f'{R07}/scripts/joint_cycle_training.py'],
        [gpu,f'{R07}/scripts/tabpfn_experiment.py','--smoke'],
        [gpu,f'{R07}/scripts/tabpfn_experiment.py'],
        [cpu,f'{R07}/scripts/evaluate_experiments.py'],
        [cpu,f'{R08}/scripts/prepare_bases.py'],
        [gpu,f'{R08}/scripts/tabpfn_crossfit.py'],
        [cpu,f'{R08}/scripts/verify_structure.py'],
        [cpu,f'{R08}/scripts/fit_decoupled.py'],
        [cpu,f'{R08}/scripts/evaluate_results.py'],
        [gpu,f'{R11}/scripts/explain_frozen_models.py'],
        [cpu,f'{R11}/scripts/extract_tree_attributions.py'],
        [cpu,f'{R11}/scripts/compute_extensions.py'],
        [cpu,'-c',f"import sys;sys.path.insert(0,{(R11+'/scripts')!r});from interpretation_figures import prepare;prepare()"],
    ]
    if args.figures:commands.append([cpu,f'{R17}/scripts/make_figures.py'])
    for i,cmd in enumerate(commands,1):
        print(f'Stage {i}/{len(commands)}: '+str(cmd[1]),flush=True)
        subprocess.run(cmd,check=True,cwd=out,env=env)
    print('Training workspace:',out)

if __name__=='__main__':main()
