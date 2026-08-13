#!/usr/bin/env python3
"""DeepGuard DF40 benchmark orchestrator.

Runs the official DeepfakeBench test entry point after validating the Drive
workspace. It never installs the obsolete `lir` package and never downloads
DF40 automatically. Detector outputs are kept separate from DeepGuard LR
calibration/evaluation.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

PROTOCOLS={
    "p2":["FSAll_cdf","FRAll_cdf","EFSAll_cdf"],
    "p3":["deepfacelab","heygen","whichisreal","MidJourney","stargan","starganv2","styleclip","e4e","CollabDiff"],
}

def sha256(path: Path, chunk=1024*1024):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--drive-root',required=True,type=Path)
    ap.add_argument('--df40-root',required=True,type=Path)
    ap.add_argument('--weights',required=True,type=Path)
    ap.add_argument('--protocol',choices=['p2','p3'],default='p3')
    ap.add_argument('--detector',default='xception')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()

    root=args.drive_root; df40=args.df40_root; weights=args.weights
    out=root/'benchmark'/'df40'/args.detector/args.protocol
    out.mkdir(parents=True,exist_ok=True)
    log=out/'run.log'
    dataset_json=root/'preprocessing'/'dataset_json'
    detector_cfg=df40/'training'/'config'/'detector'/f'{args.detector}.yaml'
    test_py=df40/'training'/'test.py'

    checks={
        'drive_root':root.exists(), 'df40_root':df40.exists(),
        'dataset_json':dataset_json.exists(), 'detector_config':detector_cfg.exists(),
        'weights':weights.exists(), 'test_py':test_py.exists()
    }
    missing=[k for k,v in checks.items() if not v]
    print('DeepGuard DF40 preflight')
    for k,v in checks.items(): print(f'  {k:20} {"PASS" if v else "MISSING"}')
    if missing:
        print('STOP — missing:',', '.join(missing)); return 2

    datasets=PROTOCOLS[args.protocol]
    missing_json=[d for d in datasets if not (dataset_json/f'{d}.json').exists()]
    if missing_json:
        print('STOP — missing JSON manifests:',', '.join(missing_json)); return 3

    manifest={
        'utc':datetime.now(timezone.utc).isoformat(),'protocol':args.protocol,
        'detector':args.detector,'datasets':datasets,
        'df40_root':str(df40),'dataset_json':str(dataset_json),
        'detector_config':str(detector_cfg),'weights':str(weights),
        'weights_sha256':sha256(weights),'python':sys.version,'cwd':os.getcwd(),
        'note':'Detector output is not LR-calibrated here.'
    }
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))

    cmd=[sys.executable,str(test_py),'--detector_path',str(detector_cfg),
         '--weights_path',str(weights),'--test_dataset',*datasets]
    print('COMMAND:',' '.join(map(str,cmd)))
    if args.dry_run: return 0
    with log.open('a') as lf:
        lf.write('\n$ '+' '.join(map(str,cmd))+'\n')
        p=subprocess.run(cmd,cwd=str(df40),stdout=lf,stderr=subprocess.STDOUT)
    (out/'exit_code.txt').write_text(str(p.returncode))
    print('Exit code:',p.returncode)
    print('Results:',out)
    return p.returncode

if __name__=='__main__': raise SystemExit(main())
