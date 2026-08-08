from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from detectors import HeuristicVideoDetector

VIDEO_EXT={'.mp4','.mov','.avi','.mkv','.webm','.m4v'}

def main():
    p=argparse.ArgumentParser(description='Run a DeepGuard detector adapter over videos')
    p.add_argument('input',help='video file or directory')
    p.add_argument('--output',default='deepguard_features.csv')
    args=p.parse_args()
    root=Path(args.input)
    paths=[root] if root.is_file() else sorted(x for x in root.rglob('*') if x.suffix.lower() in VIDEO_EXT)
    detector=HeuristicVideoDetector()
    rows=[]
    for path in paths:
        try:
            rows.append(detector.score_video(path).to_dict())
            print('OK',path)
        except Exception as e:
            print('ERROR',path,e)
    if not rows: raise SystemExit('No videos processed successfully')
    pd.DataFrame(rows).to_csv(args.output,index=False)
    print(f'Wrote {len(rows)} rows to {args.output}')

if __name__=='__main__': main()
