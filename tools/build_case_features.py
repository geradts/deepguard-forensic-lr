from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

BASE=['video_id','label','source_id','subject_id','generator_id','dataset']

def main():
    p=argparse.ArgumentParser(description='Build final DeepGuard LiR feature table')
    p.add_argument('inputs',nargs='+')
    p.add_argument('--output',default='deepguard_features.csv')
    a=p.parse_args()
    frames=[]
    for f in a.inputs:
        df=pd.read_csv(f)
        df['source_file']=Path(f).name
        frames.append(df)
    if not frames: raise SystemExit('No input files')
    out=pd.concat(frames,ignore_index=True,sort=False)
    if 'video_id' not in out: raise SystemExit('video_id is required')
    out=out.drop_duplicates(subset=['video_id'],keep='first')
    out.to_csv(a.output,index=False)
    print(f'Wrote {len(out)} unique videos to {a.output}')

if __name__=='__main__': main()
