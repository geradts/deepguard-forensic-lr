from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

META=['video_id','label','source_id','subject_id','generator_id','dataset']

def normalise(df,name):
    if 'video_id' not in df.columns:
        for c in ['video','filename','file','name']:
            if c in df.columns:
                df=df.rename(columns={c:'video_id'}); break
    if 'video_id' not in df.columns:
        raise ValueError(f'{name}: no video_id column')
    score_col=None
    for c in ['score','prediction','probability','prob','fake_score']:
        if c in df.columns: score_col=c; break
    if score_col is None:
        raise ValueError(f'{name}: no score column')
    out=df[['video_id']+[c for c in META if c in df.columns and c!='video_id']].copy()
    out[f'{name}_score']=pd.to_numeric(df[score_col],errors='coerce')
    return out

def main():
    p=argparse.ArgumentParser(description='Merge DeepfakeBench/detector outputs into a LiR feature table')
    p.add_argument('--ftcn',required=True)
    p.add_argument('--xception',required=True)
    p.add_argument('--output',default='deepguard_features.csv')
    a=p.parse_args()
    f=normalise(pd.read_csv(a.ftcn),'ftcn')
    x=normalise(pd.read_csv(a.xception),'xception')
    meta=['video_id','label','source_id','subject_id','generator_id','dataset']
    m=f.merge(x,on='video_id',how='outer',suffixes=('_ftcn','_xception'))
    for c in meta[1:]:
        left=f'{c}_ftcn'; right=f'{c}_xception'
        if left in m and right in m:
            m[c]=m[left].combine_first(m[right]); m=m.drop(columns=[left,right])
    if 'ftcn_score' in m: m['neural_temporal_score']=m['ftcn_score']
    if 'xception_score' in m: m['neural_spatial_score']=m['xception_score']
    m.to_csv(a.output,index=False)
    print(f'Wrote {len(m)} rows to {a.output}')

if __name__=='__main__': main()
