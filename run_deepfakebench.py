from __future__ import annotations
import argparse
import pandas as pd
from detectors.deepfakebench import DeepfakeBenchAdapter

EXT={'.mp4','.mov','.avi','.mkv','.webm','.m4v'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('--checkout',required=True)
    p.add_argument('--command',required=True,help='Command template with {video} and {output}')
    p.add_argument('--output',default='deepfakebench_features.csv')
    a=p.parse_args()
    from pathlib import Path
    root=Path(a.input)
    videos=[root] if root.is_file() else sorted(x for x in root.rglob('*') if x.suffix.lower() in EXT)
    det=DeepfakeBenchAdapter(a.checkout,a.command)
    rows=[]
    for v in videos:
        try:
            r=det.score_video(v)
            rows.append(r.to_dict())
            print('OK',v,r.score)
        except Exception as e:
            print('ERROR',v,e)
    if not rows: raise SystemExit('No videos scored')
    pd.DataFrame(rows).to_csv(a.output,index=False)
    print('Wrote',len(rows),'rows to',a.output)

if __name__=='__main__': main()
