#!/usr/bin/env python3
"""Fuse detector scores into DeepGuard-LR.

Input CSV: one row per case/video with `label` (1=H1 deepfake, 0=H0 genuine)
and at least two score columns. Recommended columns are xception_llr,
ftcn_llr, frequency_score and temporal_score.

This script intentionally does not calibrate on an external/open-set test set.
Use --train-csv for development/calibration and --test-csv for held-out data.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def cllr(llr,y):
    llr=np.asarray(llr,float); y=np.asarray(y,int)
    h1=llr[y==1]; h0=llr[y==0]
    if len(h1)==0 or len(h0)==0: raise ValueError('Cllr requires H1 and H0 samples')
    return float(.5*(np.logaddexp(0,-h1).mean()+np.logaddexp(0,h0).mean())/np.log(2))

def cllr_min(llr,y):
    llr=np.asarray(llr,float); y=np.asarray(y,int); o=np.argsort(llr)
    iso=IsotonicRegression(y_min=1e-6,y_max=1-1e-6,out_of_bounds='clip')
    p=iso.fit_transform(llr[o],y[o]); z=np.log(p/(1-p))
    return cllr(z,y[o])

def evaluate(model,X,y):
    z=model.decision_function(X)
    return {'n':int(len(y)),'n_H1':int((y==1).sum()),'n_H0':int((y==0).sum()),
            'cllr':cllr(z,y),'cllr_min':cllr_min(z,y)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-csv',required=True,type=Path); ap.add_argument('--test-csv',required=True,type=Path); ap.add_argument('--out-csv',required=True,type=Path); ap.add_argument('--metrics-json',required=True,type=Path); ap.add_argument('--features',nargs='+',default=['xception_llr','ftcn_llr','frequency_score','temporal_score'])
    a=ap.parse_args(); tr=pd.read_csv(a.train_csv); te=pd.read_csv(a.test_csv)
    features=[f for f in a.features if f in tr.columns and f in te.columns]
    if len(features)<2: raise SystemExit('Need at least two common detector/feature columns')
    for df,name in [(tr,'train'),(te,'test')]:
        if 'label' not in df: raise SystemExit(f'{name} CSV needs label column')
        df.dropna(subset=features+['label'],inplace=True)
    model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000).fit(tr[features],tr.label.astype(int))
    te=te.copy(); te['deepguard_llr']=model.decision_function(te[features])
    a.out_csv.parent.mkdir(parents=True,exist_ok=True); te.to_csv(a.out_csv,index=False)
    metrics=evaluate(model,te[features],te.label.astype(int)); metrics['features']=features; metrics['coef']=model.coef_[0].tolist(); metrics['intercept']=float(model.intercept_[0]); metrics['note']='LR trained only on train-csv; test-csv held out.'
    a.metrics_json.parent.mkdir(parents=True,exist_ok=True); a.metrics_json.write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))

if __name__=='__main__': raise SystemExit(main())
