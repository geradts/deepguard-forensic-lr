from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def _as_float(x):
    return np.asarray(x, dtype=float).ravel()


def _log2_1pexp(x):
    x=np.asarray(x,dtype=float)
    return np.logaddexp(0.0,x)/np.log(2.0)


def cllr(llr_h1: Sequence[float], llr_h2: Sequence[float]) -> float:
    """Log likelihood-ratio cost C_llr for H1 and H2 trials.

    LLRs use natural logarithms. The metric itself is expressed in bits.
    H1 trials should have positive evidence and H2 trials negative evidence.
    """
    a=_as_float(llr_h1); b=_as_float(llr_h2)
    if a.size==0 or b.size==0:
        raise ValueError("Both H1 and H2 must contain at least one trial")
    return float(0.5*(np.mean(_log2_1pexp(-a))+np.mean(_log2_1pexp(b))))


def cllr_min(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Minimum empirical C_llr under a monotonic score-to-LLR mapping.

    Uses isotonic regression (PAV) to estimate the empirical posterior under
    equal class priors, then converts posterior odds to LLR. This is an
    evaluation statistic, not a replacement for a separately trained
    calibration model.
    """
    s=_as_float(scores); y=np.asarray(labels,dtype=int).ravel()
    if s.size != y.size or s.size==0:
        raise ValueError("scores and labels must have equal non-zero length")
    if not set(np.unique(y)).issubset({0,1}) or len(np.unique(y))<2:
        raise ValueError("labels must contain both 0 and 1")
    iso=IsotonicRegression(y_min=0.0,y_max=1.0,increasing=True,out_of_bounds="clip")
    p=iso.fit_transform(s,y)
    eps=np.finfo(float).eps
    llr=np.log(np.clip(p,eps,1-eps)/np.clip(1-p,eps,1-eps))
    return cllr(llr[y==1],llr[y==0])


@dataclass
class LRFusionSystem:
    """Multivariate logistic LR fusion for detector scores/features.

    The model is trained with equal total weight for H1 and H2, so the
    logistic decision function is on an equal-prior log-odds scale and is
    interpreted as an LLR. This is deliberately explicit rather than hiding
    the assumptions behind a third-party LR framework.
    """
    C: float=1.0
    max_iter: int=2000

    def __post_init__(self):
        self.model=Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
            ("logit",LogisticRegression(C=self.C,solver="lbfgs",max_iter=self.max_iter))
        ])
        self.feature_names_: list[str]|None=None

    def fit(self,X,y,feature_names: Iterable[str]|None=None):
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=int).ravel()
        if X.ndim!=2 or X.shape[0]!=y.size: raise ValueError("X/y shape mismatch")
        if set(np.unique(y))!={0,1}: raise ValueError("y must contain both 0 and 1")
        counts=np.bincount(y,minlength=2).astype(float)
        weights=np.where(y==1,0.5/counts[1],0.5/counts[0])
        self.model.fit(X,y,logit__sample_weight=weights)
        self.feature_names_=list(feature_names) if feature_names is not None else None
        return self

    def transform(self,X):
        return self.model.decision_function(np.asarray(X,dtype=float))

    def predict_llr(self,X):
        return np.asarray(self.transform(X),dtype=float).ravel()

    def metrics(self,X,y):
        y=np.asarray(y,dtype=int).ravel(); llr=self.predict_llr(X)
        return {"cllr":cllr(llr[y==1],llr[y==0]),"cllr_min":cllr_min(llr,y)}

    def save(self,path):
        import joblib
        joblib.dump({"model":self.model,"feature_names":self.feature_names_,"C":self.C},path)

    @classmethod
    def load(cls,path):
        import joblib
        d=joblib.load(path); obj=cls(C=d.get("C",1.0)); obj.model=d["model"]; obj.feature_names_=d.get("feature_names"); return obj
