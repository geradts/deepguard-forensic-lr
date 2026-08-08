import numpy as np
from deepguard_lir import LRFusionSystem, cllr, cllr_min


def test_cllr_finite():
    assert np.isfinite(cllr([2.0, 1.0], [-2.0, -1.0]))


def test_fusion_and_metrics():
    rng=np.random.default_rng(7)
    X1=rng.normal(1,0.5,(80,3)); X0=rng.normal(-1,0.5,(80,3))
    X=np.vstack([X1,X0]); y=np.r_[np.ones(80,dtype=int),np.zeros(80,dtype=int)]
    s=LRFusionSystem().fit(X,y,["x1","x2","x3"])
    m=s.metrics(X,y)
    assert np.isfinite(m["cllr"])
    assert np.isfinite(m["cllr_min"])
    assert s.predict_llr(X[:4]).shape==(4,)
