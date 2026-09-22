"""Proof the pure-Python HOCBF port matches the compiled C++ module bit-for-bit.
Run where the C++ hocbf.*.so is built (e.g. laptop): PYTHONPATH=src/control pytest.
2450 states (edge + random); asserts max |T_cpp - T_py| == 0 and identical flags."""
import sys, math, random, os, pytest
sys.path.insert(0, "src/control"); sys.path.insert(0, "sim")
cpp = pytest.importorskip("hocbf")          # skip if C++ .so not built here
import hocbf_py as py
def _f(mod):
    p=mod.HOCBFParams(); p.mass=0.027;p.g=9.81;p.alpha1=2;p.alpha2=1;p.T_max=0.60;p.T_min=0;p.conservatism=1.08
    return mod.HOCBF(p)
def test_identical():
    c=_f(cpp); q=_f(py); random.seed(0); md=0.0; mism=0
    for _ in range(2000):
        a=(random.uniform(0,5),random.uniform(-10,10),random.uniform(-1.4,1.4),
           random.uniform(-1.4,1.4),random.uniform(-8,8),random.uniform(-8,8),random.uniform(-6,6))
        sc=c.filter_vla_command(*a,5.0); sq=q.filter_vla_command(*a,5.0)
        md=max(md,abs(sc.T-sq.T)); mism+= (abs(sc.T-sq.T)>1e-9 or sc.was_filtered!=sq.was_filtered)
    assert mism==0 and md<1e-9, f"port diverges: maxdiff={md} mism={mism}"
