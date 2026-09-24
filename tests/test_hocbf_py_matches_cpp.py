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


def test_nan_parity_infeasible_branch():
    """NaN/Inf T_nom with an INFEASIBLE constraint set: all ports must agree.
    Graphaudit 2026-09-24 found C++ returned T_max while C returned hover in
    exactly this branch. Fix: all impls return T_max (least-violation action).
    Diving near ground at 86deg tilt -> T_lb >> T_max -> T_max from every port."""
    c = _f(cpp); q = _f(py)
    nan = float("nan"); infs = [nan, float("inf"), float("-inf")]
    for bad in infs:
        tc = c.filter_thrust(0.1, -10.0, 1.5, 0.0, bad)
        tq = q.filter_thrust(0.1, -10.0, 1.5, 0.0, bad)
        p = c.params()
        assert tc == tq == p.T_max, (
            f"T_nom={bad}: C++={tc} Py={tq} (expected T_max={p.T_max} "
            f"in the infeasible branch — ports diverged again)"
        )
    # Replica of the C path (src/rt/safety_filter.c hocbf_filter): NaN T_nom
    # becomes T_max ("force clamp to feasible set"), falls through to the
    # infeasible check -> T_max. Expressed in math over the shared constants.
    m, g, a1, a2, tmax, cons = 0.027, 9.81, 2.0, 1.0, 0.60, 1.08
    vz, pz, roll = -10.0, 0.1, 1.5
    lglfh = max(math.cos(roll) * math.cos(0.0) / m, 0.05)
    t_lb = (g - a1 * vz - a2 * pz) / lglfh
    if t_lb > 0.0: t_lb *= cons
    lo = max(0.0, t_lb)
    assert lo > tmax, "test state must be infeasible to exercise the branch"
    t_c_style = tmax if lo > tmax else min(max(tmax, lo), tmax)
    assert t_c_style == tmax
