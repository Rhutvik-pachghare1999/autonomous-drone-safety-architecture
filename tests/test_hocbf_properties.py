"""
Property-based safety-invariant tests for the HOCBF filter (review §2 testing strategy).

Instead of hand-picked cases, hypothesis generates thousands of random states and
asserts the safety INVARIANTS hold for all of them:

  P1. Output thrust is always finite and within actuator bounds [T_min, T_max].
  P2. The filtered thrust satisfies the HOCBF constraint OR is the documented
      infeasible-fallback (hover), never an unsafe pass-through.
  P3. Non-finite inputs never produce a non-finite or out-of-bounds output (fail-safe).
  P4. A safe hover command at safe altitude is not perturbed (no unnecessary intervention).

These test the REAL compiled module (src/control/hocbf), not a reimplementation.
Run: python3.12 -m pytest tests/test_hocbf_properties.py -q
"""
import math

import pytest
from hypothesis import given, settings, strategies as st

import hocbf  # conftest.py puts src/control on sys.path


def _cbf():
    return hocbf.HOCBF(hocbf.HOCBFParams())


# Bounded, physically-meaningful ranges for the quadrotor state.
pz_s = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
vz_s = st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False)
ang_s = st.floats(min_value=-math.pi / 2, max_value=math.pi / 2,
                  allow_nan=False, allow_infinity=False)
tnom_s = st.floats(min_value=-100.0, max_value=200.0, allow_nan=False, allow_infinity=False)


@given(pz=pz_s, vz=vz_s, roll=ang_s, pitch=ang_s, T_nom=tnom_s)
@settings(max_examples=2000)
def test_output_within_actuator_bounds(pz, vz, roll, pitch, T_nom):
    """P1: filtered thrust is always finite and in [T_min, T_max]."""
    c = _cbf()
    p = c.params()
    T = c.filter_thrust(pz=pz, vz=vz, roll=roll, pitch=pitch, T_nom=T_nom)
    assert math.isfinite(T), f"non-finite thrust for {(pz, vz, roll, pitch, T_nom)}"
    assert p.T_min - 1e-6 <= T <= p.T_max + 1e-6, f"T={T} outside [{p.T_min}, {p.T_max}]"


@given(pz=pz_s, vz=vz_s, roll=ang_s, pitch=ang_s, T_nom=tnom_s)
@settings(max_examples=2000)
def test_cbf_constraint_or_infeasible_fallback(pz, vz, roll, pitch, T_nom):
    """
    P2: feasible region — output satisfies (conservatism-inflated) lower
    bound. Infeasible region — NO output can satisfy the CBF constraint
    (hover included), so the filter returns T_max, the least-violation
    action: it maximizes LgLfh*T - g + a1*vz + a2*pz over the actuator
    range. Any T < T_max (e.g. hover) violates the constraint strictly
    MORE. It is NEVER an unsafe pass-through of T_nom.
    """
    c = _cbf()
    p = c.params()
    T = c.filter_thrust(pz=pz, vz=vz, roll=roll, pitch=pitch, T_nom=T_nom)

    # Mirror the implementation EXACTLY: LgLfh floored at 0.05 (near-singular tilt
    # guard) and a conservatism factor applied to positive lower bounds. Using a
    # different LgLfh than the code would mis-decide feasibility (see review Item 1).
    CONSERVATISM = 1.08
    LgLfh = max(math.cos(roll) * math.cos(pitch) / p.mass, 0.05)
    T_lb_raw = (p.g - p.alpha1 * vz - p.alpha2 * pz) / LgLfh
    T_lb = T_lb_raw * CONSERVATISM if T_lb_raw > 0.0 else T_lb_raw

    if T_lb > p.T_max:  # infeasible region: least-violation = max thrust
        assert abs(T - p.T_max) < 1e-6, (
            f"infeasible case must return T_max (least violation), got {T}; "
            f"hover violates the CBF constraint strictly more"
        )
        # Sanity: T_max indeed dominates hover in constraint satisfaction
        lhs_tmax = LgLfh * p.T_max - p.g + p.alpha1 * vz + p.alpha2 * pz
        lhs_hover = LgLfh * min(p.mass * p.g, p.T_max) - p.g + p.alpha1 * vz + p.alpha2 * pz
        assert lhs_tmax >= lhs_hover
    else:
        # feasible: output must satisfy the (conservatism-inflated) lower bound
        assert T >= min(T_lb, p.T_max) - 1e-3, \
            f"CBF lower bound violated: T={T} < T_lb={T_lb} at {(pz, vz, roll, pitch, T_nom)}"


NON_FINITE = st.sampled_from([float("nan"), float("inf"), float("-inf")])


@given(bad=NON_FINITE, pz=pz_s, vz=vz_s)
@settings(max_examples=300)
def test_non_finite_inputs_fail_safe(bad, pz, vz):
    """P3: any non-finite input still yields a finite, in-bounds thrust."""
    c = _cbf()
    p = c.params()
    for kwargs in (
        dict(pz=bad, vz=vz, roll=0.0, pitch=0.0, T_nom=19.62),
        dict(pz=pz, vz=bad, roll=0.0, pitch=0.0, T_nom=19.62),
        dict(pz=pz, vz=vz, roll=0.0, pitch=0.0, T_nom=bad),
    ):
        T = c.filter_thrust(**kwargs)
        assert math.isfinite(T)
        assert p.T_min - 1e-6 <= T <= p.T_max + 1e-6


@given(pz=st.floats(min_value=3.0, max_value=100.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_safe_hover_not_perturbed(pz):
    """P4: hover thrust at safe altitude/zero velocity passes through unchanged."""
    c = _cbf()
    p = c.params()
    T_hover = p.mass * p.g
    T = c.filter_thrust(pz=pz, vz=0.0, roll=0.0, pitch=0.0, T_nom=T_hover)
    assert abs(T - T_hover) < 1e-6, f"safe hover perturbed at pz={pz}: {T} vs {T_hover}"
