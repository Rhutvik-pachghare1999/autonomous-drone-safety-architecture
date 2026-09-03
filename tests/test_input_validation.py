"""
Input-validation tests for the HOCBF safety filter.

A VLA/PID upstream can emit non-finite commands. The filter is a trust
boundary: it must never pass a NaN/Inf thrust to the actuators. These tests
pin the fail-safe behavior added to HOCBF.filter_thrust — every non-finite
input must yield a finite thrust inside [T_min, T_max].
"""
import math

import pytest

import hocbf  # conftest.py puts src/control on sys.path


@pytest.fixture
def cbf():
    return hocbf.HOCBF(hocbf.HOCBFParams())


NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("bad", NON_FINITE)
def test_non_finite_T_nom_is_safe(cbf, bad):
    """A non-finite nominal thrust must not propagate to the output."""
    T = cbf.filter_thrust(pz=2.0, vz=0.0, roll=0.0, pitch=0.0, T_nom=bad)
    p = cbf.params()
    assert math.isfinite(T)
    assert p.T_min <= T <= p.T_max


@pytest.mark.parametrize("field", ["pz", "vz", "roll", "pitch"])
@pytest.mark.parametrize("bad", NON_FINITE)
def test_non_finite_state_falls_back_to_hover(cbf, field, bad):
    """Non-finite state input must fall back to a finite in-range thrust."""
    args = dict(pz=2.0, vz=0.0, roll=0.0, pitch=0.0, T_nom=19.62)
    args[field] = bad
    T = cbf.filter_thrust(**args)
    p = cbf.params()
    assert math.isfinite(T)
    assert p.T_min <= T <= p.T_max


def test_all_inputs_non_finite_returns_hover(cbf):
    """Total garbage in must still yield a bounded, finite thrust."""
    nan = float("nan")
    T = cbf.filter_thrust(pz=nan, vz=nan, roll=nan, pitch=nan, T_nom=nan)
    p = cbf.params()
    assert math.isfinite(T)
    assert p.T_min <= T <= p.T_max
    # With no trustworthy state, the fallback is hover thrust (m*g), clamped.
    expected = min(max(p.mass * p.g, p.T_min), p.T_max)
    assert abs(T - expected) < 1e-9


def test_finite_input_unchanged(cbf):
    """The guard must not alter the normal (finite) path."""
    T_hover = cbf.params().mass * cbf.params().g
    T = cbf.filter_thrust(pz=5.0, vz=0.0, roll=0.0, pitch=0.0, T_nom=T_hover)
    assert abs(T - T_hover) < 1e-9
