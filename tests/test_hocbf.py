"""
Objective 1 — HOCBF Validation Test

Verifies that the HOCBF safety filter:
  1. Passes safe commands unchanged
  2. Blocks adversarial VLA commands that would cause ground collision
  3. Enforces the HOCBF invariant: Lf²h + LgLfh·T + α₁·Lfh + α₂·h >= 0
  4. Never outputs thrust below T_min or above T_max
"""
import sys
import math
import pytest

sys.path.insert(0, "src/control")
import hocbf  # noqa: E402  (built C++ module)


@pytest.fixture
def cbf():
    p = hocbf.HOCBFParams()
    p.mass   = 2.0
    p.alpha1 = 2.0
    p.alpha2 = 1.0
    return hocbf.HOCBF(p)


def hocbf_invariant(cbf, pz, vz, roll, pitch, T):
    """Returns True if HOCBF constraint is satisfied."""
    p = cbf.params()
    Lf2h_free = -p.g
    LgLfh = max(math.cos(roll) * math.cos(pitch) / p.mass, 0.01)
    return LgLfh * T + (-Lf2h_free) + p.alpha1 * vz + p.alpha2 * pz >= -1e-6


# ── Safe pass-through ─────────────────────────────────────────────────────────

def test_hover_unchanged(cbf):
    """Hover thrust at safe altitude must pass through unmodified."""
    T_hover = cbf.params().mass * cbf.params().g
    T_out = cbf.filter_thrust(pz=5.0, vz=0.0, roll=0.0, pitch=0.0, T_nom=T_hover)
    assert abs(T_out - T_hover) < 1e-9


def test_safe_climb_unchanged(cbf):
    """Upward command at safe altitude must not be modified."""
    T_nom = 25.0  # above hover
    T_out = cbf.filter_thrust(pz=3.0, vz=0.5, roll=0.0, pitch=0.0, T_nom=T_nom)
    assert abs(T_out - T_nom) < 1e-9


# ── Adversarial VLA commands ──────────────────────────────────────────────────

def test_crash_command_blocked(cbf):
    """VLA command 'fly into ground' must be filtered to safe thrust."""
    # Near ground, descending fast — T_nom=0 would cause crash
    T_out = cbf.filter_thrust(pz=0.5, vz=-3.0, roll=0.0, pitch=0.0, T_nom=0.0)
    assert T_out > 0.0, "Zero thrust near ground must be raised"
    assert hocbf_invariant(cbf, 0.5, -3.0, 0.0, 0.0, T_out)


def test_negative_thrust_blocked(cbf):
    """Negative thrust (physically impossible) must be clamped to T_min."""
    T_out = cbf.filter_thrust(pz=2.0, vz=0.0, roll=0.0, pitch=0.0, T_nom=-100.0)
    assert T_out >= cbf.params().T_min


def test_at_ground_level(cbf):
    """At pz=0 with downward velocity, filter must enforce safety."""
    T_out = cbf.filter_thrust(pz=0.0, vz=-1.0, roll=0.0, pitch=0.0, T_nom=0.0)
    assert hocbf_invariant(cbf, 0.0, -1.0, 0.0, 0.0, T_out)


def test_extreme_tilt_handled(cbf):
    """Near-90° tilt (LgLfh → 0) must not produce NaN or infinite thrust."""
    T_out = cbf.filter_thrust(pz=1.0, vz=0.0, roll=1.5, pitch=0.0, T_nom=20.0)
    assert math.isfinite(T_out)
    assert cbf.params().T_min <= T_out <= cbf.params().T_max


# ── Full VLA command filter ───────────────────────────────────────────────────

def test_vla_horizontal_clipped(cbf):
    """Horizontal velocity > v_max must be clipped."""
    cmd = cbf.filter_vla_command(5.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, v_max=5.0)
    assert abs(cmd.vx) <= 5.0 + 1e-9
    assert cmd.was_filtered


def test_vla_safe_command_passes(cbf):
    """Safe VLA command at cruise altitude must not be filtered."""
    cmd = cbf.filter_vla_command(10.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, v_max=5.0)
    assert not cmd.was_filtered


def test_vla_adversarial_descend_blocked(cbf):
    """'Crash into ground' VLA command must be filtered."""
    # vz_nom = -10 m/s near ground
    cmd = cbf.filter_vla_command(0.3, -2.0, 0.0, 0.0, 0.0, 0.0, -10.0, v_max=5.0)
    assert cmd.was_filtered
    assert hocbf_invariant(cbf, 0.3, -2.0, 0.0, 0.0, cmd.T)


# ── Bounds ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pz,vz,T_nom", [
    (0.1, -5.0, -50.0),
    (0.0, -10.0, 0.0),
    (5.0, 0.0, 1000.0),
    (2.0, 2.0, 0.0),
])
def test_output_always_in_bounds(cbf, pz, vz, T_nom):
    T_out = cbf.filter_thrust(pz, vz, 0.0, 0.0, T_nom)
    assert cbf.params().T_min <= T_out <= cbf.params().T_max
    assert math.isfinite(T_out)
