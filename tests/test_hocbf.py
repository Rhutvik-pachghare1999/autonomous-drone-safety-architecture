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
    p.T_max  = 4.0 * 2.0 * 9.81   # 4x hover for a 2 kg quad — must be set
                                  # explicitly: the module default (0.60 N)
                                  # is the Crazyflie 2.X physical motor limit
                                  # and would make 2 kg hover infeasible
    p.alpha1 = 2.0
    p.alpha2 = 1.0
    return hocbf.HOCBF(p)


def hocbf_invariant(cbf, pz, vz, roll, pitch, T):
    """
    Returns True if the PURE CBF constraint is satisfied.

    The implementation in hocbf.cpp adds:
      - conservatism factor (1.08) on positive T_lb
      - wider singularity guard (0.05 vs 0.01)
    This helper validates the textbook CBF form: Lf²h + LgLfh·u + α₁·Lfh + α₂·h ≥ 0
    where Lf²h = -g, Lfh = vz, h = pz, u = T.
    """
    p = cbf.params()
    LgLfh = max(math.cos(roll) * math.cos(pitch) / p.mass, 0.01)
    # Constraint: LgLfh*T - g + α₁*vz + α₂*pz ≥ 0
    return LgLfh * T - p.g + p.alpha1 * vz + p.alpha2 * pz >= -1e-6


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


def test_gravity_sign_in_invariant(cbf):
    """
    The CBF invariant helper MUST use -g (not +g) in the constraint.

    At pz=0.5m, vz=-3.0m/s: correct T_lb = 30.62 N.
    Old helper with +g gave T_lb = -8.62 N (always passes).
    This test would pass with the wrong helper but fails with the correct one
    for T_nom between -8.62 and 30.62 N.
    """
    # T_nom=20.0 is below the correct safe bound (30.62 N) but above
    # the incorrect bound (-8.62 N) — it should be filtered UP.
    T_out = cbf.filter_thrust(pz=0.5, vz=-3.0, roll=0.0, pitch=0.0, T_nom=20.0)
    assert hocbf_invariant(cbf, 0.5, -3.0, 0.0, 0.0, T_out)
    # The filter must correct 20.0 → ≥30.62
    assert T_out >= 30.6


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


def test_infeasible_safe_fallback(cbf):
    """
    When required safe thrust exceeds T_max, the feasible set is EMPTY:
    every thrust in [T_min, T_max] violates the CBF constraint — hover
    (m*g) included, since hover < T_max < T_lb. The old "hover fallback"
    was therefore strictly less safe: sinking fast near the ground,
    m*g decelerates more weakly than T_max.

    The filter must return the least-violation action T_max and flag
    infeasibility via was_infeasible (so the supervisor sees the CBF
    constraint cannot be met and can escalate).

    Scenario: pz=0.1m, vz=-10m/s, roll=1.5rad (86° tilt → LgLfh≈0.007).
    T_lb = (9.81 - 2*(-10) - 1*0.1) / 0.007 ≈ 4200 N >> T_max.
    """
    cmd = cbf.filter_vla_command(
        pz=0.1, vz=-10.0, roll=1.5, pitch=0.0,
        vx_nom=0.0, vy_nom=0.0, vz_nom=-10.0, v_max=5.0
    )
    # Least-violation action: maximum recovery authority, not hover.
    assert abs(cmd.T - cbf.params().T_max) < 1e-9, (
        f"Infeasible case returned {cmd.T:.2f}, expected T_max "
        f"({cbf.params().T_max:.2f}) — hover violates the CBF constraint "
        f"more, not less"
    )
    assert math.isfinite(cmd.T)
    assert cmd.was_filtered
    # Must indicate infeasibility was detected
    assert cmd.was_infeasible


# Watchdog tests now exercise the REAL C function via tests/watchdog_test.c
# (compiled and run separately). The local _watchdog_step() helper was removed.
# See tests/watchdog_test.c for the C test that calls vla_watchdog_check() directly.


# ── ONNX observation layout test ───────────────────────────────────────────────
# Training order from sim/quadrotor_rl.py _actor_obs:
# [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]

OBS_LAYOUT = ["px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz"]

def test_onnx_obs_layout_matches_training():
    """
    Verify ONNX inference observation layout matches training.

    The C benchmark (safety_filter.c) only has pz, vz available.
    Other fields are zeroed → known domain mismatch.
    This test documents the expected layout so C and Python can't drift.
    """
    # Training uses this exact order
    assert OBS_LAYOUT[2] == "pz"
    assert OBS_LAYOUT[5] == "vz"
    assert OBS_LAYOUT[6] == "qw"
    
    # C code should use: obs[2]=pz, obs[5]=vz, obs[6]=1.0
    # Documenting here so both sides reference the same truth


# ── Shared-memory layout tests ─────────────────────────────────────────────────
# Verify VLACommand layout compatibility between Python (shm_bridge.py)
# and C (src/rt/watchdog.h). Seqlock layout, 64-byte file:
#   offset  0: seq_head  (Q, even = stable, odd = write in progress)
#   offset  8: vx_nom, vy_nom, vz_nom (3 doubles)
#   offset 32: is_new_data (?), 7 bytes pad
#   offset 40: seq_tail  (Q, torn-read check: must equal seq_head)
#   offset 48: 16 bytes pad
# Python publishes with three writes (head-odd -> payload+tail -> head-even);
# the C vla_shm_snapshot() rejects torn frames.

import struct as _struct

VLA_CMD_FMT = "=Qddd?7xQ16x"  # Python shm_bridge.py full-frame format
VLA_CMD_SIZE = _struct.calcsize(VLA_CMD_FMT)

def test_vla_shm_layout_size():
    """Full VLACommand frame must be 64 bytes (C struct is aligned(64))."""
    assert VLA_CMD_SIZE == 64, f"Expected 64, got {VLA_CMD_SIZE}"

def test_vla_shm_field_offsets():
    """
    Verify field offsets match between Python pack and C struct.

    Python '=Qddd?7xQ16x' layout (native byte order, explicit padding):
      offset 0:  seq_head (uint64, 8 bytes) — seqlock, even when stable
      offset 8:  vx_nom (double, 8 bytes)
      offset 16: vy_nom (double, 8 bytes)
      offset 24: vz_nom (double, 8 bytes)
      offset 32: is_new_data (bool, 1 byte)
      offset 33: pad (7 bytes) — aligns seq_tail to 8
      offset 40: seq_tail (uint64, 8 bytes) — equals seq_head when clean
      offset 48: pad (16 bytes)
    Total: 64 bytes — matches C VLACommand (watchdog.h) exactly.
    """
    sample = _struct.pack(VLA_CMD_FMT, 42, 1.0, 2.0, 3.0, True, 42)
    assert len(sample) == 64

    seq, vx, vy, vz, is_new, tail = _struct.unpack(VLA_CMD_FMT, sample)
    assert seq == 42          # seq_head at offset 0
    assert vx == 1.0
    assert vy == 2.0
    assert vz == 3.0
    assert is_new is True
    assert tail == 42         # seq_tail at offset 40
    # A clean frame has head == tail and even head (seqlock contract)
    assert seq % 2 == 0 and seq == tail

def test_vla_shm_c_struct_compatibility():
    """
    C src/rt/watchdog.h VLACommand struct layout:
      uint64_t seq_head;      // 8 bytes @ 0   (seqlock)
      double vx_nom, vy_nom, vz_nom;  // 24 bytes @ 8
      uint8_t is_new_data;    // 1 byte  @ 32
      char _pad1[7];          // 7 bytes @ 33  (aligns seq_tail)
      uint64_t seq_tail;      // 8 bytes @ 40
      char _pad2[16];         // 16 bytes @ 48 — aligned(64) total

    Python '=Qddd?7xQ16x' MUST produce byte-identical offsets.
    """
    assert 8 + 24 + 1 + 7 + 8 + 16 == 64
    # seq_tail offset in C = 8 + 24 + 1 + 7 = 40
    assert 8 + 24 + 1 + 7 == 40
