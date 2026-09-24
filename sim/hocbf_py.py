"""
hocbf_py.py — pure-Python port of src/control/hocbf.cpp (HOCBF altitude safety
filter), faithful to the C++ control law and parameters.

WHY THIS EXISTS: NVIDIA's Isaac Sim runtime container has no C++ compiler, so the
pybind11 `hocbf` module cannot be built in-job on the cluster. The filter itself is
a small deterministic formula; this Python port computes bit-for-bit-equivalent
decisions (verified against the C++ module on a battery of states — see
tests/test_hocbf_py_matches_cpp.py). It is used ONLY for the Sol survival A/B
(a safety-effectiveness measurement). The real-time WCET/latency claim remains
backed by the C++ implementation and its separate EVT benchmark — Python is NOT
used for any timing claim.

Ported exactly from hocbf.cpp:
  HOCBFParams defaults, filter_thrust (incl. non-finite fail-safe, LgLfh floor
  0.05, conservatism 1.08 on positive T_lb, infeasibility -> T_max
  least-violation fallback),
  and filter_vla_command (T_nom = m*g + m*vz_nom*2.0, horizontal v_max clip,
  was_filtered / was_infeasible flags).
"""
import math
from dataclasses import dataclass


@dataclass
class HOCBFParams:
    # Defaults = Crazyflie 2.X vehicle, matching src/control/hocbf.cpp and
    # src/rt/safety_filter.c. All studies must use these vehicle params —
    # never a generic-quadrotor mass — so safety evidence is vehicle-consistent.
    mass: float = 0.027        # kg   (Crazyflie 2.X; sim/crazyflie_env.py MASS)
    g: float = 9.81            # m/s²
    alpha1: float = 2.0        # class-K gain on Lfh (velocity term)
    alpha2: float = 1.0        # class-K gain on h   (position term)
    T_max: float = 0.60        # N, 4 × ~0.15 N per motor (Crazyflie 2.X)
    T_min: float = 0.0
    conservatism: float = 1.08 # reality-gap factor (hocbf.cpp:54)


@dataclass
class SafeCommand:
    vx: float
    vy: float
    T: float
    was_filtered: bool
    was_infeasible: bool


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


class HOCBF:
    def __init__(self, params: HOCBFParams | None = None):
        self.p = params if params is not None else HOCBFParams()

    def params(self) -> HOCBFParams:
        return self.p

    def filter_thrust(self, pz: float, vz: float, roll: float, pitch: float,
                      T_nom: float) -> float:
        p = self.p
        # Non-finite fail-safe (hocbf.cpp:filter_thrust head)
        finite = all(math.isfinite(v) for v in (T_nom, pz, vz, roll, pitch))
        if not finite:
            state_ok = all(math.isfinite(v) for v in (pz, vz, roll, pitch))
            if not state_ok:
                return _clamp(p.mass * p.g, p.T_min, p.T_max)
            LgLfh_s = max(math.cos(roll) * math.cos(pitch) / p.mass, 0.05)
            rhs_s = p.g - p.alpha1 * vz - p.alpha2 * pz
            lb_raw = rhs_s / LgLfh_s
            lb = lb_raw * p.conservatism if lb_raw > 0.0 else lb_raw
            return _clamp(max(p.T_min, lb), p.T_min, p.T_max)

        # Lie derivatives (hocbf.cpp)
        Lf2h_free = -p.g
        LgLfh = math.cos(roll) * math.cos(pitch) / p.mass
        LgLfh_safe = max(LgLfh, 0.05)                    # singularity guard
        rhs = -Lf2h_free - p.alpha1 * vz - p.alpha2 * pz
        T_lb_raw = rhs / LgLfh_safe
        T_lb = T_lb_raw * p.conservatism if T_lb_raw > 0.0 else T_lb_raw

        # Infeasibility: T_lb > T_max means no thrust satisfies the CBF
        # constraint (hover included). Least-violation action = maximum
        # thrust (decelerates a sinking descent harder than hover).
        # Matches src/control/hocbf.cpp and src/rt/safety_filter.c exactly.
        if T_lb > p.T_max:
            return p.T_max

        # Analytical QP projection onto feasible set
        return _clamp(T_nom, max(p.T_min, T_lb), p.T_max)

    def filter_vla_command(self, pz: float, vz: float, roll: float, pitch: float,
                           vx_nom: float, vy_nom: float, vz_nom: float,
                           v_max: float = 5.0) -> SafeCommand:
        p = self.p
        T_hover = p.mass * p.g
        T_nom = T_hover + p.mass * vz_nom * 2.0          # simple P-gain

        # infeasibility check mirrors filter_thrust
        Lf2h_free = -p.g
        LgLfh = math.cos(roll) * math.cos(pitch) / p.mass
        LgLfh_safe = max(LgLfh, 0.05)
        rhs = -Lf2h_free - p.alpha1 * vz - p.alpha2 * pz
        T_lb_raw = rhs / LgLfh_safe
        T_lb = T_lb_raw * p.conservatism if T_lb_raw > 0.0 else T_lb_raw
        infeasible = (T_lb > p.T_max)

        T_safe = self.filter_thrust(pz, vz, roll, pitch, T_nom)

        h_mag = math.hypot(vx_nom, vy_nom)
        vx_safe, vy_safe = vx_nom, vy_nom
        if h_mag > v_max:
            vx_safe = vx_nom * v_max / h_mag
            vy_safe = vy_nom * v_max / h_mag

        filtered = (T_safe != T_nom) or (h_mag > v_max)
        return SafeCommand(vx_safe, vy_safe, T_safe, filtered, infeasible)
