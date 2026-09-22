# sim/hocbf_crazyflie_ab.py
# Phase 3 (crazyflie-vla-sim plan): HOCBF safety filter re-parameterized for the
# REAL 27 g Crazyflie 2.X (T_max 0.60 N, mass 0.027 kg) and run LIVE in
# Isaac Sim 5.1 — filter OFF vs ON on the SAME adversarial command sequence.
#
# Command provenance note (honesty): in Phase 2 the SmolVLM2-2.2B-4bit pilot
# refused the adversarial "dive" prompt and emitted (0,0,0). To exercise the
# safety filter against its design threat anyway, the adversarial segment here
# is a SCRIPTED worst-case hallucination command (dive at -3.0 m/s) — exactly
# the class of output the filter exists to neutralize. Scripted commands are
# identical across both arms, making the A/B perfectly controlled.
#
# Filter: src/control/hocbf.cpp (C++ pybind11 4bit module built for the Isaac
# 3.11 interpreter: sim/hocbf.cpython-311-x86_64-linux-gnu.so).
#
# Run:  ~/.local/share/ov/pkg/isaac_sim-5.1.0/python.sh sim/hocbf_crazyflie_ab.py

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sim/

import numpy as np
import crazyflie_env as cfenv                      # creates SimulationApp
from vla_crazyflie_flight import CrazyflieController, quat_to_roll_pitch, DT

import hocbf                                       # C++ module (pybind11)

RESULTS = os.path.join(REPO, "experiments", "results")
OUT     = os.path.join(RESULTS, "crazyflie_vla_ab.json")

# Worst-case adversarial VLA output (scripted; see header note).
ADV_SEQUENCES = {
    "dive_full":   [(1.0, (0.0, 0.0, 0.0)), (4.0, (0.0, 0.0, -3.0))],
    "dive_medium": [(1.0, (0.0, 0.0, 0.0)), (4.0, (0.0, 0.0, -1.2))],
}

T_HOLD = 1.0  # first segment duration (s)


def make_filter():
    p = hocbf.HOCBFParams()
    p.mass = cfenv.MASS       # 0.027 kg — REAL Crazyflie mass
    p.g = cfenv.G
    p.T_max = cfenv.T_MAX     # 0.60 N — 4x0.15 N motors
    p.T_min = 0.0
    p.alpha1 = 2.0
    p.alpha2 = 1.0
    p.conservatism = 1.08
    return hocbf.HOCBF(p)


def run_arm(name: str, seq, use_filter: bool) -> dict:
    filt = make_filter() if use_filter else None
    env = cfenv.CrazyflieEnv(start=(0.0, 0.0, 1.0))
    env.setup()
    ctl = CrazyflieController(env)
    env.reset_pose(1.0)

    n_steps = int(sum(d for d, _ in seq) / DT)
    edges = np.cumsum([d for d, _ in seq])
    min_z, crashed, n_filt = 1.0, False, 0
    trace = []
    print(f"--- ARM {name}: filter={'ON' if use_filter else 'OFF'} ---", flush=True)
    for i in range(n_steps):
        t = i * DT
        seg = int(np.searchsorted(edges, t, side="right"))
        vx_cmd, vy_cmd, vz_cmd = seq[min(seg, len(seq) - 1)][1]

        pos, quat, vel, _ = env.state()
        z, vz = float(pos[2]), float(vel[2])
        roll, pitch = quat_to_roll_pitch(quat)

        if use_filter:
            safe = filt.filter_vla_command(z, vz, roll, pitch,
                                           vx_cmd, vy_cmd, vz_cmd, 5.0)
            n_filt += 1 if safe.was_filtered else 0
            # filtered collective thrust + level attitude (safety mode)
            f1, f2, f3, f4 = ctl.mix(safe.T, 0.0, 0.0)
            ctl.apply_rotors(f1, f2, f3, f4)
            T_used = safe.T
        else:
            T_used = ctl.control(vx_cmd, vy_cmd, vz_cmd)   # no filter

        env.world.step(render=False)
        min_z = min(min_z, z)
        if i % 5 == 0:  # 0.1 s resolution — committed to JSON, drives the plot
            trace.append([round(t, 2), round(z, 4), round(vz, 4), round(T_used, 4)])
        if i % 25 == 0:
            print(f"  t={t:5.2f}s z={z:+6.3f} vz={vz:+5.2f} T={T_used:.3f}", flush=True)
        if z <= 0.03:
            crashed = True
            trace.append([round(t + DT, 3), round(z, 4), round(vz, 4), round(T_used, 4)])
            print(f"  *** CRASH at t={t+DT:.2f}s (z={z:.3f}) ***", flush=True)
            break

    crash_t = round((i + 1) * DT, 3) if crashed else None
    result = {
        "arm": name, "filter_on": use_filter,
        "min_z_m": round(min_z, 4), "crashed": crashed,
        "crash_t_s": crash_t,
        "filtered_steps_frac": round(n_filt / (i + 1), 3),
        "trace": trace,
    }
    del env
    return result


def run_phase3() -> None:
    print("=" * 78)
    print("PHASE 3 — HOCBF(mass=0.027, T_max=0.60) ON REAL CRAZYFLIE: A/B OFF vs ON")
    print("(scripted worst-case adversarial dive; 2.2B pilot refused it in Phase 2)")
    print("=" * 78, flush=True)

    out = {"phase": 3, "hocbf_params": {"mass": 0.027, "T_max": 0.60,
                                        "alpha1": 2.0, "alpha2": 1.0,
                                        "conservatism": 1.08},
           "scenarios": {}}
    for sname, seq in ADV_SEQUENCES.items():
        off = run_arm(f"{sname}/OFF", seq, use_filter=False)
        on  = run_arm(f"{sname}/ON",  seq, use_filter=True)
        out["scenarios"][sname] = {"OFF": off, "ON": on}
        print(f"[{sname}] OFF: min_z={off['min_z_m']} crashed={off['crashed']} | "
              f"ON: min_z={on['min_z_m']} crashed={on['crashed']} "
              f"filtered={on['filtered_steps_frac']:.0%}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUT}", flush=True)
    print("=" * 78, flush=True)
    cfenv._app.close()


if __name__ == "__main__":
    run_phase3()
