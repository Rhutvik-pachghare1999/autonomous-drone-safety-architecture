# sim/hocbf_crazyflie_iso.py
# SINGLE-VARIABLE (isolation) A/B: the ONLY difference between arms is whether
# the HOCBF CBF projection acts on the controller's nominal thrust.
#
# Why this exists (reviewer-grade attribution): the earlier A/B scripts
# (sim/hocbf_crazyflie_ab.py, sim/hocbf_crazyflie_ab100.py — the runs behind
# experiments/results/crazyflie_vla_ab*.json) changed TWO things between arms:
#   OFF arm: CrazyflieController.control() — velocity P-loop + attitude PD.
#   ON arm:  filt.filter_vla_command() — whose internal nominal law
#            (T_hover + m*vz*2.0, gain 2.0) REPLACED the P-controller, AND the
#            roll/pitch moments were forced to zero (ctl.mix(safe.T, 0, 0)).
# So OFF-vs-ON there measured "controller swap + CBF projection" vs
# "controller alone" — the improvement cannot be attributed to the safety
# filter alone. Results of those runs are kept as history with that caveat;
# THIS script is the attribution-grade comparison.
#
# Design (paired, seeded, domain-randomized — identical randomization both
# arms, same ranges as ab100):
#   both arms per step:  T_nom, tx, ty = ctl.compute_wrench(0, 0, vz_cmd)
#   OFF arm:             apply (T_nom, tx, ty)
#   ON arm:              apply (filter_thrust(z, vz, roll, pitch, T_nom), tx, ty)
# Controller, gains, attitude correction, vehicle, seed, wind, delay — all
# identical; ONLY the CBF projection of the collective thrust differs.
#
# FURTHER HONEST SCOPE: simulation only (Isaac PhysX), NOT a real hardware
# drone. The adversarial dive is a SCRIPTED worst-case command (the 2.2B VLA
# pilot refused dive prompts in Phase 2) — this measures whether the HOCBF
# projection prevents ground impact from unsafe vz commands, everything else
# held constant.
#
# Run (Sol Slurm):  python.sh sim/hocbf_crazyflie_iso.py 500
# Run (laptop):     ~/.local/share/ov/pkg/isaac_sim-5.1.0/python.sh \
#                       sim/hocbf_crazyflie_iso.py 4

import json, os, sys
from collections import deque

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import crazyflie_env as cfenv                      # creates SimulationApp
from vla_crazyflie_flight import CrazyflieController, quat_to_roll_pitch, DT
try:
    import hocbf                                    # compiled C++ pybind11 module (preferred)
    _HOCBF_IMPL = "cpp"
except ImportError:
    import hocbf_py as hocbf                         # verified pure-Python port (identical
    _HOCBF_IMPL = "python"                           # to C++, 2450-case cross-check) — used on
                                                     # the Isaac runtime container (no compiler)
print(f"[hocbf] using {_HOCBF_IMPL} implementation", flush=True)

RESULTS = os.path.join(REPO, "experiments", "results")
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(os.environ.get("SEED", "42"))
OUT = os.path.join(RESULTS, f"crazyflie_vla_iso{N_EP}.json")

# Domain randomization ranges — IDENTICAL to hocbf_crazyflie_ab100.py so the
# isolation run is directly comparable with the old confounded headline.
R_START_Z = (0.8, 1.5)      # m
R_DIVE_VZ = (-3.5, -1.0)    # m/s commanded worst-case dive
R_ONSET   = (0.5, 1.5)      # s until the dive starts
R_VZ0     = (-1.0, 0.5)     # m/s initial vertical velocity
R_WIND    = (0.0, 0.06)     # N horizontal disturbance (0-23% of hover thrust)
R_DELAY   = (0, 10)         # command delay steps at 50 Hz (0-0.2 s)


def make_filter():
    p = hocbf.HOCBFParams()
    p.mass = cfenv.MASS; p.g = cfenv.G
    p.T_max = cfenv.T_MAX; p.T_min = 0.0
    p.alpha1 = 2.0; p.alpha2 = 1.0; p.conservatism = 1.08
    return hocbf.HOCBF(p)


def run_episode(env, ctl, ep, use_filter):
    """One randomized episode. Returns (survived, min_z, filtered_frac).

    OFF and ON differ in EXACTLY one operation per step: ON projects the
    controller's nominal collective thrust through filter_thrust (CBF safe-set
    clamp) before mixing; OFF uses it unprojected. Attitude moments (tx, ty)
    and everything else are byte-identical across arms.
    """
    filt = make_filter() if use_filter else None
    wind = ep["wind"]             # world-frame (Fx, Fy, 0), N
    delay = ep["delay_steps"]     # 0 -> no delay line
    cmdbuf = deque([0.0] * delay, maxlen=delay) if delay > 0 else None

    env.reset_pose(ep["start_z"])
    # randomized initial vertical velocity (velocity step at t=0)
    env.cf.set_velocities(
        np.array([[0.0, 0.0, ep["vz0"], 0.0, 0.0, 0.0]], dtype=np.float32))

    n_steps = int((ep["onset"] + 3.0) / DT)
    min_z, crashed, n_filt = ep["start_z"], False, 0
    for i in range(n_steps):
        t = i * DT
        vz_now = 0.0 if t < ep["onset"] else ep["dive_vz"]
        if cmdbuf is None:
            vz_cmd = vz_now
        else:
            cmdbuf.append(vz_now)
            vz_cmd = cmdbuf[0]                       # delayed adversarial cmd
        pos, quat, vel, _ = env.state()
        z, vz = float(pos[2]), float(vel[2])
        roll, pitch = quat_to_roll_pitch(quat)

        # --- shared nominal controller (identical in both arms) ---
        T_nom, tx, ty = ctl.compute_wrench(0.0, 0.0, vz_cmd)
        # --- THE SINGLE VARIABLE: CBF projection of T_nom ---
        if use_filter:
            T_apply = filt.filter_thrust(z, vz, roll, pitch, T_nom)
            n_filt += 1 if T_apply != T_nom else 0
        else:
            T_apply = T_nom
        # --- shared mixer + application (identical in both arms) ---
        f1, f2, f3, f4 = ctl.mix(T_apply, tx, ty)
        ctl.apply_rotors(f1, f2, f3, f4, wind_w=wind)

        env.world.step(render=False)
        min_z = min(min_z, z)
        if z <= 0.03:
            crashed = True
            break
    return (not crashed), round(min_z, 4), round(n_filt / (i + 1), 3)


def run():
    rng = np.random.default_rng(SEED)
    # pre-draw identical per-episode randomization for BOTH arms (paired A/B)
    episodes = []
    for _ in range(N_EP):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        mag = float(rng.uniform(*R_WIND))
        episodes.append({
            "start_z":      float(rng.uniform(*R_START_Z)),
            "dive_vz":      float(rng.uniform(*R_DIVE_VZ)),
            "onset":        float(rng.uniform(*R_ONSET)),
            "vz0":          float(rng.uniform(*R_VZ0)),
            "wind":         (mag * np.cos(theta), mag * np.sin(theta), 0.0),
            "wind_mag_N":   mag,
            "delay_steps":  int(rng.integers(R_DELAY[0], R_DELAY[1] + 1)),
        })

    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        import subprocess as _sp
        try:
            gpu = _sp.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True).strip().splitlines()[0]
        except Exception:
            gpu = "unknown"
    platform_tag = (f"host={os.uname().nodename} gpu={gpu} "
                    f"slurm_job={os.environ.get('SLURM_JOB_ID', '-')}")

    print("=" * 78, flush=True)
    print(f"CRAZYFLIE 2.X ISOLATION A/B — {N_EP} episodes/mode, paired, "
          f"seed={SEED}", flush=True)
    print("single variable: CBF thrust projection ON vs OFF; controller, gains,"
          " attitude, randomization IDENTICAL across arms", flush=True)
    print(f"randomization: start_z{R_START_Z}, dive_vz{R_DIVE_VZ}, onset{R_ONSET},"
          f" vz0{R_VZ0}, wind_N{R_WIND}, delay_steps{R_DELAY}", flush=True)
    print(f"{platform_tag} | SIMULATION ONLY (no mass randomization — env has"
          " no set_mass)", flush=True)
    print("=" * 78, flush=True)

    env = cfenv.CrazyflieEnv(start=(0.0, 0.0, 1.0))
    env.setup()
    ctl = CrazyflieController(env)

    res = {"OFF": {"survived": 0, "crashed": 0, "min_z": []},
           "ON":  {"survived": 0, "crashed": 0, "min_z": [], "filtered_frac": []}}

    for k, ep in enumerate(episodes):
        for arm, uf in (("OFF", False), ("ON", True)):
            surv, min_z, ff = run_episode(env, ctl, ep, uf)
            res[arm]["survived" if surv else "crashed"] += 1
            res[arm]["min_z"].append(min_z)
            if arm == "ON":
                res["ON"]["filtered_frac"].append(ff)
        if (k + 1) % max(1, N_EP // 40) == 0:
            print(f"  [{k+1:4d}/{N_EP}] OFF surv={res['OFF']['survived']:4d} "
                  f"ON surv={res['ON']['survived']:4d}", flush=True)

    summary = {
        "n_episodes_per_mode": N_EP, "seed": SEED,
        "ablation": {"design": "single-variable: ONLY the CBF thrust projection "
                               "differs between arms",
                     "shared": "CrazyflieController.compute_wrench(0,0,vz_cmd) "
                               "-> (T,tx,ty); OFF applies T directly, ON applies "
                               "filter_thrust(z,vz,roll,pitch,T); same mixer, "
                               "same attitude moments, same randomization (paired)",
                     "differs_from": "hocbf_crazyflie_ab100.py (that run also "
                               "swapped the controller nominal law and zeroed "
                               "attitude moments in the ON arm — confounded "
                               "attribution; kept as history only)"},
        "simulation_only": True, "hardware_validated": False,
        "vehicle": "Crazyflie 2.X (crazyflie.usd, 27 g rigid body)",
        "engine": "Isaac Sim GPU PhysX (50 Hz)",
        "platform": platform_tag,
        "adversarial_command": "scripted worst-case dive (VLA refused dive prompts in Phase 2)",
        "randomization": {
            "start_z": list(R_START_Z), "dive_vz": list(R_DIVE_VZ),
            "onset": list(R_ONSET), "vz0": list(R_VZ0),
            "wind_mag_N": list(R_WIND), "delay_steps": list(R_DELAY),
            "mass": "NOT randomized (env has no set_mass)"},
        "filter_OFF": {"survival_rate": round(res["OFF"]["survived"] / N_EP, 3),
                       "survived": res["OFF"]["survived"],
                       "crashed": res["OFF"]["crashed"],
                       "mean_min_z_m": round(float(np.mean(res["OFF"]["min_z"])), 4),
                       "min_z_samples": res["OFF"]["min_z"]},
        "filter_ON":  {"survival_rate": round(res["ON"]["survived"] / N_EP, 3),
                       "survived": res["ON"]["survived"],
                       "crashed": res["ON"]["crashed"],
                       "mean_min_z_m": round(float(np.mean(res["ON"]["min_z"])), 4),
                       "mean_filtered_frac": round(float(np.mean(res["ON"]["filtered_frac"])), 3),
                       "min_z_samples": res["ON"]["min_z"],
                       "filtered_frac_samples": res["ON"]["filtered_frac"]},
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n" + "=" * 78, flush=True)
    print(f"OFF survival: {summary['filter_OFF']['survival_rate']:.1%} "
          f"({res['OFF']['survived']}/{N_EP})", flush=True)
    print(f"ON  survival: {summary['filter_ON']['survival_rate']:.1%} "
          f"({res['ON']['survived']}/{N_EP})", flush=True)
    print(f"Wrote {OUT}", flush=True)
    print("=" * 78, flush=True)
    cfenv._app.close()


if __name__ == "__main__":
    run()
