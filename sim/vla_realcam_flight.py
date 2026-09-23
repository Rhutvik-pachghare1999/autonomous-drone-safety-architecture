# sim/vla_realcam_flight.py
# Phase 5 (sol-wcet-vla TASK 2): REAL RENDERED CAMERA -> SmolVLM2 (GPU) -> HOCBF
# guard -> 50 Hz P controller -> motor mixer, on the REAL 27 g Crazyflie 2.X in
# Isaac Sim 5.1 (headless, GPU PhysX + RTX render).
#
# Delta vs Phase 2 (vla_crazyflie_flight.py):
#   * CAMERA IS REAL: every VLA query consumes an RTX-rendered RGB frame from a
#     camera prim rigid-mounted on the Crazyflie body (forward +X, 20 deg
#     down-tilt). Phase 2's make_cam_image() painted sky/ground rows from
#     (z, vz) — a documented synthetic stand-in. This script does not use it.
#   * GUARD IS ON: the VLA velocity command passes through the HOCBF filter
#     (sim/hocbf_py.py — verified pure-Python port of the C++ filter; the Isaac
#     container has no C++ compiler) on EVERY 50 Hz control step. The guard
#     clips |v_xy| to v_max and projects the vz->thrust map onto the CBF-safe
#     thrust interval. Phase 2 trusted the raw command; here the honest claim
#     is "the guard bounds unsafe VLA velocities while letting safe ones
#     through" with per-step filtered/infeasible provenance.
#   * GPU INFERENCE: SmolVLM2-2.2B runs on the datacenter GPU alongside PhysX
#     (no ~2-minute laptop-CPU latency of Phase 2). Actual dtype load path
#     (4-bit NF4 vs bf16 fallback) is recorded in the summary.
#
# HONEST SCOPE:
#   * "Real camera" = Isaac RTX-rendered frame readback, NOT a physical camera.
#   * parse_source may be 'prompt_echo:*' when model text is uninformative
#     (VLABridge fallback). That echo still yields the worst-case adversarial
#     velocity the guard must bound, so the safety claim holds either way.
#   * All numbers below come from this script's own JSONL output, nothing else.
#
# Run (Sol Slurm):  /isaac-sim/python.sh sim/vla_realcam_flight.py 6
# Requires: transformers/bitsandbytes on $PYTHONPATH and the model weights in
# $HF_HOME — both staged by slurm/30_vla_prep.sbatch (compute-node job).

import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sim/
sys.path.insert(0, os.path.join(REPO, "src", "perception"))       # for lazy vla_bridge

# crazyflie_env creates the headless SimulationApp at import time (must be first)
import crazyflie_env as cfenv

from vla_crazyflie_flight import (CrazyflieController, quat_to_roll_pitch,
                                  DT, MISSION)
import hocbf_py as hocbf   # pure-Python HOCBF port (container-safe; verified
                           # vs C++ in tests/test_hocbf_py_matches_cpp.py)

RESULTS = os.path.join(REPO, "experiments", "results")
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TAG = os.environ.get("RUN_TAG", "vla_realcam")
JSONL = os.path.join(RESULTS, f"{TAG}.jsonl")
SUMMARY = os.path.join(RESULTS, f"{TAG}_summary.json")

# ── VLA / camera configuration ───────────────────────────────────────────────
VLA_HZ = 2.0                        # GPU inference -> 2 Hz outer loop (Phase 2 CPU
                                    # fallback ran 0.5 Hz for wall-clock reasons)
VLA_PERIOD = int(round(1.0 / (VLA_HZ * DT)))   # physics steps between VLA queries
CAM_RES = (224, 224)                # px, matches Phase-2 synthetic input size
CAM_TILT_DEG = 20.0                 # downward tilt from forward +X body axis
CAM_FWD_OFF = 0.05                  # m, camera nudged ahead of CF center
SAVE_FRAME_EPS = (0,)               # episodes whose per-segment q0 frames get saved

# ── HOCBF guard parameters (identical to crazyflie_vla_ab500.json) ───────────
V_MAX_CMD = 5.0                     # m/s horizontal clamp inside filter


def make_filter():
    p = hocbf.HOCBFParams()
    p.mass = cfenv.MASS
    p.g = cfenv.G
    p.T_max = cfenv.T_MAX   # 0.60 N (4 x ~0.15 N motors)
    p.T_min = 0.0
    p.alpha1 = 2.0
    p.alpha2 = 1.0
    p.conservatism = 1.08
    return hocbf.HOCBF(p)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Standard 3x3 rotation matrix -> quaternion (w, x, y, z)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = 0.5 / math.sqrt(1.0 + tr)
        return np.array([0.25 / s, (m21 - m12) * s, (m02 - m20) * s,
                         (m10 - m01) * s], dtype=np.float64)
    i = int(np.argmax([m00, m11, m22]))
    if i == 0:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        return np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s,
                         (m02 + m20) / s], dtype=np.float64)
    if i == 1:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        return np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s,
                         (m12 + m21) / s], dtype=np.float64)
    s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
    return np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s,
                     0.25 * s], dtype=np.float64)


def _camera_mount_quat() -> np.ndarray:
    """Camera looks along -Z with +Y up (USD convention). Build R: body-frame
    mount with optical axis -> +X (forward), up -> +Z, then tilt DOWN by
    CAM_TILT_DEG about the camera's image-right axis (intrinsic local X)."""
    col1 = np.array([0.0, -1.0, 0.0])   # R e_x  (image right -> body -Y)
    col2 = np.array([0.0, 0.0, 1.0])    # R e_y  (image up    -> body +Z)
    col3 = np.array([-1.0, 0.0, 0.0])   # R e_z  (optical back -> body -X)
    R = np.stack([col1, col2, col3], axis=1)
    th = math.radians(-CAM_TILT_DEG)    # negative pitch = nose-down view
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(th), -math.sin(th)],
                   [0.0, math.sin(th), math.cos(th)]])
    return _rotmat_to_quat_wxyz(R @ Rx)


def attach_camera(env) -> object:
    """Rigid-mount an RTX camera under the Crazyflie prim and light the scene.

    The prim is a CHILD of the rigid body, so it follows every physics step
    and reset_pose() teleport automatically.
    """
    from isaacsim.sensors.camera import Camera
    import omni.usd
    from pxr import UsdLux

    stage = omni.usd.get_context().get_stage()
    # lighting: headless container scene has no sky; make frames meaningful
    if not stage.GetPrimAtPath("/World/DomeLight"):
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1500.0)
        dome.CreateColorAttr((0.55, 0.7, 1.0))          # sky-blue ambient
    if not stage.GetPrimAtPath("/World/SunLight"):
        sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        sun.CreateIntensityAttr(2500.0)
        sun.CreateAngleAttr(2.0)

    cam_path = f"{env.prim_path}/vla_cam"
    # NOTE: no `frequency=` — the sensor raises unless the requested frequency
    # divides the app's rendering frequency (1/60 here, not 50 Hz physics).
    # We capture on demand via world.render()+get_rgba(), so the sensor tick
    # frequency is irrelevant.
    cam = Camera(prim_path=cam_path, resolution=CAM_RES)
    q = _camera_mount_quat()
    t = np.array([CAM_FWD_OFF, 0.0, 0.0], dtype=np.float64)
    mounted = False
    errs = []
    for attempt in (
        lambda: cam.set_local_pose(translation=t, orientation=q),
        lambda: cam.set_local_pose(t[None], q[None]),        # batched fallback
    ):
        try:
            attempt()
            mounted = True
            break
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}")
    if not mounted:
        raise RuntimeError(f"camera mount failed: {errs}")
    cam.initialize()
    # a couple of rendered warmup steps so the render product is live
    for _ in range(3):
        env.world.step(render=True)
    pos, ori = cam.get_world_pose()
    print(f"[cam] prim={cam_path} res={CAM_RES} tilt_deg={CAM_TILT_DEG} "
          f"world_pos={np.round(pos, 3).tolist()}", flush=True)
    return cam


def grab_frame(env, cam) -> np.ndarray:
    """Render the CURRENT state (no extra physics step) and return uint8 RGB."""
    env.world.render()
    rgb = None
    try:
        rgb = cam.get_rgb()
    except Exception:
        pass
    if rgb is None or np.asarray(rgb).size == 0:
        rgba = np.asarray(cam.get_rgba())
        rgb = rgba[..., :3]
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return np.ascontiguousarray(rgb)


def control_guarded(ctl: CrazyflieController, vx_c: float, vy_c: float,
                    T_safe: float, wind_w=None) -> None:
    """Identical attitude/velocity P loop to CrazyflieController.control(),
    except collective thrust T comes from the HOCBF guard (not from vz)."""
    pos, quat, vel, omega = ctl.env.state()
    roll, pitch = quat_to_roll_pitch(quat)
    vx, vy = vel[0], vel[1]
    wx, wy = omega[0], omega[1]

    roll_ref = ctl.s2x * max(-0.25, min(0.25, ctl.kxy * (vy_c - vy)))
    pitch_ref = ctl.s2y * max(-0.25, min(0.25, ctl.kxy * (vx_c - vx)))
    tx = ctl.s1x * ctl.kp_at * (roll_ref - roll) - ctl.kd_at * wx
    ty = ctl.s1y * ctl.kp_at * (pitch_ref - pitch) - ctl.kd_at * wy

    f1, f2, f3, f4 = ctl.mix(T_safe, tx, ty)
    ctl.apply_rotors(f1, f2, f3, f4, wind_w=wind_w)


def run() -> None:
    from vla_bridge import VLABridge, MODEL_ID      # lazy: needs transformers
    print("=" * 78)
    print("PHASE 5 — REAL CAMERA -> SmolVLM2 (GPU) -> HOCBF GUARD -> CRAZYFLIE")
    print("=" * 78)
    print("Frames: Isaac RTX render of a camera prim mounted on the drone body.\n"
          "Guard:  sim/hocbf_py.py (verified C++ port) at every 50 Hz step.\n"
          "All reported numbers come from the JSONL this run writes.", flush=True)

    device = os.environ.get("VLA_DEVICE", "cuda")

    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[env] gpu={gpu_name}  torch={torch.__version__}  device={device}",
          flush=True)

    env = cfenv.CrazyflieEnv(start=(0.0, 0.0, 1.0))
    env.setup()
    cam = attach_camera(env)
    ctl = CrazyflieController(env)
    ctl.calibrate()

    t0 = time.time()
    vla = VLABridge(device=device)
    print(f"[vla] model={MODEL_ID} quant={getattr(vla, 'quant', '?')} "
          f"load_s={time.time() - t0:.1f}", flush=True)

    from PIL import Image
    filt = make_filter()
    os.makedirs(RESULTS, exist_ok=True)
    jf = open(JSONL, "w")
    header = {
        "type": "header", "script": os.path.basename(__file__),
        "model": MODEL_ID, "quant": getattr(vla, "quant", None),
        "device": device, "gpu": gpu_name,
        "transformers": __import__("transformers").__version__,
        "camera": {"kind": "isaac_rtx_render", "res": list(CAM_RES),
                   "mount": {"fwd_off_m": CAM_FWD_OFF, "tilt_deg": CAM_TILT_DEG},
                   "note": "simulated render, NOT a physical camera"},
        "vla_hz": VLA_HZ, "control_hz": int(1.0 / DT),
        "guard_impl": "hocbf_py",
        "guard_params": filt.p.__dict__,
        "mission": [{"dur_s": d, "prompt": p} for d, p in MISSION],
        "n_episodes": N_EP,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    jf.write(json.dumps(header) + "\n")
    print("[header] " + json.dumps({k: v for k, v in header.items()
                                   if k not in ("guard_params", "mission")}),
          flush=True)

    seg_edges = np.cumsum([d for d, _ in MISSION])
    n_steps = int(sum(d for d, _ in MISSION) / DT)
    episodes = []
    lat_all: list[float] = []
    parse_all: dict[str, int] = {}

    for ep in range(N_EP):
        start_z = 0.9 + 0.15 * ep     # small honest per-episode altitude variation
        env.reset_pose(start_z)
        vx_cmd = vy_cmd = vz_cmd = 0.0
        min_z = float(start_z)
        crashed = False
        n_q = 0
        n_steps_filtered = 0
        n_steps_infeasible = 0
        t_ep0 = time.time()

        for i in range(n_steps):
            t = i * DT
            pos, quat, vel, _ = env.state()
            x, y = float(pos[0]), float(pos[1])
            z, vz = float(pos[2]), float(vel[2])
            roll, pitch = quat_to_roll_pitch(quat)
            min_z = min(min_z, z)

            if i % VLA_PERIOD == 0:
                seg_idx = int(np.searchsorted(seg_edges, t, side="right"))
                seg_idx = min(seg_idx, len(MISSION) - 1)
                seg_label = ("hover", "forward", "dive")[seg_idx]
                prompt = MISSION[seg_idx][1]

                frame = grab_frame(env, cam)
                if ep in SAVE_FRAME_EPS and n_q % max(1, int(6 * VLA_HZ)) == 0 \
                        and seg_idx < 3:
                    # one frame per segment of episode 0
                    fname = os.path.join(RESULTS, f"{TAG}_ep{ep}_{seg_label}.png")
                    if not os.path.exists(fname):
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        plt.imsave(fname, frame)
                        print(f"[cam] saved {fname}", flush=True)

                img = Image.fromarray(frame)
                cmd = vla.query(prompt, image=img)
                vx_cmd, vy_cmd, vz_cmd = cmd.vx, cmd.vy, cmd.vz
                n_q += 1
                lat_all.append(float(cmd.latency_ms))
                parse_all[cmd.parse_source] = parse_all.get(cmd.parse_source, 0) + 1

                safe_q = filt.filter_vla_command(z, vz, roll, pitch,
                                                 vx_cmd, vy_cmd, vz_cmd,
                                                 V_MAX_CMD)
                line = {
                    "ep": ep, "seg": seg_label, "t": round(float(t), 3),
                    "z": round(z, 4), "vz": round(vz, 4),
                    "x": round(x, 4), "y": round(y, 4),
                    "prompt": prompt,
                    "raw_out": cmd.text_output,
                    "vx": float(cmd.vx), "vy": float(cmd.vy), "vz": float(cmd.vz),
                    "parse_source": cmd.parse_source,
                    "confidence": round(float(cmd.confidence), 3),
                    "latency_ms": round(float(cmd.latency_ms), 1),
                    "guard_at_query": {
                        "T_safe": round(float(safe_q.T), 5),
                        "was_filtered": bool(safe_q.was_filtered),
                        "was_infeasible": bool(safe_q.was_infeasible),
                        "vx_applied": round(float(safe_q.vx), 4),
                        "vy_applied": round(float(safe_q.vy), 4),
                    },
                }
                jf.write(json.dumps(line) + "\n")
                jf.flush()
                print(f"ep{ep} t={t:5.2f}s [{seg_label:7s}] z={z:+5.2f} "
                      f"vz={vz:+5.2f} | VLA[{cmd.parse_source}] "
                      f"'{cmd.text_output[:48]}' -> [{cmd.vx:+.2f},{cmd.vy:+.2f},"
                      f"{cmd.vz:+.2f}] guard(f={int(safe_q.was_filtered)},"
                      f"i={int(safe_q.was_infeasible)}) {cmd.latency_ms/1000:4.1f}s",
                      flush=True)

            # 50 Hz inner loop: latest VLA cmd ALWAYS through the guard
            safe = filt.filter_vla_command(z, vz, roll, pitch,
                                           vx_cmd, vy_cmd, vz_cmd, V_MAX_CMD)
            n_steps_filtered += 1 if safe.was_filtered else 0
            n_steps_infeasible += 1 if safe.was_infeasible else 0
            control_guarded(ctl, safe.vx, safe.vy, safe.T)
            env.world.step(render=False)   # force buffer owned by ctl, Phase-2 NOTE

            if z <= 0.03:
                print(f"\n*** GROUND CONTACT ep{ep} t={t:.2f}s z={z:.3f} "
                      f"vz={vz:+.2f} ***", flush=True)
                crashed = True
                break

        ep_row = {
            "ep": ep, "start_z": round(start_z, 3), "min_z_m": round(min_z, 4),
            "crashed": crashed, "n_vla_queries": n_q,
            "n_ctrl_steps": i + 1,
            "guard_filtered_frac": round(n_steps_filtered / (i + 1), 4),
            "guard_infeasible_steps": n_steps_infeasible,
            "wall_s": round(time.time() - t_ep0, 1),
        }
        episodes.append(ep_row)
        jf.write(json.dumps({"type": "episode", **ep_row}) + "\n")
        jf.flush()
        print(f"[episode {ep}] {ep_row}", flush=True)

    lat = np.asarray(lat_all) if lat_all else np.asarray([0.0])
    summary = {
        "task": "sol-wcet-vla TASK 2 (real-camera VLA + HOCBF guard)",
        "model": MODEL_ID, "quant": getattr(vla, "quant", None),
        "device": device, "gpu": gpu_name,
        "camera": header["camera"], "vla_hz": VLA_HZ,
        "guard_impl": "hocbf_py", "guard_params": filt.p.__dict__,
        "n_episodes": N_EP,
        "n_vla_queries": int(len(lat_all)),
        "parse_source_distribution": parse_all,
        "parse_rate_model_structured": round(
            parse_all.get("model_structured", 0) / max(1, len(lat_all)), 4),
        "latency_ms": {
            "mean": round(float(lat.mean()), 1),
            "p50": round(float(np.percentile(lat, 50)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1),
            "max": round(float(lat.max()), 1),
        },
        "episodes": episodes,
        "n_crashes": sum(1 for e in episodes if e["crashed"]),
        "guard_filtered_frac_overall": round(
            float(np.mean([e["guard_filtered_frac"] for e in episodes])), 4)
        if episodes else None,
        "guard_infeasible_steps_total": sum(e["guard_infeasible_steps"]
                                            for e in episodes),
        "wall_s_total": round(time.time() - t0, 1),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    jf.write(json.dumps({"type": "summary", **summary}) + "\n")
    jf.close()

    print("\n" + "=" * 78)
    print("PHASE 5 SUMMARY")
    print("=" * 78)
    for k, v in summary.items():
        if k != "episodes":
            print(f"  {k}: {v}")
    sys.stdout.flush()
    cfenv._app.close()


if __name__ == "__main__":
    # kit python swallows script exceptions and still exits 0 — convert any
    # failure into a REAL nonzero exit so Slurm marks the job FAILED.
    try:
        run()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        print("REALCAM_FATAL: run aborted", flush=True)
        try:
            cfenv._app.close()
        except Exception:
            pass
        os._exit(1)
