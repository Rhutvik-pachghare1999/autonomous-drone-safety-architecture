# slurm/dbg_render.py — TEMPORARY diagnostics: minimal RTX camera render tests.
# NOT part of the reproducible pipeline (runs only via dbg_render.sbatch).
#
# Job 63824988 (sg048 MIG slice): Vulkan dead  -> expected.
# Job 63826532 (sg028 full A100): Vulkan UP ("Graphics API: Vulkan") yet the
# cf2-mounted Camera returns all-None frames via get_rgb/get_rgba
# (replicator _resize_data_for_overscan NoneType).  20x USD
# "metrics:UnitsAdjust ... not in the local LayerStack" coding errors fired
# exactly at the first render.
#
# Isolate the trigger with 4 variants, FIRST camera product that produces a
# valid frame wins.  Prints one PASS/FAIL line per variant (grep-friendly).

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sim"))

# crazyflie_env creates the SimulationApp (headless) at import time.
import crazyflie_env as cfenv  # noqa: E402  (config identical to prod runs)


def _try_grab(cam, world, tag):
    """Warmup + render + every readback API; print PASS line or the error."""
    from omni import replicator as rep
    try:
        for _ in range(3):
            world.step(render=True)
        info = ["not attempted", "not attempted"]
        rgb = cam.get_rgb()
        info[0] = f"shape={None if rgb is None else rgb.shape}"
        rgba = cam.get_rgba()
        info[1] = f"shape={None if rgba is None else rgba.shape}"
        fr = cam.get_current_frame()
        keys = list(fr.keys()) if isinstance(fr, dict) else type(fr).__name__
        arr = np.asarray(rgb if rgb is not None and np.size(rgb) else rgba)
        finite = bool(np.isfinite(arr.astype(np.float32)).all()) if arr.size else False
        mean = float(arr[..., :3].mean()) if arr.size else float("nan")
        print(f"PASS {tag}: get_rgb {info[0]} | get_rgba {info[1]} | "
              f"frame_keys={keys} | finite={finite} mean_rgb={mean:.2f}",
              flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"FAIL {tag}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=3)
        return False


def _fresh_world():
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import create_new_stage
    create_new_stage()
    w = World(physics_dt=cfenv.PHYS_DT, rendering_dt=cfenv.PHYS_DT)
    w.scene.add_default_ground_plane()
    return w


def _lights(stage):
    from pxr import UsdLux
    d = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    d.CreateIntensityAttr(1500.0)
    s = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    s.CreateIntensityAttr(2500.0)


def variant_A():
    """Static Camera on /World, ground plane, dome light: renderer sanity."""
    import omni.usd
    from isaacsim.sensors.camera import Camera
    world = _fresh_world()
    _lights(omni.usd.get_context().get_stage())
    cam = Camera("/World/dbg_cam", resolution=(224, 224))
    cam.set_local_pose(translation=np.array([2.0, 2.0, 2.0]),
                       orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                       camera_axes="usd")
    cam.initialize()
    return _try_grab(cam, world, "A-static")


def variant_B():
    """Camera as CHILD of a dynamic rigid cube (same pattern as cf2 mount)."""
    import omni.usd
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.api.objects import DynamicCuboid
    world = _fresh_world()
    _lights(omni.usd.get_context().get_stage())
    DynamicCuboid("/World/box", position=np.array([0.0, 0.0, 1.0]),
                  scale=np.array([0.09, 0.09, 0.03]), mass=0.027)
    cam = Camera("/World/box/vla_cam", resolution=(224, 224))
    q = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../sim")
        from vla_realcam_flight import _camera_mount_quat  # reuse prod mount
        q = _camera_mount_quat()
    except Exception:
        q = np.array([0.5792, 0.4056, -0.4056, -0.5792])
    cam.set_local_pose(translation=np.array([0.05, 0.0, 0.0]), orientation=q)
    cam.initialize()
    return _try_grab(cam, world, "B-onbox")


def variant_C():
    """EXACT prod path: CrazyflieEnv + reference USD + our attach/grab."""
    from vla_realcam_flight import attach_camera, grab_frame  # prod functions
    env = cfenv.CrazyflieEnv()
    env.setup()
    cam = attach_camera(env)
    try:
        rgb = grab_frame(env, cam)
        a = np.asarray(rgb)
        print(f"PASS C-cf2prod: shape={a.shape} dtype={a.dtype} "
              f"mean={a[..., :3].mean():.2f}", flush=True)
        return True
    except Exception as e:
        print(f"FAIL C-cf2prod: {type(e).__name__}: {e}", flush=True)
        return False


def variant_D():
    """Crazyflie present, but camera at ABSOLUTE /World prim (not a child)."""
    import omni.usd
    from isaacsim.sensors.camera import Camera
    env = cfenv.CrazyflieEnv()
    env.setup()
    _lights(omni.usd.get_context().get_stage())
    cam = Camera("/World/dbg_cam_world", resolution=(224, 224))
    cam.set_world_pose(position=np.array([1.5, 1.5, 1.5]),
                       orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                       camera_axes="usd")
    cam.initialize()
    return _try_grab(cam, env.world, "D-cam-world")


def main():
    results = {}
    for name, fn in (("A", variant_A), ("B", variant_B),
                     ("C", variant_C), ("D", variant_D)):
        try:
            results[name] = fn()
        except Exception as e:
            print(f"FAIL {name}: outer {type(e).__name__}: {e}", flush=True)
            results[name] = False
    print("DBG_RENDER_RESULT: " + " ".join(
        f"{k}={'ok' if v else 'FAIL'}" for k, v in results.items()), flush=True)
    cfenv._app.close()


if __name__ == "__main__":
    main()
