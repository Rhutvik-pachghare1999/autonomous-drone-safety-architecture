# slurm/isaac6_probe.py — compat probe for Isaac Sim 6.0.1 on Sol.
# (a) RTX render proof: static camera, ground plane, dome light -> save one
#     rendered frame as raw PPM + npy under experiments/results/ (no PIL/matplotlib
#     dependency; pure-python PPM writer) and print pixel stats.
# (b) environment sanity: python/torch versions, GPU name, vulkan up.
# Runs via 61_isaac6_compat.sbatch.  Temporary-but-committable diagnostic;
# the realcam script reuses the same mount/settings helpers.

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim"))

import crazyflie_env as cfenv  # creates SimulationApp (headless) at import


def patch_rtx_settings():
    """Headless kit session lacks the rtx data-window settings that
    replicator's _resize_data_for_overscan dereferences (NoneType crash in
    5.1 headless).  Short-circuit the branch + give sane NDC defaults."""
    from carb import settings
    s = settings.get_settings()
    s.set("/rtx/dataWindow/fitOutputToDataWindow", True)
    for k, v in (("/rtx/dataWindowNDC/0", 0.0), ("/rtx/dataWindowNDC/1", 0.0),
                 ("/rtx/dataWindowNDC/2", 1.0), ("/rtx/dataWindowNDC/3", 1.0)):
        if s.get(k) is None:
            s.set(k, v)


def save_ppm(path, rgb_uint8):
    h, w = rgb_uint8.shape[:2]
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(rgb_uint8.tobytes())


def main():
    import torch
    print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    patch_rtx_settings()

    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    import omni.usd
    from pxr import UsdLux

    world = World(physics_dt=cfenv.PHYS_DT, rendering_dt=cfenv.PHYS_DT)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    d = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    d.CreateIntensityAttr(1500.0)

    cam = Camera("/World/isaac6_cam", resolution=(224, 224))
    cam.set_local_pose(translation=np.array([2.0, 2.0, 2.0]),
                       orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                       camera_axes="usd")
    cam.initialize()
    for _ in range(5):
        world.step(render=True)
    rgba = np.asarray(cam.get_rgba())
    if rgba.size == 0:
        print("ISAAC6_PROBE: EMPTY_FRAME", flush=True)
        cfenv._app.close()
        os._exit(1)
    rgb = rgba[..., :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    outdir = os.path.join(REPO, "experiments", "results")
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "isaac6_probe.npy"), rgb)
    save_ppm(os.path.join(outdir, "isaac6_probe.ppm"), rgb)
    print("ISAAC6_PROBE: OK shape=%s dtype=%s mean=%.2f min=%d max=%d"
          % (tuple(rgb.shape), str(rgb.dtype), float(rgb.mean()),
             int(rgb.min()), int(rgb.max())), flush=True)
    cfenv._app.close()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        try:
            cfenv._app.close()
        finally:
            os._exit(1)
