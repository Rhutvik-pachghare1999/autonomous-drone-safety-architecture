# slurm/isaac6_probe2.py — RTX render proof #2 on Isaac Sim 6.0.1, using the
# custom camera experience file (slurm/isaac6_headless_render.kit), which
# pulls in omni.kit.viewport.rtx + omni.replicator.core + synchronous-render
# settings that the bare headless experience lacks (job 63842534: no crash,
# but rgb annotator stayed empty for 60 warm steps).
#
# Standalone (does NOT import crazyflie_env) so it can own SimulationApp.
# Writes experiments/results/isaac6_probe2.{ppm,npy} on success.

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIENCE = "/repo/slurm/isaac6_headless_render.kit"
PHYS_DT = 0.02

from isaacsim import SimulationApp
app = SimulationApp({"headless": True}, experience=EXPERIENCE)


def patch_rtx_settings():
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
    patch_rtx_settings()

    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    import omni.usd
    from pxr import UsdLux

    world = World(physics_dt=PHYS_DT, rendering_dt=PHYS_DT)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    d = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    d.CreateIntensityAttr(1500.0)

    cam = Camera("/World/isaac6_cam", resolution=(224, 224))
    cam.set_local_pose(translation=np.array([2.0, 2.0, 2.0]),
                       orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                       camera_axes="usd")
    cam.initialize()
    rgba = None
    for k in range(60):
        world.step(render=True)
        if k >= 4 and k % 2 == 0:
            r = np.asarray(cam.get_rgba())
            if r.ndim == 3 and r.size:
                rgba = r
                print(f"[probe2] first valid rgba at step {k}", flush=True)
                break
    if rgba is None:
        print("ISAAC6_PROBE2: EMPTY_FRAME", flush=True)
        app.close()
        os._exit(1)
    rgb = rgba[..., :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    outdir = os.path.join(REPO, "experiments", "results")
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "isaac6_probe2.npy"), rgb)
    save_ppm(os.path.join(outdir, "isaac6_probe2.ppm"), rgb)
    print("ISAAC6_PROBE2: OK shape=%s dtype=%s mean=%.2f min=%d max=%d"
          % (tuple(rgb.shape), str(rgb.dtype), float(rgb.mean()),
             int(rgb.min()), int(rgb.max())), flush=True)
    app.close()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        try:
            app.close()
        finally:
            os._exit(1)
