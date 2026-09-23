# slurm/isaac6_probe3.py — parametrized RTX render proof on Isaac Sim 6.0.1.
#   argv[1] = v1|v2|v3   (one SimulationApp per process)
#   v1: bare headless + manual replicator orchestrator capture
#       (create render_product + rgb annotator + orchestrator.step())
#   v2: bare headless + /exts/isaacsim.core.throttling/enable_async=false set
#       post-startup + Camera.get_rgba with warm steps
#       (replicator troubleshooting fix for missing annotator frames)
#   v3: experience slurm/isaac6_headless_render.kit WITHOUT waitIdle/serial trio
#       (variant v3.kit keeps viewport.rtx + asyncRendering=false only)
# Prints "ISAAC6_P3:<mode>: OK ..." or "ISAAC6_P3:<mode>: EMPTY_FRAME".

import os
import sys

import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "v1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYS_DT = 0.02
RES = (224, 224)

if MODE == "v3":
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True},
                        experience="/repo/slurm/isaac6_headless_render_v3.kit")
else:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})


def save_ppm(path, rgb_uint8):
    h, w = rgb_uint8.shape[:2]
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(rgb_uint8.tobytes())


def report(rgb):
    rgb = np.asarray(rgb)[..., :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    outdir = os.path.join(REPO, "experiments", "results")
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"isaac6_probe3_{MODE}.npy"), rgb)
    save_ppm(os.path.join(outdir, f"isaac6_probe3_{MODE}.ppm"), rgb)
    print("ISAAC6_P3:%s: OK shape=%s mean=%.2f min=%d max=%d"
          % (MODE, tuple(rgb.shape), float(rgb.mean()), int(rgb.min()),
             int(rgb.max())), flush=True)


def build_scene():
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    import omni.usd
    from pxr import UsdLux
    world = World(physics_dt=PHYS_DT, rendering_dt=PHYS_DT)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1500.0)
    cam = Camera("/World/isaac6_cam", resolution=RES)
    cam.set_local_pose(translation=np.array([2.0, 2.0, 2.0]),
                       orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                       camera_axes="usd")
    cam.initialize()
    return world, cam


def main_v1():
    import omni.replicator.core as rep
    world, cam = build_scene()
    rp = rep.create.render_product(cam.prim_path, RES)
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach(rp)
    data = None
    for k in range(60):
        rep.orchestrator.step()
        if k >= 4 and k % 2 == 0:
            d = np.asarray(annot.get_data())
            if d.ndim == 3 and d.size:
                data = d
                print(f"[p3-{MODE}] first valid annotator data at step {k}",
                      flush=True)
                break
    if data is None:
        print(f"ISAAC6_P3:{MODE}: EMPTY_FRAME", flush=True)
        return 1
    report(data)
    return 0


def main_v2():
    from carb import settings as cs
    s = cs.get_settings()
    s.set("/exts/isaacsim.core.throttling/enable_async", False)
    s.set("/omni/replicator/asyncRendering", False)
    s.set("/rtx/dataWindow/fitOutputToDataWindow", True)
    for k, v in (("/rtx/dataWindowNDC/0", 0.0), ("/rtx/dataWindowNDC/1", 0.0),
                 ("/rtx/dataWindowNDC/2", 1.0), ("/rtx/dataWindowNDC/3", 1.0)):
        if s.get(k) is None:
            s.set(k, v)
    world, cam = build_scene()
    rgba = None
    for k in range(60):
        world.step(render=True)
        if k >= 4 and k % 2 == 0:
            r = np.asarray(cam.get_rgba())
            if r.ndim == 3 and r.size:
                rgba = r
                print(f"[p3-{MODE}] first valid rgba at step {k}", flush=True)
                break
    if rgba is None:
        print(f"ISAAC6_P3:{MODE}: EMPTY_FRAME", flush=True)
        return 1
    report(rgba)
    return 0


def main_v3():
    s = None
    world, cam = build_scene()
    rgba = None
    for k in range(60):
        world.step(render=True)
        if k >= 4 and k % 2 == 0:
            r = np.asarray(cam.get_rgba())
            if r.ndim == 3 and r.size:
                rgba = r
                print(f"[p3-{MODE}] first valid rgba at step {k}", flush=True)
                break
    if rgba is None:
        print(f"ISAAC6_P3:{MODE}: EMPTY_FRAME", flush=True)
        return 1
    report(rgba)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = {"v1": main_v1, "v2": main_v2, "v3": main_v3}[MODE]()
    except BaseException:
        import traceback
        traceback.print_exc()
    try:
        app.close()
    except BaseException:
        pass
    os._exit(rc)
