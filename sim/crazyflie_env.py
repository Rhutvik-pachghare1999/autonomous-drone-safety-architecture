#!/usr/bin/env python3
"""
sim/crazyflie_env.py — Real Crazyflie 2.X in Isaac Sim 5.1 (Phase 1)
=====================================================================
Replaces the DynamicCuboid "flying box" with the real multirotor mesh
(crazyflie.usd — propeller_ccw/cw, motor mounts, ~205 KB) and proves the
27 g airframe can hover via PhysX body forces.

Crazyflie 2.X physical parameters (Bitcraze docs):
  mass               = 0.027 kg                (27 g)
  hover thrust       = m*g = 0.027*9.81 = 0.26487 N
  max total thrust   ~ 0.60 N                  (4 x ~0.15 N per 7 mm coreless motor)
  arm length         = 0.046 m

CRITICAL (verified live in Phase 0): the crazyflie.usd carries an INVALID
default inertia / negative mass. RigidBodyAPI + MassAPI with an explicit
mass of 0.027 kg MUST be applied or PhysX computes nonsense dynamics.

Run ONLY via Isaac Sim's python:
  ~/.local/share/ov/pkg/isaac_sim-5.1.0/python.sh sim/crazyflie_env.py
"""

# SimulationApp MUST be created before any other isaacsim/omni imports
# (established pattern from sim/isaac_sil_eval.py).
from isaacsim import SimulationApp
_app = SimulationApp({"headless": True})

import math
import os
import sys

import numpy as np
import omni.usd
from pxr import UsdPhysics, UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage

# ── Crazyflie 2.X constants ─────────────────────────────────────────────────
CF_USD = os.environ.get(
    "CRAZYFLIE_USD",
    os.path.expanduser(
        "~/.local/share/ov/pkg/isaac_sim-5.1.0/extscache/"
        "omni.warp.core-1.8.2+lx64/warp/examples/assets/crazyflie.usd"
    ),
)
MASS = 0.027              # kg
G = 9.81                  # m/s^2
T_HOVER = MASS * G        # 0.26487 N
T_MAX = 0.60              # N, 4 x ~0.15 N per motor
ARM = 0.046               # m
PHYS_DT = 0.02            # 50 Hz


class CrazyflieEnv:
    """Real Crazyflie 2.X rigid body in Isaac Sim 5.1.

    Thrust is applied as a PhysX body force along the body +Z axis
    (rotated into the world frame from the live quaternion) — the same
    point-mass-force abstraction the old cuboid used, but now acting on
    the real 27 g multirotor geometry/inertia.
    """

    def __init__(self, prim_path: str = "/World/cf2", start=(0.0, 0.0, 1.0)):
        self.prim_path = prim_path
        self.start = np.asarray(start, dtype=np.float64)
        self.world = World(physics_dt=PHYS_DT, rendering_dt=PHYS_DT)
        self.cf: RigidPrim | None = None
        self._thrust = 0.0  # N, total body-Z thrust applied each step

    # ── setup ──────────────────────────────────────────────────────────────
    def setup(self) -> None:
        self.world.scene.add_default_ground_plane()

        # Load the REAL Crazyflie mesh (not a cuboid).
        add_reference_to_stage(usd_path=CF_USD, prim_path=self.prim_path)
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Failed to load Crazyflie prim at {self.prim_path}")

        # FIX the invalid USD inertia/negative mass (Phase 0-issued fix):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(MASS)

        # crazyflie.usd is a VISUAL-ONLY asset: zero CollisionAPIs on any
        # mesh (verified by stage traversal). A shapeless rigid body ignores
        # external forces and falls through the ground plane. Give it a real
        # collision volume: a box matching the CF2 frame (92x92x30 mm), welded
        # to the root rigid body. The 4 propeller meshes stay visual-only.
        from pxr import UsdGeom, Gf
        collision_path = f"{self.prim_path}/collision_box"
        cube_prim = stage.DefinePrim(collision_path, "Cube")
        cube = UsdGeom.Cube(cube_prim)
        cube.CreateSizeAttr(1.0)
        cube_xf = UsdGeom.Xformable(cube_prim)
        cube_xf.ClearXformOpOrder()
        cube_xf.AddScaleOp().Set(Gf.Vec3f(0.092, 0.092, 0.030))
        UsdGeom.Gprim(cube_prim).CreatePurposeAttr().Set("guide")
        UsdPhysics.CollisionAPI.Apply(cube_prim)

        # Move spawn point before physics starts.
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*self.start.tolist()))

        self.cf = RigidPrim(self.prim_path)
        self.world.reset()
        self.cf.initialize()

        # Never let PhysX sleep this body: a sleeping rigid body ignores
        # apply_forces() and reports a stale cached velocity.
        self.cf.set_sleep_thresholds(np.zeros(1, dtype=np.float32))

        # --- post-setup verification ---
        meff = float(self.cf.get_masses()[0])
        pos, quat = self.cf.get_world_poses()
        children = [c.GetName() for c in prim.GetChildren()]
        print(f"[setup] prim={self.prim_path}  start_z={pos[0][2]:.3f}")
        print(f"[setup] effective mass after MassAPI fix = {meff:.4f} kg (target {MASS})")
        print(f"[setup] stage children of cf2: {children}")
        sys.stdout.flush()

    def reset_pose(self, z: float) -> None:
        """Teleport to (0,0,z), level attitude, zero velocity; wake the body.

        Order matters: zero the velocities LAST so the drone does not pick up
        an initial -g*dt kick from a zero-thrust step before the test begins.
        """
        self.set_thrust(0.0)
        self.world.step(render=False)          # let the pose/velocity write land
        self.cf.set_world_poses(
            positions=np.array([[0.0, 0.0, z]], dtype=np.float32),
            orientations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        self.cf.set_velocities(np.zeros((1, 6), dtype=np.float32))
        _, _, vel, omega = self.state()
        print(f"[reset] z={z:.3f}  vel_z={vel[2]:+.4f}  |omega|={np.linalg.norm(omega):.4f}")
        sys.stdout.flush()

    # ── dynamics helpers ───────────────────────────────────────────────────
    def set_thrust(self, total_newtons: float) -> None:
        """Command total collective thrust [N] (clamped to [0, T_MAX])."""
        self._thrust = float(np.clip(total_newtons, 0.0, T_MAX))

    def state(self):
        """Returns pos(3,), quat(wxyz), vel(3,), omega(3,) as numpy arrays."""
        pos, quat = self.cf.get_world_poses()
        vel = self.cf.get_linear_velocities()
        omega = self.cf.get_angular_velocities()
        return pos[0], quat[0], vel[0], omega[0]

    def hover_controller(self, z_ref: float) -> float:
        """Feedforward mg + P/D on altitude -> thrust command [N].

        acc_cmd = kp*(z_ref - z) - kd*vz ;  T = m*(g + acc_cmd)
        """
        pos, _, vel, _ = self.state()
        acc_cmd = 4.0 * (z_ref - pos[2]) - 3.0 * vel[2]
        return MASS * (G + acc_cmd)

    # ── stepping ───────────────────────────────────────────────────────────
    def step(self) -> None:
        """Apply current thrust along body +Z, then advance physics 1 step."""
        _, quat, _, _ = self.state()
        w, x, y, z = quat
        # body +Z axis expressed in the world frame
        bz = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        force_world = bz * self._thrust
        self.cf.apply_forces(
            force_world.reshape(1, 3).astype(np.float32),
            np.array([0], dtype=np.int32),
            True,
        )
        self.world.step(render=False)


# ── Phase-1 test battery ─────────────────────────────────────────────────────

def _trace(env: CrazyflieEnv, label: str, thrust_fn, n_steps: int, every: int = 25,
           start_z: float = 1.0) -> None:
    print(f"--- {label} ---")
    sys.stdout.flush()
    env.reset_pose(start_z)
    for i in range(n_steps):
        T = thrust_fn(env)
        env.set_thrust(T)
        env.step()
        if i % every == 0 or i == n_steps - 1:
            pos, _, vel, _ = env.state()
            t = (i + 1) * PHYS_DT
            print(
                f"  t={t:5.2f}s  z={pos[2]:+.4f} m  vz={vel[2]:+.4f} m/s  "
                f"T={env._thrust:.4f} N"
            )
            sys.stdout.flush()


def run_phase1() -> None:
    print("=" * 72)
    print("PHASE 1 — REAL CRAZYFLIE 2.X: HOVER vs FALL (Isaac Sim 5.1 headless)")
    print("=" * 72)
    print(f"Crazyflie params: mass={MASS} kg  hover={T_HOVER:.5f} N  "
          f"T_max={T_MAX} N  arm={ARM} m")
    print(f"Asset: {CF_USD}")
    sys.stdout.flush()

    import torch  # Isaac's bundled torch — reports the physics GPU
    print(f"GPU_DEVICES: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    sys.stdout.flush()

    env = CrazyflieEnv(start=(0.0, 0.0, 1.0))
    env.setup()

    # TEST 1 — exact open-loop hover thrust (validates the mass fix; if the
    # mass were wrong the drone would drift up/down immediately).
    _trace(env, "TEST 1: OPEN-LOOP HOVER  T = m*g = 0.26487 N  (spawn z=1.0)",
           lambda e: T_HOVER, n_steps=150, every=30, start_z=1.0)

    # TEST 2 — closed-loop altitude hold at 1.0 m from 0.5 m (what Phase 2/3 use).
    _trace(env, "TEST 2: CLOSED-LOOP CLIMB->HOLD AT 1.0 m  (spawn z=0.5, P controller)",
           lambda e: e.hover_controller(1.0), n_steps=150, every=30, start_z=0.5)

    # TEST 3 — climb with 1.2x hover thrust from 1.0 m.
    # Expected: a = (1.2-1)*g = 1.962 m/s^2 up  ->  z = 1 + 0.981*t^2
    _trace(env, "TEST 3: CLIMB  T = 1.2*m*g = 0.31784 N  (spawn z=1.0)",
           lambda e: 1.2 * T_HOVER, n_steps=60, every=12, start_z=1.0)

    # TEST 4 — descend with 0.8x hover thrust from 1.5 m.
    # Expected: a = -0.2*g = -1.962 m/s^2  ->  z = 1.5 - 0.981*t^2
    _trace(env, "TEST 4: DESCEND  T = 0.8*m*g = 0.21190 N  (spawn z=1.5)",
           lambda e: 0.8 * T_HOVER, n_steps=60, every=12, start_z=1.5)

    # TEST 5 — zero thrust from 1.0 m -> free fall & ground contact.
    # Expected: z = 1 - 4.905*t^2 until contact at t=0.45 s.
    _trace(env, "TEST 5: FREE FALL  T = 0 N  (spawn z=1.0)",
           lambda e: 0.0, n_steps=50, every=5, start_z=1.0)

    print("=" * 72)
    print("PHASE 1 COMPLETE")
    print("=" * 72)
    sys.stdout.flush()
    _app.close()


if __name__ == "__main__":
    run_phase1()
