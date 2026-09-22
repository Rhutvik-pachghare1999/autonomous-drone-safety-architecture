#!/bin/bash
#SBATCH --job-name=isaac-gpucheck
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# ---------------------------------------------------------------------------
# STEP 0 (Sol): prove GPU PhysX actually initializes before any big run.
# Isaac 5.0 had a Warp/CUDA-13 cuDeviceGetUuid bug on our 4GB laptop — verify
# it does NOT recur on Sol's driver. If wp.init() fails here, switch to the
# apptainer Isaac 5.1 container (see slurm/README_SOL.md).
# Submit:  cd /scratch/$USER/autonomous-drone-safety-architecture && sbatch slurm/00_gpu_verify.sbatch
# ---------------------------------------------------------------------------
set -e
module purge
module load isaacsim/5.0

REPO=/scratch/$USER/autonomous-drone-safety-architecture
CF=$(find /packages/apps/isaacsim/5.0 -name crazyflie.usd 2>/dev/null | head -1)
[ -z "$CF" ] && CF=$REPO/assets/crazyflie.usd   # fallback: repo-vendored copy

# keep all caches off /home (only 26 GB free) — everything on /scratch
export HOME_ORIG=$HOME
export XDG_CACHE_HOME=/scratch/$USER/.cache
export HF_HOME=/scratch/$USER/.hf
export OMNI_KIT_ALLOW_ROOT=1
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME"

cat > /tmp/gpu_verify_$$.py <<'PY'
import sys
def log(*a): print(*a); sys.stdout.flush()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import warp as wp
try:
    wp.init(); log("GPU_DEVICES:", wp.get_cuda_devices())
except Exception as e:
    log("WARP_FAIL:", str(e)[:200]); app.close(); sys.exit(2)
import os, numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdPhysics
import omni.usd
CF = os.environ["CF_USD"]
world = World(stage_units_in_meters=1.0, physics_dt=1/240.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=CF, prim_path="/World/cf2")
prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/cf2")
UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.027)
if not prim.HasAPI(UsdPhysics.RigidBodyAPI): UsdPhysics.RigidBodyAPI.Apply(prim)
from isaacsim.core.prims import RigidPrim
cf = RigidPrim("/World/cf2"); world.reset()
cf.set_world_poses(positions=np.array([[0.0,0.0,1.0]]))
log("STEP z(m)")
for i in range(120):
    world.step(render=False)
    if i % 15 == 0:
        p,_ = cf.get_world_poses(); log(f"{i:4d} {float(np.array(p)[0][2]):.4f}")
app.close(); log("CLEAN_EXIT")
PY

export CF_USD="$CF"
echo "Using Crazyflie USD: $CF"
srun isaacsim.sh --no-window --/app/quitAfter=0 || true   # noop guard for some builds
python.sh /tmp/gpu_verify_$$.py
rm -f /tmp/gpu_verify_$$.py
