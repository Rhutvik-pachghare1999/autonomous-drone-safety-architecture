#!/bin/bash
#SBATCH --job-name=isaac6-compat
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/isaac6-compat-%j.out
#SBATCH --exclude=sg048,sg049
# ---------------------------------------------------------------------------
# Compat gate BEFORE any Phase-5 port (per user-approved plan):
#   (a) RTX render proof on Isaac Sim 6.0.1 (one full frame -> PPM+npy,
#       no librtx.scenedb segfault — the 5.1/driver-595 regression).
#   (b) crazyflie_env + hocbf_py + VLA pipeline run a 2-episode A/B UNCHANGED
#       on 6.0. If OFF-crash / ON-survive holds, physics+filter path intact.
# Writable binds: /isaac-sim is squashed read-only even under --fakeroot;
# kit must write shaderdb/DerivedDataCache (5.1 without these -> HydraEngine
# rtx "failed creating scene renderer" + empty annotator frames; with them ->
# scenedb crash reaches the real RTX path).  Separate cache tree for 6.0.
# Requires /scratch/$USER/isaac-sim-6.0.1.sif (from 60_isaac6_pull.sbatch).
# Submit: sbatch slurm/61_isaac6_compat.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-6.0.1.sif
[ -f "$SIF" ] || { echo "SIF_MISSING: $SIF — run 60_isaac6_pull.sbatch first"; exit 1; }
REPO=/scratch/$USER/autonomous-drone-safety-architecture
cd "$REPO"

export PYTHONPATH=/repo:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export OMNI_KIT_ALLOW_ROOT=1
export TOKENIZERS_PARALLELISM=false
CACHE=/scratch/$USER/isaac_cache6
mkdir -p "$CACHE"/{kit_cache,kit_data,ov,glcache,computecache,omni_logs,pip}
# Do NOT set XDG_CACHE_HOME=$CACHE: only its SUBDIRS are bind-mounted into the
# container, so $CACHE itself appears read-only in-container and 6.0's
# omni.warp.core os.makedirs(kernel_cache_dir) raises EROFS at World() init
# (job 63841944).  Under --home /scratch/$USER/solhome the default
# $HOME/.cache/warp is writable and scratch-persistent.

run() {
  apptainer exec --nv --fakeroot \
    --home /scratch/$USER/solhome \
    --bind "$REPO":/repo,/tmp:/tmp,$CACHE/kit_cache:/isaac-sim/kit/cache,$CACHE/kit_data:/isaac-sim/kit/data,$CACHE/ov:/root/.cache/ov,$CACHE/glcache:/root/.cache/nvidia/GLCache,$CACHE/computecache:/root/.nv/ComputeCache,$CACHE/omni_logs:/root/.nvidia-omniverse/logs,$CACHE/pip:/root/.cache/pip \
    --env CF_USD=/repo/assets/crazyflie.usd,HF_HOME=/scratch/$USER/.hf,OMNI_KIT_ALLOW_ROOT=1,TOKENIZERS_PARALLELISM=false,PYTHONUNBUFFERED=1 \
    "$SIF" "$@"
}

echo "===== isaac6 compat on $(hostname), SIF=$SIF ====="

echo "===== step 1: RTX render probe ====="
run /isaac-sim/python.sh /repo/slurm/isaac6_probe.py 2>&1 | tail -40

echo "===== step 2: 2-episode A/B (unchanged 5.1-era code path) ====="
run /isaac-sim/python.sh /repo/sim/hocbf_crazyflie_ab100.py 2 2>&1 | tail -40

echo "===== artifacts ====="
ls -la "$REPO/experiments/results/isaac6_probe.ppm" "$REPO/experiments/results/isaac6_probe.npy" "$REPO/experiments/results/crazyflie_vla_ab2.json" 2>&1 || true
echo "ISAAC6_COMPAT_DONE"
