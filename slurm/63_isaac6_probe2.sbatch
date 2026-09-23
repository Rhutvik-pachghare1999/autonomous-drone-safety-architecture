#!/bin/bash
#SBATCH --job-name=isaac6-probe2
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/isaac6-probe2-%j.out
#SBATCH --exclude=sg048,sg049
# ---------------------------------------------------------------------------
# Render proof #2 on Isaac Sim 6.0.1 with slurm/isaac6_headless_render.kit
# (offscreen camera pipeline: omni.kit.viewport.rtx + omni.replicator.core +
# synchronous render settings, adapted from IsaacLab's headless.rendering kit).
# Gate for porting Phase 5 realcam to 6.0.
# ---------------------------------------------------------------------------
set -euo pipefail
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-6.0.1.sif
REPO=/scratch/$USER/autonomous-drone-safety-architecture
cd "$REPO"

CACHE=/scratch/$USER/isaac_cache6
mkdir -p "$CACHE"/{kit_cache,kit_data,ov,glcache,computecache,omni_logs,pip}

run() {
  apptainer exec --nv --fakeroot \
    --home /scratch/$USER/solhome \
    --bind "$REPO":/repo,/tmp:/tmp,$CACHE/kit_cache:/isaac-sim/kit/cache,$CACHE/kit_data:/isaac-sim/kit/data,$CACHE/ov:/root/.cache/ov,$CACHE/glcache:/root/.cache/nvidia/GLCache,$CACHE/computecache:/root/.nv/ComputeCache,$CACHE/omni_logs:/root/.nvidia-omniverse/logs,$CACHE/pip:/root/.cache/pip \
    --env OMNI_KIT_ALLOW_ROOT=1,PYTHONUNBUFFERED=1 \
    "$SIF" "$@"
}

echo "===== isaac6 probe2 (camera experience) on $(hostname) ====="
run /isaac-sim/python.sh /repo/slurm/isaac6_probe2.py 2>&1 | tail -30
ls -la "$REPO/experiments/results/isaac6_probe2.ppm" 2>&1 || true
echo "PROBE2_DONE"
