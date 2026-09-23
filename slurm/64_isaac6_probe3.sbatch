#!/bin/bash
#SBATCH --job-name=isaac6-probe3
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/isaac6-probe3-%j.out
#SBATCH --exclude=sg048,sg049
# ---------------------------------------------------------------------------
# 3-variant RTX render proof on Isaac Sim 6.0.1 (one SimulationApp per proc):
#   v1: bare headless + manual rep.orchestrator capture (rep-troubleshooting)
#   v2: bare headless + throttling.enable_async=false (missing-frames fix)
#   v3: camera experience w/o waitIdle/forceSerial trio (job 63843303 aborted
#       with "Destroying busy TaskGroup!" on teardown with those settings)
# Any variant that prints ISAAC6_P3:<mode>: OK unblocks Phase-5 port to 6.0.
# ---------------------------------------------------------------------------
set -uo pipefail
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

echo "===== isaac6 probe3 on $(hostname) ====="
for m in v1 v2 v3; do
  echo "===== variant $m ====="
  run /isaac-sim/python.sh /repo/slurm/isaac6_probe3.py $m 2>&1 | tail -12
done
echo "PROBE3_DONE"
