#!/bin/bash
#SBATCH --job-name=vla-rc6
#SBATCH --partition=public
#SBATCH --exclude=sg048,sg049
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/vla-realcam6-%j.out
# ---------------------------------------------------------------------------
# TASK 2 run on IsaIM 6.0.1 — REAL rendered Isaac camera -> SmolVLM2 (GPU) ->
# HOCBF guard -> 50 Hz controller on the 27 g Crazyflie 2.X.
#
# Prereq: slurm/32_vla_prep6.sbatch completed with PREP_RESULT: OK
#         (deps in /scratch/$USER/pylibs6, weights in /scratch/$USER/.hf).
# Sol topology: sg048/049 serve ONLY MIG a100.20gb slices (no Vulkan graphics)
# -> RTX renderer dies there (carb::graphics::createInstance failed, job
# 63824988). Must run on a full a100 from partition=public, excl sg048/049.
# 6.0-specific env: CRAZYFLIE_USD (crazyflie_env.py reads THIS name), writable
# kit/cache binds (NO XDG_CACHE_HOME pointing at non-bound paths -> EROFS).
#
# Usage:
#   SMALL proof (2 episodes):  sbatch --export=ALL,N_EP=2 slurm/33_vla_realcam6.sbatch
#   FULL run    (6 episodes):  sbatch --export=ALL,N_EP=6 slurm/33_vla_realcam6.sbatch
#
# Outputs: experiments/results/{vla_realcam.jsonl,vla_realcam_summary.json,
#           vla_realcam_ep0_*.png} -> scp back to laptop.
# ---------------------------------------------------------------------------
set -euo pipefail
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-6.0.1.sif
REPO=/scratch/$USER/autonomous-drone-safety-architecture
PYLIBS=/scratch/$USER/pylibs6

if [ ! -f "$SIF" ]; then
  echo "MISSING CONTAINER: $SIF" >&2; exit 1
fi

N_EP="${N_EP:-2}"
cd "$REPO"
echo "=== NODE $(hostname) GPU $SLURM_JOB_GPUS  N_EP=$N_EP  SIF=$(basename $SIF)"
echo "=== git: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

# fail fast if prep6 was not run
ls "$PYLIBS/transformers" >/dev/null 2>&1 || {
  echo "MISSING $PYLIBS/transformers — run slurm/32_vla_prep6.sbatch first" >&2
  exit 1; }
ls /scratch/$USER/.hf/hub/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct >/dev/null 2>&1 || {
  echo "MISSING SmolVLM2 weights in /scratch/$USER/.hf — run slurm/32_vla_prep6.sbatch" >&2
  exit 1; }

CACHE=/scratch/$USER/isaac_cache6
mkdir -p "$CACHE"/{kit_cache,kit_data,ov,glcache,computecache,omni_logs,pip}

run() {
  apptainer exec --nv --fakeroot \
    --home /scratch/$USER/solhome \
    --bind "$REPO":/repo,/tmp:/tmp,$CACHE/kit_cache:/isaac-sim/kit/cache,$CACHE/kit_data:/isaac-sim/kit/data,$CACHE/ov:/root/.cache/ov,$CACHE/glcache:/root/.cache/nvidia/GLCache,$CACHE/computecache:/root/.nv/ComputeCache,$CACHE/omni_logs:/root/.nvidia-omniverse/logs,$CACHE/pip:/root/.cache/pip \
    --bind "$PYLIBS":/scratch/$USER/pylibs6,/scratch/$USER/.hf:/scratch/$USER/.hf \
    --env OMNI_KIT_ALLOW_ROOT=1,PYTHONUNBUFFERED=1 \
    --env CF_USD=/repo/assets/crazyflie.usd \
    --env CRAZYFLIE_USD=/repo/assets/crazyflie.usd \
    --env HF_HOME=/scratch/$USER/.hf \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env PYLIBS=/scratch/$USER/pylibs6 \
    "$SIF" "$@"; RC=$?; echo "APPTAINER_RC=$RC"; return $RC; }
# NOTE: $PYLIBS is NOT on PYTHONPATH at kit startup on purpose — the 6.0 kit's
# deprecation_manager would try to import torch at ext-scan time, the dormant
# ml_archive prebundle libtorch_cuda.so then fails against pip NCCL and kit
# exits(0) without running the script (job 63847627).  vla_realcam_flight.py
# appends $PYLIBS to sys.path AFTER SimulationApp is up (first torch import
# happens via the lazy vla_bridge import).

run /isaac-sim/python.sh /repo/sim/vla_realcam_flight.py "$N_EP"; RC=$?
echo "RUN_EXIT_CODE=$RC"
exit $RC
