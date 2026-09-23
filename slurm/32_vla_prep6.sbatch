#!/bin/bash
#SBATCH --job-name=vla-prep6
#SBATCH --partition=public,htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/vla-prep6-%j.out
# ---------------------------------------------------------------------------
# TASK 2 prep FOR THE 6.0.1 CONTAINER — runs ONCE, compute node only.
# Same steps as slurm/30_vla_prep.sbatch but against
# /scratch/$USER/isaac-sim-6.0.1.sif (bundle python is 3.12, NOT 3.11 ->
# pylibs wheels are ABI-incompatible; separate target dir pylibs6).
# HF_HOME cache is shared (weights are ABI-agnostic): /scratch/$USER/.hf.
#
# If .out ends with "PREP_RESULT: NO_INTERNET" -> STOP and ask the user.
# ---------------------------------------------------------------------------
set -euo pipefail
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-6.0.1.sif
REPO=/scratch/$USER/autonomous-drone-safety-architecture

if [ ! -f "$SIF" ]; then
  echo "MISSING CONTAINER: $SIF — run slurm/60_isaac6_pull.sbatch first" >&2
  exit 1
fi
if [ ! -f "$REPO/slurm/vla_prep6.py" ]; then
  echo "MISSING $REPO/slurm/vla_prep6.py — git pull in $REPO first" >&2
  exit 1
fi

CACHE=/scratch/$USER/isaac_cache6
PYLIBS=/scratch/$USER/pylibs6
# reset: previous attempt installed a redundant pip torch (cu13) which broke
# bnb's cublasLt discovery — minimal --no-deps stack on container torch only.
rm -rf "$PYLIBS"
mkdir -p "$CACHE"/{kit_cache,kit_data,ov,glcache,computecache,omni_logs,pip} "$PYLIBS"

run() {
  apptainer exec --fakeroot --nv \
    --home /scratch/$USER/solhome \
    --bind "$REPO":/repo,/tmp:/tmp,$CACHE/kit_cache:/isaac-sim/kit/cache,$CACHE/kit_data:/isaac-sim/kit/data,$CACHE/ov:/root/.cache/ov,$CACHE/glcache:/root/.cache/nvidia/GLCache,$CACHE/computecache:/root/.nv/ComputeCache,$CACHE/omni_logs:/root/.nvidia-omniverse/logs,$CACHE/pip:/root/.cache/pip \
    --bind "$PYLIBS":/scratch/$USER/pylibs6,/scratch/$USER/.hf:/scratch/$USER/.hf \
    --env OMNI_KIT_ALLOW_ROOT=1,HF_HOME=/scratch/$USER/.hf,PYLIBS=/scratch/$USER/pylibs6,PYTHONUNBUFFERED=1 \
    "$SIF" "$@"; RC=$?; echo "APPTAINER_RC=$RC"; return $RC; }

echo "=== NODE $(hostname) GPU $SLURM_JOB_GPUS ==="
run /isaac-sim/python.sh /repo/slurm/vla_prep6.py; RC=$?
echo "PREP_EXIT_CODE=$RC"
exit $RC
