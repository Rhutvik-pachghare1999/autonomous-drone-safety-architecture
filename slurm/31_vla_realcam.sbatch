#!/bin/bash
#SBATCH --job-name=vla-rc
#SBATCH --partition=public
#SBATCH --exclude=sg048,sg049
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/vla-realcam-%j.out
#
# TASK 2 run — REAL rendered Isaac camera -> SmolVLM2 (GPU) -> HOCBF guard ->
# 50 Hz controller on the real 27 g Crazyflie in Isaac Sim 5.1.
#
# Prereq: slurm/30_vla_prep.sbatch completed with PREP_RESULT: OK
#         (deps in /scratch/$USER/pylibs, weights in /scratch/$USER/.hf).
# NOTE Sol topology: sg048/049 serve ONLY MIG a100.20gb slices (no Vulkan
# graphics) and htc serves a100.40gb slices — the RTX renderer dies there
# (carb::graphics::createInstance failed, job 63824988).  Must run on a full
# a100 from partition=public, excluding sg048/049.
#
# Usage:
#   SMALL proof (2 episodes):  sbatch --export=ALL,N_EP=2 slurm/31_vla_realcam.sbatch
#   FULL run    (6 episodes):  sbatch --export=ALL,N_EP=6 slurm/31_vla_realcam.sbatch
#
# Outputs land in the Sol repo clone experiments/results/ (vla_realcam.jsonl,
# vla_realcam_summary.json, vla_realcam_ep0_*.png) → scp back to laptop.

set -euo pipefail
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-5.1.sif
REPO=/scratch/$USER/autonomous-drone-safety-architecture
PYLIBS=/scratch/$USER/pylibs

if [ ! -f "$SIF" ]; then
  echo "MISSING CONTAINER: $SIF" >&2; exit 1
fi

N_EP="${N_EP:-2}"
cd "$REPO"
echo "=== NODE $(hostname) GPU $SLURM_JOB_GPUS  N_EP=$N_EP ==="
echo "=== git: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

# fail fast if prep was not run
ls "$PYLIBS/transformers" >/dev/null 2>&1 || {
  echo "MISSING $PYLIBS/transformers — run slurm/30_vla_prep.sbatch first" >&2
  exit 1; }
ls /scratch/$USER/.hf/hub/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct >/dev/null 2>&1 || {
  echo "MISSING SmolVLM2 weights in /scratch/$USER/.hf — run slurm/30_vla_prep.sbatch" >&2
  exit 1; }

run() { apptainer exec --fakeroot --nv \
  --home /scratch/$USER/solhome \
  --bind /scratch:/scratch \
  --bind "$REPO":/repo \
  --bind /tmp:/tmp \
  --env "CF_USD=/repo/assets/crazyflie.usd" \
  --env "CRAZYFLIE_USD=/repo/assets/crazyflie.usd" \
  --env "XDG_CACHE_HOME=/tmp/xdg_$SLURM_JOB_ID" \
  --env "HF_HOME=/scratch/$USER/.hf" \
  --env "HF_HUB_OFFLINE=1" \
  --env "TRANSFORMERS_OFFLINE=1" \
  --env "PYTHONPATH=$PYLIBS" \
  --env "OMNI_KIT_ALLOW_ROOT=1" \
  "$SIF" "$@"; RC=$?; echo "APPTAINER_RC=$RC"; return $RC; }

pushd "$REPO" >/dev/null
run /isaac-sim/python.sh sim/vla_realcam_flight.py "$N_EP"; RC=$?
popd >/dev/null
echo "RUN_EXIT_CODE=$RC"
exit $RC
