#!/bin/bash
#SBATCH --job-name=vla-prep
#SBATCH --partition=public,htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/vla-prep-%j.out
#
# TASK 2 prep — runs EXACTLY ONCE, on a compute node (never login node):
#   * checks compute-node internet reachability (result printed in .out)
#   * pip-installs transformers/bitsandbytes/accelerate/pillow into
#     /scratch/$USER/pylibs  (container python is read-only -> --target)
#   * downloads SmolVLM2-2.2B-Instruct weights into /scratch/$USER/.hf
#   * GPU smoke: loads the model + one real generation (catches bnb/torch
#     ABI issues here instead of inside the experiment run)
#
# If .out ends with "PREP_RESULT: NO_INTERNET" -> STOP and ask the user
# (weights/deps would need a different staging route, e.g. scp from laptop).

set -euo pipefail
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-5.1.sif
REPO=/scratch/$USER/autonomous-drone-safety-architecture

if [ ! -f "$SIF" ]; then
  echo "MISSING CONTAINER: $SIF — pull onto /scratch first (see slurm/README_SOL.md)" >&2
  exit 1
fi
if [ ! -f "$REPO/slurm/vla_prep.py" ]; then
  echo "MISSING $REPO/slurm/vla_prep.py — git pull in $REPO first" >&2
  exit 1
fi

run() { apptainer exec --fakeroot --nv \
  --home /scratch/$USER/solhome \
  --bind /scratch:/scratch \
  --bind "$REPO":/repo \
  --bind /tmp:/tmp \
  --env "XDG_CACHE_HOME=/tmp/xdg_$SLURM_JOB_ID" \
  --env "HF_HOME=/scratch/$USER/.hf" \
  --env "PYLIBS=/scratch/$USER/pylibs" \
  --env "OMNI_KIT_ALLOW_ROOT=1" \
  "$SIF" "$1"; RC=$?; echo "APPTAINER_RC=$RC"; return $RC; }

echo "=== NODE $(hostname) GPU $SLURM_JOB_GPUS ==="
run "/isaac-sim/python.sh /repo/slurm/vla_prep.py"; RC=$?
echo "PREP_EXIT_CODE=$RC"
exit $RC
