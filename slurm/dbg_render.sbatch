#!/bin/bash
#SBATCH --job-name=dbg-rend
#SBATCH --partition=public
#SBATCH --exclude=sg048,sg049
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/dbg-render-%j.out
#
# One-off: debug the camera render-product failure (see dbg_render.py).
# Temporary — delete once the fix lands in sim/vla_realcam_flight.py.

set -euo pipefail
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-5.1.sif
REPO=/scratch/$USER/autonomous-drone-safety-architecture
: "${CF_USD:=/scratch/$USER/assets/crazyflie.usd}"
: "${CRAZYFLIE_USD:=$CF_USD}"
export CF_USD CRAZYFLIE_USD

run() {
  apptainer exec --fakeroot --nv \
    --home /scratch/$USER/solhome \
    --bind /scratch:/scratch \
    --bind /tmp:/tmp \
    --env ACCEPT_EULA=Y \
    --env CF_USD="$CF_USD" --env CRAZYFLIE_USD="$CRAZYFLIE_USD" \
    --env XDG_CACHE_HOME=/tmp/xdg_${SLURM_JOB_ID:-manual} \
    --env OMNI_KIT_ALLOW_ROOT=1 \
    --env OPENCODE_OBS="$OPENCODE_OBS" \
    "$SIF" "$@"
}

: "${OPENCODE_OBS:=$REPO/.opencode-obs}"
run /isaac-sim/python.sh "$REPO/slurm/dbg_render.py"
