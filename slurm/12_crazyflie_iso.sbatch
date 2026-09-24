#!/bin/bash
#SBATCH --job-name=cf-iso
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# ---------------------------------------------------------------------------
# SINGLE-VARIABLE (isolation) A/B on the REAL Crazyflie 2.X (Isaac GPU PhysX
# on Sol, apptainer isaac-sim-6.0.1.sif — the 5.1 SIF on Sol became unusable
# (stat /isaac-sim/python.sh: permission denied inside the image, dbg5 probe
# 63853352); engine version is identical across both arms of the test, and
# HOCBF runs as the verified pure-Python port hocbf_py here (2450-case exact
# cross-check vs C++), so the comparison stays single-variable.
#
# Responds to the attribution critique: in the earlier A/B
# (hocbf_crazyflie_ab100.py, results/crazyflie_vla_ab*.json) the ON arm also
# swapped the controller's nominal law AND zeroed the attitude moments, so
# OFF-vs-ON measured "controller swap + CBF", not the filter alone. Here BOTH
# arms use CrazyflieController.compute_wrench() for the nominal (T, tx, ty);
# the ONLY difference is whether T passes through filter_thrust (CBF safe-set
# clamp) before the shared mixer. Same seed, same paired randomization as the
# old headline (start_z, dive_vz, onset, vz0, wind, delay; mass NOT
# randomized). Writes experiments/results/crazyflie_vla_iso<N>.json (+ PNG).
# SIMULATION ONLY — not a hardware drone. Adversarial dive is scripted.
# Submit:  sbatch --export=ALL,N_EP=500 slurm/12_crazyflie_iso.sbatch
# ---------------------------------------------------------------------------
set -e
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac601.dir   # sandbox with chmod-fixed /isaac-sim (built by fix_isaac_sandbox2, job 63856233)
[ -e "$SIF" ] || { echo "SIF_MISSING: $SIF"; exit 1; }
REPO=/scratch/$USER/autonomous-drone-safety-architecture
cd "$REPO"
export N_EP=${N_EP:-500}
mkdir -p experiments/results

run() {
  apptainer exec --nv --writable \
    --home /scratch/$USER/solhome \
    --bind /scratch/$USER/autonomous-drone-safety-architecture:/repo,/tmp:/tmp \
    --env CF_USD=/repo/assets/crazyflie.usd,XDG_CACHE_HOME=/scratch/$USER/.cache,HF_HOME=/scratch/$USER/.hf,OMNI_KIT_ALLOW_ROOT=1 \
    "$SIF" "$@"
}

echo "Running $N_EP-episode/mode ISOLATION A/B on $(hostname), GPU=$CUDA_VISIBLE_DEVICES"
echo "(6.0.1 kit python is 3.12 — the compiled hocbf extension is py3.11-only;"
echo " hocbf_crazyflie_iso.py auto-falls-back to the verified hocbf_py port.)"

run /isaac-sim/python.sh /repo/sim/hocbf_crazyflie_iso.py "$N_EP"

# regenerate the plot strictly from the JSON we just wrote (no fabricated data)
run /isaac-sim/python.sh /repo/experiments/plot_crazyflie_ab_large.py \
      --json /repo/experiments/results/crazyflie_vla_iso${N_EP}.json \
      --out  /repo/experiments/results/crazyflie_vla_iso${N_EP}.png \
  || echo "PLOT_SKIPPED_ON_CLUSTER (regenerate locally from the JSON)"
echo "DONE — artifacts in experiments/results/ (commit from a login node)"
