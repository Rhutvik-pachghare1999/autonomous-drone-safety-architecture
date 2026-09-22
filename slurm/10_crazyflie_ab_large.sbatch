#!/bin/bash
#SBATCH --job-name=cf-ab1000
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# ---------------------------------------------------------------------------
# Large randomized A/B on the REAL Crazyflie 2.X (Isaac Sim GPU PhysX on Sol).
# N episodes/mode, paired (same randomized episode fed to filter OFF and ON):
# randomize start altitude, dive vz, dive onset, wind force, command delay.
# Writes experiments/results/crazyflie_vla_ab1000.json (+ plot via plot script).
# SIMULATION ONLY — not a hardware drone. Adversarial dive is scripted worst-case.
# Submit:  sbatch --export=ALL,N_EP=1000 slurm/10_crazyflie_ab_large.sbatch
# ---------------------------------------------------------------------------
set -e
module purge
module load isaacsim/5.0

REPO=/scratch/$USER/autonomous-drone-safety-architecture
cd "$REPO"
export XDG_CACHE_HOME=/scratch/$USER/.cache
export HF_HOME=/scratch/$USER/.hf
export OMNI_KIT_ALLOW_ROOT=1
export N_EP=${N_EP:-1000}
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" experiments/results

echo "Running $N_EP-episode/mode A/B on $(hostname), GPU=$CUDA_VISIBLE_DEVICES"
# hocbf_crazyflie_ab100.py accepts N as argv[1]. Randomization: start_z,
# dive_vz, onset, vz0, wind (N), command delay (steps). Mass is NOT randomized
# (env has no set_mass — see script header).
python.sh sim/hocbf_crazyflie_ab100.py "$N_EP"

# regenerate the plot strictly from the JSON we just wrote (no fabricated data).
# matplotlib may be absent in Isaac's bundled python; fall back to system python.
python.sh experiments/plot_crazyflie_ab_large.py \
      --json experiments/results/crazyflie_vla_ab${N_EP}.json \
      --out  experiments/results/crazyflie_vla_ab${N_EP}.png \
  || python3 experiments/plot_crazyflie_ab_large.py \
      --json experiments/results/crazyflie_vla_ab${N_EP}.json \
      --out  experiments/results/crazyflie_vla_ab${N_EP}.png \
  || echo "PLOT_SKIPPED_ON_CLUSTER (regenerate locally from the JSON)"
echo "DONE — artifacts in experiments/results/ (copy back / commit from a login node)"
