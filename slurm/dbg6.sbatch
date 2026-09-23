#!/bin/bash
#SBATCH --job-name=dbg6
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -x
module purge
module load apptainer/1.4.5

REPO=/scratch/$USER/autonomous-drone-safety-architecture
run() {
  apptainer exec --nv \
    --home /scratch/$USER/solhome \
    --bind $REPO:/repo,/tmp:/tmp \
    --env CF_USD=/repo/assets/crazyflie.usd,XDG_CACHE_HOME=/scratch/$USER/.cache,HF_HOME=/scratch/$USER/.hf,OMNI_KIT_ALLOW_ROOT=1 \
    "$1" "${@:2}"
}

for S in /scratch/$USER/isaac-sim-5.1.sif /scratch/$USER/isaac-sim-6.0.1.sif; do
  echo "===== $S ====="
  run "$S" ls -la / | head -20
  run "$S" bash -c "ls -la /isaac-sim/python.sh; head -3 /isaac-sim/python.sh" 2>&1 | head -8
  run "$S" /isaac-sim/python.sh -V 2>&1 | tail -2
done
echo JOB_DONE