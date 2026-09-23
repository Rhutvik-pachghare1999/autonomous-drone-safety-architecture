#!/bin/bash
#SBATCH --job-name=dbg5
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -x
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-5.1.sif
run() {
  apptainer exec --nv \
    --home /scratch/$USER/solhome \
    --bind /scratch/$USER/autonomous-drone-safety-architecture:/repo,/tmp:/tmp \
    --env OMNI_KIT_ALLOW_ROOT=1 \
    "$SIF" "$@"
}

echo "=== external ===" && ls -la "$SIF" && id
echo "=== in-container id ===" && run id
echo "=== /isaac-sim listing ===" && run ls -la /isaac-sim 2>&1 | head -15
echo "=== stat python.sh ===" && run stat /isaac-sim/python.sh 2>&1
echo "=== python.sh -V ===" && run /isaac-sim/python.sh -V 2>&1
echo JOB_DONE