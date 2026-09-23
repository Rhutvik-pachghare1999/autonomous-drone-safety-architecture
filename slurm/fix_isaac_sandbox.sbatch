#!/bin/bash
#SBATCH --job-name=fix-isaac
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Fix (once): both Isaac SIFs on Sol contain /isaac-sim with mode 0750 owned
# by root -> "stat /isaac-sim/python.sh: permission denied" for our userns
# user (dbg6 probe, job 63853654). Extract the 6.0.1 SIF to a sandbox dir
# under fakeroot, make /isaac-sim world-readable+executable, verify, then
# point slurm/12_crazyflie_iso.sbatch at the sandbox dir (apptainer exec
# accepts sandbox dirs directly; no need to repack a 10GB SIF).
set -ex
module purge
module load apptainer/1.4.5

DIR=/scratch/$USER/isaac-sim-6.0.1.dir
rm -rf "$DIR"
apptainer build --sandbox --fakeroot "$DIR" /scratch/$USER/isaac-sim-6.0.1.sif
chmod -R a+rX "$DIR/isaac-sim"
ls -ld "$DIR/isaac-sim" "$DIR/isaac-sim/python.sh"

# verify exec from the sandbox dir (same flags the A/B uses)
apptainer exec --nv \
  --home /scratch/$USER/solhome \
  --bind /scratch/$USER/autonomous-drone-safety-architecture:/repo,/tmp:/tmp \
  --env OMNI_KIT_ALLOW_ROOT=1 \
  "$DIR" /isaac-sim/python.sh -c "import sys; print('PY_OK', sys.version)"
echo FIX_DONE