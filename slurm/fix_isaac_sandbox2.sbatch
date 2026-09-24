#!/bin/bash
#SBATCH --job-name=fix-isaac2
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Fix (second attempt): 'apptainer build --sandbox' needs unsquashfs on the
# host (absent -> job 63855291 failed). Instead exec the SIF under
# --fakeroot (container-root can traverse /isaac-sim which is mode 0750)
# and tar-stream the whole rootfs out onto a scratch directory. Files
# written by container-root land owned by the calling user (fakeroot
# mapping), so a plain host chmod a+rX then fixes the root-only /isaac-sim.
# apptainer exec accepts a sandbox DIRECTORY directly — no SIF repack.
set -ex
module purge
module load apptainer/1.4.5

OUT=/scratch/$USER/isaac601.dir
rm -rf "$OUT"
mkdir -p "$OUT"

apptainer exec --fakeroot --bind "$OUT":/out /scratch/$USER/isaac-sim-6.0.1.sif \
  bash -c 'tar -C / --exclude=./proc --exclude=./sys --exclude=./dev --exclude=./out -cf - . | tar -C /out -xf -'

chmod -R a+rX "$OUT"
ls -ld "$OUT/isaac-sim" "$OUT/isaac-sim/python.sh"

# verify exec from the sandbox dir with the same flags the A/B uses
apptainer exec --nv \
  --home /scratch/$USER/solhome \
  --bind /scratch/$USER/autonomous-drone-safety-architecture:/repo,/tmp:/tmp \
  --env OMNI_KIT_ALLOW_ROOT=1 \
  "$OUT" /isaac-sim/python.sh -c "import sys; print('PY_OK', sys.version)"
echo FIX_DONE