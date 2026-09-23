#!/bin/bash
#SBATCH --job-name=inv6-torch
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/%u/inv6-torch-%j.out
#SBATCH --exclude=sg048,sg049
# Probe: does the 6.0.1 kit python actually ship torch/nvidia? (user directive
# assumed it does; prep6 job 63844741 printed FATAL container torch missing)
set -uo pipefail
module purge
module load apptainer/1.4.5
SIF=/scratch/$USER/isaac-sim-6.0.1.sif
CACHE=/scratch/$USER/isaac_cache6
apptainer exec --fakeroot --nv \
  --home /scratch/$USER/solhome \
  --bind /tmp:/tmp,$CACHE/ov:/root/.cache/ov \
  --env OMNI_KIT_ALLOW_ROOT=1 \
  "$SIF" bash -c '
echo "--- site-packages torch-ish entries:"
ls /isaac-sim/kit/python/lib/python3.12/site-packages/ 2>/dev/null | grep -iE "^(torch|nvidia|triton|transformers|accelerate)" || echo NO_TORCH_PKGS_IN_SITE_PACKAGES
echo "--- python.sh import torch:"
/isaac-sim/python.sh -c "import torch; print(\"TORCH_OK\", torch.__version__)" 2>&1 | tail -3 || true
'
echo INV_TORCH_DONE
