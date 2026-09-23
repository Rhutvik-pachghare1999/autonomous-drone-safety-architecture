#!/bin/bash
#SBATCH --job-name=isaac6-inventory
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/isaac6-inventory-%j.out
#SBATCH --exclude=sg048,sg049
# ---------------------------------------------------------------------------
# Inventory ONLY (no render): what knobs does Isaac Sim 6.0.1's SimulationApp
# offer for headless+camera rendering, and which camera/headless experience
# .kit files ship in the container?  Feeds the next compat iteration.
# ---------------------------------------------------------------------------
set -euo pipefail
module purge
module load apptainer/1.4.5
SIF=/scratch/$USER/isaac-sim-6.0.1.sif
REPO=/scratch/$USER/autonomous-drone-safety-architecture

run() {
  apptainer exec --nv --fakeroot \
    --home /scratch/$USER/solhome \
    --bind "$REPO":/repo,/tmp:/tmp \
    --env OMNI_KIT_ALLOW_ROOT=1 \
    "$SIF" "$@"
}

echo "===== apps dir (headless/remote/render related) ====="
run bash -c 'ls /isaac-sim/apps/ | grep -iE "headless|remote|render|stream" || ls /isaac-sim/apps/'

echo "===== 6.0 SimulationApp signature ====="
run /isaac-sim/python.sh -c "
import inspect
from isaacsim import SimulationApp
sig = inspect.signature(SimulationApp.__init__)
print('SIMAPP_SIG:', sig)
doc = (SimulationApp.__init__.__doc__ or '')[:3000]
print('SIMAPP_DOC:', doc)
"

echo "INVENTORY_DONE"
