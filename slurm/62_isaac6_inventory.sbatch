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

echo "===== apps dir (full) ====="
run bash -c 'ls /isaac-sim/apps/'

echo "===== SimulationApp __init__ source (default experience resolution) ====="
run /isaac-sim/python.sh -c "
import inspect
import isaacsim.simulation_app.simulation_app as m
src = inspect.getsource(m.SimulationApp.__init__)
import re
keep = [l for l in src.splitlines() if re.search(r'experience|headless|exp\.|app\.', l)]
print('SIMAPP_INIT_EXPERIENCE_LINES:')
print('\n'.join(keep[:60]))
"

echo "===== base python kit package block ====="
run bash -c 'for f in /isaac-sim/apps/isaacsim.exp.base.python.kit /isaac-sim/apps/isaacsim.exp.base.kit /isaac-sim/apps/isaacsim.exp.full.kit; do [ -f "$f" ] && { echo "--- $f"; sed -n "1,40p" "$f"; }; done'

echo "INVENTORY_DONE"
