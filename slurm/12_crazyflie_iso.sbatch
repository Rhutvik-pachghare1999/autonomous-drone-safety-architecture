#!/bin/bash
#SBATCH --job-name=cf-iso
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# ---------------------------------------------------------------------------
# SINGLE-VARIABLE (isolation) A/B on the REAL Crazyflie 2.X (Isaac Sim 5.1
# GPU PhysX on Sol, apptainer container — same environment as the original
# headline run, slurm/README_SOL.md). Responds to the attribution critique:
# in the earlier A/B (hocbf_crazyflie_ab100.py) the ON arm also swapped the
# controller's nominal law and zeroed the attitude moments, so OFF-vs-ON
# measured "controller swap + CBF" not the filter alone. Here BOTH arms use
# CrazyflieController.compute_wrench() for the nominal (T, tx, ty); the ONLY
# difference is whether T passes through filter_thrust (CBF safe-set clamp)
# before the shared mixer. Same seed, same paired randomization as the old
# headline (start_z, dive_vz, onset, vz0, wind, delay; mass NOT randomized).
# Writes experiments/results/crazyflie_vla_iso<N>.json (+ plot PNG).
# SIMULATION ONLY — not a hardware drone. Adversarial dive is scripted.
# Submit:  sbatch --export=ALL,N_EP=500 slurm/12_crazyflie_iso.sbatch
# ---------------------------------------------------------------------------
set -e
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-5.1.sif
[ -f "$SIF" ] || { echo "SIF_MISSING: $SIF"; exit 1; }
REPO=/scratch/$USER/autonomous-drone-safety-architecture
cd "$REPO"
export N_EP=${N_EP:-500}
mkdir -p experiments/results

run() {
  apptainer exec --nv \
    --home /scratch/$USER/solhome \
    --bind /scratch/$USER/autonomous-drone-safety-architecture:/repo,/tmp:/tmp,/scratch/$USER/pybind11_include:/pybind \
    --env CF_USD=/repo/assets/crazyflie.usd,XDG_CACHE_HOME=/scratch/$USER/.cache,HF_HOME=/scratch/$USER/.hf,OMNI_KIT_ALLOW_ROOT=1 \
    "$SIF" "$@"
}

echo "Running $N_EP-episode/mode ISOLATION A/B on $(hostname), GPU=$CUDA_VISIBLE_DEVICES"

# HOCBF python extension: compile in-job iff not already importable
# (pybind11 headers staged at /scratch/$USER/pybind11_include from the login
# node; compute nodes have no internet). Falls back to the verified pure-
# Python port (sim/hocbf_py.py) automatically if the extension is absent.
if ! run /isaac-sim/python.sh -c "import sys; sys.path.insert(0,'/repo/sim'); import hocbf" 2>/dev/null; then
  echo "Building hocbf extension in-job..."
  run /isaac-sim/python.sh -c "import sysconfig; print(sysconfig.get_paths()['include'])" > /tmp/pyinc_$$.txt
  PYINC=$(tail -1 /tmp/pyinc_$$.txt)
  run bash -c "g++ -O2 -std=c++17 -shared -fPIC -DBUILD_PYTHON_BINDINGS \
      -I/repo/src/control -I/pybind -I$PYINC /repo/src/control/hocbf.cpp \
      -o /repo/sim/hocbf.cpython-311-x86_64-linux-gnu.so && echo HOCBF_BUILT"
  run /isaac-sim/python.sh -c "import sys; sys.path.insert(0,'/repo/sim'); import hocbf; print('hocbf import OK')"
fi

run /isaac-sim/python.sh /repo/sim/hocbf_crazyflie_iso.py "$N_EP"

# regenerate the plot strictly from the JSON we just wrote (no fabricated data)
run /isaac-sim/python.sh /repo/experiments/plot_crazyflie_ab_large.py \
      --json /repo/experiments/results/crazyflie_vla_iso${N_EP}.json \
      --out  /repo/experiments/results/crazyflie_vla_iso${N_EP}.png \
  || echo "PLOT_SKIPPED_ON_CLUSTER (regenerate locally from the JSON)"
echo "DONE — artifacts in experiments/results/ (commit from a login node)"
