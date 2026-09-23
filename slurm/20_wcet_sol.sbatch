#!/bin/bash
#SBATCH --job-name=wcet-sol
#SBATCH --partition=public,htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/wcet_sol-%j.out
# ---------------------------------------------------------------------------
# TASK 1 — WCET / latency benchmark of src/rt/safety_filter.c on ASU Sol.
#
# WCET is a property of the C IMPLEMENTATION, and the benchmark needs NO
# Isaac Sim and NO GPU physics — it is a CPU hot-path measurement. The GPU
# allocation is only to satisfy Sol's fastest-scheduling path
# (`--gres=gpu:1 --mem>=24G`); the GPU itself is unused by the benchmark.
#
# The binary is compiled ON THE COMPUTE NODE with -march=native (never on
# the login node), run from a scratch RUNDIR so the committed
# experiments/results/latency_raw.csv (the laptop RT-run artifact of record)
# is NOT overwritten, pinned to one allocated core via taskset. The Slurm
# job is unprivileged → SCHED_FIFO/mlockall fail with EPERM → this is a
# Linux USERSPACE measurement, honestly labelled as such (NOT RTOS).
#
# Statistics + EVT Gumbel tail bound are computed by
# experiments/exp_wcet_sol.py using the container's python (numpy), written
# to $RUNDIR/experiments/results/wcet_sol.json. scp the JSON/CSV back to the
# laptop and commit from there.
#
# Submit (proof run):  sbatch --export=ALL,N_TRIALS=10000  slurm/20_wcet_sol.sbatch
# Submit (full run):   sbatch --export=ALL,N_TRIALS=100000 slurm/20_wcet_sol.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

module purge
# Need a C compiler ON THE COMPUTE NODE (container has none; build happens here)
module load gcc/12.5.0 2>/dev/null || module load gcc/15.2.0 2>/dev/null || true
command -v gcc >/dev/null || { echo "FATAL: no gcc available on compute node"; exit 1; }
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-5.1.sif
REPO=/scratch/$USER/autonomous-drone-safety-architecture
N_TRIALS=${N_TRIALS:-10000}

[ -f "$SIF" ] || { echo "SIF_MISSING: $SIF"; exit 1; }
[ -d "$REPO" ] || { echo "REPO_MISSING: $REPO"; exit 1; }

RUNDIR=/scratch/$USER/wcet_sol_run/job_${SLURM_JOB_ID}
mkdir -p "$RUNDIR/experiments/results"
cd "$RUNDIR"

echo "================================================================"
echo "WCET benchmark on Sol — $(hostname)"
echo "Slurm job: $SLURM_JOB_ID   N_TRIALS=$N_TRIALS"
echo "================================================================"

# ---- machine spec (written into the JSON artifact) -------------------------
CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')
AFFINITY_LIST=$(taskset -pc $$ | sed 's/.*list: //')
PIN_CPU=$(echo "$AFFINITY_LIST" | tr ',' '\n' | head -1 | cut -d- -f1)
GCC_VER=$(gcc --version | head -1)
{
  echo "hostname: $(hostname)"
  echo "slurm_job_id: ${SLURM_JOB_ID}"
  echo "cpu_model: ${CPU_MODEL}"
  echo "kernel: $(uname -sr)"
  echo "gcc: ${GCC_VER}"
  echo "nproc_visible: $(nproc)"
  echo "slurm_affinity_list: ${AFFINITY_LIST}"
  echo "pinned_cpu: ${PIN_CPU}"
  echo "n_trials: ${N_TRIALS}"
  echo "rt_scheduling: unavailable (unprivileged Slurm job — sched_setscheduler/mlockall EPERM expected)"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > machine_spec.txt
cat machine_spec.txt

# ---- compile on the compute node ---------------------------------------------
echo ""
echo "── Compiling safety_filter.c (gcc, -O3 -march=native -DNO_ONNX) ──"
gcc -O3 -march=native -DNO_ONNX \
    -o "$RUNDIR/safety_filter" \
    "$REPO/src/rt/safety_filter.c" "$REPO/src/rt/watchdog.c" -lm -lrt
echo "BUILD_OK"

# ---- run pinned to one allocated core -----------------------------------------
echo ""
echo "── Running ${N_TRIALS} trials pinned to CPU ${PIN_CPU} (taskset) ──"
set +e
taskset -c "$PIN_CPU" "$RUNDIR/safety_filter" "$N_TRIALS" "$PIN_CPU" \
    > bench.stdout 2> bench.stderr
RC=$?
set -e
echo "── bench stdout ──"
cat bench.stdout
echo "── bench stderr (RT-setup warnings expected under unprivileged Slurm) ──"
cat bench.stderr
echo "bench exit code: $RC"
[ -s experiments/results/latency_raw.csv ] || { echo "FATAL: no latency_raw.csv written"; exit 1; }
LINES=$(wc -l < experiments/results/latency_raw.csv)
echo "latency_raw.csv rows (incl header): $LINES"

# ---- statistics + EVT (container python: has numpy; matplotlib/scipy optional) ─
echo ""
echo "── Statistics + EVT Gumbel fit (container python) ──"
apptainer exec --fakeroot --nv \
  --bind /scratch:/scratch,/tmp:/tmp \
  --env XDG_CACHE_HOME=/scratch/$USER/.cache,HF_HOME=/scratch/$USER/.hf,OMNI_KIT_ALLOW_ROOT=1 \
  "$SIF" /isaac-sim/python.sh \
  "$REPO/experiments/exp_wcet_sol.py" \
    --csv        "$RUNDIR/experiments/results/latency_raw.csv" \
    --spec-file  "$RUNDIR/machine_spec.txt" \
    --out-json   "$RUNDIR/experiments/results/wcet_sol.json" \
    --png        "$RUNDIR/experiments/results/wcet_sol.png"

echo ""
echo "DONE — artifacts in $RUNDIR/experiments/results/"
echo "Next (login node / laptop): scp wcet_sol.json + latency_raw.csv back and commit."
