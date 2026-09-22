# Running on ASU Sol (Slurm GPU cluster)

Verified cluster facts (2026-09-22):
- Login: `ssh sol` (sol.asu.edu). Login node = sol-login03. **Never run heavy work on the
  login node — submit Slurm jobs.**
- Scheduler: Slurm. GPU partitions: `public` (`general` is now privately-owned. GPUs: A100 (40/80GB), A30, L40, H100.
  Request e.g. `--gres=gpu:a100:1`. Max walltime 7–14 days.
- Isaac Sim in-flight: `module load isaacsim/5.0` is ONLY the compatibility-check
  kit (verified 2026-09-22: `/packages/apps/isaacsim/5.0` has no `isaacsim` python
  package, no physics exts — `from isaacsim import SimulationApp` fails).
  PyPI install fails too: the wheel is `manylinux_2_35` but Sol is glibc 2.28
  (RHEL 8.10). **Working route: apptainer 1.4.5 + `nvcr.io/nvidia/isaac-sim:5.1.0`**
  (anonymous pull via Docker-v2 token confirmed 2026-09-22; no NGC login needed).
  Pull once on the login node to `/scratch/$USER/isaac-sim-5.1.sif`.
- Storage: `/home` = 100 GB (nearly full — DO NOT put data/models/caches here).
  `/scratch` = 3.7 PB BeeGFS — clone the repo, HF cache, datasets, and results here.
- Python envs: `module load mamba/latest`. cudnn cuda12/13 available.

## First-time setup (run on a login node — cheap, no GPU)
```bash
ssh sol
mkdir -p /scratch/$USER && cd /scratch/$USER
git clone -b crazyflie-vla-sim https://github.com/Rhutvik-pachghare1999/autonomous-drone-safety-architecture.git
export XDG_CACHE_HOME=/scratch/$USER/.cache HF_HOME=/scratch/$USER/.hf

# Isaac Sim 5.1 container (one-time, ~9 GB download; I/O only, login node OK).
# squashfs module required — mksquashfs is not on the login node by default:
module load apptainer/1.4.5 squashfs-4.6.1-gcc-11.2.0
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer APPTAINER_TMPDIR=/scratch/$USER/tmp
apptainer pull --name /scratch/$USER/isaac-sim-5.1.sif docker://nvcr.io/nvidia/isaac-sim:5.1.0

# pybind11 headers for the in-job HOCBF build (compute nodes have no internet):
cd /scratch/$USER
/scratch/$USER/isaac311/bin/pip download -q --no-deps pybind11 -d /tmp/pb \
  && unzip -o -q /tmp/pb/pybind11-*.whl -d /tmp/pbx \
  && mkdir -p pybind11_include && cp -r /tmp/pbx/pybind11/include/pybind11 pybind11_include/pybind11
```

## Step 0 — PROVE GPU PhysX works on Sol before any big run
```bash
sbatch slurm/00_gpu_verify.sbatch
# watch: squeue --me ; then read isaac-gpucheck-<jobid>.out
# PASS = "GPU_DEVICES: ['cuda:0']" + a falling z-trace + CLEAN_EXIT.
# FAIL (CUDA error 36 / cuDeviceGetUuid) = Isaac 5.0 Warp bug recurs on Sol →
#   use the apptainer Isaac 5.1 container instead (see below).
```

## Isaac 5.1 apptainer fallback (only if 5.0 GPU init fails)
```bash
cd /scratch/$USER
apptainer pull isaac-sim-5.1.sif docker://nvcr.io/nvidia/isaac-sim:5.1.0   # needs NGC login
# then run scripts with: apptainer exec --nv isaac-sim-5.1.sif <isaac python> ...
```

## Large A/B run
```bash
sbatch --export=ALL,N_EP=1000 slurm/10_crazyflie_ab_large.sbatch
squeue --me
# results -> experiments/results/crazyflie_vla_ab1000.json (+ plot)
```

## Committing results back
Do `git add/commit/push` from a **login node** (compute nodes have no internet).
Only commit small artifacts (JSON, PNG); keep raw traces/datasets on /scratch (gitignored).

## Honest scope (keep in all artifacts)
Simulation only (Isaac PhysX), NOT a hardware drone. The adversarial dive is a scripted
worst-case command (the 2.2B VLA refused dive prompts). Label every Sol artifact accordingly.
