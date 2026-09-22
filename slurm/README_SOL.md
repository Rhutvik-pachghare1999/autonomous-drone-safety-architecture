# Running on ASU Sol (Slurm GPU cluster)

Verified cluster facts (2026-09-22):
- Login: `ssh sol` (sol.asu.edu). Login node = sol-login03. **Never run heavy work on the
  login node — submit Slurm jobs.**
- Scheduler: Slurm. GPU partitions: `public` (`general` is now privately-owned. GPUs: A100 (40/80GB), A30, L40, H100.
  Request e.g. `--gres=gpu:a100:1`. Max walltime 7–14 days.
- Isaac Sim: `module load isaacsim/5.0` → adds `/packages/apps/isaacsim/5.0/kit` to PATH;
  use `python.sh` / `isaacsim.sh` from there.
- Containers: `apptainer/1.4.5` (fallback for Isaac 5.1 if 5.0 GPU init fails).
- Storage: `/home` = 100 GB (nearly full — DO NOT put data/models/caches here).
  `/scratch` = 3.7 PB BeeGFS — clone the repo, HF cache, datasets, and results here.
- Python envs: `module load mamba/latest`. cudnn cuda12/13 available.

## First-time setup (run on a login node — cheap, no GPU)
```bash
ssh sol
mkdir -p /scratch/$USER && cd /scratch/$USER
git clone -b crazyflie-vla-sim https://github.com/Rhutvik-pachghare1999/autonomous-drone-safety-architecture.git
cd autonomous-drone-safety-architecture
export XDG_CACHE_HOME=/scratch/$USER/.cache HF_HOME=/scratch/$USER/.hf
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
