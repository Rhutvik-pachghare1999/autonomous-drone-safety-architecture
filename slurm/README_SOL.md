# Running on ASU Sol (Slurm GPU cluster)

**2026-09-23 sandbox note (supersedes the SIF paths below):** both shipped SIFs
(`isaac-sim-5.1.sif` rebuilt 09-22 15:37, `isaac-sim-6.0.1.sif` 09-23 09:31)
contain `/isaac-sim` as `root:nogroup 0750` → plain `apptainer exec` hits
`stat /isaac-sim/python.sh: permission denied` (probe: dbg6, job 63853654).
Two fixes, choose either:
1. `apptainer exec --fakeroot ...` — userns root bypasses the perms (this is
   what `slurm/20_ab_fakeroot.sbatch` did for the Phase-3b runs; cheap, no copy).
2. Tar-extracted sandbox dir `/scratch/$USER/isaac601.dir` (built by
   `slurm/fix_isaac_sandbox2.sbatch`, job 63856233; `apptainer build --sandbox`
   does NOT work — host lacks unsquashfs). Exec it with `--writable` (RTX
   shaderdb/kit cache need a writable `/isaac-sim`) and keep
   `/dev /proc /sys /repo /tmp` as existing dirs inside the sandbox (mount-hook
   targets — tar excluded them, re-created manually).
The isolation A/B (Phase 3c) uses option 2 (`slurm/12_crazyflie_iso.sbatch`).

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

## WCET / latency benchmark (CPU-only, no Isaac needed)
```bash
sbatch --export=ALL,N_TRIALS=100000 slurm/20_wcet_sol.sbatch
# compiles src/rt/safety_filter.c on the compute node (gcc -O3 -march=native),
# pins to one allocated core (taskset), writes wcet_sol.json + latency_raw.csv
# under /scratch/$USER/wcet_sol_run/job_<id>/ — scp small artifacts back and
# commit from the laptop. Userspace measurement (no SCHED_FIFO); see
# docs/WCET_BENCHMARK.md §3.
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

## Isaac Sim 6.0.1 (Phase 5 real-camera flight — ONLY under 6.0)

5.1 is retained for: WCET benchmark (TASK 1 numbers), large A/B runs, all
non-camera work. **6.0.1 is required for RTX camera rendering on Sol**: 5.1's
RTX stack segfaults on the 595.71.05 compute-node driver
(`librtx.scenedb.plugin.so`, slug `libtx.scenedb.plugin.so` in
`omni.hydra.pxr.settings` ui mode — log `dbg4-63829568.out`, upstream issue
IsaacSim#677). 6.0.1 = 2025.12 Long-Term-Support release with the fixed
driver contract.

Compat evidence (both `sbatch`-job verified, never login node):
```
gate (a) render:  job 63843563  ISAAC6_P3:v1: OK shape=(224,224,3) mean=176.31
                                  (probe3 = rep-orchestrator capture; bare
                                   headless SimulationApp config)
gate (b) A/B:     job 63842534  2-ep A/B unchanged on 6.0 (OFF 0/2, ON 2/2,
                                  hocbf_py guard) → ISAAC6_COMPAT_DONE
```
6.0.1 facts that differ from 5.1 and were learned the hard way (job refs in
slurm script comments):
- Container python is 3.12 (5.1 was 3.11) → separate dep tree:
  `/scratch/$USER/pylibs6` (5.1's `pylibs` is cp311 — not reusable).
- The kit ships **NO torch at all** (verified job 63844887; 5.1 shipped
  torch+torchvision). `slurm/vla_prep6.py` installs a self-contained
  `torch==2.9.1`+`torchvision==0.24.1` (PyPI cu128 wheels) **inside pylibs6**.
  Torch 2.14 CANNOT be used: its `_native` triton kernels need a host C
  compiler, and the SIF has none (`Failed to find C compiler`, job 63846729).
  2.9.1+cu128 additionally restores bitsandbytes 4-bit NF4 (bnb has no cu130
  binary for torch 2.14; job 63845630).
- Pylibs must NOT be on PYTHONPATH at kit startup (deprecation_manager then
  tries the dormant ml_archive prebundle torch → NCCL symbol clash → kit
  exits(0) silently, job 63847627). `sim/vla_realcam_flight.py` appends
  `$PYLIBS` to sys.path AFTER SimulationApp is up.
- Camera capture: `Camera.get_rgba()` returns EMPTY under headless on 6.0
  (probe3 v2/v3 EMPTY_FRAME). Working recipe (probe3 v1): manual
  `omni.replicator.core` render_product + rgb annotator +
  `rep.orchestrator.step()` until the annotator yields a non-empty frame.
  waitIdle/forceSerial in a custom .kit experience → abort
  "Destroying busy TaskGroup!" (job 63843303).
- Writable binds required for all kit caches (`/scratch/$USER/isaac_cache6/`);
  host parent dirs must exist AND be explicitly bound (unbound /scratch
  subpaths are EROFS inside the SIF).

Workflow:
```bash
sbatch slurm/32_vla_prep6.sbatch                      # one-time deps+weights (≈10 min)
sbatch --export=ALL,N_EP=2 slurm/33_vla_realcam6.sbatch  # 2-episode proof
sbatch --export=ALL,N_EP=6 slurm/33_vla_realcam6.sbatch  # demo run
# outputs: experiments/results/vla_realcam.{jsonl,summary.json...} + ep*_*.png
```
