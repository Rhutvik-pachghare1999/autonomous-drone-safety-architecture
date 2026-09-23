#!/bin/bash
#SBATCH --job-name=dbg-vk
#SBATCH --partition=public,htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/%u/dbg-vulkan-%j.out
#
# One-off diagnostics: why does carb::graphics::createInstance fail under
# apptainer on Sol compute nodes?  Prints everything needed to point
# VK_ICD_FILENAMES / VK_DRIVER_FILES at a working NVIDIA vulkan ICD.
# NOT part of the reproducible pipeline — delete once resolved.

set -uo pipefail
module purge
module load apptainer/1.4.5

SIF=/scratch/$USER/isaac-sim-5.1.sif

echo "===================== HOST ====================="
hostname; nvidia-smi -L; nvidia-smi -q | grep -A3 "Virtualization\|Persistence\|Compute Mode" | head -12
echo "--- host icd dirs ---"
ls -la /usr/share/vulkan/icd.d/ 2>&1
ls -la /etc/vulkan/icd.d/ 2>&1
ls /etc/vulkan/implicit_layer.d/ 2>&1 | head -5
for f in /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json; do
  [ -f "$f" ] && echo "-- $f:" && cat "$f"; done
echo "--- host GL/VK libs ---"
ls -la /usr/lib/x86_64-linux-gnu/libGLX_nvidia* /usr/lib/x86_64-linux-gnu/libnvidia-*vulkan* 2>&1 | head

echo "===================== CONTAINER ====================="
apptainer exec --fakeroot --nv \
  --home /scratch/$USER/solhome \
  --bind /scratch:/scratch \
  --bind /tmp:/tmp \
  --env XDG_CACHE_HOME=/tmp/xdg_$SLURM_JOB_ID \
  --env OMNI_KIT_ALLOW_ROOT=1 \
  "$SIF" bash -c '
set -u
echo "--- container nvidia bind libs ---"
ls /.singularity.d/libs/ 2>/dev/null | head -30
echo "count: $(ls /.singularity.d/libs/ 2>/dev/null | wc -l)"
echo "--- container vulkan loader + ICD search dirs ---"
find / -maxdepth 4 -name "*nvidia_icd*" 2>/dev/null | head
ls -la /usr/share/vulkan/icd.d/ 2>&1
ls /isaac-sim/kit/ | grep -i vulkan; find /isaac-sim -maxdepth 3 -name "libvulkan*" 2>/dev/null | head
echo "--- ld cache: vulkan/GLX ---"
ldconfig -p | grep -iE "vulkan|GLX_nvidia" | head
echo "--- relevant env ---"
env | grep -iE "^(VK_|__EGL|NVIDIA_DRIVER)" || echo "(none set)"
echo "--- EGL platforms ---"
ls /usr/share/glvnd/egl_vendor.d/ 2>&1; cat /usr/share/glvnd/egl_vendor.d/*.json 2>/dev/null | head
'
echo "DONE rc=$?"
