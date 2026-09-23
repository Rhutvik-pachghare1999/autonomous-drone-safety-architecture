#!/bin/bash
#SBATCH --job-name=isaac6-pull
#SBATCH --partition=public
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/isaac6-pull-%j.out
# ---------------------------------------------------------------------------
# One-off: pull Isaac Sim 6.0.1 container for Phase-5 RTX rendering.
# WHY (documented, issue isaac-sim/IsaacSim#677): Isaac Sim 5.1.0's validated
# Linux driver is 580.65.06; Sol compute nodes run 595.71.05 (R595 branch),
# which SEGFAULTS 5.1's librtx.scenedb.plugin.so at first scene render
# (reproduced on sg039 in /scratch/$USER/dbg4-63829568.out).  Isaac Sim 6.0.0+
# runs the RTX renderer correctly on the SAME 595.71.05 driver.
# 5.1 stays the version for WCET + 1000-ep A/B (rendering-independent, already
# committed — NOT migrated/re-run).
# Compute node only: SIF build via apptainer pull (mksquashfs) is heavy;
# /scratch has internet on compute nodes (verified in prep jobs).
# ---------------------------------------------------------------------------
set -euo pipefail
module purge
module load apptainer/1.4.5

OUT=/scratch/$USER/isaac-sim-6.0.1.sif
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer_cache
mkdir -p "$APPTAINER_CACHEDIR"

echo "=== pull on $(hostname) -> $OUT ==="
df -hT /scratch | tail -1
apptainer pull --force "$OUT" docker://nvcr.io/nvidia/isaac-sim:6.0.1
ls -la "$OUT"
echo "PULL_RESULT: OK"
