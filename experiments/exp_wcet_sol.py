"""
Sol WCET benchmark analysis
===========================

Computes WCET / latency statistics from a raw CSV produced by
`src/rt/safety_filter.c` running on an ASU Sol compute node (see
`slurm/20_wcet_sol.sbatch`). Outputs a single JSON artifact
(`experiments/results/wcet_sol.json`) that records:

  - empirical percentiles (min / mean / p50 / p99 / p99.9 / p99.99 / max)
  - EVT Gumbel tail bound at P=1e-9 (Block Maxima method, reused from
    experiments/exp_wcet_evt.py — the standard avionics method)
  - OS inter-cycle jitter (max, p99)
  - machine spec of the Sol compute node (from a spec file written by the
    sbatch job: CPU model, kernel, gcc, pinned core, Slurm job id)

HONESTY NOTE: this is a Linux USERSPACE measurement on shared cluster
hardware. On Sol the Slurm job runs unprivileged, so `sched_setscheduler`
(SCHED_FIFO) and `mlockall` fail with EPERM — the run measures the
arithmetic cost pinned to one allocated core, WITHOUT real-time scheduling
isolation. It is NOT an RTOS or flight-controller WCET number. The RT-run
reference number (2,725 ns max outlier) remains the committed
`experiments/results/latency_raw.csv` captured on the author's laptop under
SCHED_FIFO + mlockall (see docs/WCET_BENCHMARK.md).

Usage:
    python experiments/exp_wcet_sol.py --csv <latency_raw.csv> \
        --spec-file <machine_spec.txt> --out-json <wcet_sol.json> [--png out.png]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# Reuse the Block-Maxima Gumbel fit from the existing EVT experiment so the
# Sol analysis uses the exact same statistical method as the committed one.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_wcet_evt import gumbel_fit  # noqa: E402

DEADLINE_NS = 100_000  # 100 µs hard deadline (matches exp_wcet_evt.py)


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    lats, jits = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lats.append(int(row["latency_ns"]))
            jits.append(int(row.get("jitter_ns", 0)))
    return np.array(lats), np.array(jits)


def load_spec(path: str | None) -> dict:
    spec = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if ": " in line:
                    k, _, v = line.partition(": ")
                    spec[k.strip()] = v.strip()
    return spec


def run(csv_path: str, spec_file: str | None, out_json: str, png_path: str | None) -> dict:
    print("=" * 60)
    print("Sol WCET analysis — HOCBF safety_filter.c on ASU Sol compute node")
    print("=" * 60)

    lats, jits = load_csv(csv_path)
    n = len(lats)
    if n < 1000:
        print(f"WARNING: only {n} samples — EVT tail fit not meaningful")
    print(f"\nLoaded {n:,} latency samples from {csv_path}")

    spec = load_spec(spec_file)
    if spec:
        print("\n── Machine spec (Sol compute node) ─────────────────────────")
        for k, v in spec.items():
            print(f"  {k}: {v}")

    p = lambda q: float(np.percentile(lats, q))
    empirical = {
        "n_trials": n,
        "min_ns": int(lats.min()),
        "mean_ns": float(lats.mean()),
        "p50_ns": p(50),
        "p99_ns": p(99),
        "p999_ns": p(99.9),
        "p9999_ns": p(99.99),
        "wcet_ns": int(lats.max()),
        "deadline_ns": DEADLINE_NS,
        "deadline_pass": bool(lats.max() < DEADLINE_NS),
    }

    print("\n── Empirical percentiles (HOCBF hot path) ──────────────────")
    print(f"  N          : {n:,}")
    print(f"  Min        : {empirical['min_ns']} ns")
    print(f"  Mean       : {empirical['mean_ns']:.1f} ns")
    print(f"  P50        : {empirical['p50_ns']:.0f} ns")
    print(f"  P99        : {empirical['p99_ns']:.0f} ns")
    print(f"  P99.9      : {empirical['p999_ns']:.0f} ns")
    print(f"  P99.99     : {empirical['p9999_ns']:.0f} ns")
    print(f"  WCET (max) : {empirical['wcet_ns']} ns  ({empirical['wcet_ns']/1000:.2f} µs)")
    print(f"  < 100µs    : {'PASS' if empirical['deadline_pass'] else 'FAIL'}")

    fit = gumbel_fit(lats)
    evt = {
        "method": "block_maxima_gumbel",
        "block_size": fit["block_size"],
        "n_blocks": fit["n_blocks"],
        "gumbel_mu_ns": fit["mu"],
        "gumbel_beta_ns": fit["beta"],
        "p999_bound_ns": fit["p999_ns"],
        "p9999_bound_ns": fit["p9999_ns"],
        "p99999_bound_ns": fit["p99999_ns"],
        "p1e9_bound_ns": fit["p1e9_ns"],
        "p1e9_pass": bool(fit["p1e9_ns"] < DEADLINE_NS),
    }

    print(f"\n── EVT Gumbel tail extrapolation (Block Maxima, block={fit['block_size']}) ──")
    print(f"  Blocks          : {fit['n_blocks']} × {fit['block_size']} samples")
    print(f"  Block-max mean  : {fit['maxima_mean']:.1f} ns")
    print(f"  Block-max max   : {fit['maxima_max']:.1f} ns  (= empirical WCET)")
    print(f"  Gumbel µ        : {fit['mu']:.2f} ns")
    print(f"  Gumbel β        : {fit['beta']:.2f} ns")
    print(f"  P=1e-3 bound    : {fit['p999_ns']:.1f} ns")
    print(f"  P=1e-4 bound    : {fit['p9999_ns']:.1f} ns")
    print(f"  P=1e-9 bound    : {fit['p1e9_ns']:.1f} ns  ({fit['p1e9_ns']/1000:.3f} µs)")
    print(f"  < 100µs (P=1e-9): {'PASS' if evt['p1e9_pass'] else 'FAIL'}")

    jitter = {
        "max_jitter_ns": int(jits.max()),
        "p99_jitter_ns": float(np.percentile(jits, 99)),
        "jitter_pass_50us": bool(jits.max() < 50_000),
    }
    print("\n── OS inter-cycle jitter (deviation proxy) ─────────────────")
    print(f"  Max jitter : {jitter['max_jitter_ns']} ns  ({jitter['max_jitter_ns']/1000:.1f} µs)")
    print(f"  P99 jitter : {jitter['p99_jitter_ns']:.0f} ns")
    print(f"  < 50µs     : {'PASS' if jitter['jitter_pass_50us'] else 'WARN'}")

    result = {
        "measurement_context": (
            "Linux USERSPACE measurement on an ASU Sol compute node; Slurm job "
            "ran unprivileged so SCHED_FIFO/mlockall were unavailable (EPERM). "
            "Process pinned to one allocated core via taskset. This is NOT an "
            "RTOS / flight-controller WCET number."
        ),
        "source_csv": csv_path,
        "machine": spec,
        "empirical": empirical,
        "evt": evt,
        "jitter": jitter,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {out_json}")

    if png_path:
        try:
            _plot(lats, jits, fit, empirical, evt, spec, png_path)
        except ImportError:
            print("PNG skipped (matplotlib/scipy not importable in this python)")
    return result


def _plot(lats, jits, fit, empirical, evt, spec, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gumbel_r

    cpu = spec.get("cpu_model", "Sol compute node")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"HOCBF Safety Filter — WCET on ASU Sol  (N={len(lats):,} trials, "
        f"userspace, pinned core)\n{cpu}",
        fontsize=10,
    )

    lats_us = lats / 1000.0

    ax = axes[0]
    ax.hist(lats_us, bins=200, color="steelblue", alpha=0.7,
            density=True, label="Measured latency")
    x_fit = np.linspace(0, lats_us.max() * 1.5, 500)
    pdf = gumbel_r.pdf(x_fit, loc=fit["mu"] / 1000, scale=fit["beta"] / 1000)
    ax.plot(x_fit, pdf, "r-", linewidth=2, label="Gumbel EVT fit")
    ax.axvline(100, color="red", linestyle="--", linewidth=2, label="100µs deadline")
    ax.axvline(empirical["wcet_ns"] / 1000, color="orange", linestyle=":",
               linewidth=1.5, label=f"WCET={empirical['wcet_ns']/1000:.2f}µs")
    ax.set_xlabel("Latency (µs)")
    ax.set_ylabel("Density")
    ax.set_title("Latency Distribution + Gumbel Fit (Sol)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    sorted_lats = np.sort(lats_us)
    exceedance = np.maximum(1.0 - np.arange(1, len(sorted_lats) + 1) / len(sorted_lats), 1e-10)
    ax2.semilogy(sorted_lats, exceedance, "steelblue", linewidth=1.5, label="Empirical CCDF")
    p_range = np.logspace(-2, -9, 200)
    t_evt = fit["mu"] / 1000 - fit["beta"] / 1000 * np.log(-np.log(1 - p_range))
    ax2.semilogy(t_evt, p_range, "r--", linewidth=2, label="Gumbel EVT extrapolation")
    ax2.axvline(100, color="red", linestyle="--", linewidth=2, label="100µs deadline")
    ax2.axhline(1e-9, color="gray", linestyle=":", linewidth=1, label="P=1e-9 (DO-178C DAL-A)")
    ax2.plot(evt["p1e9_bound_ns"] / 1000, 1e-9, "r*", markersize=12,
             label=f"EVT bound={evt['p1e9_bound_ns']/1000:.3f}µs")
    ax2.set_xlabel("Latency (µs)")
    ax2.set_ylabel("Exceedance probability")
    ax2.set_title("CCDF Tail + EVT Extrapolation (Sol, log scale)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    ax3 = axes[2]
    jit_us = jits[jits > 0] / 1000.0
    if len(jit_us) > 0:
        ax3.hist(jit_us, bins=100, color="seagreen", alpha=0.7)
    ax3.axvline(50, color="red", linestyle="--", linewidth=2, label="50µs jitter SLA")
    ax3.axvline(jits.max() / 1000, color="orange", linestyle=":", linewidth=1.5,
                label=f"Max={jits.max()/1000:.1f}µs")
    ax3.set_xlabel("Inter-cycle jitter (µs)")
    ax3.set_ylabel("Count")
    ax3.set_title("OS Scheduler Jitter (Sol, userspace)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"Plot saved: {png_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="latency_raw.csv from safety_filter.c")
    ap.add_argument("--spec-file", default=None, help="machine spec file (key: value lines)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    run(a.csv, a.spec_file, a.out_json, a.png)


if __name__ == "__main__":
    main()
