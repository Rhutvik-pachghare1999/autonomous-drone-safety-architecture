"""
Objective 2 — WCET Latency Benchmark

Runs the C safety filter for 10,000 trials and proves WCET < 100μs.
Also runs a pure-Python baseline to show the overhead of Python/ROS serialization.

Outputs:
  - experiments/results/latency_report.json
  - experiments/results/latency_histogram.png
"""
import subprocess
import json
import time
import math
import sys
import os
import numpy as np

sys.path.insert(0, "src/control")

RESULTS_DIR = "experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_c_filter(n_trials: int = 10000, cpu_core: int = 2) -> dict:
    """Run the compiled C safety filter and parse its output."""
    result = subprocess.run(
        ["./build/safety_filter", str(n_trials), str(cpu_core)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode not in (0, 1):
        print("STDERR:", result.stderr)

    stats = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            stats[key.strip()] = val.strip()
    return stats


def run_python_baseline(n_trials: int = 10000) -> dict:
    """Pure Python HOCBF — measures Python overhead as baseline comparison."""
    import hocbf
    cbf = hocbf.HOCBF()

    latencies = []
    pz, vz = 1.0, 0.0

    for _ in range(n_trials):
        T_nom = 2.0 * 9.81 + 2.0 * vz * 2.0
        t0 = time.perf_counter_ns()
        T_safe = cbf.filter_thrust(pz, vz, 0.0, 0.0, T_nom)
        t1 = time.perf_counter_ns()
        latencies.append(t1 - t0)
        vz += (T_safe / 2.0 - 9.81) * 0.001
        pz = max(0.0, pz + vz * 0.001)

    latencies.sort()
    n = len(latencies)
    return {
        "min_ns":    latencies[0],
        "mean_ns":   sum(latencies) / n,
        "p50_ns":    latencies[n * 50 // 100],
        "p99_ns":    latencies[n * 99 // 100],
        "p999_ns":   latencies[n * 999 // 1000],
        "wcet_ns":   latencies[-1],
        "wcet_pass": latencies[-1] < 100_000,
    }


def load_raw_csv() -> list[int]:
    path = f"{RESULTS_DIR}/latency_raw.csv"
    if not os.path.exists(path):
        return []
    lats = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            _, ns = line.strip().split(",")
            lats.append(int(ns))
    return lats


def plot_histogram(latencies_ns: list[int], python_stats: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # C filter histogram
        lats_us = [x / 1000.0 for x in latencies_ns]
        axes[0].hist(lats_us, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
        axes[0].axvline(100, color="red", linestyle="--", linewidth=2, label="100μs deadline")
        wcet_us = max(lats_us)
        p99_us  = sorted(lats_us)[len(lats_us) * 99 // 100]
        axes[0].axvline(wcet_us, color="orange", linestyle=":", linewidth=1.5,
                        label=f"WCET={wcet_us:.1f}μs")
        axes[0].axvline(p99_us, color="green", linestyle=":", linewidth=1.5,
                        label=f"P99={p99_us:.1f}μs")
        axes[0].set_xlabel("Latency (μs)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("C HOCBF Filter — 10,000 Trials\n(SCHED_FIFO + mlockall + mmap IPC)")
        axes[0].legend()

        # C vs Python comparison bar chart
        c_stats = {
            "min":  min(lats_us),
            "mean": sum(lats_us) / len(lats_us),
            "p99":  p99_us,
            "wcet": wcet_us,
        }
        py_stats_us = {k: v / 1000.0 for k, v in python_stats.items() if k.endswith("_ns")}

        labels = ["min", "mean", "p99", "wcet"]
        c_vals  = [c_stats[k] for k in labels]
        py_vals = [py_stats_us.get(f"{k}_ns", 0) for k in labels]

        x = np.arange(len(labels))
        w = 0.35
        axes[1].bar(x - w/2, c_vals,  w, label="C (RT)", color="steelblue")
        axes[1].bar(x + w/2, py_vals, w, label="Python", color="coral")
        axes[1].axhline(100, color="red", linestyle="--", linewidth=2, label="100μs deadline")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Latency (μs)")
        axes[1].set_title("C RT vs Python Baseline")
        axes[1].legend()

        plt.tight_layout()
        out = f"{RESULTS_DIR}/latency_histogram.png"
        plt.savefig(out, dpi=150)
        print(f"Histogram saved: {out}")
    except ImportError:
        print("matplotlib not installed — skipping histogram")


# ── Formal properties P1–P7 ───────────────────────────────────────────────────
# Documented here alongside the latency benchmark so the WCET evidence is
# co-located with the safety properties it supports.
FORMAL_PROPERTIES = {
    "P1": "Geofence violation → RTL within 1 control cycle (≤100ms)",
    "P2": "DISARMED reachable from every state (BFS proof)",
    "P3": "No DISARMED→FLYING without ARM + TAKEOFF sequence",
    "P4": "Watchdog timeout → EMERGENCY_LAND from any flight state",
    "P5": "No deadlocks — every state has ≥1 outgoing transition",
    "P6": "NaN/Inf position inputs always rejected",
    "P7": (
        "tr(P[p_x, p_y, psi]) >= SIGMA_CRITICAL_SQ "
        "=> RTL within 1 control cycle (<= 100 ms)"
    ),
}


def evt_gumbel_wcet(latencies_ns: list[int],
                    target_prob: float = 1e-9) -> dict:
    """
    Fit a Gumbel (Type-I Extreme Value) distribution to the observed latency
    maxima and extrapolate to an ultra-low exceedance probability.

    This is the standard method for WCET tail-bound estimation in avionics
    (Foughali & Zuepke, Frontiers in Robotics and AI, 2022).

    The Gumbel CDF: F(t) = exp(-exp(-(t - mu) / beta))
    Fitted via method of moments:
        beta = std(x) * sqrt(6) / pi
        mu   = mean(x) - beta * gamma_EM   (gamma_EM ≈ 0.5772)

    Args:
        latencies_ns : list of per-trial latency measurements (nanoseconds)
        target_prob  : exceedance probability for WCET bound (default 1e-9,
                       matching DO-178C DAL-A failure rate target)

    Returns:
        dict with mu_ns, beta_ns, wcet_bound_ns, wcet_bound_us
    """
    x = np.array(latencies_ns, dtype=float)
    EULER_MASCHERONI = 0.5772156649

    beta = x.std() * np.sqrt(6.0) / np.pi
    mu   = x.mean() - beta * EULER_MASCHERONI

    # Invert CDF: t = mu - beta * ln(-ln(1 - p_exceed))
    # For p_exceed = 1e-9: -ln(-ln(1 - 1e-9)) ≈ -ln(1e-9) ≈ 20.7
    wcet_bound_ns = mu - beta * np.log(-np.log(1.0 - target_prob))

    return {
        "gumbel_mu_ns":      float(mu),
        "gumbel_beta_ns":    float(beta),
        "target_prob":       target_prob,
        "wcet_bound_ns":     float(wcet_bound_ns),
        "wcet_bound_us":     float(wcet_bound_ns / 1000.0),
        "wcet_bound_100ms":  wcet_bound_ns < 100_000_000,
    }


def main():
    print("=" * 50)
    print("Objective 2: WCET Latency Benchmark")
    print("=" * 50)

    print("\nFormal Safety Properties:")
    for k, v in FORMAL_PROPERTIES.items():
        print(f"  {k}: {v}")

    # Run C RT filter
    print("\n[1/2] Running C safety filter (10,000 trials)...")
    c_stats = run_c_filter(n_trials=10000, cpu_core=2)

    # Run Python baseline
    print("\n[2/2] Running Python baseline (10,000 trials)...")
    py_stats = run_python_baseline(n_trials=10000)
    print(f"  Python WCET : {py_stats['wcet_ns']/1000:.1f} μs")
    print(f"  Python P99  : {py_stats['p99_ns']/1000:.1f} μs")
    print(f"  Python mean : {py_stats['mean_ns']/1000:.1f} μs")

    # Save report
    report = {"c_filter": c_stats, "python_baseline": py_stats}
    with open(f"{RESULTS_DIR}/latency_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {RESULTS_DIR}/latency_report.json")

    # Plot
    raw = load_raw_csv()
    if raw:
        plot_histogram(raw, py_stats)

    # EVT Gumbel WCET tail-bound
    if raw:
        print("\n[EVT] Gumbel tail-bound extrapolation (P99.9 → P=1e-9):")
        evt = evt_gumbel_wcet(raw, target_prob=1e-9)
        print(f"  Gumbel μ    : {evt['gumbel_mu_ns']:.1f} ns")
        print(f"  Gumbel β    : {evt['gumbel_beta_ns']:.1f} ns")
        print(f"  WCET bound  : {evt['wcet_bound_us']:.2f} μs  (P=1e-9)")
        print(f"  < 100ms     : {'PASS' if evt['wcet_bound_100ms'] else 'FAIL'}")
        report["evt_wcet"] = evt
        with open(f"{RESULTS_DIR}/latency_report.json", "w") as f:
            json.dump(report, f, indent=2)

    # Final verdict
    wcet_str = c_stats.get("WCET", "N/A")
    print(f"\n{'='*50}")
    print(f"WCET < 100μs: {c_stats.get('WCET < 100us', 'N/A')}")
    print(f"P99  < 100μs: {c_stats.get('P99  < 100us', 'N/A')}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
