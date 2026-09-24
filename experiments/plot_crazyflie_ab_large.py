#!/usr/bin/env python3
"""Plot the LARGE real-Crazyflie HOCBF A/B (N paired, randomized episodes).

Reads ONLY a committed artifact experiments/results/crazyflie_vla_ab<N>.json
(produced by sim/hocbf_crazyflie_ab100.py on the real Crazyflie 2.X USD,
GPU PhysX @50 Hz, paired domain randomization: start_z, dive_vz, onset, vz0,
horizontal wind force, command delay). No data is modified or fabricated.

Run:  python3 experiments/plot_crazyflie_ab_large.py [--json PATH] [--out PATH]
Out:  experiments/results/crazyflie_vla_ab<N>.png
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    src = args.json or os.path.join(REPO, "experiments", "results",
                                    "crazyflie_vla_ab1000.json")
    out = args.out or src.replace(".json", ".png")

    with open(src) as fh:
        d = json.load(fh)
    n = d["n_episodes_per_mode"]
    off, on = d["filter_OFF"], d["filter_ON"]
    rand = d["randomization"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    # --- 1: survival rates with Wilson 95% CI -------------------------------
    ax = axes[0]
    labels = ["Filter OFF", "Filter ON (HOCBF)"]
    rates = [off["survival_rate"], on["survival_rate"]]
    lo_hi = [wilson_ci(off["survived"], n), wilson_ci(on["survived"], n)]
    # max(0, ..): Wilson bounds can land a float-epsilon outside [0, r]
    errs = [[max(0.0, r - l) for r, (l, _) in zip(rates, lo_hi)],
            [max(0.0, u - r) for r, (_, u) in zip(rates, lo_hi)]]
    bars = ax.bar(labels, [100 * r for r in rates],
                  color=["#c0392b", "#1e7f4f"], alpha=0.88, width=0.55)
    ax.errorbar(labels, [100 * r for r in rates],
                yerr=[[100 * e for e in errs[0]], [100 * e for e in errs[1]]],
                fmt="none", ecolor="k", capsize=6, lw=1.6)
    for b, (k, _) in zip(bars, [(off["survived"], n), (on["survived"], n)]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4,
                f"{k}/{n}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_ylabel("survival rate (%)")
    ax.set_title(f"Paired A/B, {n} episodes/mode (Wilson 95% CI)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # --- 2: min-altitude distributions --------------------------------------
    ax = axes[1]
    ax.hist(off["min_z_samples"], bins=40, color="#c0392b", alpha=0.65,
            label=f"OFF (mean {off['mean_min_z_m']:.2f} m)", density=True)
    ax.hist(on["min_z_samples"], bins=40, color="#1e7f4f", alpha=0.65,
            label=f"ON  (mean {on['mean_min_z_m']:.2f} m)", density=True)
    ax.axvline(0.03, color="k", ls=":", lw=1)
    ax.text(0.045, ax.get_ylim()[1] * 0.9, "ground", fontsize=8, color="#555",
            rotation=90, va="top")
    ax.set_xlabel("minimum altitude per episode (m)")
    ax.set_ylabel("density")
    ax.set_title("Per-episode minimum altitude", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- 3: filtered fraction distribution (ON arm) --------------------------
    ax = axes[2]
    ax.hist(on["filtered_frac_samples"], bins=40, color="#1e7f4f", alpha=0.8)
    ax.set_xlabel("fraction of steps where HOCBF modified the command")
    ax.set_ylabel("episodes")
    ax.set_title(f"Filter activity, ON arm (mean {on['mean_filtered_frac']:.0%})",
                 fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Real Crazyflie 2.X (27 g) HOCBF A/B — {n} paired episodes "
        f"(simulation only, ISAAC GPU PhysX 50 Hz)\n"
        f"randomized: start_z{rand['start_z']}m, dive_vz{rand['dive_vz']}m/s, "
        f"onset{rand['onset']}s, vz0{rand.get('vz0')}m/s, "
        f"wind_N{rand.get('wind_mag_N')}, delay{rand.get('delay_steps')}steps "
        f"— adversarial dive is scripted (seed {d['seed']})\n"
        f"platform: {d.get('platform', 'n/a')} | source: {os.path.basename(src)}",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    print(f"[OFF] {off['survived']}/{n} survived "
          f"({100*off['survival_rate']:.1f}%, "
          f"95% CI [{100*lo_hi[0][0]:.1f}, {100*lo_hi[0][1]:.1f}])")
    print(f"[ON ] {on['survived']}/{n} survived "
          f"({100*on['survival_rate']:.1f}%, "
          f"95% CI [{100*lo_hi[1][0]:.1f}, {100*lo_hi[1][1]:.1f}])")


if __name__ == "__main__":
    main()
