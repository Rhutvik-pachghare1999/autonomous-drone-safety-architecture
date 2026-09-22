#!/usr/bin/env python3
"""Plot the real 500-episode Crazyflie A/B from the committed JSON (no fabricated data).
Left: survival-rate bars with Wilson 95% CI (asymmetric, clamped >=0 so errorbar is valid).
Right: min-altitude distribution (filter OFF vs ON) vs the ground line.
Usage: python3 plot_crazyflie_ab_survival.py --json <ab500.json> --out <png>"""
import argparse, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return p, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.json))
    n = d["n_episodes_per_mode"]
    off, on = d["filter_OFF"], d["filter_ON"]

    p_off, lo_off, hi_off = wilson(off["survived"], n)
    p_on, lo_on, hi_on = wilson(on["survived"], n)
    # asymmetric error bars, clamped non-negative (fixes 'yerr must not be negative')
    yerr = np.array([[max(0.0, p_off - lo_off), max(0.0, p_on - lo_on)],
                     [max(0.0, hi_off - p_off), max(0.0, hi_on - p_on)]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    bars = ax1.bar(["Filter OFF", "Filter ON"], [p_off * 100, p_on * 100],
                   color=["#c0392b", "#27ae60"], yerr=yerr * 100, capsize=8)
    ax1.set_ylabel("Survival rate (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title(f"Crazyflie 2.X HOCBF A/B — {n} paired episodes\n"
                  f"Isaac Sim GPU PhysX (simulation only)")
    for b, k in zip(bars, [off["survived"], on["survived"]]):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                 f"{k}/{n}", ha="center", fontsize=10, fontweight="bold")

    off_z = off.get("min_z_samples", [])
    on_z = on.get("min_z_samples", [])
    ax2.hist(off_z, bins=30, alpha=0.6, color="#c0392b", label="OFF (crashes)")
    ax2.hist(on_z, bins=30, alpha=0.6, color="#27ae60", label="ON (survives)")
    ax2.axvline(0.0, color="k", ls="--", lw=1, label="ground")
    ax2.set_xlabel("min altitude reached (m)")
    ax2.set_ylabel("episodes")
    ax2.set_title("Min-altitude distribution")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"Wrote {a.out}")
    print(f"OFF survival {p_off:.1%} ({off['survived']}/{n}), "
          f"ON survival {p_on:.1%} ({on['survived']}/{n})")


if __name__ == "__main__":
    main()
