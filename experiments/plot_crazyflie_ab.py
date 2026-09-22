#!/usr/bin/env python3
"""Plot the real Crazyflie HOCBF A/B result.

Reads ONLY the committed artifact experiments/results/crazyflie_vla_ab.json
(produced by sim/hocbf_crazyflie_ab.py on the real NVIDIA Crazyflie 2.X USD,
Isaac Sim 5.1 GPU PhysX, 50 Hz) and renders altitude-vs-time for the two
worst-case VLA dive commands, filter OFF vs ON.

No data is modified or fabricated; traces are drawn verbatim from the JSON.

Run:  python3 experiments/plot_crazyflie_ab.py
Out:  experiments/results/crazyflie_vla_ab.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "experiments", "results", "crazyflie_vla_ab.json")
DST = os.path.join(REPO, "experiments", "results", "crazyflie_vla_ab.png")

TITLES = {
    "dive_full":   "Adversarial dive vz = -3.0 m/s (worst case)",
    "dive_medium": "Adversarial dive vz = -1.2 m/s",
}


def arm_xy(arm):
    tr = arm["trace"]
    return [r[0] for r in tr], [r[1] for r in tr]


def main() -> None:
    with open(SRC) as fh:
        data = json.load(fh)
    hp = data["hocbf_params"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, (name, scn) in zip(axes, sorted(data["scenarios"].items())):
        off, on = scn["OFF"], scn["ON"]

        # Filter OFF: full dive to crash
        t, z = arm_xy(off)
        ax.plot(t, z, color="#c0392b", lw=2.2,
                label=f"Filter OFF (crash t={off['crash_t_s']}s)")
        ax.plot([off["crash_t_s"]], [off["min_z_m"]], marker="x",
                markersize=13, markeredgewidth=3, color="#c0392b")
        ax.axvspan(off["crash_t_s"], ax.get_xlim()[1], color="#c0392b",
                   alpha=0.06)

        # Filter ON: HOCBF enforces safe thrust
        t, z = arm_xy(on)
        ax.plot(t, z, color="#1e7f4f", lw=2.2,
                label=f"Filter ON: HOCBF (min z={on['min_z_m']} m)")

        ax.axhline(0.03, color="k", ls=":", lw=1)
        ax.text(0.08, 0.08, "ground contact", fontsize=8, color="#555")
        ax.set_xlabel("time (s)")
        ax.set_title(TITLES.get(name, name), fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 1.1)

    axes[0].set_ylabel("altitude z (m)")
    fig.suptitle(
        "Real Crazyflie 2.X (27 g) in Isaac Sim 5.1 GPU PhysX — HOCBF OFF vs ON\n"
        f"HOCBF: mass={hp['mass']} kg, T_max={hp['T_max']} N, "
        f"alpha1={hp['alpha1']}, alpha2={hp['alpha2']}" +
        f", conservatism={hp.get('conservatism', 'n/a')}\n"
        "Source: experiments/results/crazyflie_vla_ab.json",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(DST, dpi=150)
    print(f"Wrote {DST}")

    # Print a compact summary so the numbers on the figure are auditable.
    for name, scn in sorted(data["scenarios"].items()):
        off, on = scn["OFF"], scn["ON"]
        print(f"[{name}] OFF: crashed={off['crashed']} crash_t={off['crash_t_s']}s "
              f"min_z={off['min_z_m']}m | ON: crashed={on['crashed']} "
              f"min_z={on['min_z_m']}m filtered={on['filtered_steps_frac']:.0%}")


if __name__ == "__main__":
    main()
