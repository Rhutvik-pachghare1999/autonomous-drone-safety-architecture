#!/usr/bin/env python3
"""
SIL (Software-in-the-Loop) Harness for Drone Safety Architecture.

Plays scripted trajectories and writes VLA commands to /dev/shm/aisp_vla_cmd,
while the C sil_runner reads them and runs the watchdog + HOCBF filter.

Scenarios:
  (a) normal hover/climb
  (b) adversarial ground-dive (vz_nom = -8 m/s near pz=0.5)
  (c) VLA dropout (stop writing for >100ms to trigger STALE)
  (d) NaN/garbage command injection
"""

import argparse
import os
import sys
import time
import subprocess
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.shm_bridge import VLASharedMemoryPublisher

SHM_PATH = "/dev/shm/aisp_vla_cmd"
DEFAULT_CSV = "experiments/results/sil_replay.csv"
DEFAULT_PLOT = "experiments/results/sil_intervention.png"
# SIL_RUNNER path relative to project root (script location)
_SIL_RUNNER_REL = "build/sil_runner"


class SILHarness:
    def __init__(self, csv_out=DEFAULT_CSV, plot_out=DEFAULT_PLOT):
        self.csv_out = csv_out
        self.plot_out = plot_out
        self.publisher = VLASharedMemoryPublisher(SHM_PATH)
        self.sil_proc = None

    def cleanup(self):
        if self.sil_proc and self.sil_proc.poll() is None:
            self.sil_proc.terminate()
            self.sil_proc.wait(timeout=2)
        self.publisher.close()

    def run_sil_runner(self, n_cycles, csv_path):
        """Start the C sil_runner as a subprocess."""
        # Resolve SIL_RUNNER relative to project root (script location)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sil_runner_path = os.path.join(project_root, _SIL_RUNNER_REL)
        if not os.path.exists(sil_runner_path):
            raise FileNotFoundError(f"sil_runner not found at {sil_runner_path}. Build it first.")

        # Pre-populate shared memory with initial hover command
        # so the C reader sees valid data immediately on first read
        self.publisher.publish(0.0, 0.0, 0.0)
        time.sleep(0.01)  # Let it settle

        self.sil_proc = subprocess.Popen(
            [sil_runner_path, str(n_cycles), csv_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Give it a moment to open shm and read first command
        time.sleep(0.05)

    def run_scenario_hover(self, n_cycles=1000):
        """Scenario (a): Normal hover at 2m altitude."""
        print(f"  Running hover scenario ({n_cycles} cycles)...")
        for i in range(n_cycles):
            # Hover at 2m: vz_nom = 0
            self.publisher.publish(0.0, 0.0, 0.0)
            time.sleep(0.001)  # ~1 kHz
        print("  Hover scenario done")

    def run_scenario_climb(self, n_cycles=1000):
        """Scenario (a): Normal climb."""
        print(f"  Running climb scenario ({n_cycles} cycles)...")
        for i in range(n_cycles):
            # Gentle climb: vz_nom = 0.5 m/s
            self.publisher.publish(0.0, 0.0, 0.5)
            time.sleep(0.001)
        print("  Climb scenario done")

    def run_scenario_adversarial_dive(self, n_cycles=1500):
        """
        Scenario (b): Adversarial ground-dive.
        VLA commands -8 m/s descent near ground (pz~0.5m).
        The safety filter should raise thrust and prevent ground impact.
        """
        print(f"  Running adversarial dive scenario ({n_cycles} cycles)...")
        # First 200 cycles: normal hover at 5m
        for i in range(200):
            self.publisher.publish(0.0, 0.0, 0.0)
            time.sleep(0.001)

        # Next 800 cycles: aggressive dive command (-8 m/s) while near ground
        # The sim will be at low altitude by then
        for i in range(800):
            self.publisher.publish(0.0, 0.0, -8.0)
            time.sleep(0.001)

        # Final 500 cycles: return to hover
        for i in range(500):
            self.publisher.publish(0.0, 0.0, 0.0)
            time.sleep(0.001)
        print("  Adversarial dive scenario done")

    def run_scenario_vla_dropout(self, n_cycles=1200):
        """
        Scenario (c): VLA dropout.
        Stop writing to shm for >100ms to trigger STALE state.
        Safety filter should fall back to hover thrust (m*g).
        """
        print(f"  Running VLA dropout scenario ({n_cycles} cycles)...")
        # First 300 cycles: normal hover
        for i in range(300):
            self.publisher.publish(0.0, 0.0, 0.0)
            time.sleep(0.001)

        # Next 300 cycles: DROP VLA (stop publishing, but keep time advancing)
        print("    VLA DROPOUT START (no commands for ~300ms)")
        start = time.time()
        while time.time() - start < 0.3:
            time.sleep(0.001)
        print("    VLA DROPOUT END (resuming commands)")

        # Resume normal hover
        for i in range(600):
            self.publisher.publish(0.0, 0.0, 0.0)
            time.sleep(0.001)
        print("  VLA dropout scenario done")

    def run_scenario_nan_injection(self, n_cycles=500):
        """
        Scenario (d): NaN/garbage command injection.
        Safety filter should output finite thrust within [T_MIN, T_MAX].
        """
        print(f"  Running NaN injection scenario ({n_cycles} cycles)...")
        for i in range(n_cycles):
            if i == 100:
                # Inject NaN via the same seqlock write path as normal
                # publishes (raw struct.pack would desync the wire layout)
                self.publisher.publish(0.0, 0.0, float('nan'))
            elif i == 200:
                # Inject Inf
                self.publisher.publish(0.0, 0.0, float('inf'))
            elif i == 300:
                # Inject negative Inf
                self.publisher.publish(0.0, 0.0, float('-inf'))
            else:
                self.publisher.publish(0.0, 0.0, 0.0)
            time.sleep(0.001)
        print("  NaN injection scenario done")

    def run_scenario(self, name, n_cycles, csv_path):
        """Run a single scenario with the C sil_runner."""
        print(f"\n=== Scenario: {name} ===")
        self.run_sil_runner(n_cycles, csv_path)

        # Run scenario-specific publisher logic
        if name == "hover":
            self.run_scenario_hover(n_cycles)
        elif name == "climb":
            self.run_scenario_climb(n_cycles)
        elif name == "adversarial_dive":
            self.run_scenario_adversarial_dive(n_cycles)
        elif name == "vla_dropout":
            self.run_scenario_vla_dropout(n_cycles)
        elif name == "nan_injection":
            self.run_scenario_nan_injection(n_cycles)
        else:
            raise ValueError(f"Unknown scenario: {name}")

        # Wait for sil_runner to finish
        stdout, stderr = self.sil_proc.communicate(timeout=30)
        if stdout:
            print(stdout.strip())
        if stderr:
            print("STDERR:", stderr.strip(), file=sys.stderr)

        if self.sil_proc.returncode != 0:
            raise RuntimeError(f"sil_runner exited with code {self.sil_proc.returncode}")

        # Verify CSV was written
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Expected CSV not found: {csv_path}")
        return csv_path

    def run_all(self):
        """Run all scenarios and produce combined outputs."""
        os.makedirs(os.path.dirname(self.csv_out), exist_ok=True)
        os.makedirs(os.path.dirname(self.plot_out), exist_ok=True)

        scenarios = [
            ("hover", 1000),
            ("climb", 1000),
            ("adversarial_dive", 1500),
            ("vla_dropout", 1200),
            ("nan_injection", 500),
        ]

        all_dfs = []
        for name, n_cycles in scenarios:
            csv_path = self.csv_out.replace('.csv', f'_{name}.csv')
            self.run_scenario(name, n_cycles, csv_path)
            # Read for plotting
            df = np.genfromtxt(csv_path, delimiter=',', names=True)
            df = np.array([tuple(row) for row in df],
                          dtype=[('t_ns', 'u8'), ('pz', 'f8'), ('vz', 'f8'),
                                 ('vla_vz_nom', 'f8'), ('is_fresh', 'i4'),
                                 ('vla_state', 'i4'), ('T_nom', 'f8'),
                                 ('T_safe', 'f8'), ('was_infeasible', 'i4'),
                                 ('intervened', 'i4')])
            all_dfs.append((name, df))

        # Combine into master CSV
        master_csv = self.csv_out
        with open(master_csv, 'w') as f:
            f.write("scenario,t_ns,pz,vz,vla_vz_nom,is_fresh,vla_state,T_nom,T_safe,was_infeasible,intervened\n")
            for name, df in all_dfs:
                for row in df:
                    f.write(f"{name},{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]},{row[5]},{row[6]:.6f},{row[7]:.6f},{row[8]},{row[9]}\n")
        print(f"\nMaster CSV written: {master_csv}")

        # Generate plot
        self.generate_plot(all_dfs)

        # Run assertions (acceptance criteria)
        self.assert_criteria(all_dfs)

        print("\n✓ All scenarios passed acceptance criteria!")

    def generate_plot(self, all_dfs):
        """Generate matplotlib plot: altitude, T_nom vs T_safe, state timeline."""
        n = len(all_dfs)
        fig, axes = plt.subplots(n, 3, figsize=(18, 4*n), squeeze=False)
        fig.suptitle('SIL Replay Results', fontsize=14)

        state_names = {0: 'STARTUP', 1: 'FRESH', 2: 'STALE'}

        for idx, (name, df) in enumerate(all_dfs):
            t_ms = df['t_ns'] / 1e6  # Convert to ms for readability

            # Plot 1: Altitude
            ax = axes[idx, 0]
            ax.plot(t_ms, df['pz'], 'b-', label='Altitude (pz)')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Ground')
            ax.set_ylabel('Altitude (m)')
            ax.set_title(f'{name}: Altitude')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Plot 2: Thrust comparison
            ax = axes[idx, 1]
            ax.plot(t_ms, df['T_nom'], 'r--', label='T_nom (VLA)', alpha=0.7)
            ax.plot(t_ms, df['T_safe'], 'g-', label='T_safe (filtered)', alpha=0.7)
            ax.axhline(y=19.62, color='k', linestyle=':', alpha=0.5, label='Hover (m*g)')
            ax.axhline(y=78.48, color='orange', linestyle=':', alpha=0.5, label='T_MAX')
            ax.set_ylabel('Thrust (N)')
            ax.set_title(f'{name}: T_nom vs T_safe')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Plot 3: VLA state timeline
            ax = axes[idx, 2]
            colors = {0: 'gray', 1: 'green', 2: 'red'}
            for state_val, color in colors.items():
                mask = df['vla_state'] == state_val
                if np.any(mask):
                    ax.scatter(t_ms[mask], df['vla_state'][mask],
                               c=color, label=state_names[state_val], s=10, alpha=0.6)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['STARTUP', 'FRESH', 'STALE'])
            ax.set_ylabel('VLA State')
            ax.set_xlabel('Time (ms)')
            ax.set_title(f'{name}: VLA Watchdog State')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_out, dpi=150)
        print(f"Plot saved: {self.plot_out}")

    def assert_criteria(self, all_dfs):
        """Verify acceptance criteria from PHASE 1 spec."""
        print("\n=== Verifying Acceptance Criteria ===")

        for name, df in all_dfs:
            pz = df['pz']
            T_nom = df['T_nom']
            T_safe = df['T_safe']
            vla_state = df['vla_state']
            was_infeasible = df['was_infeasible']
            intervened = df['intervened']

            if name == "adversarial_dive":
                # T_safe raised above T_nom near ground AND altitude never < 0
                min_alt = np.min(pz)
                assert min_alt >= 0.0, f"FAIL: altitude went below ground: min={min_alt:.4f}m"
                print(f"  ✓ {name}: min altitude = {min_alt:.4f}m >= 0")

                # Check that T_safe > T_nom during dive (intervention occurred)
                intervention_mask = intervened == 1
                assert np.any(intervention_mask), f"FAIL: no intervention during adversarial dive"
                print(f"  ✓ {name}: filter intervened {np.sum(intervention_mask)} times")

            elif name == "vla_dropout":
                # vla_state reaches STALE
                stale_mask = vla_state == 2
                assert np.any(stale_mask), f"FAIL: VLA state never reached STALE"
                print(f"  ✓ {name}: VLA state reached STALE ({np.sum(stale_mask)} cycles)")

                # thrust falls back to hover (m*g = 19.62 N) during STALE
                stale_T_safe = T_safe[stale_mask]
                hover_thrust = 19.62
                assert np.allclose(stale_T_safe, hover_thrust, atol=0.1), \
                    f"FAIL: thrust during STALE not at hover: mean={np.mean(stale_T_safe):.2f}"
                print(f"  ✓ {name}: thrust at hover ({hover_thrust:.2f} N) during STALE")

            elif name == "nan_injection":
                # T_safe stays finite and within [T_MIN, T_MAX]
                assert np.all(np.isfinite(T_safe)), "FAIL: T_safe produced non-finite values"
                assert np.all(T_safe >= 0.0), "FAIL: T_safe below T_MIN (0)"
                assert np.all(T_safe <= 78.48), "FAIL: T_safe above T_MAX (78.48)"
                print(f"  ✓ {name}: T_safe always finite and in bounds [0, 78.48]")

            elif name in ("hover", "climb"):
                # Normal flight: no unnecessary intervention
                unnecessary = intervened == 1
                # Allow some intervention due to initial transients
                intervention_rate = np.sum(unnecessary) / len(unnecessary)
                assert intervention_rate < 0.1, \
                    f"FAIL: excessive intervention rate {intervention_rate:.2%} in {name}"
                print(f"  ✓ {name}: intervention rate = {intervention_rate:.2%} (< 10%)")


def main():
    parser = argparse.ArgumentParser(description="SIL Harness for Drone Safety")
    parser.add_argument("--scenario", choices=["all", "hover", "climb", "adversarial_dive",
                                                "vla_dropout", "nan_injection"],
                        default="all", help="Scenario to run")
    parser.add_argument("--cycles", type=int, default=1000,
                        help="Number of cycles per scenario")
    parser.add_argument("--out", default="experiments/results/",
                        help="Output directory")
    args = parser.parse_args()

    csv_out = os.path.join(args.out, "sil_replay.csv")
    plot_out = os.path.join(args.out, "sil_intervention.png")
    harness = SILHarness(csv_out, plot_out)

    try:
        if args.scenario == "all":
            harness.run_all()
        else:
            n_cycles = args.cycles
            csv_path = os.path.join(args.out, f"sil_replay_{args.scenario}.csv")
            harness.run_scenario(args.scenario, n_cycles, csv_path)
            # Verify single scenario
            df = np.genfromtxt(csv_path, delimiter=',', names=True)
            # Quick checks
            print(f"Scenario {args.scenario} completed: {len(df)} cycles")
    finally:
        harness.cleanup()


if __name__ == "__main__":
    main()