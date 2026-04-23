"""
P99.99 WCET Histogram with EVT Gumbel Tail Extrapolation
=========================================================
Produces the publication-grade latency histogram for the HOCBF safety filter.

Inputs:  experiments/results/latency_raw.csv  (100k trials from safety_filter.c)
Outputs: experiments/results/wcet_evt.png
         experiments/results/wcet_evt.json

EVT Method (Gumbel / Type-I Extreme Value Distribution):
    The maximum of n i.i.d. latency samples converges to a Gumbel distribution:
        F(t) = exp(-exp(-(t - mu) / beta))
    Fitted via method of moments:
        beta = std(x) * sqrt(6) / pi
        mu   = mean(x) - beta * gamma_EM   (gamma_EM = 0.5772...)
    WCET bound at exceedance probability p:
        t_wcet = mu - beta * ln(-ln(1 - p))

    This is the standard avionics WCET tail-bound method.
    Reference: Foughali & Zuepke (2022). Frontiers in Robotics and AI.

Author: Rhutvik Prashant Pachghare, ASU Robotics & Autonomous Systems
"""

import csv, json, os
import numpy as np

RESULTS_DIR = "experiments/results"
DEADLINE_NS = 100_000   # 100 µs hard deadline


def load_csv() -> tuple[np.ndarray, np.ndarray]:
    path = f"{RESULTS_DIR}/latency_raw.csv"
    lats, jitters = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lats.append(int(row["latency_ns"]))
            jitters.append(int(row.get("jitter_ns", 0)))
    return np.array(lats), np.array(jitters)


def gumbel_fit(x: np.ndarray, block_size: int = 100) -> dict:
    """
    Fit a Gumbel (Type-I EVD) distribution to latency data using the
    Block Maxima method — the statistically correct approach for WCET.

    Method-of-moments on the full sample is INVALID for WCET because:
      - The latency distribution is heavily right-skewed (spike at mode,
        long tail from OS preemptions).
      - MoM fits the bulk, not the tail; mu ends up below the minimum
        observed value, making the extrapolation meaningless.

    Block Maxima (Gumbel / GEV Type-I):
      1. Partition n samples into blocks of size `block_size`.
      2. Take the maximum of each block → block_maxima array.
      3. Fit Gumbel to block_maxima via MLE (or MoM on maxima only).
      4. Extrapolate to P=1e-9 using the fitted parameters.

    This is the standard avionics WCET tail-bound method.
    Reference: Foughali & Zuepke (2022). Frontiers in Robotics and AI.
    """
    # Block maxima
    n_blocks = len(x) // block_size
    maxima = np.array([x[i*block_size:(i+1)*block_size].max()
                       for i in range(n_blocks)])

    # MoM on block maxima (valid because maxima are approximately Gumbel)
    EULER = 0.5772156649
    beta  = maxima.std() * np.sqrt(6) / np.pi
    mu    = maxima.mean() - beta * EULER

    def wcet_bound(p_exceed: float) -> float:
        # Right-tail bound: t such that P(X > t) = p_exceed
        # For Gumbel CDF F(t) = exp(-exp(-(t-mu)/beta)):
        #   1 - F(t) = p_exceed  →  t = mu - beta*ln(-ln(1-p_exceed))
        # At small p_exceed, ln(1-p_exceed) ≈ -p_exceed, so:
        #   t ≈ mu - beta*ln(p_exceed)  which is >> mu  ✓
        return mu - beta * np.log(-np.log(1.0 - p_exceed))

    return {
        'mu': mu, 'beta': beta,
        'n_blocks': n_blocks, 'block_size': block_size,
        'maxima_mean': float(maxima.mean()),
        'maxima_max':  float(maxima.max()),
        'p999_ns':    wcet_bound(1e-3),
        'p9999_ns':   wcet_bound(1e-4),
        'p99999_ns':  wcet_bound(1e-5),
        'p1e9_ns':    wcet_bound(1e-9),
        'wcet_bound': wcet_bound,
    }


def run() -> dict:
    print('=' * 60)
    print('Experiment 3: P99.99 WCET Histogram + EVT Gumbel Fit')
    print('=' * 60)

    lats, jitters = load_csv()
    n = len(lats)
    print(f'\nLoaded {n:,} latency samples')

    # Empirical percentiles
    p = lambda q: float(np.percentile(lats, q))
    stats = {
        'n_trials':  n,
        'min_ns':    int(lats.min()),
        'mean_ns':   float(lats.mean()),
        'p50_ns':    p(50),
        'p99_ns':    p(99),
        'p999_ns':   p(99.9),
        'p9999_ns':  p(99.99),
        'wcet_ns':   int(lats.max()),
        'deadline_ns': DEADLINE_NS,
        'deadline_pass': bool(lats.max() < DEADLINE_NS),
    }

    print(f"\n── Empirical percentiles ────────────────────────────────────")
    print(f"  N          : {n:,}")
    print(f"  Min        : {stats['min_ns']} ns")
    print(f"  Mean       : {stats['mean_ns']:.1f} ns")
    print(f"  P99        : {stats['p99_ns']:.0f} ns")
    print(f"  P99.9      : {stats['p999_ns']:.0f} ns")
    print(f"  P99.99     : {stats['p9999_ns']:.0f} ns")
    print(f"  WCET (max) : {stats['wcet_ns']} ns  ({stats['wcet_ns']/1000:.2f} µs)")
    print(f"  < 100µs    : {'PASS ✓' if stats['deadline_pass'] else 'FAIL ✗'}")

    # EVT Gumbel fit
    fit = gumbel_fit(lats)
    evt = {
        'method':            'block_maxima_gumbel',
        'block_size':        fit['block_size'],
        'n_blocks':          fit['n_blocks'],
        'gumbel_mu_ns':      fit['mu'],
        'gumbel_beta_ns':    fit['beta'],
        'p999_bound_ns':     fit['p999_ns'],
        'p9999_bound_ns':    fit['p9999_ns'],
        'p99999_bound_ns':   fit['p99999_ns'],
        'p1e9_bound_ns':     fit['p1e9_ns'],
        'p1e9_pass':         fit['p1e9_ns'] < DEADLINE_NS,
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
    print(f"  < 100µs (P=1e-9): {'PASS ✓' if evt['p1e9_pass'] else 'FAIL ✗'}")

    # Jitter stats
    jitter_stats = {
        'max_jitter_ns':  int(jitters.max()),
        'p99_jitter_ns':  float(np.percentile(jitters, 99)),
        'jitter_pass_50us': bool(jitters.max() < 50_000),
    }
    print(f"\n── OS jitter (inter-cycle) ──────────────────────────────────")
    print(f"  Max jitter : {jitter_stats['max_jitter_ns']} ns  ({jitter_stats['max_jitter_ns']/1000:.1f} µs)")
    print(f"  P99 jitter : {jitter_stats['p99_jitter_ns']:.0f} ns")
    print(f"  < 50µs     : {'PASS ✓' if jitter_stats['jitter_pass_50us'] else 'WARN'}")

    _plot(lats, jitters, fit, stats, evt)

    result = {'empirical': stats, 'evt': evt, 'jitter': jitter_stats}
    path = f'{RESULTS_DIR}/wcet_evt.json'
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda o: bool(o) if hasattr(o, 'item') else str(o))
    print(f'\nResults saved: {path}')
    return result


def _plot(lats, jitters, fit, stats, evt):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import gumbel_r

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f'HOCBF Safety Filter — WCET Analysis  (N={len(lats):,} trials, '
            f'SCHED_FIFO prio 99, CPU core 2)',
            fontsize=11
        )

        lats_us = lats / 1000.0

        # ── Left: histogram + EVT fit ─────────────────────────────────────────
        ax = axes[0]
        ax.hist(lats_us, bins=200, color='steelblue', alpha=0.7,
                density=True, label='Measured latency')

        # Gumbel PDF overlay
        x_fit = np.linspace(0, lats_us.max() * 1.5, 500)
        mu_us, beta_us = fit['mu'] / 1000, fit['beta'] / 1000
        pdf = gumbel_r.pdf(x_fit, loc=mu_us, scale=beta_us)
        ax.plot(x_fit, pdf, 'r-', linewidth=2, label='Gumbel EVT fit')

        ax.axvline(100, color='red', linestyle='--', linewidth=2,
                   label='100µs deadline')
        ax.axvline(stats['wcet_ns']/1000, color='orange', linestyle=':',
                   linewidth=1.5, label=f"WCET={stats['wcet_ns']/1000:.2f}µs")
        ax.set_xlabel('Latency (µs)')
        ax.set_ylabel('Density')
        ax.set_title('Latency Distribution + Gumbel Fit')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Middle: log-scale tail CDF ────────────────────────────────────────
        ax2 = axes[1]
        sorted_lats = np.sort(lats_us)
        exceedance  = 1.0 - np.arange(1, len(sorted_lats)+1) / len(sorted_lats)
        exceedance  = np.maximum(exceedance, 1e-10)
        ax2.semilogy(sorted_lats, exceedance, 'steelblue', linewidth=1.5,
                     label='Empirical CCDF')

        # EVT extrapolation
        p_range = np.logspace(-2, -9, 200)
        t_evt   = fit['mu']/1000 - fit['beta']/1000 * np.log(-np.log(1 - p_range))
        ax2.semilogy(t_evt, p_range, 'r--', linewidth=2,
                     label='Gumbel EVT extrapolation')

        ax2.axvline(100, color='red', linestyle='--', linewidth=2,
                    label='100µs deadline')
        ax2.axhline(1e-9, color='gray', linestyle=':', linewidth=1,
                    label='P=1e-9 (DO-178C DAL-A)')

        # Mark EVT bound at P=1e-9
        ax2.plot(evt['p1e9_bound_ns']/1000, 1e-9, 'r*', markersize=12,
                 label=f"EVT bound={evt['p1e9_bound_ns']/1000:.3f}µs")

        ax2.set_xlabel('Latency (µs)')
        ax2.set_ylabel('Exceedance probability')
        ax2.set_title('CCDF Tail + EVT Extrapolation\n(log scale)')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)

        # ── Right: jitter histogram ───────────────────────────────────────────
        ax3 = axes[2]
        jit_us = jitters[jitters > 0] / 1000.0
        if len(jit_us) > 0:
            ax3.hist(jit_us, bins=100, color='seagreen', alpha=0.7)
        ax3.axvline(50, color='red', linestyle='--', linewidth=2,
                    label='50µs jitter SLA')
        ax3.axvline(jitters.max()/1000, color='orange', linestyle=':',
                    linewidth=1.5, label=f"Max={jitters.max()/1000:.1f}µs")
        ax3.set_xlabel('Inter-cycle jitter (µs)')
        ax3.set_ylabel('Count')
        ax3.set_title('OS Scheduler Jitter\n(SCHED_FIFO, CPU core 2)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out = f'{RESULTS_DIR}/wcet_evt.png'
        plt.savefig(out, dpi=150)
        print(f'Plot saved: {out}')
    except ImportError:
        pass


if __name__ == '__main__':
    run()
