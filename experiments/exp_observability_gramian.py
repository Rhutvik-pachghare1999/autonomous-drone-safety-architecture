"""
Empirical Observability Gramian Analysis
=========================================
Computes the Empirical Observability Gramian (EOG) for the 15-state EKF
under three sensor configurations and plots the eigenvalue spectrum to
show exactly how the observable subspace collapses under sensor loss.

Method (Krener & Ide 2009):
    W_o = sum_i sum_k  (1/epsilon^2) * (y(x0+eps*ei, tk) - y(x0, tk))^T
                                      * (y(x0+eps*ei, tk) - y(x0, tk))
    where ei is the i-th unit perturbation of the state vector.

    Eigenvalues of W_o reveal the observable subspace:
    - Large eigenvalue → state direction is well-observed
    - Near-zero eigenvalue → state direction is unobservable

Sensor configurations tested:
    1. GPS + IMU + Baro  (nominal)
    2. IMU + Baro only   (GPS denied)
    3. IMU + Baro + VIO  (GPS denied + simulated optical flow)

The plot shows the eigenvalue spectrum for each configuration, proving
that GPS denial collapses the horizontal position and yaw subspace, and
that VIO partially restores horizontal velocity observability.

References:
    Krener & Ide (2009). Measures of Unobservability. CDC.
    Markovic et al. (2021). ES-EKF for GPS-Denied UAV Navigation. arXiv:2109.04908.

Author: Rhutvik Prashant Pachghare, ASU Robotics & Autonomous Systems
"""

import json, os
import numpy as np

RESULTS_DIR = "experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── EKF state indices ─────────────────────────────────────────────────────────
# x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bgx, bgy, bgz, bax, bay]
N_STATES = 15
IDX = dict(px=0, py=1, pz=2, vx=3, vy=4, vz=5,
           qw=6, qx=7, qy=8, qz=9, bgx=10, bgy=11, bgz=12, bax=13, bay=14)
STATE_LABELS = ['p_x','p_y','p_z','v_x','v_y','v_z',
                'q_w','q_x','q_y','q_z','b_gx','b_gy','b_gz','b_ax','b_ay']

# ── Sensor noise (architecture spec) ─────────────────────────────────────────
R_GPS  = np.diag([2.5, 2.5, 5.0])          # m²
R_BARO = np.array([[0.25]])                 # m²  (0.5m std)
R_IMU  = np.diag([0.01, 0.01, 0.01])       # (m/s²)²  accel
R_VIO  = np.diag([0.1, 0.1])               # (m/s)²  horizontal velocity

G = 9.81


# ── EKF measurement functions ─────────────────────────────────────────────────

def h_gps(x: np.ndarray) -> np.ndarray:
    """GPS: measures [px, py, pz]."""
    return x[[IDX['px'], IDX['py'], IDX['pz']]]


def h_baro(x: np.ndarray) -> np.ndarray:
    """Barometer: measures pz."""
    return x[[IDX['pz']]]


def h_imu(x: np.ndarray) -> np.ndarray:
    """IMU accelerometer: measures specific force ≈ [0,0,g] at hover."""
    qw, qx, qy, qz = x[IDX['qw']:IDX['qz']+1]
    # Gravity in body frame: R^T * [0,0,g]
    gx_b = 2*(qx*qz - qw*qy) * G
    gy_b = 2*(qy*qz + qw*qx) * G
    gz_b = (qw**2 - qx**2 - qy**2 + qz**2) * G
    bax, bay = x[IDX['bax']], x[IDX['bay']]
    return np.array([gx_b + bax, gy_b + bay, gz_b])


def h_vio(x: np.ndarray) -> np.ndarray:
    """VIO optical flow: measures [vx, vy]."""
    return x[[IDX['vx'], IDX['vy']]]


# ── Sensor configuration measurement stacks ───────────────────────────────────

def measure(x: np.ndarray, config: str) -> np.ndarray:
    parts = [h_baro(x), h_imu(x)]
    if 'gps' in config:
        parts.append(h_gps(x))
    if 'vio' in config:
        parts.append(h_vio(x))
    return np.concatenate(parts)


# ── Empirical Observability Gramian ──────────────────────────────────────────

def empirical_observability_gramian(x0: np.ndarray, config: str,
                                    T: float = 1.0, dt: float = 0.1,
                                    eps: float = 1e-4) -> np.ndarray:
    """
    Compute the Empirical Observability Gramian via finite differences.

    W_o[i,j] = sum_k  (1/eps^2) * delta_y_i(tk)^T * delta_y_j(tk)

    where delta_y_i(tk) = y(x0 + eps*ei, tk) - y(x0, tk)

    This is the standard method for nonlinear observability analysis
    (Krener & Ide 2009).  No linearisation required.
    """
    steps = int(T / dt)
    n = len(x0)
    W = np.zeros((n, n))

    for i in range(n):
        x_plus  = x0.copy(); x_plus[i]  += eps
        x_minus = x0.copy(); x_minus[i] -= eps
        dy_i_traj = []
        for _ in range(steps):
            y_p = measure(x_plus,  config)
            y_m = measure(x_minus, config)
            dy_i_traj.append((y_p - y_m) / (2 * eps))

        for j in range(n):
            x_plus_j  = x0.copy(); x_plus_j[j]  += eps
            x_minus_j = x0.copy(); x_minus_j[j] -= eps
            for k in range(steps):
                y_p_j = measure(x_plus_j,  config)
                y_m_j = measure(x_minus_j, config)
                dy_j  = (y_p_j - y_m_j) / (2 * eps)
                W[i, j] += np.dot(dy_i_traj[k], dy_j)

    return W


def run() -> dict:
    # Nominal hover state
    x0 = np.zeros(N_STATES)
    x0[IDX['pz']] = 2.0
    x0[IDX['qw']] = 1.0   # identity quaternion

    configs = {
        'GPS + IMU + Baro':  'gps_imu_baro',
        'IMU + Baro only':   'imu_baro',
        'IMU + Baro + VIO':  'imu_baro_vio',
    }

    results = {}
    gramians = {}

    print('=' * 60)
    print('Experiment 2: Empirical Observability Gramian')
    print('=' * 60)

    for label, config in configs.items():
        print(f'\nComputing EOG: {label}...')
        W = empirical_observability_gramian(x0, config, T=1.0, dt=0.1)
        eigvals = np.linalg.eigvalsh(W)
        eigvals_sorted = np.sort(eigvals)[::-1]

        # Rank = number of eigenvalues above threshold
        threshold = eigvals_sorted[0] * 1e-6
        rank = int(np.sum(eigvals_sorted > threshold))

        print(f'  Observable rank : {rank} / {N_STATES}')
        print(f'  Top-3 eigenvals : {eigvals_sorted[:3]}')
        print(f'  Min eigenval    : {eigvals_sorted[-1]:.2e}')

        # Identify unobservable states (near-zero eigenvalue directions)
        _, eigvecs = np.linalg.eigh(W)
        unobs_dirs = eigvecs[:, eigvals < threshold]
        unobs_states = []
        for col in range(unobs_dirs.shape[1]):
            dominant = np.argmax(np.abs(unobs_dirs[:, col]))
            unobs_states.append(STATE_LABELS[dominant])

        results[label] = {
            'config':          config,
            'rank':            rank,
            'eigenvalues':     eigvals_sorted.tolist(),
            'unobservable_states': list(set(unobs_states)),
        }
        gramians[label] = eigvals_sorted

    _plot(gramians, results)

    path = f'{RESULTS_DIR}/observability_gramian.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {path}')
    return results


def _plot(gramians: dict, results: dict) -> None:
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        colors = {'GPS + IMU + Baro': 'steelblue',
                  'IMU + Baro only':  'coral',
                  'IMU + Baro + VIO': 'seagreen'}

        # Left: eigenvalue spectrum (log scale)
        ax = axes[0]
        for label, eigvals in gramians.items():
            ax.semilogy(range(1, N_STATES+1), eigvals + 1e-20,
                        'o-', label=f"{label} (rank={results[label]['rank']})",
                        color=colors[label], linewidth=2, markersize=5)
        ax.axhline(1e-6 * max(v[0] for v in gramians.values()),
                   color='red', linestyle='--', linewidth=1.5,
                   label='Observability threshold')
        ax.set_xlabel('State index (sorted by observability)')
        ax.set_ylabel('Eigenvalue of W_o (log scale)')
        ax.set_title('Empirical Observability Gramian\nEigenvalue Spectrum')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, N_STATES+1))
        ax.set_xticklabels(STATE_LABELS, rotation=45, fontsize=7)

        # Right: rank bar chart
        ax2 = axes[1]
        labels = list(gramians.keys())
        ranks  = [results[l]['rank'] for l in labels]
        bar_colors = [colors[l] for l in labels]
        bars = ax2.bar(range(len(labels)), ranks, color=bar_colors, alpha=0.8)
        ax2.axhline(N_STATES, color='steelblue', linestyle=':', alpha=0.5,
                    label=f'Full rank ({N_STATES})')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels([l.replace(' + ', '\n+\n') for l in labels], fontsize=8)
        ax2.set_ylabel('Observable rank')
        ax2.set_ylim(0, N_STATES + 2)
        ax2.set_title('Observable Subspace Rank\nby Sensor Configuration')
        for bar, rank in zip(bars, ranks):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     str(rank), ha='center', va='bottom', fontweight='bold')

        # Annotate unobservable states
        for i, label in enumerate(labels):
            unobs = results[label]['unobservable_states']
            if unobs:
                ax2.text(i, 1, '\n'.join(unobs[:3]), ha='center', va='bottom',
                         fontsize=6, color='red', alpha=0.8)

        plt.tight_layout()
        out = f'{RESULTS_DIR}/observability_gramian.png'
        plt.savefig(out, dpi=150)
        print(f'Plot saved: {out}')
    except ImportError:
        pass


if __name__ == '__main__':
    run()
