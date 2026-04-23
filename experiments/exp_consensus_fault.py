"""
aBFT Consensus Under Packet Loss and Asymmetric Latency
========================================================
Tests the observability-weighted HotStuff consensus protocol under:
  - 20% random packet loss (simulated at the message layer)
  - Asymmetric latency: 0–50ms per link, randomly assigned
  - 5-node swarm: 3 GPS-active, 1 GPS-degraded, 1 GPS-denied (Byzantine)
  - 100 consensus rounds

Metrics reported:
  - Commit rate (% of rounds reaching weighted quorum)
  - Mean commit latency (ms)
  - Byzantine rejection rate (% of rounds where Byzantine node's state
    was NOT adopted as the agreed state)
  - Weighted quorum fraction per round

The Byzantine node reports a fabricated position [100, 100, 2] while
GPS-active nodes report the true position [0, 0, 2].  The weighted
quorum rule should reject the Byzantine state in all rounds.

Author: Rhutvik Prashant Pachghare, ASU Robotics & Autonomous Systems
"""

import json, os, random, time
import numpy as np

RESULTS_DIR = "experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.consensus_node import (
    ConsensusMessage, EKFSnapshot, _state_hash, Phase, SIGMA_WARN_SQ, BFT_THRESHOLD
)

# ── Network simulation ────────────────────────────────────────────────────────
PACKET_LOSS_RATE = 0.20   # 20%
LATENCY_MAX_MS   = 50.0   # asymmetric: 0–50ms per link
N_NODES          = 5
N_ROUNDS         = 100
TRUE_STATE       = [0.0, 0.0, 2.0]
BYZANTINE_STATE  = [100.0, 100.0, 2.0]

# Node configurations: (var_px, var_py, var_psi, gps_active, is_byzantine)
NODE_CONFIGS = [
    (0.1,  0.1,  0.04, True,  False),   # Node 0: GPS active, honest
    (0.1,  0.1,  0.04, True,  False),   # Node 1: GPS active, honest
    (0.1,  0.1,  0.04, True,  False),   # Node 2: GPS active, honest
    (8.0,  8.0,  3.2,  False, False),   # Node 3: GPS degraded, honest
    (20.0, 20.0, 8.0,  False, True),    # Node 4: GPS denied, Byzantine
]


def make_vote(node_id: int, round_num: int, rng: np.random.Generator) -> ConsensusMessage:
    vpx, vpy, vpsi, gps, is_byz = NODE_CONFIGS[node_id]
    snap = EKFSnapshot(0.0, 0.0, 2.0, vpx, vpy, vpsi, gps)
    state = BYZANTINE_STATE if is_byz else TRUE_STATE
    return ConsensusMessage(
        node_id=node_id, round_num=round_num, phase=Phase.PREPARE.name,
        state_hash=_state_hash(state), state_vector=state,
        trust_weight=snap.trust_weight,
        var_px=vpx, var_py=vpy, var_psi=vpsi, gps_active=gps,
    )


def simulate_network(votes: list[ConsensusMessage],
                     rng: np.random.Generator) -> list[ConsensusMessage]:
    """Apply packet loss and asymmetric latency to a vote set."""
    delivered = []
    for v in votes:
        if rng.random() < PACKET_LOSS_RATE:
            continue   # packet dropped
        # Asymmetric latency: simulate by adding noise to timestamp
        latency_ms = rng.uniform(0, LATENCY_MAX_MS)
        v.timestamp += latency_ms / 1000.0
        delivered.append(v)
    return delivered


def weighted_quorum(votes: list[ConsensusMessage],
                    my_msg: ConsensusMessage) -> tuple[list | None, float]:
    """Returns (agreed_state, quorum_fraction) or (None, fraction)."""
    all_votes = votes + [my_msg]
    total_w = sum(v.trust_weight for v in all_votes)
    if total_w < 1e-9:
        return None, 0.0

    hash_w: dict[str, float] = {}
    hash_s: dict[str, list]  = {}
    for v in all_votes:
        hash_w[v.state_hash] = hash_w.get(v.state_hash, 0.0) + v.trust_weight
        hash_s[v.state_hash] = v.state_vector

    best = max(hash_w, key=lambda h: hash_w[h])
    frac = hash_w[best] / total_w
    if frac >= BFT_THRESHOLD:
        return hash_s[best], frac
    return None, frac


def run() -> dict:
    rng = np.random.default_rng(42)
    print('=' * 60)
    print('Experiment 4: aBFT Consensus Under Packet Loss')
    print(f'  Nodes: {N_NODES}  |  Rounds: {N_ROUNDS}')
    print(f'  Packet loss: {PACKET_LOSS_RATE*100:.0f}%  |  Max latency: {LATENCY_MAX_MS}ms')
    print('=' * 60)

    commits = 0
    byzantine_rejected = 0
    quorum_fracs = []
    latencies_ms = []
    round_results = []

    for r in range(N_ROUNDS):
        t0 = time.perf_counter()

        # Each node broadcasts its vote; simulate network for each receiver
        all_votes = [make_vote(i, r, rng) for i in range(N_NODES)]

        # Node 0 is the "leader" / aggregator for this round
        my_msg = all_votes[0]
        peer_votes = simulate_network(all_votes[1:], rng)

        agreed, frac = weighted_quorum(peer_votes, my_msg)
        latency_ms = (time.perf_counter() - t0) * 1000

        committed = agreed is not None
        byz_rejected = (agreed != BYZANTINE_STATE) if committed else True

        if committed:
            commits += 1
            latencies_ms.append(latency_ms)
        if byz_rejected:
            byzantine_rejected += 1

        quorum_fracs.append(frac)
        round_results.append({
            'round': r, 'committed': committed,
            'agreed_state': agreed, 'quorum_frac': frac,
            'byzantine_rejected': byz_rejected,
            'n_delivered': len(peer_votes) + 1,
            'latency_ms': latency_ms,
        })

    commit_rate   = commits / N_ROUNDS
    byz_rej_rate  = byzantine_rejected / N_ROUNDS
    mean_lat      = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p99_lat       = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

    print(f'\n── Results ──────────────────────────────────────────────────')
    print(f'  Commit rate          : {commit_rate*100:.1f}%  ({commits}/{N_ROUNDS})')
    print(f'  Byzantine rejection  : {byz_rej_rate*100:.1f}%  ({byzantine_rejected}/{N_ROUNDS})')
    print(f'  Mean commit latency  : {mean_lat:.3f} ms')
    print(f'  P99 commit latency   : {p99_lat:.3f} ms')
    print(f'  Mean quorum fraction : {np.mean(quorum_fracs):.3f}')
    print(f'  Min quorum fraction  : {np.min(quorum_fracs):.3f}')

    # Node trust weights
    print(f'\n── Node trust weights ───────────────────────────────────────')
    for i, (vpx, vpy, vpsi, gps, byz) in enumerate(NODE_CONFIGS):
        snap = EKFSnapshot(0,0,2, vpx, vpy, vpsi, gps)
        tag = ' [BYZANTINE]' if byz else (' [DEGRADED]' if not gps else '')
        print(f'  Node {i}: w={snap.trust_weight:.4f}  gps={gps}{tag}')

    _plot(round_results, quorum_fracs)

    result = {
        'config': {
            'n_nodes': N_NODES, 'n_rounds': N_ROUNDS,
            'packet_loss_rate': PACKET_LOSS_RATE,
            'max_latency_ms': LATENCY_MAX_MS,
        },
        'commit_rate':          commit_rate,
        'byzantine_rejection_rate': byz_rej_rate,
        'mean_commit_latency_ms': mean_lat,
        'p99_commit_latency_ms':  p99_lat,
        'mean_quorum_fraction':   float(np.mean(quorum_fracs)),
    }
    path = f'{RESULTS_DIR}/consensus_fault.json'
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nResults saved: {path}')
    return result


def _plot(round_results: list, quorum_fracs: list) -> None:
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(
            f'aBFT Consensus Under {PACKET_LOSS_RATE*100:.0f}% Packet Loss + '
            f'Asymmetric Latency (0–{LATENCY_MAX_MS}ms)\n'
            f'{N_NODES}-node swarm: 3 GPS-active, 1 GPS-degraded, 1 Byzantine GPS-denied',
            fontsize=10
        )

        rounds = [r['round'] for r in round_results]
        committed = [int(r['committed']) for r in round_results]
        byz_rej   = [int(r['byzantine_rejected']) for r in round_results]

        # Left: commit and Byzantine rejection per round
        ax = axes[0]
        ax.fill_between(rounds, committed, alpha=0.4, color='steelblue',
                        label='Committed')
        ax.fill_between(rounds, byz_rej, alpha=0.3, color='coral',
                        label='Byzantine rejected')
        ax.set_xlabel('Round')
        ax.set_ylabel('Binary (1=yes)')
        ax.set_title('Commit & Byzantine Rejection\nper Round')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Middle: quorum fraction over rounds
        ax2 = axes[1]
        ax2.plot(rounds, quorum_fracs, 'steelblue', linewidth=1, alpha=0.7)
        ax2.axhline(BFT_THRESHOLD, color='red', linestyle='--', linewidth=2,
                    label=f'BFT threshold ({BFT_THRESHOLD:.2f})')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Weighted quorum fraction')
        ax2.set_title('Quorum Fraction per Round\n(weighted by EKF trust)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

        # Right: node trust weight bar chart
        ax3 = axes[2]
        weights = []
        labels  = []
        colors  = []
        for i, (vpx, vpy, vpsi, gps, byz) in enumerate(NODE_CONFIGS):
            from services.consensus_node import EKFSnapshot
            snap = EKFSnapshot(0,0,2, vpx, vpy, vpsi, gps)
            weights.append(snap.trust_weight)
            tag = 'Byzantine\n(GPS denied)' if byz else \
                  ('GPS\ndegraded' if not gps else f'GPS active\n(Node {i})')
            labels.append(tag)
            colors.append('coral' if byz else ('orange' if not gps else 'steelblue'))

        bars = ax3.bar(range(N_NODES), weights, color=colors, alpha=0.8)
        ax3.axhline(BFT_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                    label=f'2/3 threshold')
        ax3.set_xticks(range(N_NODES))
        ax3.set_xticklabels(labels, fontsize=7)
        ax3.set_ylabel('Trust weight w_i')
        ax3.set_title('Node Trust Weights\n(from EKF covariance)')
        ax3.legend(fontsize=8)
        for bar, w in zip(bars, weights):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{w:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        out = f'{RESULTS_DIR}/consensus_fault.png'
        plt.savefig(out, dpi=150)
        print(f'Plot saved: {out}')
    except ImportError:
        pass


if __name__ == '__main__':
    run()
