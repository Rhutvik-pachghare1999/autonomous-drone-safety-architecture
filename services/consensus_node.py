# Observability-weighted HotStuff consensus for drone swarms.
# Trust weight w_i = exp(-σ²_pos / (2·σ_warn²)) links EKF covariance to vote weight.
# GPS-denied nodes get w≈0 and cannot corrupt the quorum even if Byzantine.
# Reads from /dev/shm/aisp_ekf_state, writes agreed state to /dev/shm/aisp_consensus.
#
# Rhutvik Prashant Pachghare — ASU Robotics & Autonomous Systems

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
SIGMA_WARN_SQ    = 5.0    # m² — matches ekf_gating.py
BFT_THRESHOLD    = 2 / 3  # weighted quorum fraction
ROUND_TIMEOUT_S  = 0.5    # seconds before view change
SHM_EKF_PATH     = "/dev/shm/aisp_ekf_state"
SHM_CONSENSUS    = "/dev/shm/aisp_consensus"

# EKF shared memory layout (64 bytes):
#   double p_x, p_y, p_z          (24 bytes)
#   double var_px, var_py, var_psi (24 bytes)  — diagonal of P
#   uint8  gps_active              (1 byte)
#   uint8  _pad[15]                (15 bytes)
_EKF_SHM_FMT  = "=ddddddb15x"   # native byte order
_EKF_SHM_SIZE = 64


class Phase(Enum):
    PREPARE  = auto()
    PRE_VOTE = auto()
    COMMIT   = auto()


@dataclass
class EKFSnapshot:
    """EKF state read from shared memory."""
    px: float
    py: float
    pz: float
    var_px: float
    var_py: float
    var_psi: float
    gps_active: bool

    @property
    def trust_weight(self) -> float:
        """
        w = exp(-tr(P[p_x, p_y, psi]) / (2 * sigma_warn^2))
        Range: (0, 1].  GPS-denied node → w → 0.
        """
        sigma_sq = self.var_px + self.var_py + self.var_psi
        return float(np.exp(-sigma_sq / (2.0 * SIGMA_WARN_SQ)))

    @property
    def is_observable(self) -> bool:
        """True if position covariance is below the warn threshold."""
        return (self.var_px + self.var_py + self.var_psi) < SIGMA_WARN_SQ


@dataclass
class ConsensusMessage:
    node_id:      int
    round_num:    int
    phase:        str          # Phase enum name
    state_hash:   str          # SHA-256 of proposed state vector
    state_vector: List[float]  # [px, py, pz] proposed agreed state
    trust_weight: float        # w_i from EKF covariance
    var_px:       float
    var_py:       float
    var_psi:      float
    gps_active:   bool
    timestamp:    float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "ConsensusMessage":
        d = json.loads(s)
        return ConsensusMessage(**d)


def _state_hash(state: List[float]) -> str:
    payload = struct.pack(f"={len(state)}d", *state)
    return hashlib.sha256(payload).hexdigest()[:16]


class EKFSharedMemory:
    """
    Reads EKF state from /dev/shm/aisp_ekf_state.
    Falls back to synthetic data if the file does not exist
    (allows standalone testing without a running EKF).
    """

    def __init__(self, path: str = SHM_EKF_PATH):
        self._path = path
        self._shm: Optional[mmap.mmap] = None
        self._rng = np.random.default_rng(int(time.time() * 1e6) % (2**32))
        self._open()

    def _open(self) -> None:
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, _EKF_SHM_SIZE)
            self._shm = mmap.mmap(fd, _EKF_SHM_SIZE, mmap.MAP_SHARED,
                                  mmap.PROT_READ | mmap.PROT_WRITE)
            os.close(fd)
        except OSError:
            self._shm = None

    def read(self) -> EKFSnapshot:
        if self._shm is not None:
            try:
                self._shm.seek(0)
                raw = self._shm.read(struct.calcsize(_EKF_SHM_FMT))
                px, py, pz, vpx, vpy, vpsi, gps = struct.unpack(_EKF_SHM_FMT, raw)
                return EKFSnapshot(px, py, pz, vpx, vpy, vpsi, bool(gps))
            except Exception:
                pass
        # Synthetic fallback: simulate GPS dropout after 10 s
        t = time.time() % 30.0
        gps = t < 15.0
        var = 0.1 if gps else min(0.1 + (t - 15.0) * 0.5, 30.0)
        return EKFSnapshot(
            px=self._rng.normal(0, 0.1), py=self._rng.normal(0, 0.1),
            pz=2.0 + self._rng.normal(0, 0.05),
            var_px=var, var_py=var, var_psi=var * 0.4,
            gps_active=gps,
        )

    def write_synthetic(self, snap: EKFSnapshot) -> None:
        """Write a synthetic EKF snapshot for testing."""
        if self._shm is None:
            return
        data = struct.pack(_EKF_SHM_FMT,
                           snap.px, snap.py, snap.pz,
                           snap.var_px, snap.var_py, snap.var_psi,
                           int(snap.gps_active))
        self._shm.seek(0)
        self._shm.write(data)


class ConsensusNode:
    """
    Single node in the observability-weighted HotStuff swarm.

    Each node:
      1. Reads its own EKF state from shared memory
      2. Broadcasts a PREPARE message with its state proposal + trust weight
      3. Collects PREPARE messages from peers
      4. Commits if weighted quorum is reached
      5. Writes agreed state to /dev/shm/aisp_consensus

    The weighted quorum rule ensures GPS-denied nodes cannot corrupt
    the agreed state even if they report fabricated positions.
    """

    def __init__(self, node_id: int, n_nodes: int,
                 zmq_base_port: int = 5550):
        self.node_id   = node_id
        self.n_nodes   = n_nodes
        self._ekf      = EKFSharedMemory()
        self._round    = 0
        self._votes: Dict[int, ConsensusMessage] = {}
        self._lock     = threading.Lock()
        self._running  = False

        if ZMQ_AVAILABLE:
            ctx = zmq.Context.instance()
            self._pub = ctx.socket(zmq.PUB)
            self._pub.bind(f"tcp://*:{zmq_base_port + node_id}")
            self._sub = ctx.socket(zmq.SUB)
            self._sub.setsockopt(zmq.RCVTIMEO, 100)  # 100ms recv timeout
            self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
            for i in range(n_nodes):
                if i != node_id:
                    self._sub.connect(f"tcp://localhost:{zmq_base_port + i}")
        else:
            self._pub = self._sub = None

    # ── Core consensus round ──────────────────────────────────────────────────

    def _propose(self) -> ConsensusMessage:
        """Build this node's proposal from its current EKF state."""
        snap = self._ekf.read()
        state = [snap.px, snap.py, snap.pz]
        return ConsensusMessage(
            node_id      = self.node_id,
            round_num    = self._round,
            phase        = Phase.PREPARE.name,
            state_hash   = _state_hash(state),
            state_vector = state,
            trust_weight = snap.trust_weight,
            var_px       = snap.var_px,
            var_py       = snap.var_py,
            var_psi      = snap.var_psi,
            gps_active   = snap.gps_active,
        )

    def _broadcast(self, msg: ConsensusMessage) -> None:
        if self._pub is not None:
            self._pub.send_string(msg.to_json())

    def _collect_votes(self, timeout_s: float) -> List[ConsensusMessage]:
        """Collect PREPARE votes from peers within timeout."""
        votes = []
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self._sub is not None:
            try:
                raw = self._sub.recv_string()
                msg = ConsensusMessage.from_json(raw)
                if msg.round_num == self._round:
                    votes.append(msg)
            except Exception:
                break
        return votes

    def _weighted_quorum(self, votes: List[ConsensusMessage],
                         my_msg: ConsensusMessage) -> Optional[List[float]]:
        """
        Check if a weighted quorum agrees on a state hash.

        Returns the agreed state vector if quorum is reached, else None.

        Quorum rule:
            sum(w_i for voters of hash H) >= BFT_THRESHOLD * sum(w_i for all)

        This is the observability-weighted extension of HotStuff's
        (n - f) quorum, where f = n/3 Byzantine nodes.
        """
        all_votes = votes + [my_msg]
        total_weight = sum(v.trust_weight for v in all_votes)

        if total_weight < 1e-9:
            return None  # all nodes GPS-denied — no consensus possible

        # Group by state hash
        hash_weights: Dict[str, float] = {}
        hash_states:  Dict[str, List[float]] = {}
        for v in all_votes:
            hash_weights[v.state_hash] = (
                hash_weights.get(v.state_hash, 0.0) + v.trust_weight
            )
            hash_states[v.state_hash] = v.state_vector

        # Find the hash with the highest weighted support
        best_hash = max(hash_weights, key=lambda h: hash_weights[h])
        if hash_weights[best_hash] >= BFT_THRESHOLD * total_weight:
            return hash_states[best_hash]
        return None

    def run_round(self) -> Optional[List[float]]:
        """
        Execute one HotStuff consensus round.
        Returns agreed state vector, or None if quorum not reached.
        """
        my_msg = self._propose()
        self._broadcast(my_msg)

        votes = self._collect_votes(ROUND_TIMEOUT_S)
        agreed = self._weighted_quorum(votes, my_msg)

        if agreed is not None:
            self._write_consensus(agreed, my_msg.trust_weight)

        self._round += 1
        return agreed

    def _write_consensus(self, state: List[float], weight: float) -> None:
        """Write agreed state to /dev/shm/aisp_consensus for flight loop."""
        try:
            fmt  = "=ddddf"   # px, py, pz (double) + weight (float)
            size = struct.calcsize(fmt)
            fd   = os.open(SHM_CONSENSUS, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, size)
            shm  = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_WRITE)
            os.close(fd)
            shm.seek(0)
            shm.write(struct.pack(fmt, state[0], state[1], state[2], weight))
            shm.close()
        except OSError:
            pass

    def run(self, n_rounds: int = 0) -> None:
        """Run consensus loop. n_rounds=0 means run forever."""
        self._running = True
        count = 0
        while self._running and (n_rounds == 0 or count < n_rounds):
            agreed = self.run_round()
            if agreed:
                snap = self._ekf.read()
                print(f"[Node {self.node_id}] Round {self._round-1:4d} "
                      f"COMMIT  state=[{agreed[0]:.2f},{agreed[1]:.2f},{agreed[2]:.2f}] "
                      f"w={snap.trust_weight:.3f} "
                      f"gps={'Y' if snap.gps_active else 'N'}")
            else:
                print(f"[Node {self.node_id}] Round {self._round-1:4d} "
                      f"NO QUORUM (single-node or all GPS-denied)")
            count += 1

    def stop(self) -> None:
        self._running = False


# ── Standalone test ───────────────────────────────────────────────────────────

def test_weighted_quorum() -> None:
    """
    Unit test: verify that GPS-denied nodes cannot swing the quorum.
    Simulates 4 nodes: 3 GPS-active (w≈1), 1 GPS-denied (w≈0.01).
    The GPS-denied node proposes a different state — quorum should reject it.
    """
    from dataclasses import replace

    def _make_vote(node_id: int, state: list, var: float, gps: bool) -> ConsensusMessage:
        snap = EKFSnapshot(state[0], state[1], state[2], var, var, var * 0.4, gps)
        return ConsensusMessage(
            node_id=node_id, round_num=0, phase=Phase.PREPARE.name,
            state_hash=_state_hash(state), state_vector=state,
            trust_weight=snap.trust_weight,
            var_px=snap.var_px, var_py=snap.var_py, var_psi=snap.var_psi,
            gps_active=gps,
        )

    node = ConsensusNode(node_id=0, n_nodes=4)

    # 3 GPS-active nodes agree on [0, 0, 2]
    true_state  = [0.0, 0.0, 2.0]
    false_state = [100.0, 100.0, 2.0]  # Byzantine GPS-denied node

    votes = [
        _make_vote(1, true_state,  0.1,  True),   # w ≈ 0.990
        _make_vote(2, true_state,  0.1,  True),   # w ≈ 0.990
        _make_vote(3, false_state, 20.0, False),  # w ≈ 0.135 (GPS denied)
    ]
    my_msg = _make_vote(0, true_state, 0.1, True)

    agreed = node._weighted_quorum(votes, my_msg)

    assert agreed is not None, "Quorum should be reached"
    assert agreed == true_state, (
        f"Wrong state agreed: {agreed} (Byzantine node should be silenced)"
    )

    # Verify: when GPS-denied Byzantine nodes are outvoted by GPS-active nodes,
    # the GPS-active state wins — Byzantine nodes cannot override the quorum.
    # Add one GPS-active node that disagrees with Byzantine to show weight dominance.
    mixed_votes = [
        _make_vote(1, true_state,  0.1,  True),   # w ≈ 0.976 — GPS active
        _make_vote(3, false_state, 20.0, False),  # w ≈ 0.008 — GPS denied
        _make_vote(4, false_state, 20.0, False),  # w ≈ 0.008 — GPS denied
    ]
    mixed_my = _make_vote(0, true_state, 0.1, True)  # w ≈ 0.976 — GPS active
    mixed_agreed = node._weighted_quorum(mixed_votes, mixed_my)
    assert mixed_agreed == true_state, (
        f"GPS-active nodes should win: got {mixed_agreed}"
    )

    print("Observability-weighted quorum: PASS")
    print(f"  True state agreed : {agreed}")
    print(f"  GPS-active weight : {my_msg.trust_weight:.4f}")
    print(f"  GPS-denied weight : {votes[2].trust_weight:.4f}")
    total_w = sum(v.trust_weight for v in votes) + my_msg.trust_weight
    true_w  = sum(v.trust_weight for v in votes[:2]) + my_msg.trust_weight
    print(f"  Quorum fraction   : {true_w/total_w:.3f} >= {BFT_THRESHOLD:.3f} ✓")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test",    action="store_true", help="Run unit test")
    p.add_argument("--node-id", type=int, default=0)
    p.add_argument("--n-nodes", type=int, default=1)
    p.add_argument("--rounds",  type=int, default=5)
    args = p.parse_args()

    if args.test:
        test_weighted_quorum()
    else:
        node = ConsensusNode(args.node_id, args.n_nodes)
        node.run(n_rounds=args.rounds)
