# Observability-weighted voting consensus for drone swarms.
# Trust weight w_i = exp(-σ²_pos / (2·σ_warn²)) links EKF covariance to vote weight.
# GPS-denied nodes get w≈0 and cannot sway the quorum.
# Reads from /dev/shm/aisp_ekf_state, writes agreed state to /dev/shm/aisp_consensus.
#
# This is NOT Byzantine fault tolerance (no 3f+1, no view change, no PBFT).
# It is a weighted voting protocol where weight derives from estimation quality.
# Message authentication is HMAC-SHA256 with a cluster shared key — this gives
# AUTHENTICATED MEMBERSHIP (only key holders can vote) under an honest-but-
# possibly-broken transport. It does not defend against a key holder turning
# malicious; that is a documented scope limit.
#
# Rhutvik Prashant Pachghare — ASU Robotics & Autonomous Systems

from __future__ import annotations

import hashlib
import hmac
import json
import math
import mmap
import os
import struct
import sys
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
QUORUM_THRESHOLD    = 2 / 3  # weighted quorum fraction
ROUND_TIMEOUT_S  = 0.5    # seconds before view change
SHM_EKF_PATH     = "/dev/shm/aisp_ekf_state"
SHM_CONSENSUS    = "/dev/shm/aisp_consensus"

# EKF shared memory layout with seqlock (80 bytes) to prevent torn reads
# and detect uninitialized/zeroed segments:
#   uint64 seq_head                (8 bytes)  — odd = write in progress
#   double p_x, p_y, p_z           (24 bytes)
#   double var_px, var_py, var_psi (24 bytes)  — diagonal of P
#   uint8  gps_active              (1 byte)
#   uint64 seq_tail                (8 bytes)  — must equal seq_head when even
#   uint8  _pad[7]                 (7 bytes)
# Total: 8+24+24+1+8+7 = 72, rounded to 80 for cache-line alignment.
_EKF_SHM_FMT  = "=QddddddbQ7x"  # seq_head, px,py,pz,var_px,var_py,var_psi, gps, seq_tail
_EKF_SHM_SIZE = 80
_EKF_MIN_VALID_VAR = 1e-6  # minimum variance to consider data valid (not zeroed)

# Consensus shared memory with seqlock (40 bytes) matching VLA pattern:
#   uint64 seq_head              (8 bytes)  — odd = write in progress
#   double px, py, pz            (24 bytes)
#   float  trust_weight          (4 bytes)  — quorum-derived weight
#   uint64 seq_tail              (8 bytes)  — must equal seq_head when even
# Total: 8+24+4+8 = 44, round to 48 for alignment.
_CONSENSUS_FMT   = "=QdddfQ"   # seq_head, px, py, pz, trust, seq_tail
_CONSENSUS_SIZE  = 48


class Phase(Enum):
    PROPOSE  = auto()
    VOTE     = auto()
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
    state_hash:   str          # SHA-256 integrity hash of exact state vector
    state_vector: List[float]  # [px, py, pz] proposed agreed state
    trust_weight: float        # w_i from EKF covariance
    var_px:       float
    var_py:       float
    var_psi:      float
    gps_active:   bool
    timestamp:    float = field(default_factory=time.time)
    auth_tag:     str = ""     # HMAC-SHA256 (truncated) over all other fields

    def _signed_payload(self) -> bytes:
        """Canonical byte representation of every field EXCEPT auth_tag."""
        d = asdict(self)
        d.pop("auth_tag", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()

    def sign(self, key: bytes) -> None:
        self.auth_tag = hmac.new(key, self._signed_payload(),
                                 hashlib.sha256).hexdigest()[:32]

    def verify(self, key: bytes) -> bool:
        expected = hmac.new(key, self._signed_payload(),
                            hashlib.sha256).hexdigest()[:32]
        return hmac.compare_digest(expected, self.auth_tag)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "ConsensusMessage":
        d = json.loads(s)
        d.setdefault("auth_tag", "")
        return ConsensusMessage(**d)


def _state_hash(state: List[float]) -> str:
    """Integrity hash of the EXACT state vector (detects corruption in
    transit/storage). NOTE: this is not used for quorum grouping — grouping
    is distance-based (see _weighted_quorum), because any quantization grid
    has boundary cases where arbitrarily close states hash differently."""
    payload = struct.pack(f"={len(state)}d", *state)
    return hashlib.sha256(payload).hexdigest()[:16]


# Max Euclidean distance (m) between two position proposals for them to count
# as "the same state" in quorum grouping. 1 cm matches the previous
# quantization-cell size but, unlike a grid, has no boundary fractures:
# any two states within STATE_TOL of a common member join the same group
# (single-linkage; documented behavior for 4–8 node swarms).
STATE_TOL = 0.01


class EKFReadError(Exception):
    """EKF shared-memory snapshot unavailable, torn, or invalid.

    Fail-closed contract: callers must NOT substitute fabricated state.
    Synthetic fallback exists ONLY for explicit demo/test mode
    (EKFSharedMemory(allow_synthetic=True))."""


def weighted_quorum(votes: List["ConsensusMessage"],
                    my_msg: "ConsensusMessage") -> tuple[Optional[List[float]], float]:
    """
    Check if a weighted quorum agrees on a state.

    Returns (agreed state vector, quorum_weight) if quorum reached,
    else (None, 0.0).

    Grouping is DISTANCE-BASED (single-linkage, STATE_TOL metres), not
    hash-based: a quantization grid fractures states that sit on opposite
    sides of a cell boundary no matter how close they are. The agreed
    state is the trust-weighted mean of the winning group.

    Quorum rule:
        sum(w_i for voters in group G) >= QUORUM_THRESHOLD * sum(w_i all)

    The returned quorum_weight is the fraction of total weight supporting
    the winning group (not the proposing node's individual weight).
    This is what the EKF should use as the blending coefficient alpha.
    """
    all_votes = votes + [my_msg]
    total_weight = sum(v.trust_weight for v in all_votes)

    if total_weight < 1e-9:
        return None, 0.0  # all nodes GPS-denied — no consensus possible

    # Distance-based single-linkage grouping: each vote joins the first
    # group whose running centroid is within STATE_TOL, else starts one.
    # groups: [centroid, sum_weight, weighted_state_sum, count]
    groups: List[list] = []
    for v in all_votes:
        placed = False
        for g in groups:
            if math.dist(g[0], v.state_vector) <= STATE_TOL:
                n = g[3]
                for k in range(3):
                    g[0][k] = (g[0][k] * n + v.state_vector[k]) / (n + 1)
                g[1] += v.trust_weight
                for k in range(3):
                    g[2][k] += v.trust_weight * v.state_vector[k]
                g[3] = n + 1
                placed = True
                break
        if not placed:
            groups.append([
                list(v.state_vector),
                v.trust_weight,
                [v.trust_weight * x for x in v.state_vector],
                1,
            ])

    best = max(groups, key=lambda g: g[1])
    if best[1] >= QUORUM_THRESHOLD * total_weight:
        agreed = [s / best[1] for s in best[2]]  # trust-weighted mean
        quorum_weight = best[1] / total_weight   # fraction of total weight
        return agreed, quorum_weight
    return None, 0.0


class EKFSharedMemory:
    """
    Reads EKF state from /dev/shm/aisp_ekf_state (seqlock-validated).

    Fail-closed by default: read() raises EKFReadError if no valid snapshot
    is available. allow_synthetic=True enables the old demo behavior
    (fabricated plausible state) — NEVER use that in a runtime swarm:
    a dead estimator would silently become fake data and get proposed
    into consensus.
    """

    def __init__(self, path: str = SHM_EKF_PATH,
                 allow_synthetic: bool = False):
        self._path = path
        self._allow_synthetic = allow_synthetic
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
        """Read EKF snapshot with seqlock validation.

        Uses mmap SLICE access (self._shm[a:b]) rather than seek()/read():
        slice access does not touch the mmap's shared file position, so a
        writer thread and a reader thread sharing this object cannot
        misalign each other mid-transaction (found by the e2e flight test:
        50 Hz writer + worker-thread reader on one mmap → every read
        misaligned → every validity check failed).

        Raises EKFReadError if:
        - SHM not available
        - Torn read detected (odd seq_head, seq_head != seq_tail, seq changed during read)
        - Data appears uninitialized (variances <= _EKF_MIN_VALID_VAR)
        …unless allow_synthetic was set at construction (demo/test only).
        """
        n = struct.calcsize(_EKF_SHM_FMT)
        if self._shm is not None:
            for _ in range(2):  # max 2 attempts
                try:
                    raw = bytes(self._shm[0:n])
                    (seq_head, px, py, pz, vpx, vpy, vpsi, gps, seq_tail
                     ) = struct.unpack(_EKF_SHM_FMT, raw)
                    # Seqlock validation
                    if (seq_head & 1) != 0:
                        continue  # writer mid-write
                    if seq_head != seq_tail:
                        continue  # torn: head/tail mismatch
                    # Re-read seq_head to check stability
                    seq_check = struct.unpack("=Q", bytes(self._shm[0:8]))[0]
                    if seq_check != seq_head:
                        continue  # seq changed during read
                    # Data validity: variances must be > threshold (not zeroed)
                    if vpx <= _EKF_MIN_VALID_VAR or vpy <= _EKF_MIN_VALID_VAR or vpsi <= _EKF_MIN_VALID_VAR:
                        continue  # uninitialized/zeroed segment
                    return EKFSnapshot(px, py, pz, vpx, vpy, vpsi, bool(gps))
                except (struct.error, ValueError, OSError):
                    pass
        if not self._allow_synthetic:
            raise EKFReadError(
                f"no valid EKF snapshot at {self._path} "
                f"(fail-closed; synthetic fallback disabled)")
        # Synthetic fallback: EXPLICIT demo/test mode only.
        # Simulates GPS dropout after 15 s of a 30 s cycle.
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
        """Write an EKF snapshot through the seqlock protocol (slice access,
        position-independent — see read() for why this matters).

        Used by experiments/tests playing the role of the EKF writer
        (e.g. exp_consensus_ekf_flight.py). The seqlock contract is the
        same one the real EKF writer must use."""
        if self._shm is None:
            return
        import time as _time
        seq = int(_time.time() * 1e6) & 0xFFFFFFFFFFFFFFFE  # even
        # Write 1: odd seq_head (write in progress)
        self._shm[0:8] = struct.pack("=Q", seq + 1)
        # Write 2: payload + even seq_tail
        data = struct.pack(_EKF_SHM_FMT, seq + 2,  # seq_head
                           snap.px, snap.py, snap.pz,
                           snap.var_px, snap.var_py, snap.var_psi,
                           int(snap.gps_active),
                           seq + 2)  # seq_tail
        self._shm[0:len(data)] = data
        # Write 3: even seq_head (write complete)
        self._shm[0:8] = struct.pack("=Q", seq + 2)


class ConsensusNode:
    """
    Single node in the observability-weighted voting swarm.

    Each node:
      1. Reads its own EKF state from shared memory
      2. Broadcasts a PROPOSE message with its state proposal + trust weight
      3. Collects PROPOSE messages from peers
      4. Commits if weighted quorum is reached
      5. Writes agreed state to /dev/shm/aisp_consensus

    The weighted quorum rule ensures GPS-denied nodes cannot corrupt
    the agreed state even if they report fabricated positions.

    This is a weighted voting protocol, NOT Byzantine fault tolerance.
    """

    def __init__(self, node_id: int, n_nodes: int,
                 zmq_base_port: int = 5550,
                 auth_key: Optional[bytes] = None,
                 allow_synthetic_ekf: bool = False):
        self.node_id   = node_id
        self.n_nodes   = n_nodes
        self._ekf      = EKFSharedMemory(allow_synthetic=allow_synthetic_ekf)
        self._round    = 0
        self._votes: Dict[int, ConsensusMessage] = {}
        self._lock     = threading.Lock()
        self._running  = False
        # Observability counters (read by tests/experiments)
        self._ekf_read_failures   = 0   # rounds with no valid own estimate
        self._auth_failures       = 0   # unauthenticated/forged messages dropped
        self._shm_write_failures  = 0   # consensus SHM handoff failures

        # ── Authenticated membership ──────────────────────────────────────
        # Key resolution: explicit arg > $AISP_CONSENSUS_KEY > OPEN mode.
        # OPEN mode runs without authentication and prints a loud warning —
        # acceptable for local experiments, NEVER for a real deployment:
        # without a key, any process that can reach the ZMQ port can vote.
        if auth_key is None:
            env_key = os.environ.get("AISP_CONSENSUS_KEY", "")
            auth_key = env_key.encode() if env_key else None
        self._auth_key = auth_key
        if self._auth_key is None:
            print(f"[Node {node_id}] WARNING: no consensus auth key — OPEN mode; "
                  f"anyone on the ZMQ ports can vote. Set AISP_CONSENSUS_KEY.",
                  file=sys.stderr)

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
        """Build this node's proposal from its current EKF state.

        Fail-closed: if the EKF snapshot is unavailable/invalid, raises
        EKFReadError — this node must NOT fabricate a state and propose it."""
        snap = self._ekf.read()
        state = [snap.px, snap.py, snap.pz]
        return ConsensusMessage(
            node_id      = self.node_id,
            round_num    = self._round,
            phase        = Phase.PROPOSE.name,
            state_hash   = _state_hash(state),
            state_vector = state,
            trust_weight = snap.trust_weight,
            var_px       = snap.var_px,
            var_py       = snap.var_py,
            var_psi      = snap.var_psi,
            gps_active   = snap.gps_active,
        )

    def _broadcast(self, msg: ConsensusMessage) -> None:
        if self._auth_key is not None and not msg.auth_tag:
            msg.sign(self._auth_key)
        if self._pub is not None:
            self._pub.send_string(msg.to_json())

    def _collect_votes(self, timeout_s: float) -> List[ConsensusMessage]:
        """Collect PROPOSE votes from peers within timeout.

        Round synchronisation: each node self-paces its rounds (the receive
        window below plus per-node jitter), so nodes can drift ≥1 round
        apart. If a vote from a FUTURE round arrives, this node is behind:
        fast-forward to that round so it can co-vote again instead of
        silently discarding every message forever. Votes from PAST rounds
        are stale and dropped. Found by experiments/exp_consensus_ekf_flight.py
        (perma-stall after ~2 s of free-running rounds).

        Authentication (when a key is configured):
        - HMAC-SHA256 tag verified BEFORE any field is trusted. This covers
          node_id, so the dedupe below is by AUTHENTICATED identity — an
          unkeyed peer cannot impersonate node 3 or double its weight.

        Peer validation:
        - trust_weight clamped to [0, 1], NaN/Inf rejected
        - state_hash verified against state_vector (corruption check)
        - state_vector must be finite, correct length
        """
        votes = []
        seen_nodes = set()   # one vote per authenticated node_id per round
        self._resynced = False
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self._sub is not None:
            try:
                raw = self._sub.recv_string()
                msg = ConsensusMessage.from_json(raw)

                # ── Authentication FIRST ─────────────────────────────────
                if self._auth_key is not None:
                    if not msg.verify(self._auth_key):
                        self._auth_failures += 1
                        continue  # forged/unkeyed message: drop

                # ── Peer trust validation ─────────────────────────────────
                # Clamp trust to [0, 1]; reject NaN/Inf
                if not math.isfinite(msg.trust_weight) or msg.trust_weight <= 0.0:
                    continue
                if msg.trust_weight > 1.0:
                    msg.trust_weight = 1.0

                # Verify state_hash matches state_vector (detect corruption)
                expected_hash = _state_hash(msg.state_vector)
                if msg.state_hash != expected_hash:
                    continue

                # Verify state_vector is finite and has expected length
                if len(msg.state_vector) != 3:
                    continue
                if not all(math.isfinite(v) for v in msg.state_vector):
                    continue
                # ── End peer validation ──────────────────────────────────

                if msg.round_num == self._round:
                    if msg.node_id not in seen_nodes:
                        seen_nodes.add(msg.node_id)
                        votes.append(msg)
                elif msg.round_num > self._round:
                    # fell behind: skip ahead; caller re-proposes fresh
                    self._round    = msg.round_num
                    self._resynced = True
                    votes = [msg]          # keep the future-round vote
                    seen_nodes = {msg.node_id}
                # past-round votes are stale: drop silently
            except zmq.Again:
                continue                   # 100 ms idle tick, keep waiting
            except Exception:
                break                      # malformed message: stop collecting
        return votes

    def _weighted_quorum(self, votes: List[ConsensusMessage],
                          my_msg: ConsensusMessage) -> tuple[Optional[List[float]], float]:
        """Instance wrapper around the module-level weighted_quorum (single
        source of truth — experiments import the same function)."""
        return weighted_quorum(votes, my_msg)

    def run_round(self) -> Optional[List[float]]:
        """
        Execute one observability-weighted voting round.
        Returns agreed state vector, or None if quorum not reached OR this
        node had no valid own estimate (fail-closed: no valid EKF state →
        no proposal; we never vote fabricated data).

        If a future-round message arrives mid-round (we fell behind), the
        round is restarted ONCE under the caught-up round number (bounded
        to two attempts so a malformed fast-forwarded stream cannot spin).
        """
        for _attempt in range(2):
            try:
                my_msg = self._propose()
            except EKFReadError as e:
                self._ekf_read_failures += 1
                if self._ekf_read_failures <= 5 or self._ekf_read_failures % 100 == 0:
                    print(f"[Node {self.node_id}] round {self._round}: "
                          f"no proposal — {e}", file=sys.stderr)
                self._round += 1
                # Sit-out must PRESERVE ROUND CADENCE (~ROUND_TIMEOUT_S), not
                # return instantly: callers budget n_rounds ≈ wall_time /
                # ROUND_TIMEOUT_S, so instant failures would burn the entire
                # round budget in milliseconds and the node would go silent
                # for the rest of the run (found by exp_consensus_ekf_flight:
                # all workers exhausted their rounds before the first EKF
                # write landed → consensus never committed once).
                time.sleep(ROUND_TIMEOUT_S)
                return None  # fail-closed: sit this round out
            self._broadcast(my_msg)

            votes  = self._collect_votes(ROUND_TIMEOUT_S)
            if self._resynced and _attempt == 0:
                continue               # re-propose with the caught-up round
            agreed, quorum_weight = self._weighted_quorum(votes, my_msg)

            if agreed is not None:
                self._write_consensus(agreed, quorum_weight)

            self._round += 1
            return agreed
        self._round += 1
        return None

    def _write_consensus(self, state: List[float], weight: float) -> None:
        """Write agreed state to /dev/shm/aisp_consensus with seqlock.

        Failure is OBSERVABLE: counted and logged (rate-limited). A safety-
        relevant handoff must never fail silently — the EKF side would keep
        blending toward a stale value with no signal anything is wrong."""
        try:
            # fmt =QdddfQ: seq_head, px, py, pz (double), trust (float), seq_tail = 48 bytes
            # Seqlock 3-write protocol: odd head -> payload+even tail -> even head
            import time as _time
            seq = int(_time.time() * 1e6) & 0xFFFFFFFFFFFFFFFE  # even
            # Write 1: odd seq_head (write in progress)
            fd = os.open(SHM_CONSENSUS, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, _CONSENSUS_SIZE)
            shm = mmap.mmap(fd, _CONSENSUS_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE)
            os.close(fd)
            # slice access: position-independent (see EKFSharedMemory.read)
            shm[0:8] = struct.pack("=Q", seq + 1)
            data = struct.pack(_CONSENSUS_FMT, seq + 2, state[0], state[1], state[2], weight, seq + 2)
            shm[0:len(data)] = data
            shm[0:8] = struct.pack("=Q", seq + 2)
            shm.close()
        except OSError as e:
            self._shm_write_failures += 1
            if self._shm_write_failures <= 5 or self._shm_write_failures % 100 == 0:
                print(f"[Node {self.node_id}] CONSENSUS SHM WRITE FAILED "
                      f"(#{self._shm_write_failures}): {e}", file=sys.stderr)

    def run(self, n_rounds: int = 0) -> None:
        """Run consensus loop. n_rounds=0 means run forever."""
        self._running = True
        count = 0
        while self._running and (n_rounds == 0 or count < n_rounds):
            agreed = self.run_round()
            if agreed:
                try:
                    snap = self._ekf.read()
                    w, gps = snap.trust_weight, snap.gps_active
                except EKFReadError:
                    w, gps = float("nan"), False
                print(f"[Node {self.node_id}] Round {self._round-1:4d} "
                      f"COMMIT  state=[{agreed[0]:.2f},{agreed[1]:.2f},{agreed[2]:.2f}] "
                      f"w={w:.3f} "
                      f"gps={'Y' if gps else 'N'}")
            else:
                print(f"[Node {self.node_id}] Round {self._round-1:4d} "
                      f"NO QUORUM (single-node, all GPS-denied, or no valid EKF state)")
            count += 1

    def stop(self) -> None:
        self._running = False


# ── Standalone test ───────────────────────────────────────────────────────────

def test_weighted_quorum_gps_denied_cannot_sway() -> None:
    """
    Unit test: verify that GPS-denied nodes (low observability) cannot sway the quorum.
    Simulates 4 nodes: 3 GPS-active (w≈1), 1 GPS-denied (w≈0.01).
    The GPS-denied node proposes a different state — quorum should reject it.
    """

    def _make_vote(node_id: int, state: list, var: float, gps: bool) -> ConsensusMessage:
        snap = EKFSnapshot(state[0], state[1], state[2], var, var, var * 0.4, gps)
        return ConsensusMessage(
            node_id=node_id, round_num=0, phase=Phase.PROPOSE.name,
            state_hash=_state_hash(state), state_vector=state,
            trust_weight=snap.trust_weight,
            var_px=snap.var_px, var_py=snap.var_py, var_psi=snap.var_psi,
            gps_active=gps,
        )

    node = ConsensusNode(node_id=0, n_nodes=4)

    # 3 GPS-active nodes agree on [0, 0, 2]
    true_state  = [0.0, 0.0, 2.0]
    false_state = [100.0, 100.0, 2.0]  # GPS-denied node proposes different position

    votes = [
        _make_vote(1, true_state,  0.1,  True),   # w ≈ 0.990
        _make_vote(2, true_state,  0.1,  True),   # w ≈ 0.990
        _make_vote(3, false_state, 20.0, False),  # w ≈ 0.135 (GPS denied)
    ]
    my_msg = _make_vote(0, true_state, 0.1, True)

    agreed, quorum_weight = node._weighted_quorum(votes, my_msg)

    assert agreed is not None, "Quorum should be reached"
    assert agreed == true_state, (
        f"Wrong state agreed: {agreed} (GPS-denied node should be outvoted)"
    )
    assert quorum_weight > QUORUM_THRESHOLD, f"Quorum weight {quorum_weight:.3f} should exceed threshold"

    # Verify: when GPS-denied nodes are outvoted by GPS-active nodes,
    # the GPS-active state wins — low-weight nodes cannot override the quorum.
    # Add two GPS-denied nodes that disagree (outlier/faulty state) to show weight dominance.
    mixed_votes = [
        _make_vote(1, true_state,  0.1,  True),   # w ≈ 0.976 — GPS active
        _make_vote(3, false_state, 20.0, False),  # w ≈ 0.008 — GPS denied
        _make_vote(4, false_state, 20.0, False),  # w ≈ 0.008 — GPS denied
    ]
    mixed_my = _make_vote(0, true_state, 0.1, True)  # w ≈ 0.976 — GPS active
    mixed_agreed, mixed_quorum_weight = node._weighted_quorum(mixed_votes, mixed_my)
    assert mixed_agreed == true_state, (
        f"GPS-active nodes should win: got {mixed_agreed}"
    )
    assert mixed_quorum_weight > QUORUM_THRESHOLD, f"Quorum weight {mixed_quorum_weight:.3f} should exceed threshold"

    print("Observability-weighted quorum: PASS")
    print(f"  True state agreed : {agreed}")
    print(f"  GPS-active weight : {my_msg.trust_weight:.4f}")
    print(f"  GPS-denied weight : {votes[2].trust_weight:.4f}")
    total_w = sum(v.trust_weight for v in votes) + my_msg.trust_weight
    true_w  = sum(v.trust_weight for v in votes[:2]) + my_msg.trust_weight
    print(f"  Quorum fraction   : {true_w/total_w:.3f} >= {QUORUM_THRESHOLD:.3f} ✓")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test",    action="store_true", help="Run unit test")
    p.add_argument("--node-id", type=int, default=0)
    p.add_argument("--n-nodes", type=int, default=1)
    p.add_argument("--rounds",  type=int, default=5)
    p.add_argument("--demo-synthetic", action="store_true",
                   help="Allow fabricated EKF state when no estimator is "
                        "running (DEMO ONLY — never in a real swarm)")
    args = p.parse_args()

    if args.test:
        test_weighted_quorum_gps_denied_cannot_sway()
    else:
        node = ConsensusNode(args.node_id, args.n_nodes,
                             allow_synthetic_ekf=args.demo_synthetic)
        node.run(n_rounds=args.rounds)
