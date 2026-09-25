"""Consensus security & fail-closed regression tests (2026-09-25 audit).

Covers the four findings from the fresh audit of main @ 311a776:
  1. EKF SHM read must fail CLOSED (EKFReadError), not silently substitute
     fabricated "synthetic" flight state into consensus proposals.
  2. Consensus votes must be authenticated (HMAC-SHA256, cluster shared key);
     unkeyed/wrong-key messages are dropped and counted. This is what makes
     node_id deduplication meaningful (authenticated identity).
  3. Quorum grouping must be distance-based: two states 2 mm apart that fall
     on opposite sides of the OLD 1 cm quantization grid boundary must still
     group (grid hashing fractured them).
  4. Consensus SHM write failure must be observable (counter), never silent.

Requires pyzmq for the wire tests — skipped where it isn't installed.
"""
import os
import sys
import time

import pytest

zmq = pytest.importorskip("zmq")  # noqa: F401

sys.path.insert(0, "services")
import consensus_node  # noqa: E402
from consensus_node import (  # noqa: E402
    ConsensusMessage, ConsensusNode, EKFReadError, EKFSharedMemory,
    EKFSnapshot, Phase, QUORUM_THRESHOLD, _state_hash, weighted_quorum,
)

BASE_PORT = 18700
KEY = b"test-cluster-key"


def _vote(node_id: int, state: list, trust: float) -> ConsensusMessage:
    return ConsensusMessage(
        node_id=node_id, round_num=0, phase=Phase.PROPOSE.name,
        state_hash=_state_hash(state), state_vector=state,
        trust_weight=trust, var_px=0.1, var_py=0.1, var_psi=0.04,
        gps_active=True,
    )


# ── 1. EKF fail-closed ───────────────────────────────────────────────────────

def test_ekf_read_fails_closed_on_zeroed_shm(tmp_path):
    """Zeroed/uninitialized SHM must raise, never fabricate flight state."""
    path = str(tmp_path / "ekf_zeroed")
    shm = EKFSharedMemory(path=path)  # creates + truncates → all zeros
    with pytest.raises(EKFReadError):
        shm.read()


def test_ekf_read_succeeds_after_valid_write(tmp_path):
    path = str(tmp_path / "ekf_valid")
    shm = EKFSharedMemory(path=path)
    shm.write_synthetic(EKFSnapshot(1.0, 2.0, 3.0, 0.1, 0.1, 0.04, True))
    snap = shm.read()
    assert (snap.px, snap.py, snap.pz) == (1.0, 2.0, 3.0)
    assert snap.gps_active is True


def test_ekf_synthetic_only_when_explicitly_allowed(tmp_path):
    path = str(tmp_path / "ekf_synth")
    shm = EKFSharedMemory(path=path, allow_synthetic=True)
    snap = shm.read()  # zeroed SHM, but demo mode → synthetic
    assert abs(snap.pz - 2.0) < 1.0  # plausible fabricated hover


def test_node_propose_fails_closed(tmp_path, monkeypatch):
    """A node with no valid estimate must sit the round out (no fake vote)."""
    monkeypatch.setattr(consensus_node, "SHM_EKF_PATH",
                        str(tmp_path / "ekf_missing"))
    node = ConsensusNode(0, 2, zmq_base_port=BASE_PORT + 20,
                         auth_key=KEY)
    agreed = node.run_round()  # EKF read fails → no proposal → None
    assert agreed is None
    assert node._ekf_read_failures >= 1
    node.stop()


# ── 2. Authenticated membership ──────────────────────────────────────────────

def test_wrong_key_votes_dropped_and_counted():
    """A peer with a different key is not a member: its votes must vanish."""
    a = ConsensusNode(0, 2, zmq_base_port=BASE_PORT,
                      auth_key=KEY, allow_synthetic_ekf=True)
    b = ConsensusNode(1, 2, zmq_base_port=BASE_PORT,
                      auth_key=b"WRONG-KEY", allow_synthetic_ekf=True)
    try:
        time.sleep(0.6)
        b._broadcast(b._propose())
        votes = a._collect_votes(0.8)
        assert all(v.node_id != 1 for v in votes), \
            "forged (wrong-key) vote accepted into quorum"
        assert a._auth_failures >= 1, "auth failure not counted"
    finally:
        a.stop()
        b.stop()


def test_same_key_votes_accepted():
    a = ConsensusNode(0, 2, zmq_base_port=BASE_PORT + 10,
                      auth_key=KEY, allow_synthetic_ekf=True)
    b = ConsensusNode(1, 2, zmq_base_port=BASE_PORT + 10,
                      auth_key=KEY, allow_synthetic_ekf=True)
    try:
        time.sleep(0.6)
        b._broadcast(b._propose())
        votes = a._collect_votes(0.8)
        assert any(v.node_id == 1 for v in votes), \
            "authenticated peer vote was dropped"
        assert a._auth_failures == 0
    finally:
        a.stop()
        b.stop()


def test_hmac_covers_node_id():
    """Re-tagging a valid message under a spoofed node_id must fail verify."""
    msg = _vote(3, [0.0, 0.0, 2.0], 0.9)
    msg.sign(KEY)
    assert msg.verify(KEY)
    forged = ConsensusMessage(**{**msg.__dict__, "node_id": 7})
    assert not forged.verify(KEY), "HMAC does not bind node_id (spoofable)"


# ── 3. Distance-based grouping (no grid-boundary fracture) ──────────────────

def test_grouping_has_no_quantization_boundary():
    """0.0049 vs 0.0051: 2 mm apart, opposite sides of the old 1-cm cell
    boundary (0.005). Old grid hashing split them and LOST quorum here;
    distance grouping must keep them together."""
    votes = [
        _vote(1, [0.0049, 0.0, 2.0], 0.4),
        _vote(2, [0.0049, 0.0, 2.0], 0.4),
    ]
    my = _vote(0, [0.0051, 0.0, 2.0], 0.5)
    agreed, qw = weighted_quorum(votes, my)
    assert agreed is not None, \
        "quorum lost across a 2 mm spread — grouping still grid-fractured"
    assert qw >= QUORUM_THRESHOLD
    # trust-weighted mean of {0.0049 (0.8 total), 0.0051 (0.5)}
    assert abs(agreed[0] - (0.0049 * 0.8 + 0.0051 * 0.5) / 1.3) < 1e-9


def test_far_apart_states_do_not_group():
    votes = [_vote(1, [0.0, 0.0, 2.0], 0.9)]
    my = _vote(0, [50.0, 0.0, 2.0], 0.9)
    agreed, _ = weighted_quorum(votes, my)
    assert agreed is None, "states 50 m apart must never form one group"


def test_agreed_state_is_trust_weighted_mean():
    votes = [_vote(1, [1.004, 0.0, 2.0], 1.0)]
    my = _vote(0, [1.000, 0.0, 2.0], 1.0)
    agreed, qw = weighted_quorum(votes, my)
    assert agreed is not None
    assert abs(agreed[0] - 1.002) < 1e-9, \
        f"expected weighted mean 1.002, got {agreed[0]}"


# ── 4. SHM write failure observability ───────────────────────────────────────

def test_consensus_write_failure_is_counted(monkeypatch):
    monkeypatch.setattr(consensus_node, "SHM_CONSENSUS",
                        "/proc/definitely-not-writable/consensus")
    node = ConsensusNode(0, 1, zmq_base_port=BASE_PORT + 30,
                         auth_key=KEY, allow_synthetic_ekf=True)
    try:
        node._write_consensus([0.0, 0.0, 2.0], 0.9)  # must NOT raise
        assert node._shm_write_failures == 1, \
            "SHM write failure was swallowed silently again"
    finally:
        node.stop()


def test_consensus_write_success_leaves_counter_clean(tmp_path, monkeypatch):
    out = str(tmp_path / "consensus_ok")
    monkeypatch.setattr(consensus_node, "SHM_CONSENSUS", out)
    node = ConsensusNode(0, 1, zmq_base_port=BASE_PORT + 40,
                         auth_key=KEY, allow_synthetic_ekf=True)
    try:
        node._write_consensus([1.0, 2.0, 3.0], 0.9)
        assert node._shm_write_failures == 0
        # and the reader side must see the value through the seqlock
        from src.estimation import ekf_gating
        monkeypatch.setattr(ekf_gating, "_CONSENSUS_SHM", out)
        pos, trust = ekf_gating.read_consensus_shm()
        assert pos is not None and abs(pos[0] - 1.0) < 1e-9
        assert abs(trust - 0.9) < 1e-6
    finally:
        node.stop()
