"""Consensus duplicate-vote regression test.

Fixed 2026-09-24 (node-audit Item C): a resynced peer can re-broadcast its
proposal for a round this node is still collecting (fast-forward restart in
run_round). Without dedupe, the same node_id would enter the quorum weight
twice. _collect_votes now keeps only the first vote per node_id per round.

Requires pyzmq — skipped where it isn't installed (CI installs it via the
'Vote analysis' experiment step deps).
"""
import sys
import time

import pytest

zmq = pytest.importorskip("zmq")  # noqa: F401

sys.path.insert(0, "services")
from consensus_node import ConsensusNode  # noqa: E402

BASE_PORT = 18650


def test_resync_rebroadcast_counts_once():
    """Two broadcasts of the same (node_id, round) proposal -> one vote."""
    # allow_synthetic_ekf: these are wire-protocol tests, not estimator tests;
    # the proposal content is irrelevant, only dedupe behavior matters.
    a = ConsensusNode(0, 2, zmq_base_port=BASE_PORT, allow_synthetic_ekf=True)
    b = ConsensusNode(1, 2, zmq_base_port=BASE_PORT, allow_synthetic_ekf=True)
    try:
        time.sleep(0.6)  # let SUB connect + subscription propagate
        for _ in range(2):  # the resynced-peer duplicate
            b._broadcast(b._propose())
            time.sleep(0.05)
        votes = a._collect_votes(0.8)
        from_b = [v for v in votes if v.node_id == 1]
        assert len(from_b) == 1, (
            f"duplicate vote counted: got {len(from_b)} from node 1 "
            f"(quorum weight would double-count)"
        )
    finally:
        a.stop()
        b.stop()


def test_same_round_from_both_peers_kept():
    """Distinct node_ids in the same round must BOTH be kept."""
    a = ConsensusNode(0, 3, zmq_base_port=BASE_PORT + 10, allow_synthetic_ekf=True)
    b = ConsensusNode(1, 3, zmq_base_port=BASE_PORT + 10, allow_synthetic_ekf=True)
    c = ConsensusNode(2, 3, zmq_base_port=BASE_PORT + 10, allow_synthetic_ekf=True)
    try:
        time.sleep(0.6)
        b._broadcast(b._propose())
        c._broadcast(c._propose())
        votes = a._collect_votes(0.8)
        ids = sorted(v.node_id for v in votes)
        assert ids == [1, 2], f"expected votes from nodes 1 and 2, got {ids}"
    finally:
        a.stop()
        b.stop()
        c.stop()
