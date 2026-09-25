"""EKF SHM wire-contract test: production writer ↔ consensus reader.

services/consensus_node.py (reader) and src/estimation/ekf_shm_writer.py
(writer) intentionally define their own format constants so neither side
can drift silently — this test pins them together and verifies the
round-trip through a real file.
"""
import sys

import pytest

zmq = pytest.importorskip("zmq")  # consensus_node imports zmq  # noqa: F401

sys.path.insert(0, "services")
import consensus_node  # noqa: E402
from consensus_node import EKFReadError, EKFSharedMemory  # noqa: E402
from src.estimation.ekf_shm_writer import (  # noqa: E402
    EKF_SHM_FMT, EKF_SHM_SIZE, EKF_MIN_VALID_VAR, EKFShmWriter,
)


def test_format_constants_match_reader():
    assert EKF_SHM_FMT == consensus_node._EKF_SHM_FMT
    assert EKF_SHM_SIZE == consensus_node._EKF_SHM_SIZE
    assert EKF_MIN_VALID_VAR == consensus_node._EKF_MIN_VALID_VAR


def test_round_trip_writer_to_reader(tmp_path):
    path = str(tmp_path / "ekf_rt")
    w = EKFShmWriter(path)
    r = EKFSharedMemory(path=path)  # fail-closed reader (no synthetic)
    w.write(1.5, -2.5, 3.0, 0.2, 0.3, 0.08, True)
    snap = r.read()
    assert (snap.px, snap.py, snap.pz) == (1.5, -2.5, 3.0)
    assert (snap.var_px, snap.var_py, snap.var_psi) == (0.2, 0.3, 0.08)
    assert snap.gps_active is True
    assert w.write_failures == 0
    w.close()


def test_reader_sees_latest_write_and_valid_seq(tmp_path):
    path = str(tmp_path / "ekf_seq")
    w = EKFShmWriter(path)
    r = EKFSharedMemory(path=path)
    w.write(0.0, 0.0, 2.0, 0.1, 0.1, 0.04, True)
    w.write(1.0, 0.0, 2.0, 0.1, 0.1, 0.04, False)
    snap = r.read()
    assert snap.px == 1.0 and snap.gps_active is False
    w.close()


def test_gating_publishes_when_enabled(tmp_path):
    """CovarianceGating with ekf_shm_path publishes the safe state each cycle."""
    import numpy as np
    from src.estimation.ekf_gating import CovarianceGating, IDX_PX, IDX_PZ, IDX_QW

    path = str(tmp_path / "ekf_gating_pub")
    gating = CovarianceGating(seed=0, ekf_shm_path=path)
    x = np.zeros(15)
    x[IDX_PX] = 1.25
    x[IDX_PZ] = 2.0
    x[IDX_QW] = 1.0
    P = np.eye(15) * 0.1
    gating.evaluate(x, P, gps_active=True)

    snap = EKFSharedMemory(path=path).read()
    assert abs(snap.px - 1.25) < 1e-12
    assert abs(snap.var_psi - 4.0 * 0.1) < 1e-12  # 4·P[qz,qz] contract
    assert snap.gps_active is True


def test_gating_without_path_stays_silent():
    from src.estimation.ekf_gating import CovarianceGating
    gating = CovarianceGating(seed=0)
    assert gating._ekf_writer is None
