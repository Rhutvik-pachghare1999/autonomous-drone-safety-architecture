"""EKF → consensus shared-memory writer (production writer for
/dev/shm/aisp_ekf_state).

The consensus node (services/consensus_node.py EKFSharedMemory) reads each
vehicle's estimator snapshot from shared memory, fail-closed. Until now no
component in the repo wrote that file — the reader could only ever sit out
rounds. This module is the writer side: CovarianceGating publishes the
vehicle's own estimate (position + covariance diagonal + GPS status) after
every gating cycle.

Wire contract (pinned by tests/test_ekf_shm_contract.py against the reader
in services/consensus_node.py — the two files intentionally each define
their own constants so neither side can drift silently):

  segment size : 80 bytes (72 used)
  fmt "=QddddddbQ7x":
    uint64 seq_head     — odd while a write is in progress
    double px, py, pz   — estimated position (m)
    double var_px       — P[px,px]
    double var_py       — P[py,py]
    double var_psi      — linearised yaw variance 4·P[qz,qz] (matches
                          CovarianceGating._health_scalar so the swarm's
                          trust in this node equals its own health metric)
    uint8  gps_active
    uint64 seq_tail     — equals seq_head when the write is complete

Protocol: single-producer seqlock, 3 writes (odd head → payload+even tail →
even head), mmap SLICE access only — seek/read on a shared mmap object
misaligns concurrent threads (found by the e2e flight test).

Rhutvik Prashant Pachghare — ASU Robotics & Autonomous Systems
"""

import mmap
import os
import struct

EKF_SHM_FMT = "=QddddddbQ7x"
EKF_SHM_SIZE = 80
EKF_MIN_VALID_VAR = 1e-6   # reader rejects variances <= this (zeroed segment)

DEFAULT_EKF_SHM_PATH = "/dev/shm/aisp_ekf_state"


class EKFShmWriter:
    """Single-producer seqlock writer for one vehicle's EKF snapshot.

    Construction fails closed (raises OSError) if the segment cannot be
    created — a setup error must not pass silently. Per-write failures are
    counted (write_failures) and rate-limited-logged rather than raised:
    the estimator loop must not crash on a transient /dev/shm problem, but
    the failure must be observable.
    """

    def __init__(self, path: str = DEFAULT_EKF_SHM_PATH):
        self._path = path
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, EKF_SHM_SIZE)
        self._shm = mmap.mmap(fd, EKF_SHM_SIZE, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        self._seq = 0              # always even; odd values mean "writing"
        self.write_failures = 0

    def write(self, px: float, py: float, pz: float,
              var_px: float, var_py: float, var_psi: float,
              gps_active: bool) -> None:
        try:
            seq = self._seq
            # Write 1: odd seq_head (write in progress)
            self._shm[0:8] = struct.pack("=Q", seq | 1)
            # Write 2: payload + even seq_tail
            data = struct.pack(EKF_SHM_FMT, seq + 2,
                               px, py, pz, var_px, var_py, var_psi,
                               int(gps_active), seq + 2)
            self._shm[0:len(data)] = data
            # Write 3: even seq_head (write complete)
            self._shm[0:8] = struct.pack("=Q", seq + 2)
            self._seq = (seq + 2) & 0x7FFFFFFFFFFFFFFF
        except (OSError, struct.error, ValueError) as e:
            self.write_failures += 1
            if self.write_failures <= 5 or self.write_failures % 100 == 0:
                import sys
                print(f"[EKFShmWriter] write to {self._path} failed "
                      f"(#{self.write_failures}): {e}", file=sys.stderr)

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass
