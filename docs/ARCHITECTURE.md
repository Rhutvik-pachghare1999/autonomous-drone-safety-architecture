# Architecture

This document describes the data and control flow that exists in the
repository today. Each block is tagged with its maturity:

- **implemented-tested** — code exists and at least one test or experiment
  reproduced here passes.
- **implemented-unvalidated** — code exists but was not run in this environment.
- **planned** — specified in design docs or roadmap, but no implementation.

---

## 1. Top-level data flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Best-effort cores (SCHED_OTHER)                                            │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────────────────┐  │
│  │ VLA bridge   │      │ RL policy    │      │ EKF + covariance gating    │  │
│  │ (planned/    │      │ (ONNX /      │      │ (implemented-tested)       │  │
│  │  unvalidated)│      │  unvalidated)│      │                            │  │
│  └───┬──────────┘      └──────┬───────┘      └─────────────┬──────────────┘  │
│      │                        │                            │                 │
│      │ writes                 │ writes /dev/shm            │ writes          │
│      ▼                        ▼                            ▼                 │
│  /dev/shm/aisp_vla_cmd    /dev/shm/aisp_ekf_state      /dev/shm/aisp_consensus│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │                            │
                                    │ mmap read                  │ mmap read
                                    ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Hard-RT core (SCHED_FIFO prio 99, isolated, memory-locked)                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ safety_filter.c (implemented-tested in -DNO_ONNX mode)                │  │
│  │  1. mmap read VLA command from /dev/shm/aisp_vla_cmd                  │  │
│  │  2. (optional) ONNXRuntime C API RL policy forward → nominal thrust   │  │
│  │  3. HOCBF clamp: T_safe = clamp(T_nom, T_lb, T_max)                   │  │
│  │  4. non-blocking UDP SITL packet → actuator (Isaac Sim / hardware)    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. VLA / high-level command path

| Block | File | Status | Notes |
|---|---|---|---|
| SmolVLM2-500M VLA model load & inference | `src/perception/vla_bridge.py` | implemented-unvalidated | Requires `transformers`, `torch`, `bitsandbytes`, GPU, and a ~500M HF download. Not run here. |
| Free-text velocity parser | `src/perception/vla_bridge.py` `_parse_velocity()` | implemented-unvalidated | Unit-tested indirectly only through inspection. |
| Write velocity to `/dev/shm/aisp_vla_cmd` | `src/utils/shm_bridge.py` | implemented-tested | Layout matches `safety_filter.c` `VLACommand` struct. |

**Data format:** `/dev/shm/aisp_vla_cmd` is a 64-byte POSIX shared memory
region laid out as `src/rt/safety_filter.c:VLACommand`:

- `uint64_t sequence_number`
- `double vx_nom, vy_nom, vz_nom`
- `uint8_t is_new_data`
- padding to 64 bytes

The safety filter checks `is_new_data` and a 100 ms staleness window before
consuming the command.

---

## 3. Real-time safety filter hot path

| Block | File | Status | Notes |
|---|---|---|---|
| RT setup (`mlockall`, `SCHED_FIFO`, CPU affinity) | `src/rt/safety_filter.c` `rt_setup()` | implemented-tested | Runs with `sudo` in the local RT run; gracefully degrades without `CAP_SYS_NICE`. |
| `/dev/shm` mmap open | `src/rt/safety_filter.c` `main()` | implemented-tested | `open()` + `mmap()` are outside the per-trial hot path. |
| HOCBF clamp `hocbf_filter()` | `src/rt/safety_filter.c` | implemented-tested | O(1), no heap, no syscalls; benchmarked in `experiments/results/latency_raw.csv`. |
| ONNXRuntime C API policy forward | `src/rt/safety_filter.c` `onnx_*()` | implemented-unvalidated | Disabled by default (`-DNO_ONNX`); requires ONNX Runtime shared library. |
| Non-blocking UDP SITL output | `src/rt/safety_filter.c` `sitl_send()` | implemented-unvalidated | Sends to `127.0.0.1:14550`; no Isaac Sim instance was available here. |
| Inter-cycle jitter watchdog | `src/rt/safety_filter.c` | implemented-tested | Reports max/p99 jitter in `experiments/results/wcet_evt.json`. |

**HOCBF lower bound computation (single-trial path):**

```
LgLfh = max(cos(roll) * cos(pitch) / mass, 0.01)   // control effectiveness
rhs   = g - alpha1 * vz - alpha2 * pz               // CBF constraint RHS
T_lb  = rhs / LgLfh                                 // safe minimum thrust
T_safe = clamp(T_nom, max(0, T_lb), 4 * m * g)
```

The pybind11 wrapper in `src/control/hocbf.cpp` exposes the same filter to
Python tests and experiments.

---

## 4. EKF gating and P7

| Block | File | Status | Notes |
|---|---|---|---|
| 15-state EKF mode ladder (NOMINAL / DEGRADED / COLLAPSED) | `src/estimation/ekf_gating.py` | implemented-tested | `test_covariance_gating()` smoke test passes. |
| Health scalar `tr(P[px,py,ψ])` | `src/estimation/ekf_gating.py` `_health_scalar()` | implemented-tested | Used by `CovarianceGating.evaluate()`. |
| P7 threshold check | `src/estimation/ekf_gating.py` | implemented-tested | Sets `p7_triggered=True` when `sigma_sq ≥ SIGMA_CRITICAL_SQ`. |
| RTL FSM action on P7 | — | planned | No FSM code exists; see `ROADMAP.md`. |
| VIO measurement injection | `src/estimation/ekf_gating.py` `inject_vio_factor()` | implemented-tested | Reads ground-truth velocity from `/dev/shm/aisp_gt_state` (air-gap). |
| Consensus position blend | `src/estimation/ekf_gating.py` `read_consensus_shm()` | implemented-tested | Reads `/dev/shm/aisp_consensus`; no live consensus node exercised in pytest. |

**Air-gap:** the physics plant writes ground-truth `[vx, vy]` to
`/dev/shm/aisp_gt_state`. The EKF reads it as the VIO measurement `y_true`
but never writes back, preventing the filter from confirming its own drift.

---

## 5. Swarm consensus

| Block | File | Status | Notes |
|---|---|---|---|
| Observability-weighted trust `w_i = exp(-σ²_pos / (2 σ_warn²))` | `services/consensus_node.py` `EKFSnapshot.trust_weight` | implemented-tested | Matches `src/estimation/ekf_gating.py` thresholds. |
| HotStuff-style PREPARE / quorum logic | `services/consensus_node.py` `ConsensusNode` | implemented-tested | `python3 services/consensus_node.py --test` passes. |
| Weighted 2/3 quorum | `services/consensus_node.py` `_weighted_quorum()` | implemented-tested | Byzantine GPS-denied node cannot reach quorum. |
| ZMQ PUB/SUB inter-node messaging | `services/consensus_node.py` | implemented-tested | Used in `experiments/exp_consensus_fault.py`. |
| Write agreed state to `/dev/shm/aisp_consensus` | `services/consensus_node.py` `_write_consensus()` | implemented-tested | Layout matches `src/estimation/ekf_gating.py` reader. |

**Consensus fault test:** `experiments/exp_consensus_fault.py` simulates 5
nodes (3 GPS-active, 1 GPS-degraded, 1 Byzantine GPS-denied) with 20% packet
loss and 0–50 ms latency. Result: 100% commit rate, 100% Byzantine rejection.

---

## 6. Formal verification layer

| Block | File | Status | Notes |
|---|---|---|---|
| Safety properties P1–P7 | `ROADMAP.md`, `README.md` | planned | Specified; no FSM or Z3 implementation exists. |
| P6 NaN/Inf input guard | `src/control/hocbf.cpp`, `src/rt/safety_filter.c` | implemented-tested | 17 pytest cases in `tests/test_input_validation.py`. |
| P7 covariance threshold | `src/estimation/ekf_gating.py` | partial | Threshold computed; RTL FSM action not implemented. |

---

## 7. Boundary conditions and known gaps

1. **ONNX path:** the C99 file contains a complete ONNXRuntime C API
   inference path, but it is compiled out by default because the ONNX Runtime
   library is not in this checkout.

2. **VLA bridge:** depends on a 500M Hugging Face download, GPU, and
   `bitsandbytes`. It is not exercised here.

3. **SITL / Isaac Sim:** the UDP packet format is defined, but no simulator
   launch script or Isaac Sim instance is committed or run here.

4. **HOCBF4 torque filter:** a 4th-order CBF is implemented in
   `src/control/hocbf.cpp`, but no test or experiment currently calls it.

5. **Magnetometer:** yaw remains unobservable under GPS denial because no
   magnetometer model is implemented.

---

## 8. Files by responsibility

| File | Responsibility |
|---|---|
| `src/control/hocbf.cpp` | HOCBF filter, HOCBF4 filter, pybind11 module |
| `src/rt/safety_filter.c` | RT harness, ONNX path, `/dev/shm` IPC, UDP SITL output, benchmark |
| `src/estimation/ekf_gating.py` | EKF covariance gating, VIO injection, consensus read, P7 threshold |
| `services/consensus_node.py` | Weighted HotStuff consensus node |
| `src/perception/vla_bridge.py` | VLA model loading and velocity parsing |
| `src/utils/shm_bridge.py`, `shm_bridge.h` | Shared-memory layout helpers |
| `experiments/*.py` | Reproducible measurement scripts |
| `tests/test_hocbf.py` | HOCBF Python binding tests |
| `tests/test_input_validation.py` | NaN/Inf fail-safe tests |
