# Autonomous Drone Safety Architecture

[![Build Status](https://github.com/Rhutvik-pachghare1999/autonomous-drone-safety-architecture/actions/workflows/ci.yml/badge.svg)](https://github.com/Rhutvik-pachghare1999/autonomous-drone-safety-architecture/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

Research prototype: a hard real-time safety kernel that wraps an AI-driven
quadrotor flight stack. The goal is to intercept every high-level command,
project it through a physical safety filter, and clamp the actuator command
before it reaches the motors.

This is **not a certified flight controller**. It is a simulation-first
research codebase used to study how deterministic safety filters can bound
foundation-model/RL hallucinations in real time.

---

## What is implemented

| Component | Status | Evidence |
|---|---|---|
| C99 + pybind11 HOCBF altitude safety filter | implemented, tested | `src/control/hocbf.cpp`, `src/rt/safety_filter.c`, `tests/test_hocbf.py`, `tests/test_input_validation.py` |
| NaN/Inf input validation / fail-safe | implemented, tested | `src/control/hocbf.cpp`, `src/rt/safety_filter.c`, `tests/test_input_validation.py` |
| POSIX `/dev/shm` zero-copy command IPC | implemented | `src/utils/shm_bridge.py`, `src/rt/safety_filter.c` |
| EKF covariance gating / mode ladder (P7) | partially implemented | `src/estimation/ekf_gating.py` computes threshold; RTL FSM action is planned |
| Observability-weighted HotStuff consensus | implemented, tested | `services/consensus_node.py`, `experiments/exp_consensus_fault.py` |
| VLA bridge (SmolVLM2) | implemented, not validated here | `src/perception/vla_bridge.py` — requires GPU + HF model download |
| PPO policy + ONNXRuntime C hot-path | implemented, not validated here | `experiments/results/ppo_policy.onnx`, `src/rt/safety_filter.c` — requires ONNX build |
| Formal FSM / Z3 invariants (P1–P7) | planned, not implemented | `ROADMAP.md` § "Formal verification roadmap" |
| Isaac Sim / SITL closed-loop flight | planned / dev-tool only | No simulator launch scripts are committed; see `docs/DEMO_RUNBOOK.md` |

---

## System overview

```
Best-effort cores (SCHED_OTHER)
  ├── VLA bridge ..................... parses image+text → [vx, vy, vz]
  │                                    (best-effort, ~2.5 s latency)
  └── RL policy ...................... PPO actor-critic → nominal thrust
                                       (optional ONNXRuntime path)
                │
                ▼ /dev/shm/aisp_vla_cmd
Hard-RT core (SCHED_FIFO prio 99, isolated)
  └── safety_filter.c ................ mmap read → HOCBF clamp → actuator
       ├── HOCBF filter .............. O(1) arithmetic, WCET 2,725 ns
       ├── jitter watchdog ........... per-cycle deviation from expected
       └── stale-VLA fallback ........ RL hover if VLA silent > 100 ms

State estimation:
  └── ekf_gating.py .................. 15-state EKF mode ladder (NOMINAL / DEGRADED / COLLAPSED)
       └── P7 threshold: tr(P[px,py,ψ]) ≥ 25 m² → set p7_triggered

Swarm consensus:
  └── consensus_node.py .............. observability-weighted HotStuff quorum
       └── writes /dev/shm/aisp_consensus for EKF gating
```

Only the HOCBF clamp and jitter watchdog run in the hard-RT path and are
covered by tests. The VLA bridge, RL/ONNX policy, EKF gating FSM action, and
Isaac Sim / SITL loop either require external runtime dependencies (Isaac Sim,
ONNX runtime library, GPU, downloaded VLA weights) or are planned; see the
status column in "What is implemented" and `docs/ARCHITECTURE.md`.

---

## Repository layout

| Path | What it contains |
|---|---|
| `src/control/hocbf.cpp` | C++ HOCBF filter with pybind11 Python wrapper |
| `src/rt/safety_filter.c` | Standalone C99 real-time filter + ONNX path + benchmark harness |
| `src/estimation/ekf_gating.py` | EKF covariance gating, P7 threshold, VIO injection, consensus read |
| `src/perception/vla_bridge.py` | SmolVLM2 VLA bridge (requires GPU + transformers) |
| `src/utils/shm_bridge.py`, `shm_bridge.h` | `/dev/shm` layout definitions |
| `services/consensus_node.py` | Observability-weighted HotStuff node |
| `experiments/*.py` | Reproducible experiments; see "Quick start" for which ones run here |
| `experiments/results/*.json`, `*.csv`, `*.png` | Committed measurement artifacts |
| `docs/WCET_BENCHMARK.md` | How the 2,725 ns WCET was produced and reproduced |
| `docs/LATENCY_BUDGET.md` | Subsystem latency budget and SLA definitions |
| `docs/CLAIM_EVIDENCE_AUDIT.md` | Claim-by-claim evidence map |
| `docs/ARCHITECTURE.md` | Code-verified data/control flow |
| `docs/DEMO_RUNBOOK.md` | Procedures for paths that cannot run in this environment |
| `ROADMAP.md` | Development roadmap including formal verification P1–P7 |
| `tests/` | `test_hocbf.py` (13 tests) + `test_input_validation.py` (17 tests) |

---

## Requirements

**For the verified, CI-friendly subset:**

- Python 3.12
- `numpy`, `scipy`, `pyzmq`, `pytest`, `pybind11`
- GCC or Clang for the C99 benchmark harness
- CMake 3.20+ (for the C++/pybind11 build)

**For optional paths (not validated here):**

- `casadi` → `experiments/exp_lie_derivatives.py`
- `transformers`, `torch`, `pillow`, `bitsandbytes`, GPU → `src/perception/vla_bridge.py`
- ONNX Runtime C library → ONNX-linked `safety_filter.c` build
- NVIDIA Isaac Sim → closed-loop SITL

Install the core set:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy casadi pyzmq osqp pytest pybind11
```

---

## Quick start

Only the commands below were run in this environment.

### 1. Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j2 && make install
cd ..
```

This compiles the pybind11 `hocbf` module and installs `hocbf.so` into
`src/control/`.

### 2. Run tests

```bash
python3 -m pytest tests/ -q
```

Expected result: **30 passed** (13 from `test_hocbf.py`, 17 from
`test_input_validation.py`).

### 3. Run the C99 filter benchmark

```bash
gcc -O3 -DNO_ONNX -o /tmp/sf src/rt/safety_filter.c -lm -lrt
/tmp/sf 1000 0
```

This is the no-root, no-SCHED_FIFO run. It measures the arithmetic cost of
the filter without OS scheduler isolation. On the author's machine it reports
WCET ≈ 31 ns for 1,000 trials. The committed 2,725 ns WCET was captured under
`SCHED_FIFO` + isolated core; see `docs/WCET_BENCHMARK.md`.

### 4. Run experiments that execute here

```bash
python3 experiments/exp_hallucination_1000.py
python3 experiments/exp_wcet_evt.py
python3 experiments/exp_observability_gramian.py
python3 experiments/exp_consensus_fault.py
python3 experiments/exp_battery_validation.py   # requires experiments/results/nasa_pcoe_discharge.csv
python3 services/consensus_node.py --test
```

`experiments/exp_lie_derivatives.py` needs `casadi`, which is not installed in
this environment. ONNX-linked and Isaac-Sim paths are documented in
`docs/DEMO_RUNBOOK.md`.

---

## Tests & validation

| Test / check | Command | Result | Evidence |
|---|---|---|---|
| HOCBF unit tests | `pytest tests/test_hocbf.py -q` | 13 passed | `tests/test_hocbf.py` |
| NaN/Inf fail-safe | `pytest tests/test_input_validation.py -q` | 17 passed | `tests/test_input_validation.py` |
| C filter no-ONNX run | `gcc -O3 -DNO_ONNX -o /tmp/sf src/rt/safety_filter.c -lm -lrt && /tmp/sf 1000 0` | builds and runs | terminal output |
| Consensus quorum | `python3 services/consensus_node.py --test` | PASS | terminal output |

All 30 pytest cases pass with Python 3.12 in a clean checkout after the
`conftest.py` fix.

---

## Results

| Claim | Value | Committed artifact | Notes |
|---|---|---|---|
| HOCBF filter WCET | 2,725 ns | `experiments/results/latency_raw.csv` (100,000 rows), `experiments/results/wcet_evt.json` | Captured under `SCHED_FIFO` prio 99, core 2, isolated core. Fresh-clone no-root run ≈ 31 ns. |
| EVT tail bound (P=10⁻⁹) | 1,733 ns | `experiments/results/wcet_evt.json` | Gumbel block-maxima fit from committed CSV. |
| Filter latency p99 | 31 ns | `experiments/results/wcet_evt.json` | Empirical p99 from committed CSV. |
| Filter latency mean / p50 | 22.31 ns / 20 ns | `experiments/results/wcet_evt.json` | From committed CSV. |
| Adversarial survival | 100% (1,000 / 1,000) | `experiments/results/hallucination_1000.json`, `hallucination_1000.png` | vz swept −0.1 to −100 m/s. |
| Worst corrected command | T_nom = −380.4 N → T_safe = 16.9 N, correction = 397.25 N | `experiments/results/hallucination_1000.json` | **Previous README value "78.5 N" was the actuator ceiling (T_max), not the HOCBF-corrected thrust.** The filter returns the safe lower bound 16.9 N. |
| EKF rank GPS denied | 6 → 4 | `experiments/results/observability_gramian.json` | IMU+Baro only loses horizontal position. |
| VIO restores rank | 4 → 6 | `experiments/results/observability_gramian.json` | With OpenVINS-derived noise model. |
| Consensus commit rate | 100% (100 / 100 rounds) | `experiments/results/consensus_fault.json` | 20% packet loss + 50 ms max latency. |
| Byzantine rejection | 100% (100 / 100) | `experiments/results/consensus_fault.json` | Byzantine GPS-denied node cannot reach 2/3 weighted quorum. |
| Battery poly-4 RMSE | 0.016 Ah (B0005), 0.030 Ah (B0006), 0.014 Ah (B0007) | `experiments/results/battery_validation.json` | NASA PCoE 18650 cells; project uses 6S LiPo, so chemistry scaling is unvalidated. |
| Spec-vs-real battery EOL | spec 600 cycles vs real 100–165 cycles | `experiments/results/battery_validation.json` | Linear spec model overestimates usable life by ~4–6×. |

---

## Engineering decisions

1. **Closed-form HOCBF instead of OSQP in the hot path.** The altitude-only
   safety constraint reduces to a 1-D clamp, so the C99 hot path avoids any
   QP solver or heap allocation. This keeps the deterministic latency in the
   tens of nanoseconds instead of the tens of microseconds seen in the early
   Python/OSQP prototype.

2. **Fail-safe on non-finite inputs.** Upstream models can emit NaN/Inf.
   Both `hocbf.cpp` and `safety_filter.c` detect non-finite state or nominal
   commands and fall back to hover thrust or the computed safe lower bound.
   This is tested by 17 dedicated pytest cases.

3. **Ground-truth air-gap for VIO.** The physics plant writes true velocity
   to `/dev/shm/aisp_gt_state`; the EKF reads it as a VIO measurement but
   never writes back. This prevents the filter from confirming its own drift
   under GPS denial.

4. **Observability-weighted consensus.** Vote weight is tied to EKF covariance
   rather than adding a separate fault detector. GPS-denied nodes naturally
   receive near-zero weight, so a Byzantine node in that state cannot sway
   the 2/3 quorum.

5. **Separate best-effort and hard-RT paths.** VLA inference (~seconds) and
   RL policy inference (~milliseconds) run on best-effort cores; only the
   O(1) HOCBF clamp runs under `SCHED_FIFO` priority 99 with memory locked.

---

## Limitations & known gaps

- **Not certified.** The codebase is DO-178C-inspired in structure only. There
  is no requirements traceability, MC/DC evidence, or certification authority
  involvement.

- **Formal verification is planned, not done.** Properties P1–P5 are
  specified but unproven — there is no finite-state machine or Z3 harness in
  the repo. The NaN/Inf-rejection behaviour described by P6 is implemented and
  tested (`tests/test_input_validation.py`), but not as part of a formal
  proof. P7 computes the covariance threshold but the RTL FSM action is not
  implemented. See `ROADMAP.md`.

- **Simulation-only.** No hardware-in-the-loop, flight logs, or real-vehicle
   validation. The zero-copy IPC path has not been exercised against a live
   flight controller.

- **Yaw unobservable without magnetometer.** Under GPS denial the EKF cannot
  observe yaw; the current model does not include a magnetometer.

- **Battery model chemistry gap.** Validation used NASA PCoE 18650 cells
  (~2 Ah). The project assumes a 6S LiPo (~5 Ah); direct chemistry and
  capacity scaling is unvalidated.

- **VLA bridge unvalidated here.** `src/perception/vla_bridge.py` depends on
  Hugging Face `transformers`, `bitsandbytes`, a CUDA GPU, and a ~500M
  download. It has not been run in this environment.

- **ONNX hot-path unvalidated here.** The ONNXRuntime C API path in
  `safety_filter.c` is compiled out by default (`-DNO_ONNX`) because the
  ONNX Runtime shared library is not present in this checkout.

- **Cyclictest OS jitter images not committed.** The 2.0–28.0 µs scheduler
  jitter numbers in `docs/LATENCY_BUDGET.md` were measured locally but the
  raw data and CDF plots were not committed; treat them as indicative.

- **Static README badges removed.** The old "13/13 tests" and "WCET 2725 ns"
  shields were hardcoded and are now removed. Only the live GitHub Actions
  badge remains.

---

## Safety & scope

This repository is a **research prototype**. It demonstrates a real-time
safety-filter architecture in simulation and provides reproducible evidence
for the latency and adversarial-blocking claims above. It is **not intended
for deployment** on real aircraft without substantial additional work:
requirements engineering, independent verification, hardware testing,
fail-operational analysis, and regulatory review.

If you use ideas from this codebase in a safety-critical system, assume every
claim is unproven until you reproduce and validate it in your own environment.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Author: Rhutvik Prashant Pachghare, ASU Robotics & Autonomous Systems.*
*Advisor: Prof. Shenghan Guo.*
