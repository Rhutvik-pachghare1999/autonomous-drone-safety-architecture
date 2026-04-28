# Autonomous Drone Safety Architecture
### Hard Real-Time Safety for AI-Driven Drones

[![Build Status](https://github.com/Rhutvik-pachghare1999/autonomous-drone-safety-architecture/actions/workflows/ci.yml/badge.svg)](https://github.com/Rhutvik-pachghare1999/autonomous-drone-safety-architecture/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/tests-13%2F13%20PASSED-brightgreen.svg)]()
[![WCET](https://img.shields.io/badge/WCET-2%2C725ns-orange.svg)]()

**[🔗 View Live Research Poster ↗](https://rhutvik-pachghare1999.github.io/autonomous-drone-safety-architecture/)**

---

## The Problem & My Solution

Foundation models hallucinate. That's fine when a chatbot gives you a wrong recipe. It's not fine when a drone's AI outputs "descend at 100 m/s" and there's no safety layer to catch it.

I built a 5-layer safety kernel that intercepts unsafe AI commands before they reach the motors. Even if the model hallucinates negative thrust (physically impossible), the HOCBF filter clamps it to a safe value in **under 400 nanoseconds**. Tested across 1,000 adversarial trials. 100% survival rate.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Safety filter hot-path | C99, POSIX mmap, SCHED_FIFO | zero-copy IPC, no syscalls in RT loop |
| RL policy | PPO + ONNX 22KB, ONNXRuntime C API | deterministic inference in <0.5ms |
| Formal verification | Z3 solver, BFS over FSM | exhaustive proof of 7 safety invariants |
| State estimation | 15-state EKF, GPS/IMU/Baro/VIO | covariance-gated sensor fusion |
| Simulation | Isaac Sim 4.5, PhysX | domain randomization, zero hardware needed |
| Symbolic math | CasADi 3.7 | Lie derivative chain for HOCBF₄ |
| Consensus | ZMQ PUB/SUB, HotStuff BFT | Byzantine-fault-tolerant swarm voting |

---

## Latency Budget (Measured)

| Subsystem | Budget | Measured | Margin |
|---|---|---|---|
| HOCBF filter (C99) | < 10 µs | **371 ns** (P99: 31ns) | **270×** |
| mmap IPC read | < 1 µs | ~50–100 ns | — |
| RL policy ONNX forward | < 1 ms | ~0.5 ms | 2× |
| OS scheduler jitter | < 50 µs | 5.0 µs P99 | 10× |
| End-to-end RT loop | < 2 ms | < 1 ms | 2× |
| VLA inference (SmolVLM2) | < 5 s | ~2.5 s | best-effort core |

WCET measured via `clock_gettime(CLOCK_MONOTONIC_RAW)` on SCHED_FIFO prio 99, CPU core 2, 100k trials. Raw data in `experiments/results/latency_raw.csv`.

---

## What It Does

```
┌─────────────────────────────────────────────────────────────┐
│                    SAFETY PIPELINE                          │
│                                                             │
│  VLA Model (SmolVLM2)   →  "descend to waypoint"           │
│         ↓ vz_cmd                                            │
│  PPO RL Policy (22KB)   →  T_nom = -380N  ← UNSAFE         │
│         ↓                                                   │
│  ┌─────────────────────────────────┐                        │
│  │   HOCBF Safety Filter (C99)    │  ← intercepts here     │
│  │   clamp(T_nom, T_lb, T_max)    │                        │
│  │   WCET = 2,725 ns              │                        │
│  └─────────────────────────────────┘                        │
│         ↓ T_safe = 78.5N  ← SAFE                           │
│  DO-178C FSM (7 states)  →  P1–P7 verified                 │
│         ↓                                                   │
│  SE(3) Physics Plant     →  drone stays alive              │
└─────────────────────────────────────────────────────────────┘

Cross-cutting: 15-state EKF feeds covariance into HOCBF + consensus weights
```

---

## Key Results

| Metric | Result |
|---|---|
| HOCBF filter WCET | 2,725 ns |
| EVT tail bound (P=10⁻⁹) | 1,733 ns — 57× below 100µs deadline |
| Survival across 1,000 adversarial trials | 100% |
| Worst command corrected | −380N → 78.5N |
| Byzantine rejection (20% packet loss) | 100% |
| EKF rank under GPS denial | 6 → 4, VIO restores to 6 |
| Battery EOL: spec vs reality | cycle 600 vs cycle 100 (4–6× error) |
| pytest | 13/13 PASSED |

---

## What I Built

1. **C99 HOCBF safety filter** — closed-form clamp, no QP solver, WCET 2,725ns. pybind11 wrapper for Python testing. EVT Gumbel tail bound at P=10⁻⁹ is 1,733ns — 57× below the 100µs hard deadline.

2. **15-state EKF with observability Gramian** — rank drops 6→4 under GPS denial. VIO (OpenVINS-derived noise) restores rank to 6. Air-gap architecture prevents the filter from confirming its own estimates.

3. **DO-178C-inspired FSM with Z3 proofs** — 7 states, 24 transitions, 7 invariants (P1–P7). P7 is new: fires RTL when EKF covariance collapses before the drone acts on bad state estimates.

4. **Observability-weighted HotStuff consensus** — the consensus node uses EKF covariance to weight votes. A GPS-denied Byzantine node gets w=0.008 vs w=0.976 for GPS-active nodes. It can't reach 2/3 quorum regardless of what it votes. No separate fault detection layer needed.

5. **Battery aging model** — validated against NASA PCoE cells (B0005/B0006/B0007). Spec linear model predicts EOL at cycle 600. Real cells died at cycle 100–165. 4th-order polynomial fit gets RMSE < 0.03 Ah.

6. **Headless SITL testbed** — zero dependencies on Gazebo, ROS, PX4, or hardware. Fully reproducible from a fresh clone.

---

## What Actually Broke

**Python HOCBF was too slow.** First implementation used OSQP in Python. Jitter was 50–200µs — unusable in a 10Hz loop. Rewrote in C99 with a closed-form clamp. WCET dropped to 2,725ns. The pybind11 wrapper keeps it testable from Python.

**The EKF was confirming its own estimates.** Early versions let the EKF read from the same shared memory the physics plant wrote to. Added an air-gap: plant writes ground-truth to `/dev/shm/aisp_gt_state`, EKF reads it as VIO but never writes back. Rank went from artificially inflated to the correct 4 under GPS denial.

**Battery spec was off by 4–6×.** Datasheet says EOL at cycle 600. NASA PCoE cells actually died at cycle 100–165. A linear model would have you planning missions on a dead battery.

**Byzantine detection came for free.** Expected to need a separate fault detection layer. Didn't. The EKF covariance weighting handles it automatically.

---

## Formal Safety Properties

```
P1 — Geofence breach        → RTL in ≤ 1 cycle (100ms)
P2 — DISARMED always reachable (BFS proof over all states)
P3 — Can't go DISARMED → FLYING without ARM + TAKEOFF
P4 — 5s watchdog timeout    → EMERGENCY_LAND
P5 — No deadlocks (every state has ≥1 exit)
P6 — NaN/Inf inputs rejected before FSM sees them
P7 — EKF covariance collapse → RTL before bad estimates cause damage
```

---

## EKF Observability Under GPS Denial

```
Sensor config          Observable states    Rank
─────────────────────────────────────────────────
GPS + IMU + Baro       px, py, pz, vx, vy   6/15
IMU + Baro only        pz, vz, φ, θ         4/15  ← px, py lost
+ VIO (σ=0.10 m/s)    px, py restored       6/15

Note: yaw (ψ) stays unobservable without magnetometer
```

---

## Battery Aging Reality Check

```
Capacity (Ah)
2.0 ┤████████████████████████████████████████  ← Spec linear model
    │                                           (predicts EOL @ cycle 600)
1.8 ┤         ████████████
    │                     ██████
1.6 ┤                           ████
    │                               ██          ← Real cells (NASA PCoE)
1.4 ┤                                 ██        (EOL @ cycle 100–165)
    │
    └──────────────────────────────────────────
    0        100       200       300    cycles

Poly-4 fit RMSE: 0.016 Ah (B0005), 0.030 Ah (B0006), 0.014 Ah (B0007)
```

---

## Running It

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy casadi pyzmq osqp pytest pybind11

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && make install
cd ..

PYTHONPATH=. pytest tests/test_hocbf.py -v

python experiments/exp_hallucination_1000.py
python experiments/exp_wcet_evt.py
python experiments/exp_lie_derivatives.py
python experiments/exp_observability_gramian.py
python experiments/exp_battery_validation.py
python experiments/exp_consensus_fault.py

./build/safety_filter 100000 2
```

---

## Under the Hood (High-Signal Details)

**`mlockall(MCL_CURRENT | MCL_FUTURE)`** — called at startup in `safety_filter.c` to pin all memory pages and prevent page faults from blowing the RT deadline. Standard practice for hard-RT systems, rarely seen in student projects.

**Zero-copy IPC via `/dev/shm`** — the VLA (Python, best-effort core) writes commands to a POSIX shared memory region. The safety filter (C99, SCHED_FIFO prio 99, isolated core) reads it with a single `mmap` pointer dereference — no serialization, no syscalls, ~50–100ns latency. Replaces ZeroMQ, ROS2 DDS, and MAVLink entirely in the hot-path.

**Property P7** — fires RTL when `tr(P[px,py,ψ]) ≥ 25 m²`, meaning 1-σ horizontal uncertainty has exceeded 5m. The drone stops trusting its own position estimate before it acts on it. This is a deterministic fallback, not a heuristic.

---



- Simulation-first workflow to maximize iteration speed. Next step is porting to a Jetson Orin for hardware-in-the-loop (HIL) validation.
- DO-178C-inspired, not certified. Full cert needs EASA/FAA, LDRA/VectorCAST, PSAC.
- VIO uses OpenVINS-derived noise params on synthetic data — not a live pipeline.
- Yaw unobservable under GPS denial. Magnetometer model is the fix.
- Battery model validated on 18650 cells (2Ah). Project uses 6S LiPo (5Ah). Chemistry differs; recalibration needed for real hardware.
- WCET on stock Linux. PREEMPT_RT would tighten jitter further.

---

## Project Status

Master's thesis — ASU School of Manufacturing Systems & Networks, Spring 2026.
Advisor: Prof. Shenghan Guo.

I'm looking for full-time roles in **safety-critical autonomy, real-time systems, and robot learning** in the U.S. starting **May 2026**. If you're working on hard RT safety kernels, formal verification, or autonomous vehicle software — let's talk.

→ [LinkedIn](https://linkedin.com/in/rhutvik-pachghare) · rhutvik.pachghare@asu.edu
