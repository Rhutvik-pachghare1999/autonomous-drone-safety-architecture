# Autonomous Drone Safety Architecture
### Hard Real-Time Safety for AI-Driven Drones

[![Build Status](https://github.com/Rhutvik-pachghare1999/autonomous-drone-safety-architecture/actions/workflows/ci.yml/badge.svg)](https://github.com/Rhutvik-pachghare1999/autonomous-drone-safety-architecture/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[🔗 View Live Research Poster ↗](https://rhutvik-pachghare1999.github.io/autonomous-drone-safety-architecture/)**

---

## The Mission

I built this to solve one specific problem: **how do you stop an AI from crashing a drone?**

Foundation models (VLAs) are powerful planners, but they hallucinate. If a model outputs "descend at 100 m/s," a real drone hits the ground. I wanted to build a safety layer that intercepts those commands and overrides them in **under 400 nanoseconds** — fast enough to run inside a 10Hz real-time control loop.

The result: a 5-layer safety kernel that sits between the AI and the motors. Even if the AI commands negative thrust (physically impossible), the HOCBF filter clamps it to a safe value before it ever reaches the plant. Tested across 1,000 adversarial trials. **100% survival rate.**

---

## Key Numbers

| Metric | Result |
|---|---|
| HOCBF safety filter WCET | **2,725 ns** |
| EVT tail bound (P=10⁻⁹) | **1,733 ns** — 57× below 100µs deadline |
| Hallucination blocking | **100%** survival across 1,000 trials |
| Max unsafe command corrected | T_nom = −380N → T_safe = 78.5N |
| Byzantine consensus rejection | **100%** under 20% packet loss |
| EKF observability rank (GPS denied) | drops 6→4, VIO restores to 6 |
| Battery EOL prediction error (spec vs real) | **4–6×** — spec says cycle 600, cells die at 100–165 |
| pytest | **13/13 PASSED** |

---

## System Architecture — 5 Layers

```
VLA Model (SmolVLM2)          ← strategic planner, ~2.5s/query
      ↓ vz_cmd
PPO RL Policy (ONNX 22KB)     ← low-level control, ~0.5ms
      ↓ T_nom
HOCBF Safety Filter (C99)     ← clamp(T_nom, T_lb, T_max), WCET=2,725ns
      ↓ T_safe
DO-178C FSM (7 states, P1–P7) ← formal safety kernel
      ↓
SE(3) Physics Plant (RK4)     ← rigid-body dynamics, SO(3) projection
```

**Cross-cutting:** 15-state EKF (GPS/IMU/Baro/VIO) feeds covariance into both the HOCBF filter and the HotStuff consensus vote weights.

---

## Engineering Challenges & Trade-offs

**The Python OSQP problem.** I originally implemented the HOCBF QP solver in Python using OSQP. Jitter was 50–200µs — way too slow for a 10Hz loop. I rewrote the safety filter in C99 with a closed-form clamp (no QP needed for the altitude CBF), which dropped WCET to 2,725ns. The C++ pybind11 wrapper keeps it callable from Python for testing.

**The EKF air-gap.** Early versions let the EKF read from the same shared memory the physics plant wrote to. That created a feedback loop — the filter was "confirming" its own estimates. I separated them: the plant writes ground-truth velocity to `/dev/shm/aisp_gt_state`, the EKF reads it as a VIO measurement but never writes back. Rank went from artificially high to the correct 4 under GPS denial.

**The battery model gap.** I validated the Coulomb-counting + 4th-order OCV model against NASA PCoE data (B0005/B0006/B0007). The spec linear model predicts EOL at cycle 600. Real cells died at cycle 100–165. That's a 4–6× mission planning error. The poly-4 fit gets RMSE < 0.03 Ah.

**Byzantine consensus weight collapse.** A GPS-denied node naturally gets a low EKF trust weight (w=0.008 vs w=0.976 for GPS-active nodes). I didn't need to add a separate Byzantine detection layer — the observability-weighted quorum handles it automatically. The Byzantine node can't reach 2/3 threshold regardless of what it votes.

---

## Formal Safety Properties (P1–P7)

```
P1: Geofence violation → RTL within 1 control cycle (≤100ms)
P2: DISARMED reachable from every state (BFS exhaustive proof)
P3: No DISARMED→FLYING without ARM + TAKEOFF sequence
P4: Watchdog timeout (5s) → EMERGENCY_LAND from any flight state
P5: No deadlocks — every state has ≥1 outgoing transition
P6: NaN/Inf position inputs always rejected
P7: tr(P[px,py,ψ]) ≥ 25.0 m² → RTL ≤100ms  ← EKF collapse detection
```

All 7 properties verified via Z3 BFS. P7 is new — it triggers RTL when the EKF covariance collapses (GPS denied + VIO failed), before the drone acts on bad state estimates.

---

## Running

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy casadi pyzmq osqp pytest pybind11

# Build C++ HOCBF
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && make install && cd ..

# Run all tests
PYTHONPATH=. pytest tests/test_hocbf.py -v

# Experiments (any order)
python experiments/exp_lie_derivatives.py
python experiments/exp_observability_gramian.py
python experiments/exp_battery_validation.py
python experiments/exp_hallucination_1000.py
python experiments/exp_consensus_fault.py

# WCET benchmark
./build/safety_filter 100000 2
python experiments/exp_wcet_evt.py
```

---

## Honest Limitations

- No physical hardware tested — SITL only. Real sensor noise will differ.
- DO-178C-inspired, not certified. Full cert needs EASA/FAA, LDRA/VectorCAST, PSAC.
- VIO uses OpenVINS-derived noise params (σ_v=0.10 m/s) on synthetic data — not a live OpenVINS pipeline.
- Yaw is unobservable under GPS denial regardless of VIO. Known gap.
- Battery model validated on 18650 cells (2Ah). Project uses 6S LiPo (5Ah). Chemistry differs.
- WCET on unpatched Linux. PREEMPT_RT would reduce jitter further.

---

## Stack

`C99` · `C++20` · `Python 3.12` · `CasADi` · `OSQP` · `ONNXRuntime` · `ZMQ` · `Z3` · `CMake` · `pytest` · `Isaac Sim 4.5`

---

## Project Status

Master's thesis, ASU School of Manufacturing Systems & Networks — Spring 2026.
Advisor: Prof. Shenghan Guo.

I'm transitioning to full-time roles in robotics/autonomy in the U.S. starting **May 2026**.
Open to roles in safety-critical autonomy, real-time systems, and robot learning.
→ [LinkedIn](https://linkedin.com/in/rhutvik-pachghare) · rhutvik.pachghare@asu.edu
