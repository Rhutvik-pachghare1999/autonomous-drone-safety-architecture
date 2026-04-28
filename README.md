# Autonomous Drone Safety Architecture
### Hard Real-Time Safety for AI-Driven Drones

[![Build Status](https://github.com/Rhutvik-pachghare1999/autonomous-drone-safety-architecture/actions/workflows/ci.yml/badge.svg)](https://github.com/Rhutvik-pachghare1999/autonomous-drone-safety-architecture/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/tests-13%2F13%20PASSED-brightgreen.svg)]()
[![WCET](https://img.shields.io/badge/WCET-2%2C725ns-orange.svg)]()

**[🔗 View Live Research Poster ↗](https://rhutvik-pachghare1999.github.io/autonomous-drone-safety-architecture/)**

---

## The Problem I Was Trying to Solve

Foundation models hallucinate. That's fine when a chatbot gives you a wrong recipe. It's not fine when a drone's AI outputs "descend at 100 m/s" and there's no safety layer to catch it.

I wanted to know: can you build something that sits between the AI and the motors, catches every unsafe command, and does it fast enough that it doesn't break the real-time loop? Under 400 nanoseconds fast.

Turns out you can. Here's how.

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

## Numbers That Matter

```
┌─────────────────────────────────────────┬──────────────────────────┐
│ Metric                                  │ Result                   │
├─────────────────────────────────────────┼──────────────────────────┤
│ HOCBF filter WCET                       │ 2,725 ns                 │
│ EVT tail bound (P=10⁻⁹)                 │ 1,733 ns  (57× margin)   │
│ Survival across 1,000 adversarial trials│ 100%                     │
│ Worst command corrected                 │ -380N → 78.5N            │
│ Byzantine rejection (20% packet loss)   │ 100%                     │
│ EKF rank under GPS denial               │ 6 → 4, VIO restores to 6 │
│ Battery EOL: spec vs reality            │ cycle 600 vs cycle 100   │
│ pytest                                  │ 13/13 PASSED             │
└─────────────────────────────────────────┴──────────────────────────┘
```

---

## Engineering Challenges (What Actually Broke)

**HOCBF in Python was too slow.** My first implementation used OSQP in Python. Jitter was 50–200µs — completely unusable in a 10Hz loop. I rewrote the safety filter in C99 with a closed-form clamp instead of a QP solve. WCET dropped to 2,725ns. The pybind11 wrapper keeps it testable from Python.

**The EKF was lying to itself.** Early on, the EKF was reading from the same shared memory the physics plant wrote to. It was essentially confirming its own estimates. I added an air-gap: plant writes ground-truth to `/dev/shm/aisp_gt_state`, EKF reads it as a VIO measurement but never writes back. Observability rank went from artificially inflated to the correct 4 under GPS denial.

**Battery spec was off by 4–6×.** The datasheet says EOL at cycle 600. NASA PCoE cells (B0005/B0006/B0007) actually died at cycle 100–165. A linear model would have you planning missions on a dead battery. The 4th-order polynomial fit gets RMSE < 0.03 Ah and catches the knee in the capacity curve.

**Byzantine detection came for free.** I expected to need a separate fault detection layer. Didn't need one. A GPS-denied node gets w=0.008 from the EKF covariance weighting. GPS-active nodes get w=0.976. The Byzantine node simply can't swing the 2/3 quorum no matter what it votes.

---

## Formal Safety Properties

```
  P1 ── Geofence breach        → RTL in ≤ 1 cycle (100ms)
  P2 ── DISARMED always reachable (BFS proof over all states)
  P3 ── Can't go DISARMED → FLYING without ARM + TAKEOFF
  P4 ── 5s watchdog timeout    → EMERGENCY_LAND
  P5 ── No deadlocks (every state has ≥1 exit)
  P6 ── NaN/Inf inputs rejected before FSM sees them
  P7 ── EKF covariance collapse → RTL before bad estimates cause damage  [NEW]
```

P7 is the one I'm most proud of. It fires when `tr(P[px,py,ψ]) ≥ 25 m²` — meaning the drone has lost confidence in its own position — and forces RTL before it acts on garbage state estimates. All 7 verified with Z3 BFS.

---

## EKF Observability Under GPS Denial

```
  Sensor config          Observable states    Rank
  ─────────────────────────────────────────────────
  GPS + IMU + Baro       px, py, pz, vx, vy   6/15
  IMU + Baro only        pz, vz, φ, θ         4/15  ← px, py lost
  + VIO (σ=0.10 m/s)    px, py restored       6/15
  
  Note: yaw (ψ) stays unobservable without magnetometer — known gap
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
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy casadi pyzmq osqp pytest pybind11

# Build the C99 safety filter
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && make install
cd ..

# Tests (should show 13/13 PASSED)
PYTHONPATH=. pytest tests/test_hocbf.py -v

# Run experiments
python experiments/exp_hallucination_1000.py   # the main result
python experiments/exp_wcet_evt.py             # timing analysis
python experiments/exp_lie_derivatives.py      # HOCBF math
python experiments/exp_observability_gramian.py
python experiments/exp_battery_validation.py
python experiments/exp_consensus_fault.py

# Raw WCET benchmark
./build/safety_filter 100000 2
```

---

## Honest Limitations

- No real hardware. SITL only. Real IMU noise will be different.
- DO-178C-*inspired*, not certified. Actual cert needs EASA/FAA, LDRA/VectorCAST, PSAC.
- VIO is synthetic — OpenVINS noise params applied to simulated data, not a live pipeline.
- Yaw unobservable under GPS denial. No magnetometer model.
- Battery model trained on 18650 cells (2Ah). Project uses 6S LiPo (5Ah). Chemistry differs.
- WCET on stock Linux. PREEMPT_RT would tighten the jitter further.

---

## Stack

`C99` · `C++20` · `Python 3.12` · `CasADi` · `OSQP` · `ONNXRuntime` · `ZMQ` · `Z3` · `CMake` · `pytest` · `Isaac Sim 4.5`

---

## Project Status

Master's thesis — ASU School of Manufacturing Systems & Networks, Spring 2026.
Advisor: Prof. Shenghan Guo.

Transitioning to full-time roles in robotics/autonomy in the U.S. starting **May 2026**.
Interested in safety-critical autonomy, real-time systems, and robot learning.

→ [LinkedIn](https://linkedin.com/in/rhutvik-pachghare) · rhutvik.pachghare@asu.edu
