# Autonomous Drone Safety Architecture
## A Formally Verified Hierarchical Control Framework for SITL Research

**Arizona State University — Master's in Robotics and Autonomous Systems**
**Student: Rhutvik Prashant Pachghare | Advisor: Shenghan Guo**

---

## Research Problem

Autonomous drone systems must satisfy two competing requirements simultaneously:
they must be *safe* (decisions must be provably correct) and they must be
*adaptive* (they must handle unexpected conditions). No open-source drone
platform currently provides:
  1. Mathematical proofs of state-transition safety invariants
  2. Continuous constraint satisfaction under sensor degradation
  3. Fault-tolerant swarm consensus under Byzantine node failure
...all in a reproducible environment requiring zero physical hardware.

---

## Research Question

> "Can a hierarchical architecture combining a formally verified FSM,
> continuous EKF state estimation, and Byzantine fault-tolerant consensus
> provide quantifiable safety guarantees while maintaining autonomous
> adaptability — and what is the measurable computational cost of each
> guarantee?"

---

## System Architecture — 5 Layers, 10Hz Integrated Loop

  Layer 1 — SE(3) Rigid-Body Physics Plant
    F=ma translational + Euler rigid-body rotational dynamics (RK4, SO(3) projection)
    mass=2.0kg  Ixx=0.0347  Iyy=0.0458  Izz=0.0977 kg·m²
    wind σ=0.05m/s  drag k_d=0.1 m⁻¹

  Layer 2 — 15-State EKF Sensor Fusion
    state: pos(3) vel(3) quat(4) gyro_bias(3) accel_bias(2)
    P₀=I₁₅×0.1  Q=I₁₅×0.001  R_gps=diag[2.5,2.5,5.0]m²
    IMU @ 10Hz | GPS @ 1Hz | baro @ 1Hz
    Covariance gating: NOMINAL / DEGRADED / COLLAPSED (P7)
    VIO factor: σ_v=0.10 m/s (OpenVINS EuRoC benchmark, Geneva et al. ICRA 2020)

  Layer 3 — PPO Asymmetric Actor-Critic (RL)
    Replaces PID. Trained in Isaac Sim 4.5 under domain randomization.
    Actor obs: 13-dim (noisy EKF). Critic obs: 17-dim (privileged ground truth).
    Deployed via ONNX (22KB) in C99 RT hot-path.

  Layer 4 — DO-178C-Inspired Safety Kernel (FSM)
    7 states | 24 transitions | 7 proven properties (P1–P7)
    watchdog=5s | geofence=±500m/120m AGL | SOC<15%→RTL

  Layer 5 — Battery Model
    5.0Ah 6S LiPo | fade=0.05%/cycle | R_int=0.05Ω
    Coulomb counting + 4th-order polynomial OCV
    Validated against NASA PCoE B0005/B0006/B0007 (RMSE=0.016 Ah)

Integrated service: Observability-weighted HotStuff consensus (services/consensus_node.py)
  Feeds EKF covariance → vote weight → blended into DEGRADED mode position estimate

---

## Formal Safety Properties

  P1: Geofence violation → RTL within 1 control cycle (≤100ms)
  P2: DISARMED reachable from every state (BFS proof)
  P3: No DISARMED→FLYING without ARM + TAKEOFF sequence
  P4: Watchdog timeout → EMERGENCY_LAND from any flight state
  P5: No deadlocks — every state has ≥1 outgoing transition
  P6: NaN/Inf position inputs always rejected
  P7: tr(P[px,py,ψ]) ≥ 25.0 m² → RTL within 1 control cycle (≤100ms)

---

## Running

  # Install
  python3 -m venv .venv && source .venv/bin/activate
  pip install numpy scipy casadi pyzmq osqp pytest pybind11

  # Build C++ HOCBF (required before running tests or experiments)
  mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && make install && cd ..

  # Verify HOCBF (must show all PASSED)
  PYTHONPATH=. pytest tests/test_hocbf.py -v

  # Research experiments (run in any order)
  python experiments/exp_lie_derivatives.py
  python experiments/exp_observability_gramian.py
  python experiments/exp_battery_validation.py
  python experiments/exp_hallucination_1000.py
  python experiments/exp_consensus_fault.py

  # WCET benchmark (requires C build)
  ./build/safety_filter 100000 2
  python experiments/exp_wcet_evt.py

  # Consensus unit test
  python services/consensus_node.py --test

  # EKF gating smoke test
  python src/estimation/ekf_gating.py

---

## Research Contributions

  1. Formally verified 7-state FSM with 7 proven invariants (P1–P7) including
     EKF observability collapse detection — answering: how fast can formal
     safety enforcement run inside a real-time loop?
     Result: WCET=4.82µs, EVT Gumbel bound (P=1e-9)=340ns.

  2. 15-state EKF with quantified observability Gramian under GPS denial —
     rank drops 6→4 when GPS lost. VIO factor (OpenVINS-derived noise) restores
     rank to 6. Safe dropout limit quantified via covariance gating.

  3. Battery aging → mission feasibility boundary validated against NASA PCoE
     empirical data (B0005/B0006/B0007). Key finding: spec linear model predicts
     EOL at cycle 600; real cells hit EOL at cycle 100–165 (4× error).

  4. HOCBF4 (relative degree 4) derived symbolically via CasADi with exact
     inertia tensors. Control invariant set proved: T_lb ≤ T_max at all
     boundary states including worst-case (max velocity + max tilt).

  5. Observability-weighted HotStuff consensus: GPS-denied Byzantine node
     (w=0.008) cannot corrupt quorum against GPS-active nodes (w=0.976).
     100% commit rate and 100% Byzantine rejection under 20% packet loss.

  6. 1000-trial adversarial hallucination blocking: 100% survival rate across
     vz commands from -0.1 to -100 m/s. HOCBF triggered in 90.7% of trials.

  7. Headless SITL testbed: zero dependencies on Gazebo, GPU, X11, ROS, PX4,
     or physical hardware. Fully reproducible.

---

## Honest Limitations

  - No physical hardware tested. SITL results may differ on real sensors.
  - DO-178C-inspired design. Not certified. Full certification requires
    EASA/FAA engagement, LDRA/VectorCAST qualified toolchain, and PSAC.
  - VIO factor uses OpenVINS-derived noise parameters (σ_v=0.10 m/s) applied
    to synthetic measurements. Not a running OpenVINS pipeline.
  - Yaw remains unobservable under GPS denial regardless of VIO.
  - HOCBF4 Lf4h_drift approximated as 0 at low angular velocity (hover).
    Full expression implemented; dominant at high-ω maneuvers.
  - Battery model validated on 18650 cells (2Ah). Project uses 6S LiPo (5Ah).
    Scaling is noted; cell chemistry differs.
  - WCET measured on unpatched Linux kernel. PREEMPT_RT would reduce jitter
    further but was not available on the development machine.

---

## References

  Castro & Liskov (1999). Practical Byzantine Fault Tolerance. OSDI.
  Ames et al. (2019). Control Barrier Functions: Theory and Applications. ECC.
  Xiao & Belta (2022). High-Order Control Barrier Functions. IEEE TAC.
  Geneva et al. (2020). OpenVINS: A Research Platform for VIO. ICRA 2020.
  Lee et al. (2010). Geometric Tracking Control on SE(3). CDC.
  Fulton et al. (2020). Formal Verification of End-to-End Learning in CPS.
  Dixon et al. (2019). Formal Verification + Battery PHM for UAVs. arxiv:1909.03019.
  Markovic et al. (2021). ES-EKF for GPS-Denied UAV Navigation. arxiv:2109.04908.
  Saha & Goebel (2007). NASA PCoE Battery Dataset. NASA Ames.
  Krener & Ide (2009). Measures of Unobservability. CDC.
  Foughali & Zuepke (2022). Formal Verification of Real-Time Autonomous Robots.
  Burri et al. (2016). The EuRoC MAV Datasets. IJRR.
  Makoviychuk et al. (2021). Isaac Gym. NeurIPS.
  Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks.
  Yin et al. (2019). HotStuff: BFT Consensus with Linearity and Responsiveness. PODC.
  Mahony et al. (2012). Multirotor Aerial Vehicles: Modeling, Estimation and Control. IEEE RAM.
