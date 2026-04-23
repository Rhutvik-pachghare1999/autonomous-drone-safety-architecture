# AISP Neurosymbolic Swarm Architecture — Development Roadmap


**Research Question:** How to mathematically bound foundation model hallucinations in safety-critical flight using microsecond-latency deterministic safety filters?

---

## Phase 1: Foundation & Safety Infrastructure (Weeks 1-2)

### 1.1 Physics & Simulation Environment
- [ ] Migrate from pure Python 6DOF to **NVIDIA Isaac Sim** or **PyBullet**
- [ ] Implement full rigid-body dynamics (motor torque → thrust chain)
- [ ] Add ground effect, turbulent wind models, and aerodynamic disturbances
- [ ] Validate against real quadrotor flight data

### 1.2 High-Order Control Barrier Function (HOCBF)
- [ ] Mathematically derive HOCBF for quadrotor (relative degree 2)
- [ ] Map position constraints → roll/pitch/thrust acceleration limits
- [ ] Implement OSQP-based QP solver for real-time constraint satisfaction
- [ ] Port to **C++** or **Rust** for deterministic microsecond latency
- [ ] Create Python bindings via `pybind11`

### 1.3 Formal Safety Kernel
- [ ] Retain existing 7-state FSM with proven invariants (P1-P6)
- [ ] Integrate HOCBF as safety filter layer
- [ ] Benchmark latency: FSM + HOCBF + OSQP solver chain

---

## Phase 2: Reinforcement Learning Controller (Weeks 3-4)

### 2.1 RL Environment Setup
- [ ] Configure **Isaac Gym/Sim** for asymmetric actor-critic training
- [ ] Define observation space: noisy EKF estimates (actor input)
- [ ] Define privileged state: ground truth wind, motor degradation (critic only)
- [ ] Define action space: thrust + roll/pitch/yaw rates

### 2.2 Domain Randomization
- [ ] Randomize mass: ±30% variation
- [ ] Randomize motor torque constants: ±20%
- [ ] Randomize inertia tensor: ±15%
- [ ] Inject turbulent wind: Dryden/von Kármán models
- [ ] Simulate rotor failure scenarios

### 2.3 PPO Training
- [ ] Train end-to-end RL policy to replace cascaded PID
- [ ] Validate zero-shot sim-to-real transfer capability
- [ ] Benchmark against classical PID under extreme conditions

---

## Phase 3: Vision-Language-Action (VLA) Integration (Weeks 5-6)

### 3.1 VLA Model Selection & Deployment
- [ ] Select lightweight VLA model: RT-2, Pixtral, or Qwen-VL
- [ ] Quantize for edge deployment (simulated Jetson Orin Nano)
- [ ] Implement semantic command interface: "Inspect perimeter, avoid vehicles"
- [ ] Output: nominal velocity vectors

### 3.2 Neurosymbolic Safety Shield
- [ ] Wire VLA outputs → HOCBF safety filter
- [ ] Ensure HOCBF blocks hallucinated/unsafe commands
- [ ] Log all VLA outputs vs. HOCBF-corrected outputs

### 3.3 Hallucination Stress Testing
- [ ] Design adversarial prompts to induce VLA hallucinations
- [ ] Test: "Fly through the wall", "Ignore geofence", "Disable safety"
- [ ] Validate 100% collision-free flight under adversarial inputs

---

## Phase 4: Neural Control Barrier Functions (NCBF) (Weeks 7-8)

### 4.1 NCBF Training
- [ ] Train neural network to learn barrier function $h(x)$
- [ ] Account for complex aerodynamic disturbances (ground effect, gusts)
- [ ] Validate against analytical HOCBF baseline

### 4.2 Integration & Benchmarking
- [ ] Replace analytical HOCBF with NCBF in safety pipeline
- [ ] Measure computational overhead: NCBF inference + OSQP solve
- [ ] Compare safety guarantees: NCBF vs. HOCBF

---

## Phase 5: Asynchronous Byzantine Fault Tolerance (aBFT) ✅ COMPLETE

### 5.1 Consensus Protocol Implementation
- [x] Implement observability-weighted HotStuff consensus (`services/consensus_node.py`)
- [x] Causally link EKF covariance to vote weight: `w_i = exp(-tr(P_i[px,py,ψ]) / (2σ²_warn))`
- [x] Wire consensus output into EKF gating (`src/estimation/ekf_gating.py`): DEGRADED mode
      blends EKF position with swarm-agreed position weighted by consensus trust
- [x] Reads/writes `/dev/shm/aisp_consensus` (zero-copy mmap, flight-loop compatible)

### 5.2 Byzantine Failure Testing ✅ COMPLETE
- [x] 5-node swarm: 3 GPS-active (w≈0.976), 1 GPS-degraded (w≈0.147), 1 Byzantine GPS-denied (w≈0.008)
- [x] 20% packet loss + asymmetric latency (0–50ms): commit rate 100%, Byzantine rejection 100%
- [x] Results: `experiments/results/consensus_fault.json`, `experiments/results/consensus_fault.png`

---

## Phase 6: Benchmarking & Data Collection (Weeks 11-12)

### 6.1 Latency Profiling ✅ COMPLETE
- [x] HOCBF C filter: WCET=4.82µs, P99.99=70ns, EVT Gumbel bound (P=1e-9)=340ns
- [x] 100,000-trial benchmark: `experiments/results/wcet_evt.json`, `wcet_evt.png`
- [x] OS jitter watchdog: max=7.3µs, 0 alerts above 50µs SLA

### 6.2 Safety Validation Experiments ✅ COMPLETE
- [x] Exp 1: VLA hallucination blocking — 1000 trials, 100% survival, `hallucination_1000.png`
- [x] Exp 2: HOCBF constraint satisfaction — 10,000 domain-rand trials, 0 violations
- [x] Exp 3: Zero-shot sim-to-real robustness — ±40% mass, σ=2.0 wind
- [x] Exp 4: aBFT consensus under Byzantine failures — 100% commit, 100% Byzantine rejection
- [x] Exp 5: NASA PCoE battery validation — B0005/B0006/B0007, RMSE poly-4=0.016 Ah
      **Key finding:** spec linear model predicts EOL at cycle 600; real cells hit EOL at
      cycle 100–165. Mission feasibility boundary is 4× tighter than the spec assumed.
- [x] Exp 6: CasADi symbolic Lie derivatives — full Lf4h derived, invariant set PROVED
- [x] Exp 7: Empirical Observability Gramian — rank drops 6→4 under GPS denial

### 6.3 Comparative Analysis ✅ COMPLETE
- [x] HOCBF RD-2 (kinematic) vs HOCBF4 RD-4 (rigid-body) — both implemented
- [x] PBFT vs weighted HotStuff — HotStuff: 100% commit at 20% loss; PBFT: O(n²) messages

---

## Phase 7: Paper Writing & Defense Preparation (Weeks 13-16)

### 7.1 Paper Structure
- **Title:** *Neurosymbolic Safety Kernels for VLA-Driven Drone Swarms: Bounding Foundation Model Hallucinations with High-Order Neural CBFs*
- **Abstract:** 250 words
- **Introduction:** Problem statement + research gap
- **Related Work:** VLA models, CBF theory, swarm consensus
- **Methodology:** Architecture diagram + mathematical formulation
- **Experiments:** Benchmarks + ablation studies
- **Results:** Graphs, tables, latency profiles
- **Discussion:** Limitations + future work
- **Conclusion:** Contributions summary

### 7.2 Supplementary Materials
- [ ] Open-source codebase on GitHub
- [ ] Docker container for reproducibility
- [ ] Video demonstrations: VLA hallucination blocking, swarm consensus
- [ ] Appendix: Formal proofs, hyperparameters, hardware specs

---

## Key Deliverables

| Deliverable | Target Date | Status |
|-------------|-------------|--------|
| Isaac Sim migration | Week 2 | ✅ Complete |
| HOCBF C++ implementation (RD-2 + RD-4) | Week 2 | ✅ Complete |
| RL policy trained | Week 4 | ✅ Complete |
| VLA integration | Week 6 | ✅ Complete |
| NASA PCoE battery validation | Week 8 | ✅ Complete — EOL at cycle ~100–165, not 600 |
| aBFT consensus (integrated into EKF gating) | Week 10 | ✅ Complete |
| All experiments complete | Week 12 | ✅ Complete (7 experiments) |
| Paper draft | Week 14 | 🔄 In Progress |
| Defense ready | Week 16 | 🔄 In Progress |

---

## Critical Path Dependencies

```
Physics/Sim → HOCBF (RD-2 + RD-4) → RL Training → VLA Integration → Benchmarking → Paper
                ↓
           Safety Kernel (FSM + HOCBF + P7)
                ↓
           aBFT Consensus → EKF Gating (DEGRADED blend) ✅ INTEGRATED
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Isaac Sim learning curve | Start with PyBullet fallback |
| HOCBF derivation complexity | Consult control theory literature (Ames et al.) |
| RL training instability | Use proven PPO hyperparameters from Isaac Gym examples |
| VLA hallucination unpredictability | Design structured adversarial test suite |
| C++/Rust porting delays | Prototype in Python, port incrementally |

---

## Success Criteria

1. **Safety:** 100% collision-free flight under adversarial VLA inputs
2. **Latency:** Safety kernel <100μs (C++/Rust implementation)
3. **Robustness:** Zero-shot sim-to-real transfer via domain randomization
4. **Consensus:** Swarm maintains semantic agreement under 30% Byzantine failures
5. **Publication:** Accepted to ICRA/IROS/RSS 2026

---

**Last Updated:** 2026-04-22  
**Project Lead:** Rhutvik Prashant Pachghare  
**Advisor:** Shenghan Guo  
**Institution:** Arizona State University — Robotics & Autonomous Systems (MS)
