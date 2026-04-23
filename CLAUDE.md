# CLAUDE.md — AISP Research Testbed

## Project Purpose

Master's research at Arizona State University investigating formally verified
hierarchical safety architectures for autonomous drone systems.
Student: Rhutvik Prashant Pachghare | Advisor: Shenghan Guo

## Running the Project

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy casadi pyzmq osqp pytest pybind11

# Build C++ HOCBF (required first)
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && make install && cd ..

# Verify HOCBF (must show all PASSED)
PYTHONPATH=. pytest tests/test_hocbf.py -v

# Research experiments
python experiments/exp_lie_derivatives.py
python experiments/exp_observability_gramian.py
python experiments/exp_battery_validation.py
python experiments/exp_hallucination_1000.py
python experiments/exp_consensus_fault.py

# WCET benchmark
./build/safety_filter 100000 2
python experiments/exp_wcet_evt.py

# Unit tests
python services/consensus_node.py --test
python src/estimation/ekf_gating.py
python sim/domain_rand.py
```

## Key Architecture Decisions

- Layer 3 is PPO RL (not PID). ONNX policy runs in C99 RT hot-path.
- HOCBF RD-2 (kinematic) and HOCBF4 RD-4 (rigid-body) both in src/control/hocbf.cpp
- Consensus output is causally linked to EKF gating via /dev/shm/aisp_consensus
- P7 is the 7th formal property: EKF covariance collapse → RTL
- VIO noise: σ_v=0.10 m/s from OpenVINS EuRoC benchmark (Geneva et al. ICRA 2020)

## File Map

    src/control/hocbf.cpp           — HOCBF + HOCBF4 C++ + pybind11 bindings
    src/rt/safety_filter.c          — Hard-RT C99 hot-path + jitter watchdog
    src/estimation/ekf_gating.py    — CovarianceGating, VIONoiseModel, P7, consensus blend
    src/utils/shm_bridge.py/.h      — VLA→C++ zero-copy mmap bridge
    src/perception/vla_bridge.py    — VLM loader + adversarial parser
    sim/domain_rand.py              — DomainRandomizer + SE(3) RigidBodyPlant
    sim/quadrotor_rl.py             — Isaac Sim PPO environment
    services/consensus_node.py      — Observability-weighted HotStuff consensus
    experiments/exp_lie_derivatives.py      — CasADi symbolic Lie chain + invariant set proof
    experiments/exp_observability_gramian.py — Empirical Observability Gramian
    experiments/exp_wcet_evt.py             — P99.99 WCET + EVT Gumbel fit
    experiments/exp_battery_validation.py   — NASA PCoE B0005/B0006/B0007 validation
    experiments/exp_consensus_fault.py      — aBFT under 20% packet loss
    experiments/exp_hallucination_1000.py   — 1000-trial adversarial blocking
    tests/test_hocbf.py             — HOCBF pytest suite (requires C++ build)

## Notes on vla_bridge.py

Requires `turboquant` for 4-bit KV cache. This package is not on PyPI.
The VLA experiments use cached commands (experiments/results/vla_commands_cache.json)
and do not require the VLA model to be loaded for the hallucination blocking experiment.
