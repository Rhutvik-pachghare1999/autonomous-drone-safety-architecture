# Claim / Evidence Audit

This file maps every significant claim in the repository to its source
location, the evidence that supports it, and an honesty status. The status
values are:

- **verified** — the claim was reproduced or inspected in this environment.
- **implemented-unvalidated** — code exists, but it was not run or tested here.
- **planned** — specified in design docs or roadmap, but not implemented.
- **stale** — previously claimed, now removed or corrected.

Last updated: 2026-09-03.

---

## Build & test infrastructure

| Claim | Source location | Evidence | Status |
|---|---|---|---|
| `pytest tests/` passes | `pyproject.toml`, `tests/conftest.py`, `tests/*.py` | `python3 -m pytest tests/ -q` → 39 passed | verified |
| `import hocbf` works on fresh clone | `tests/conftest.py` (adds `src/control` to `sys.path`) | `pytest tests/test_hocbf.py` imports module successfully | verified |
| C99 safety filter compiles without ONNX | `src/rt/safety_filter.c` | `gcc -O3 -DNO_ONNX -o /tmp/sf src/rt/safety_filter.c -lm -lrt && /tmp/sf 1000 0` | verified |
| CMake/pybind11 build works | `CMakeLists.txt` | `mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j2 && make install` | verified |
| No secrets in git history | `.env.example`, `.gitignore`, `docs/SECURITY_NOTES.md` | `git log --all -- .env` returns nothing; grep for API keys/tokens in tracked files found none | verified |

## HOCBF safety filter

| Claim | Source location | Evidence | Status |
|---|---|---|---|
| HOCBF filter is closed-form O(1) arithmetic | `src/control/hocbf.cpp`, `src/rt/safety_filter.c` | Source inspection: no heap, no syscalls, one `cos`, one `sin` | verified |
| HOCBF filter WCET = 2,725 ns | `docs/WCET_BENCHMARK.md`, `experiments/results/wcet_evt.json` | `experiments/results/latency_raw.csv` max = 2,725; `wcet_evt.json` empirical.wcet_ns = 2,725 | verified |
| HOCBF p99 = 31 ns | `experiments/results/wcet_evt.json` | Computed from committed `latency_raw.csv` | verified |
| HOCBF mean = 22.31 ns, p50 = 20 ns | `experiments/results/wcet_evt.json` | Computed from committed `latency_raw.csv` | verified |
| Filter fails safe on NaN/Inf inputs | `src/control/hocbf.cpp` lines 84–98, `src/rt/safety_filter.c` lines 71–75 | `tests/test_input_validation.py` (17 cases) | verified |
| 2,725 ns came from SCHED_FIFO/isolated-core local run | `docs/WCET_BENCHMARK.md` | CSV header/context; fresh-clone no-root run gives ~31 ns WCET | verified |
| Actuator infeasibility handled (T_lb > T_max → hover fallback) | `src/control/hocbf.cpp` lines 122–128, `src/rt/safety_filter.c` lines 136–143 | `tests/test_hocbf.py::test_infeasible_safe_fallback` | verified |
| Stale VLA watchdog with state machine (STARTUP/FRESH/STALE) | `src/rt/safety_filter.c` lines 53–83, 373–374 | `tests/watchdog_test.c` (C test calling real `vla_watchdog_check()`) | verified |
| C/C++ HOCBF consistency: floor 0.05, conservatism 1.08 | `src/control/hocbf.cpp:111,120`, `src/rt/safety_filter.c:131,134` | Both paths now match | verified |
| Vehicle-parameter consistency (mass/T_max = Crazyflie 2.X) | `src/control/hocbf.cpp`, `src/rt/safety_filter.c`, `sim/hocbf_py.py`, `sim/crazyflie_env.py` | Defaults now `mass=0.027 kg, T_max=0.60 N` in C++, C, and Python (previously C/C++ defaulted to a generic 2 kg quadrotor — truck-brakes-on-a-scooter mismatch). Verified: C++ and Python defaults agree per-key, dive case T=0.3881 N both. WCET numbers unaffected (O(1) arithmetic, parameter-independent timing) | verified (2026-09-23) |
| Consensus→EKF shared-memory wire contract | `services/consensus_node.py` `_write_consensus`, `src/estimation/ekf_gating.py` `read_consensus_shm` | Both sides now use `"=dddf"` (px, py, pz, trust = 28 B). Previously the format was `"=ddddf"` (5 slots) while the writer packed 4 values and the reader unpacked 5 into 4 names — writer raised struct.error, reader raised ValueError. Fixed and verified by round-trip write/read test | verified (2026-09-23) |
| EKF covariance carried forward between cycles | `src/estimation/ekf_gating.py` `GatingResult.covariance_out` | VIO-updated covariance is now returned and must be fed back as next cycle's P (previously recomputed inside `evaluate()` and dropped). Smoke test asserts carry-forward | verified (2026-09-23) |
| Consensus "latency" experiment models network delay | `experiments/exp_consensus_fault.py`, `experiments/results/consensus_fault.json` | **Relabelled.** The 0–50 ms value is a message-timestamp offset only (votes delivered immediately; quorum logic ignores timestamps). Reported commit-time numbers measure local compute, NOT network delay. Tests consensus logic under real packet loss only. Real network-delay testing (ns-3/emulation) remains future work | stale → corrected (2026-09-23) |
| End-to-end: flight loop + EKF gating + live swarm consensus over ZMQ+SHM | `experiments/exp_consensus_ekf_flight.py`, `experiments/results/consensus_ekf_flight.json` | 4 real `ConsensusNode` threads (real ZMQ PUB/SUB, real `/dev/shm` files), point-mass plant writing truth via the `/dev/shm/aisp_gt_state` air-gap, 50 Hz flight loop paced in real time. GPS denied at t=2 s with 0.2 m/s unmodelled drift. **Swarm: horizontal error bounded at 0.192 m (blend weight 0.98 live from the consensus file); ego-only control: 0.880 m (no bound). P7 correctly silent; 176/176 VIO updates shrank velocity covariance (carry-forward exercised).** No mocks of any inter-subsystem wire | verified (2026-09-23) |
| Consensus protocol liveness under round drift | `services/consensus_node.py` `_collect_votes` / `run_round` | **Two bugs found by the end-to-end integration test** (invisible to the original self-test, which re-synchronised every 16 rounds): (1) a 100 ms `zmq.Again` idle tick broke the 0.5 s collect window early, so rounds ran ~150 ms and vote phases desynchronised; (2) votes from any round ≠ current were silently dropped, so once drifted, the swarm stalled QUORUM forever. Fixed: catch `zmq.Again` and wait out the full window; fast-forward on future-round votes with a bounded one-time re-proposal. Re-verified: original smoke test + 6.4 s live integration run both PASS | stale → corrected (2026-09-23) |

## Experiments that run here

| Claim | Source location | Evidence | Status |
|---|---|---|---|
| 1,000 adversarial trials, 100% survival | `experiments/exp_hallucination_1000.py`, `experiments/results/hallucination_1000.json` | `python3 experiments/exp_hallucination_1000.py` → survival 100% | verified |
| Worst corrected command: T_nom = −380.4 N, T_safe = 16.9 N, correction 397.25 N | `experiments/results/hallucination_1000.json` | Re-running `exp_hallucination_1000.py` reproduces `max_correction_N = 397.2496` | verified |
| EVT Gumbel P=10⁻⁹ bound = 1,733 ns | `experiments/exp_wcet_evt.py`, `experiments/results/wcet_evt.json` | `python3 experiments/exp_wcet_evt.py` → p1e9_bound_ns = 1,733.3 | verified |
| OS inter-cycle jitter max = 9,548 ns, p99 = 11 ns | `experiments/results/wcet_evt.json` | Regenerated by `exp_wcet_evt.py` from committed CSV | verified |
| EKF rank 6 → 4 under GPS denial, VIO restores to 6 | `experiments/exp_observability_gramian.py`, `experiments/results/observability_gramian.json` | `python3 experiments/exp_observability_gramian.py` | verified |
| Consensus commit rate 100%, GPS-denied rejection 100% | `experiments/exp_consensus_fault.py`, `experiments/results/consensus_fault.json` | `python3 experiments/exp_consensus_fault.py` + `python3 services/consensus_node.py --test` | verified |
| Battery poly-4 RMSE < 0.03 Ah; spec EOL 600 vs real 100–165 | `experiments/exp_battery_validation.py`, `experiments/results/battery_validation.json` | `python3 experiments/exp_battery_validation.py` with local `nasa_pcoe_discharge.csv` | verified |

## Code paths not validated here

| Claim | Source location | Evidence | Status |
|---|---|---|---|
| VLA bridge loads SmolVLM2 and parses velocities | `src/perception/vla_bridge.py` | **Now validated in the Isaac Sim loop** (`sim/vla_crazyflie_flight.py`, `experiments/results/vla_crazyflie_episode.jsonl`): SmolVLM2-2.2B-4bit, 9/9 queries parsed `model_structured` | verified |
| **Real Crazyflie 2.X VLA flight + HOCBF A/B (HEADLINE)** | `sim/hocbf_crazyflie_ab100.py`, `slurm/20_ab_fakeroot.sbatch`, `experiments/results/crazyflie_vla_ab500.json` | **500-episode randomized A/B on ASU Sol (Isaac 5.1 GPU PhysX): filter OFF 0% survival (0/500, mean min_z 0.019 m) vs ON 100% (500/500, mean min_z 0.782 m, 88.7% filtered).** Domain-randomized (start_z, dive_vz, onset, vz0, wind, command delay). Python HOCBF port verified identical to C++ (2,450-case cross-check, `tests/test_hocbf_py_matches_cpp.py`). Sim only; adversarial dive scripted. Plot `crazyflie_vla_ab500.png`. | verified |
| Real Crazyflie single-dive trace (illustrative) | `sim/hocbf_crazyflie_ab.py`, `experiments/results/crazyflie_vla_ab.json` | Isaac 5.1 GPU PhysX; dive −3.0 m/s: OFF crashes t=1.56 s, ON min_z=0.753 m. Kept for the time-series plot; superseded as headline by the 500-episode A/B above. | verified |
| ONNXRuntime C hot-path runs policy forward pass | `src/rt/safety_filter.c` lines 86–169 | Code exists; default build uses `-DNO_ONNX` because `onnxruntime.so` is not present | implemented-unvalidated |
| ~~Isaac Sim / SITL closed-loop flight (GPU PhysX A/B, 100 eps/mode)~~ — cuboid surrogate | `sim/isaac_sil_eval.py`, `experiments/results/isaac_sil_summary.json` | **SUPERSEDED.** 86%/16% survival was flown on an ad-hoc 2 kg cuboid point-mass surrogate, not the real Crazyflie. Kept for history; real-vehicle evidence is the `crazyflie_vla_ab` row above | superseded |
| HOCBF4 4th-order torque filter | `src/control/hocbf.cpp` lines 178–392 | Code implemented and bound, but no test/experiment calls it | implemented-unvalidated |

## Formal verification / safety properties

| Claim | Source location | Evidence | Status |
|---|---|---|---|
| P1–P5 are proven invariants | Previously in `README.md` | **No FSM/Z3 code exists.** Corrected in `README.md` and `ROADMAP.md` | stale → now planned |
| P6 behaviour (NaN/Inf inputs rejected) implemented and tested | `src/control/hocbf.cpp`, `src/rt/safety_filter.c`, `tests/test_input_validation.py` | 17 pytest cases pass. Note: this is the filter's fail-safe behaviour, not a formal proof of a P6 invariant — no FSM/Z3 harness exists | verified (behaviour only) |
| P7: EKF covariance collapse triggers RTL | `src/estimation/ekf_gating.py` | `p7_triggered` is set; RTL FSM action not implemented | partial / implemented-unvalidated |
| Full FSM with Z3/BFS proof | `ROADMAP.md` § "Formal verification roadmap" | No implementation; listed as future work | planned |

## Documentation / badges

| Claim | Source location | Evidence | Status |
|---|---|---|---|
| README shows "13/13 tests passed" static badge | Old `README.md` | Removed; replaced with live GitHub Actions badge + explicit test table | stale |
| README shows "WCET 2,725 ns" static badge | Old `README.md` | Removed; WCET is now documented with evidence links | stale |
| `docs/LATENCY_BUDGET.md` referenced `latency_histogram.png` and `cyclictest_cdf.png` | Old `docs/LATENCY_BUDGET.md` | Images do not exist in repo; references removed, uncommitted cyclictest data labelled as local-only | stale → corrected |
| README said worst corrected command was 78.5 N | Old `README.md` | Reconciled to experiment output: T_safe = 16.9 N, correction = 397.25 N | stale → corrected |

## Security

| Claim | Source location | Evidence | Status |
|---|---|---|---|
| `.env` is gitignored and uncommitted | `.gitignore`, `docs/SECURITY_NOTES.md` | `git log --all -- .env` empty; `git check-ignore .env` returns `.env` | verified |
| `.env` contains only paths/versions, no secrets | `docs/SECURITY_NOTES.md` | File inspection confirmed; no rotation needed | verified |

---

## Summary counts

- **Verified:** 24
- **Implemented-unvalidated:** 3
- **Planned:** 1 (full FSM/Z3 proof)
- **Stale → corrected/removed:** 5
