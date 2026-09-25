# Claim / Evidence Audit

This file maps every significant claim in the repository to its source
location, the evidence that supports it, and an honesty status. The status
values are:

- **verified** — the claim was reproduced or inspected in this environment.
- **implemented-unvalidated** — code exists, but it was not run or tested here.
- **planned** — specified in design docs or roadmap, but not implemented.
- **stale** — previously claimed, now removed or corrected.

Last updated: 2026-09-25.

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
| Actuator infeasibility handled (T_lb > T_max → T_max least-violation fallback) | `src/control/hocbf.cpp`, `src/rt/safety_filter.c`, `sim/hocbf_py.py` (identical branch in all three) | `tests/test_hocbf.py::test_infeasible_safe_fallback`, `tests/test_hocbf_properties.py::test_cbf_constraint_or_infeasible_fallback`, `tests/test_hocbf_py_matches_cpp.py::test_nan_parity_infeasible_branch` | **corrected 2026-09-24**: hover fallback replaced — when infeasible, hover violates the CBF constraint strictly MORE than T_max (it is the least-violation action). Also fixed C/C++ divergence on NaN T_nom in the infeasible branch (both now T_max) | verified (2026-09-24) |
| Stale VLA watchdog with state machine (STARTUP/FRESH/STALE) | `src/rt/safety_filter.c` lines 53–83, 373–374 | `tests/watchdog_test.c` (C test calling real `vla_watchdog_check()`) | verified |
| C/C++ HOCBF consistency: floor 0.05, conservatism 1.08 | `src/control/hocbf.cpp:111,120`, `src/rt/safety_filter.c:131,134` | Both paths now match | verified |
| Vehicle-parameter consistency (mass/T_max = Crazyflie 2.X) | `src/control/hocbf.cpp`, `src/rt/safety_filter.c`, `sim/hocbf_py.py`, `sim/crazyflie_env.py` | Defaults now `mass=0.027 kg, T_max=0.60 N` in C++, C, and Python (previously C/C++ defaulted to a generic 2 kg quadrotor — truck-brakes-on-a-scooter mismatch). Verified: C++ and Python defaults agree per-key, dive case T=0.3881 N both. WCET numbers unaffected (O(1) arithmetic, parameter-independent timing) | verified (2026-09-23) |
| Consensus→EKF shared-memory wire contract | `services/consensus_node.py` `_write_consensus`, `src/estimation/ekf_gating.py` `read_consensus_shm` | Both sides now use `"=dddf"` (px, py, pz, trust = 28 B). Previously the format was `"=ddddf"` (5 slots) while the writer packed 4 values and the reader unpacked 5 into 4 names — writer raised struct.error, reader raised ValueError. Fixed and verified by round-trip write/read test | verified (2026-09-23) |
| EKF covariance carried forward between cycles | `src/estimation/ekf_gating.py` `GatingResult.covariance_out` | VIO-updated covariance is now returned and must be fed back as next cycle's P (previously recomputed inside `evaluate()` and dropped). Smoke test asserts carry-forward | verified (2026-09-23) |
| Consensus "latency" experiment models network delay | `experiments/exp_consensus_fault.py`, `experiments/results/consensus_fault.json` | **Relabelled.** The 0–50 ms value is a message-timestamp offset only (votes delivered immediately; quorum logic ignores timestamps). Reported commit-time numbers measure local compute, NOT network delay. Tests consensus logic under real packet loss only. Real network-delay testing (ns-3/emulation) remains future work | stale → corrected (2026-09-23) |
| End-to-end: flight loop + EKF gating + live swarm consensus over ZMQ+SHM | `experiments/exp_consensus_ekf_flight.py`, `experiments/results/consensus_ekf_flight.json` | 4 real `ConsensusNode` threads (real ZMQ PUB/SUB, real `/dev/shm` files), point-mass plant writing truth via the `/dev/shm/aisp_gt_state` air-gap, 50 Hz flight loop paced in real time. GPS denied at t=2 s with 0.2 m/s unmodelled drift. **Swarm: horizontal error bounded at 0.192 m (blend weight 0.98 live from the consensus file); ego-only control: 0.880 m (no bound). P7 correctly silent; 176/176 VIO updates shrank velocity covariance (carry-forward exercised).** No mocks of any inter-subsystem wire | verified (2026-09-23) |
| Consensus protocol liveness under round drift | `services/consensus_node.py` `_collect_votes` / `run_round` | **Two bugs found by the end-to-end integration test** (invisible to the original self-test, which re-synchronised every 16 rounds): (1) a 100 ms `zmq.Again` idle tick broke the 0.5 s collect window early, so rounds ran ~150 ms and vote phases desynchronised; (2) votes from any round ≠ current were silently dropped, so once drifted, the swarm stalled QUORUM forever. Fixed: catch `zmq.Again` and wait out the full window; fast-forward on future-round votes with a bounded one-time re-proposal. Re-verified: original smoke test + 6.4 s live integration run both PASS | stale → corrected (2026-09-23) |
| Consensus vote deduplication (double-counted quorum weight) | `services/consensus_node.py` `_collect_votes` | A resynced peer re-broadcasting its vote after our fast-forward was counted twice (quorum weight inflated). Fixed: votes deduped by `node_id` per round (first received wins; `seen_nodes` reset on future-round fast-forward). Tests: `tests/test_consensus_dedupe.py` (duplicate rebroadcast counts once; distinct peers both count) | corrected (2026-09-24) |
| VLA→safety-filter SHM torn-read hardening (seqlock) | `src/rt/watchdog.h` `vla_shm_snapshot()`, `src/rt/safety_filter.c`, `src/utils/shm_bridge.py` | The VLA publisher wrote the [seq, vx, vy, vz, is_new] payload with a single non-atomic 33-byte `pwrite`; the safety kernel read it concurrently and could consume a torn frame (mixed old/new velocities marked fresh). Fixed both sides with a single-producer seqlock: odd head seq during write, `seq_tail` echo, double-checked copy with compiler barriers (max 2 retries, never spins); a torn copy falls back to hover (not-fresh). Old readers MUST be rebuilt/writers updated together — do NOT mix layouts across processes. Tests: `tests/watchdog_test.c::test_torn_read_rejected`, wire-layout checks in `tests/test_hocbf.py` | corrected (2026-09-24) |
| NaN/Inf T_nom in infeasible-QP branch (C vs C++ divergence) | `src/rt/safety_filter.c`, `src/control/hocbf.cpp` | C++ returned T_max; C returned hover thrust — the SAME command could diverge across implementations when VLA emitted NaN/Inf while infeasible. Unified to T_max in all three implementations (C, C++, Python). Test: `tests/test_hocbf_py_matches_cpp.py::test_nan_parity_infeasible_branch` | corrected (2026-09-24) |
| VIO measurement source is ground-truth air-gap, not self-estimate | `src/estimation/ekf_gating.py` `inject_vio_factor` | `y_true` comes from `/dev/shm/aisp_gt_state` (written by the physics plant — Isaac Sim or `sim/domain_rand.py` in the bench) — never from the EKF state. Offline/unit-test path is a documented synthetic zero-velocity measurement: covariance still updated (honest, but optimistic — an offline covariance-shrink claim would overstate real VIO performance). Scope label: pipeline smoke tests are offline → VIO source = synthetic zero vector | verified + scoped (2026-09-24) |
| HOCBF4 post-saturation safety re-check | `src/control/hocbf.cpp` lines 406–424 | After clamping torques to ±tau_max, the filter re-projects onto the feasible half-space ∩ box constraints (3 iterations of project→clamp). Pitch singularity guard at |pitch| ≥ 80° returns zero torques. Tests: `tests/test_hocbf4.py::test_hocbf4_post_saturation_safety`, `test_hocbf4_pitch_singularity_guard` | verified (2026-09-25) |
| EKF shared memory trust inversion fixed (seqlock + validity + FAIL-CLOSED) | `services/consensus_node.py` `EKFSharedMemory.read/write_synthetic` | Seqlock format (=QddddddbQ7x, 80 B) with odd/even seq_head/seq_tail; reader validates seq even, head==tail, stable across reads, variances > 1e-6. **2026-09-25 PM: invalid/missing SHM now raises EKFReadError — the old silent synthetic fallback could inject fabricated flight state into consensus (found in fresh audit). Synthetic mode requires explicit `allow_synthetic=True` (demo/tests only).** Reader/writer use mmap slice access, not seek/read: shared file position between threads misaligned every read (found by e2e). | corrected (2026-09-25) |
| Consensus quorum grouping is distance-based (not hash, not quantization) | `services/consensus_node.py` `weighted_quorum` (module-level, shared with experiments) | **2026-09-25 PM: replaced quantization hashing.** The 1 cm grid had boundary fractures (0.0049 vs 0.0051 m — 2 mm apart — hashed to different cells and could lose quorum). Now single-linkage Euclidean grouping at STATE_TOL=0.01 m; agreed state is the trust-weighted mean of the winning group. `_state_hash` is now an exact-state integrity check only. Tests: `tests/test_consensus_security.py::test_grouping_has_no_quantization_boundary`, `test_far_apart_states_do_not_group`, `test_agreed_state_is_trust_weighted_mean` | corrected (2026-09-25) |
| Peer trust validation (clamp, NaN/Inf reject, hash verify) | `services/consensus_node.py` `_collect_votes` | Incoming votes: trust_weight clamped to (0,1], NaN/Inf rejected; state_hash recomputed and compared to message hash; state_vector length=3, all finite. Malformed messages dropped silently. | corrected (2026-09-25) |
| Consensus writes quorum-derived weight (not local trust) | `services/consensus_node.py` `_weighted_quorum`, `run_round`, `_write_consensus` | Returns `(state, quorum_weight = best_hash_weight / total_weight)`. EKF uses this as blending coefficient alpha. | corrected (2026-09-25) |
| Consensus SHM torn-read protection (seqlock) | `services/consensus_node.py` `_write_consensus`, `src/estimation/ekf_gating.py` `read_consensus_shm` | New format =QdddfQ (48 B): seq_head, px,py,pz, trust, seq_tail. 3-write protocol (odd head → payload+even tail → even head). Reader validates seq even, head==tail, stable. Trust clamped to [0,1]. | corrected (2026-09-25) |
| VLA velocity limits enforced (V_MAX_XY=5, V_MAX_Z=3) | `src/perception/vla_bridge.py` `parse_velocity_text` | Structured parsing clamps to V_MAX_XY/Z. Keyword heuristics return bounded values (dive → -V_MAX_Z, speed → V_MAX_XY). Previously allowed ±50 m/s. | corrected (2026-09-25) |
| EKF state contract unified (15 elements) | `tests/conftest.py` `sample_state`, `src/estimation/ekf_gating.py` index map | Fixture now matches 15-state EKF: pos(3), vel(3), quat(4), gyro_bias(3), accel_bias(2). Removed extra accel_bias[2]. | corrected (2026-09-25) |
| Post-mixer CBF feasibility check | `sim/vla_crazyflie_flight.py` `CrazyflieController.mix_and_check` | Computes T_delivered after motor clamping, verifies T_delivered >= T_lb (CBF lower bound with conservatism). Returns cbf_violated flag. | verified (2026-09-25) |
| HOCBF4 test coverage added | `tests/test_hocbf4.py` (7 tests) | Covers import, hover no-filtering, filtering when needed, saturation limits, post-saturation safety, pitch singularity guard, C++ binding. | verified (2026-09-25) |
| Production RT binary (safety_filter) with SCHED_FIFO verification | `CMakeLists.txt` `safety_filter` target, `src/rt/safety_filter.c` `rt_setup()` | Builds without SIL_REPLAY. `rt_setup()` fails closed on mlockall/sched_setscheduler/sched_setaffinity failures; verifies policy==SCHED_FIFO, priority==99, affinity==requested core via sched_getscheduler/sched_getparam/sched_getaffinity. | verified (2026-09-25) |
| Periodic deadline-driven benchmark (1 kHz, 100 µs deadline) | `src/rt/safety_filter.c` main loop | `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)` for drift-free 1 ms period. Deadline miss detection (cycle_elapsed > 100 µs) counted and reported. Jitter = deviation from ideal periodic start. | verified (2026-09-25) |
| Deadline miss detection and reporting | `src/rt/safety_filter.c` lines 480–495 | Each cycle checks `cycle_elapsed > deadline_ns`; misses counted, first 5 logged to stderr. Statistics include deadline_misses count. | verified (2026-09-25) |
| RT loop measures TRUE end-of-cycle latency | `src/rt/safety_filter.c` main loop | **2026-09-25 PM fix:** previously `lat_total` and the deadline check stopped at HOCBF completion — excluding SITL UDP send and state update from the "full loop" claim. Now measured after ALL cycle work (SHM read → policy → HOCBF → UDP send → state update). Also: ONNX inference runs exactly ONCE per cycle (was twice in SITL mode, second inference after the deadline check). | corrected (2026-09-25) |
| RT validation exit criteria are strict | `src/rt/safety_filter.c` exit path | Previously `exit(hocbf_wcet < 100µs)` — a run could miss every 100 µs deadline and still exit 0. Now fails on ANY deadline miss, any SITL send failure, or HOCBF WCET ≥ 100 µs. `--require-onnx` flag fails closed if the policy cannot load (default: loudly-marked DEGRADED VLA-only mode so numbers are never mistaken for the ONNX path). | corrected (2026-09-25) |
| UDP SITL send failures observable | `src/rt/safety_filter.c` `sitl_send` | Return value now checked; failures counted and reported (was fire-and-forget with silent drops). | corrected (2026-09-25) |
| CPU affinity vs isolation language | `src/rt/safety_filter.c` `rt_setup` + report | `sched_setaffinity` pins this process to a core but does NOT isolate the core from other work (that needs isolcpus/nohz_full boot config, unverifiable from userspace). Report now says "affinity-pinned; NOT necessarily isolated". | corrected (2026-09-25) |
| Consensus authenticated membership (HMAC-SHA256) | `services/consensus_node.py` `ConsensusMessage.sign/verify`, `_collect_votes`, `_broadcast` | Every vote carries a truncated HMAC-SHA256 over ALL fields incl. node_id and round_num; key from constructor or `$AISP_CONSENSUS_KEY`; no key → loud OPEN-mode warning (lab only). Verification happens BEFORE any field is trusted, so dedupe is by AUTHENTICATED identity (previously a sender-supplied unauthenticated node_id drove dedupe). This is authenticated membership under a shared key, NOT Byzantine fault tolerance (a key holder turning malicious is out of scope, documented in module header). Tests: `tests/test_consensus_security.py::test_wrong_key_votes_dropped_and_counted`, `test_same_key_votes_accepted`, `test_hmac_covers_node_id` | corrected (2026-09-25) |
| Consensus EKF sit-out preserves round cadence | `services/consensus_node.py` `run_round` | A node with no valid EKF estimate sits the round out — but sleeps ROUND_TIMEOUT_S first. Instant failure would burn a caller's wall-time-derived round budget in milliseconds (found by e2e: all workers exhausted their rounds before the first EKF write → consensus never committed). | corrected (2026-09-25) |
| Consensus SHM write failure observable | `services/consensus_node.py` `_write_consensus` | `except OSError: pass` replaced with counted, rate-limited stderr logging (`_shm_write_failures`). Test: `tests/test_consensus_security.py::test_consensus_write_failure_is_counted` | corrected (2026-09-25) |
| E2E flight re-verified under honest path | `experiments/exp_consensus_ekf_flight.py` | Re-run after fail-closed EKF + HMAC + distance grouping: swarm error 0.196 m bounded vs ego-only 0.880 m, alpha 1.0, P7 silent — PASS. (Previous 0.192 m was partly masked by the synthetic fallback.) E2E now runs in authenticated mode with a shared test key. | verified (2026-09-25) |
| EKF SHM production writer exists (was: "no writer in repo") | `src/estimation/ekf_shm_writer.py`, `src/estimation/ekf_gating.py` (`CovarianceGating(ekf_shm_path=...)` publishes each cycle) | Seqlock 3-write protocol, position-independent mmap slice access, monotonic even sequence, per-write failures counted (not raised into the control loop, not silent). Wire contract pinned against the reader by `tests/test_ekf_shm_contract.py` (constants equality + round-trip + gating-publishes). E2E ego node now runs on the real writer (peers remain simulated remote estimators): swarm error 0.192 m, PASS. | verified (2026-09-25) |

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

- **Verified:** 25 (incl. 1 verified + scoped; e2e re-verified 2026-09-25 PM under fail-closed+authenticated path)
- **Corrected 2026-09-24:** infeasible fallback hover → T_max; C/C++ NaN infeasible-branch parity; consensus vote dedupe; SHM seqlock torn-read
- **Corrected 2026-09-25 AM:** EKF SHM trust inversion; consensus hashing; peer trust validation; quorum-derived weight; consensus SHM seqlock; VLA velocity limits; EKF state contract; post-mixer CBF check
- **Corrected 2026-09-25 PM (fresh audit):** EKF synthetic fallback → fail-closed; mmap seek/read → slice access (thread misalignment); EKF sit-out cadence; quantization grouping → distance-based; HMAC authenticated membership; consensus SHM write observability; RT end-of-cycle measurement boundary; single ONNX inference/cycle; strict RT exit criteria; UDP send checks; affinity≠isolation
- **Implemented-unvalidated:** 3 (ONNX hot-path, HOCBF4 full validation, full FSM/Z3 proof)
- **Planned:** 1 (full FSM/Z3 proof)
- **Stale → corrected/removed:** 5
