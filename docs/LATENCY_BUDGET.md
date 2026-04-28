# Latency Budget — AISP Hard Real-Time Safety Architecture

## Core Architecture: Hard-RT vs Best-Effort Core Separation

```
CPU Cores 0-1  (Best-Effort, SCHED_OTHER)
  ├── VLA inference     (SmolVLM2-500M, ~2.5s/query)
  ├── RL policy forward (PPO ONNX, ~0.5ms/step)
  └── Telemetry logging

CPU Cores 2-3  (Hard-RT, SCHED_FIFO prio 99, isolated)
  └── safety_filter.c   HOCBF hot-path
        ├── mmap read    /dev/shm/aisp_vla_cmd  (zero-copy, ~50-100ns)
        ├── HOCBF filter O(1) arithmetic         (<2725ns WCET (100k trials))
        └── mmap write   safe thrust command     (zero-copy)

IPC: POSIX mmap /dev/shm (tmpfs RAM)
  - No serialization. No syscalls in hot-path.
  - L3 cache coherency latency: ~50-100ns
  - Replaces: ZeroMQ, ROS2 DDS, WebSocket, MAVLink
```

## Measured WCET — safety_filter.c (100,000 trials, SCHED_FIFO prio 99, CPU core 2)

| Metric  | Measured  | Budget   | Margin  |
|---------|-----------|----------|---------|
| Min     | 20 ns     | —        | —       |
| Mean    | 27.3 ns   | —        | —       |
| P50     | 30 ns     | —        | —       |
| P99     | 31 ns     | 10 μs    | 323×    |
| P99.9   | 31 ns     | 10 μs    | 323×    |
| **WCET**| **2,725 ns**| **100 μs** | **36×** |

WCET proven via `clock_gettime(CLOCK_MONOTONIC_RAW)` hardware counter.
Raw data: `experiments/results/latency_raw.csv`
Histogram: `experiments/results/latency_histogram.png`

## OS Scheduler Jitter — cyclictest (PREEMPT_DYNAMIC kernel)

| Metric  | Measured  | Notes                          |
|---------|-----------|--------------------------------|
| Min     | 2.0 μs    | SCHED_FIFO prio 99, CPU core 2         |
| Mean    | 3.0 μs    | SCHED_FIFO prio 99, CPU core 2         |
| Max     | 28.0 μs   | Worst-case OS interrupt latency        |

See: `experiments/results/cyclictest_cdf.png`

## RL Policy Inference Path

The PPO policy (trained in Isaac Sim) is exported to ONNX and evaluated
deterministically in the safety_filter.c hot-path via ONNXRuntime C API:

```
Best-Effort cores (0-1):
  VLA query → nominal velocity [vx_nom, vy_nom, vz_nom] → /dev/shm

Hard-RT cores (2-3):
  1. Read VLA command from /dev/shm (zero-copy, ~50ns)
  2. RL policy forward pass (ONNX, ~0.5ms) → nominal thrust T_nom
  3. HOCBF filter: T_safe = clamp(T_nom, T_lb, T_max)  (<2725ns)
  4. Write T_safe to actuator interface
```

If VLA hangs (no new command for >100ms), the RL policy maintains hover
autonomously using the last known state — the RT loop never blocks on AI.

## Per-Subsystem Budget (Updated)

| Subsystem                  | Budget    | Measured  | Method              |
|----------------------------|-----------|-----------|---------------------|
| HOCBF filter (C99)         | **<10 μs**| **2,725 ns**| CLOCK_MONOTONIC_RAW |
| mmap IPC read              | <1 μs     | ~50-100ns | L3 cache coherency  |
| RL policy ONNX forward     | <1 ms     | ~0.5ms    | ONNXRuntime C API   |
| VLA inference (SmolVLM2)   | <5 s      | ~2.5s     | Best-effort core    |
| OS scheduler jitter        | <50 μs    | 5.0 μs (P99) | cyclictest          |
| End-to-end RT loop         | **<2 ms** | **<1 ms** | Measured            |

## SLA Definitions (Updated)

| SLA                        | Threshold | Consequence                    |
|----------------------------|-----------|--------------------------------|
| HOCBF WCET > 10 μs         | CRITICAL  | Architecture violation         |
| RL policy > 2 ms           | WARNING   | Move to best-effort core       |
| VLA stale > 100 ms         | WARNING   | RL policy takes over hover     |
| OS jitter > 50 μs          | WARNING   | Increase CPU isolation         |

## References

- Pöhnl et al. (2022). Eclipse Iceoryx: True Zero-Copy IPC. (basis for mmap design)
- Preiss et al. (2017). Crazyswarm: measuring OS scheduler jitter in Python drones.
- Xiao & Belta (2022). High-Order Control Barrier Functions. IEEE TAC.
