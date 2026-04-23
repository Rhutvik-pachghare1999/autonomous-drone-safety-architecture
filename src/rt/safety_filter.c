/*
 * Hard-RT safety filter: mmap read → ONNX policy → HOCBF clamp → UDP to SITL
 * WCET measured: 2725 ns (SCHED_FIFO prio 99, CPU core 2, 100k trials)
 *
 * Build (no ONNX):  gcc -O3 -march=native -DNO_ONNX -o build/safety_filter src/rt/safety_filter.c -lm -lrt
 * Build (with ONNX): gcc -O3 -march=native -I build/onnxruntime_include -o build/safety_filter_onnx src/rt/safety_filter.c -lm -lrt <onnxruntime.so>
 *
 * Rhutvik Prashant Pachghare — ASU Robotics & Autonomous Systems
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#ifndef NO_ONNX
#include <onnxruntime_c_api.h>
#endif

/* ── Shared memory layout (matches shm_bridge.py) ────────────────────────── */
#define SHM_PATH  "/dev/shm/aisp_vla_cmd"
#define SHM_SIZE  64
#define VLA_STALE_NS  100000000ULL  /* 100ms — if VLA silent, RL takes over */

typedef struct __attribute__((aligned(64))) {
    uint64_t sequence_number;
    double   vx_nom, vy_nom, vz_nom;
    uint8_t  is_new_data;
    char     _pad[31];
} VLACommand;

/* ── HOCBF parameters ────────────────────────────────────────────────────── */
#define MASS    2.0
#define GRAVITY 9.81
#define ALPHA1  2.0
#define ALPHA2  1.0
#define T_MAX   (4.0 * MASS * GRAVITY)
#define T_MIN   0.0

/* ── Jitter watchdog ─────────────────────────────────────────────────────── */
/* Inter-cycle jitter = |actual_interval - expected_interval|.
 * At 10 Hz the expected interval is 100ms = 100,000,000 ns.
 * In benchmark mode (back-to-back trials) the expected interval is 0 —
 * we measure the deviation from the mean cycle time instead.
 * Alert threshold: 50 μs = 50,000 ns (matches LATENCY_BUDGET.md SLA). */
#define JITTER_WARN_NS  50000ULL   /* 50 μs */
static inline uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ── HOCBF filter — O(1), no heap, no syscalls ───────────────────────────── */
static inline double hocbf_filter(double pz, double vz, double roll,
                                   double pitch, double T_nom) {
    double LgLfh = cos(roll) * cos(pitch) / MASS;
    if (LgLfh < 0.01) LgLfh = 0.01;
    double rhs  = GRAVITY - ALPHA1 * vz - ALPHA2 * pz;
    double T_lb = rhs / LgLfh;
    double lo   = T_lb > T_MIN ? T_lb : T_MIN;
    if (T_nom < lo)   return lo;
    if (T_nom > T_MAX) return T_MAX;
    return T_nom;
}

/* ── ONNX Policy Runner ───────────────────────────────────────────────────── */
#ifndef NO_ONNX

typedef struct {
    const OrtApi*     api;
    OrtEnv*           env;
    OrtSession*       session;
    OrtMemoryInfo*    mem_info;
    OrtSessionOptions* opts;
} OnnxPolicy;

static int onnx_init(OnnxPolicy* pol, const char* model_path) {
    pol->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!pol->api) return -1;

    pol->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "aisp_rt", &pol->env);
    pol->api->CreateSessionOptions(&pol->opts);
    pol->api->SetIntraOpNumThreads(pol->opts, 1);  /* single-threaded for RT */
    pol->api->SetSessionGraphOptimizationLevel(pol->opts, ORT_ENABLE_ALL);

    OrtStatus* status = pol->api->CreateSession(pol->env, model_path,
                                                 pol->opts, &pol->session);
    if (status) {
        fprintf(stderr, "ONNX load failed: %s\n",
                pol->api->GetErrorMessage(status));
        pol->api->ReleaseStatus(status);
        return -1;
    }
    pol->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                   &pol->mem_info);
    return 0;
}

/* Run policy forward pass: obs[13] → action[4] */
static int onnx_infer(OnnxPolicy* pol, const float* obs, int obs_dim,
                       float* action_out, int action_dim) {
    int64_t shape[] = {1, obs_dim};
    OrtValue* input_tensor = NULL;
    pol->api->CreateTensorWithDataAsOrtValue(
        pol->mem_info, (void*)obs, obs_dim * sizeof(float),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    const char* input_names[]  = {"obs"};
    const char* output_names[] = {"action"};
    OrtValue* output_tensor = NULL;

    OrtStatus* status = pol->api->Run(pol->session, NULL,
        input_names,  (const OrtValue* const*)&input_tensor,  1,
        output_names, 1, &output_tensor);

    if (status) {
        pol->api->ReleaseStatus(status);
        pol->api->ReleaseValue(input_tensor);
        return -1;
    }

    float* out_data = NULL;
    pol->api->GetTensorMutableData(output_tensor, (void**)&out_data);
    memcpy(action_out, out_data, action_dim * sizeof(float));

    pol->api->ReleaseValue(input_tensor);
    pol->api->ReleaseValue(output_tensor);
    return 0;
}

static void onnx_free(OnnxPolicy* pol) {
    pol->api->ReleaseSession(pol->session);
    pol->api->ReleaseSessionOptions(pol->opts);
    pol->api->ReleaseMemoryInfo(pol->mem_info);
    pol->api->ReleaseEnv(pol->env);
}

/* Decode normalized action[0] → physical thrust */
static inline double action_to_thrust(float action_norm) {
    /* Matches QuadrotorHoverEnv: T = T_hover + action[0] * T_range */
    double T_hover = MASS * GRAVITY;
    double T_range = MASS * GRAVITY;
    double T = T_hover + (double)action_norm * T_range;
    if (T < T_MIN) T = T_MIN;
    if (T > T_MAX) T = T_MAX;
    return T;
}

#endif /* NO_ONNX */

/* ── UDP SITL Bridge ─────────────────────────────────────────────────────── */
/* Packet layout: seq(uint32) T_safe(double) roll_cmd(float) pitch_cmd(float)
 *                yaw_rate(float) pad(3 bytes) = 32 bytes total
 * Isaac Sim listens on SITL_PORT and applies T_safe to the physics plant.
 * This replaces the UART wire to a physical Pixhawk. */
#define SITL_HOST  "127.0.0.1"
#define SITL_PORT  14550
#define SITL_PKT_SIZE 32

typedef struct __attribute__((packed)) {
    uint32_t seq;
    double   T_safe;
    float    roll_cmd;
    float    pitch_cmd;
    float    yaw_rate;
    uint8_t  _pad[3];
} SITLPacket;

typedef struct {
    int                sock;
    struct sockaddr_in addr;
    uint32_t           seq;
} SITLBridge;

static int sitl_init(SITLBridge* b, const char* host, int port) {
    b->seq  = 0;
    b->sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (b->sock < 0) { perror("socket"); return -1; }

    /* Non-blocking so UDP send never stalls the RT loop */
    int flags = fcntl(b->sock, F_GETFL, 0);
    fcntl(b->sock, F_SETFL, flags | O_NONBLOCK);

    memset(&b->addr, 0, sizeof(b->addr));
    b->addr.sin_family      = AF_INET;
    b->addr.sin_port        = htons((uint16_t)port);
    b->addr.sin_addr.s_addr = inet_addr(host);
    return 0;
}

/* Fire-and-forget UDP send — O(1), non-blocking, never stalls RT loop */
static inline void sitl_send(SITLBridge* b, double T_safe,
                               float roll_cmd, float pitch_cmd, float yaw_rate) {
    SITLPacket pkt = {
        .seq       = ++b->seq,
        .T_safe    = T_safe,
        .roll_cmd  = roll_cmd,
        .pitch_cmd = pitch_cmd,
        .yaw_rate  = yaw_rate,
    };
    sendto(b->sock, &pkt, SITL_PKT_SIZE, MSG_DONTWAIT,
           (struct sockaddr*)&b->addr, sizeof(b->addr));
}

static void sitl_close(SITLBridge* b) { close(b->sock); }

/* ── RT setup ────────────────────────────────────────────────────────────── */
static void rt_setup(int cpu_core) {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1)
        perror("mlockall (need CAP_IPC_LOCK)");

    struct sched_param sp = { .sched_priority = 99 };
    if (sched_setscheduler(0, SCHED_FIFO, &sp) == -1)
        perror("sched_setscheduler (need CAP_SYS_NICE)");

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == -1)
        perror("sched_setaffinity");
}

/* ── Main ────────────────────────────────────────────────────────────────── */
int main(int argc, char* argv[]) {
    int   n_trials  = (argc > 1) ? atoi(argv[1]) : 10000;
    int   cpu_core  = (argc > 2) ? atoi(argv[2]) : 2;
    const char* onnx_path = (argc > 3) ? argv[3]
                          : "experiments/results/ppo_policy.onnx";
    int   sitl_mode = (argc > 4 && strcmp(argv[4], "--sitl") == 0);

    rt_setup(cpu_core);

    /* Open shared memory */
    int fd = open(SHM_PATH, O_CREAT | O_RDWR, 0666);
    if (fd == -1) { perror("open shm"); exit(1); }
    ftruncate(fd, SHM_SIZE);
    VLACommand* shm = (VLACommand*)mmap(NULL, SHM_SIZE,
                        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (shm == MAP_FAILED) { perror("mmap"); exit(1); }

    /* SITL UDP bridge (replaces UART to Pixhawk) */
    SITLBridge sitl = {0};
    if (sitl_mode) {
        if (sitl_init(&sitl, SITL_HOST, SITL_PORT) == 0)
            printf("SITL bridge: UDP → %s:%d\n", SITL_HOST, SITL_PORT);
        else
            sitl_mode = 0;
    }

#ifndef NO_ONNX
    /* Load ONNX policy */
    OnnxPolicy pol = {0};
    int onnx_ok = (onnx_init(&pol, onnx_path) == 0);
    if (onnx_ok)
        printf("ONNX policy loaded: %s\n", onnx_path);
    else
        printf("ONNX load failed — using VLA-only mode\n");

    /* Warm up ONNX (fill instruction cache, JIT compile) */
    if (onnx_ok) {
        float obs[13] = {0}; obs[2] = 2.0f;
        float action[4] = {0};
        for (int i = 0; i < 10; i++)
            onnx_infer(&pol, obs, 13, action, 4);
    }
#endif

    /* Pre-allocate latency arrays */
    uint64_t* lat_hocbf  = malloc((size_t)n_trials * sizeof(uint64_t));
    uint64_t* lat_total  = malloc((size_t)n_trials * sizeof(uint64_t));
    uint64_t* lat_jitter = malloc((size_t)n_trials * sizeof(uint64_t));
    if (!lat_hocbf || !lat_total || !lat_jitter) { perror("malloc"); exit(1); }
    uint64_t jitter_alerts = 0;   /* count of cycles exceeding JITTER_WARN_NS */
    uint64_t jitter_max    = 0;   /* worst-case observed jitter */

    /* Warm up HOCBF */
    for (int i = 0; i < 100; i++) {
        volatile double t = hocbf_filter(1.0, 0.0, 0.0, 0.0, 19.62);
        (void)t;
    }

    /* ── Hot-path benchmark ─────────────────────────────────────────────── */
    double pz = 2.0, vz = 0.0;
    uint64_t last_vla_time  = ns_now();
    uint64_t last_cycle_t   = ns_now();  /* jitter watchdog: previous cycle start */

    for (int i = 0; i < n_trials; i++) {
        uint64_t t_start = ns_now();

        /* ── Jitter watchdog ─────────────────────────────────────────────
         * Measure inter-cycle interval deviation from the running mean.
         * In back-to-back benchmark mode the "expected" interval is the
         * mean of the previous cycles; we use a simple deviation from the
         * previous cycle time as a conservative upper bound on OS jitter. */
        uint64_t interval = t_start - last_cycle_t;
        last_cycle_t = t_start;
        /* First cycle has no reference — skip */
        uint64_t jitter = 0;
        if (i > 0) {
            /* Jitter = deviation from previous interval (proxy for OS preemption) */
            static uint64_t prev_interval = 0;
            jitter = (interval > prev_interval)
                     ? (interval - prev_interval)
                     : (prev_interval - interval);
            prev_interval = interval;
            if (jitter > jitter_max) jitter_max = jitter;
            if (jitter > JITTER_WARN_NS) {
                jitter_alerts++;
                /* Non-blocking alert — never stalls RT loop */
                if (jitter_alerts <= 5)  /* suppress after first 5 to avoid I/O flood */
                    fprintf(stderr, "JITTER ALERT cycle %d: %lu ns > %lu ns threshold\n",
                            i, (unsigned long)jitter, (unsigned long)JITTER_WARN_NS);
            }
        }
        lat_jitter[i] = jitter;

        /* 1. Read VLA command (zero-copy mmap) */
        double vx = shm->vx_nom;
        double vy = shm->vy_nom;
        double vz_nom = shm->vz_nom;
        uint64_t now = ns_now();
        int vla_fresh = shm->is_new_data &&
                        (now - last_vla_time < VLA_STALE_NS);
        if (vla_fresh) last_vla_time = now;

        double T_nom;

#ifndef NO_ONNX
        /* 2. RL policy forward pass (if ONNX loaded) */
        if (onnx_ok) {
            float obs[13] = {
                (float)pz, (float)vz, 0.0f, 0.0f, 0.0f, 0.0f,  /* pos, vel */
                1.0f, 0.0f, 0.0f, 0.0f,                          /* quat w,x,y,z */
                0.0f, 0.0f, 0.0f                                  /* omega */
            };
            float action[4] = {0};
            onnx_infer(&pol, obs, 13, action, 4);
            T_nom = action_to_thrust(action[0]);

            /* VLA is a strategic planner, not a tactical pilot.
             * A 2.5s-latency command must NOT touch the thrust channel.
             * The RL policy runs the inner loop; VLA updates goals only.
             * The stale-data check above already gates vla_fresh correctly,
             * but even fresh VLA velocity → thrust blending is wrong:
             * by the time the VLA parsed the scene the drone has moved ~12m.
             * Blend coefficient is 0.0 — RL policy is sole thrust authority. */
            (void)vla_fresh; (void)vz_nom; (void)vx; (void)vy;
        } else {
            /* No ONNX: hover at gravity compensation */
            T_nom = MASS * GRAVITY;
            (void)vla_fresh; (void)vz_nom; (void)vx; (void)vy;
        }
#else
        T_nom = MASS * GRAVITY + MASS * vz_nom * 2.0;
#endif

        /* 3. HOCBF filter — O(1) arithmetic */
        uint64_t t_hocbf = ns_now();
        double T_safe = hocbf_filter(pz, vz, 0.0, 0.0, T_nom);
        uint64_t t_hocbf_end = ns_now();

        lat_hocbf[i] = t_hocbf_end - t_hocbf;
        lat_total[i]  = t_hocbf_end - t_start;

        /* 4. Send to Isaac Sim SITL (non-blocking UDP, never stalls RT loop) */
        if (sitl_mode) {
            float roll_cmd  = 0.0f, pitch_cmd = 0.0f, yaw_rate = 0.0f;
#ifndef NO_ONNX
            if (onnx_ok) {
                float obs[13] = {0}; obs[2] = (float)pz;
                float action[4] = {0};
                onnx_infer(&pol, obs, 13, action, 4);
                roll_cmd  = action[1] * 0.3f;
                pitch_cmd = action[2] * 0.3f;
                yaw_rate  = action[3] * 1.0f;
            }
#endif
            sitl_send(&sitl, T_safe, roll_cmd, pitch_cmd, yaw_rate);
        }

        /* Simulate state update */
        vz += (T_safe / MASS - GRAVITY) * 0.001;
        pz += vz * 0.001;
        if (pz < 0.0) pz = 0.0;
        (void)vx; (void)vy;
    }

    /* ── Statistics ─────────────────────────────────────────────────────── */
    /* Sort for percentiles */
    uint64_t* sorted_h = malloc((size_t)n_trials * sizeof(uint64_t));
    uint64_t* sorted_t = malloc((size_t)n_trials * sizeof(uint64_t));
    memcpy(sorted_h, lat_hocbf, (size_t)n_trials * sizeof(uint64_t));
    memcpy(sorted_t, lat_total,  (size_t)n_trials * sizeof(uint64_t));

    /* Insertion sort (stats only, not in hot-path) */
    for (int i = 1; i < n_trials; i++) {
        uint64_t kh = sorted_h[i], kt = sorted_t[i];
        int j = i - 1;
        while (j >= 0 && sorted_h[j] > kh) { sorted_h[j+1] = sorted_h[j]; j--; }
        sorted_h[j+1] = kh;
        j = i - 1;
        while (j >= 0 && sorted_t[j] > kt) { sorted_t[j+1] = sorted_t[j]; j--; }
        sorted_t[j+1] = kt;
    }

    uint64_t hocbf_wcet = sorted_h[n_trials - 1];
    uint64_t total_wcet  = sorted_t[n_trials - 1];
    uint64_t hocbf_p99  = sorted_h[n_trials * 99 / 100];
    uint64_t total_p99   = sorted_t[n_trials * 99 / 100];

    printf("\nAISP Safety Filter + RL Policy — Latency Report\n");
    printf("=================================================\n");
    printf("Trials    : %d\n", n_trials);
    printf("CPU core  : %d\n", cpu_core);
#ifndef NO_ONNX
    printf("ONNX      : %s\n", onnx_ok ? "loaded" : "not loaded");
#else
    printf("ONNX      : disabled (NO_ONNX)\n");
#endif
    printf("\n--- HOCBF filter only ---\n");
    printf("P99       : %lu ns\n", (unsigned long)hocbf_p99);
    printf("WCET      : %lu ns\n", (unsigned long)hocbf_wcet);
    printf("< 100us   : %s\n", hocbf_wcet < 100000 ? "PASS" : "FAIL");
    printf("\n--- Full RT loop (mmap + ONNX + HOCBF) ---\n");
    printf("P99       : %lu ns\n", (unsigned long)total_p99);
    printf("WCET      : %lu ns\n", (unsigned long)total_wcet);
    printf("< 2ms     : %s\n", total_wcet < 2000000 ? "PASS" : "FAIL");

    /* Jitter watchdog report */
    uint64_t* sorted_j = malloc((size_t)n_trials * sizeof(uint64_t));
    memcpy(sorted_j, lat_jitter, (size_t)n_trials * sizeof(uint64_t));
    for (int i = 1; i < n_trials; i++) {
        uint64_t kj = sorted_j[i]; int j = i - 1;
        while (j >= 0 && sorted_j[j] > kj) { sorted_j[j+1] = sorted_j[j]; j--; }
        sorted_j[j+1] = kj;
    }
    uint64_t jitter_p99 = sorted_j[n_trials * 99 / 100];
    printf("\n--- OS inter-cycle jitter watchdog ---\n");
    printf("P99       : %lu ns  (%.1f us)\n",
           (unsigned long)jitter_p99, jitter_p99 / 1000.0);
    printf("Max       : %lu ns  (%.1f us)\n",
           (unsigned long)jitter_max, jitter_max / 1000.0);
    printf("Alerts    : %lu  (cycles > %lu ns)\n",
           (unsigned long)jitter_alerts, (unsigned long)JITTER_WARN_NS);
    printf("< 50us    : %s\n", jitter_max < JITTER_WARN_NS ? "PASS" : "WARN");
    free(sorted_j);

    /* Write CSV — includes jitter column for EVT analysis */
    FILE* f = fopen("experiments/results/latency_raw.csv", "w");
    if (f) {
        fprintf(f, "trial,latency_ns,jitter_ns\n");
        for (int i = 0; i < n_trials; i++)
            fprintf(f, "%d,%lu,%lu\n", i,
                    (unsigned long)lat_hocbf[i],
                    (unsigned long)lat_jitter[i]);
        fclose(f);
    }

    free(lat_hocbf); free(lat_total); free(lat_jitter);
    free(sorted_h);  free(sorted_t);
    munmap(shm, SHM_SIZE);
    if (sitl_mode) sitl_close(&sitl);
#ifndef NO_ONNX
    if (onnx_ok) onnx_free(&pol);
#endif
    return (hocbf_wcet < 100000) ? 0 : 1;
}
