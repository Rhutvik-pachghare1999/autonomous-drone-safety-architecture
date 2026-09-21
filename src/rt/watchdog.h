/*
 * VLA Stale-Command Watchdog
 * State machine: STARTUP -> FRESH -> STALE (with recovery to FRESH)
 * Exposed for real-time loop and unit tests.
 */
#ifndef AISP_WATCHDOG_H
#define AISP_WATCHDOG_H

#include <stdint.h>
#include <stdbool.h>

/* Timeout for considering VLA command stale */
#define VLA_STALE_NS  100000000ULL  /* 100ms */

typedef enum {
    VLA_STATE_STARTUP = 0,  /* No valid command received yet */
    VLA_STATE_FRESH   = 1,  /* Receiving fresh commands */
    VLA_STATE_STALE   = 2   /* Timeout exceeded, using fallback */
} VLAState;

/* Shared memory command layout (must match Python shm_bridge.py) */
typedef struct __attribute__((aligned(64))) {
    uint64_t sequence_number;
    double   vx_nom, vy_nom, vz_nom;
    uint8_t  is_new_data;
    char     _pad[31];
} VLACommand;

/* Initialize watchdog state (call once at startup) */
void watchdog_init(void);

/* Check VLA command freshness and update watchdog state.
 * Returns true if command is fresh and should be used.
 * Must be called every control cycle with current monotonic time. */
bool vla_watchdog_check(const VLACommand* shm, uint64_t now);

/* Get current watchdog state (for diagnostics) */
VLAState watchdog_state(void);

#endif /* AISP_WATCHDOG_H */