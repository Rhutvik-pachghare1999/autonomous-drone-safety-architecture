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

/* Shared memory command layout (must match Python src/utils/shm_bridge.py).
 *
 * Seqlock: the Python publisher performs three mmap byte-copies, and mmap
 * writes carry NO atomicity guarantee against concurrent mmap readers — a
 * reader can observe the buffer mid-copy (e.g. new seq_head paired with
 * stale vx/vy/vz, a torn frame). Protocol:
 *   writer: seq_head = odd (write in progress)
 *           -> payload fields + seq_tail (= even seq)
 *           -> seq_head = even (stable)
 *   reader: vla_shm_snapshot() — accept iff seq_head even, seq_tail ==
 *           seq_head, and seq_head stable across the copy.
 * Offsets: 0 seq_head | 8 vx_nom | 16 vy_nom | 24 vz_nom | 32 is_new_data
 *          | 33 _pad1[7] | 40 seq_tail | 48 _pad2[16]  = 64 bytes */
typedef struct __attribute__((aligned(64))) {
    uint64_t seq_head;
    double   vx_nom, vy_nom, vz_nom;
    uint8_t  is_new_data;
    char     _pad1[7];
    uint64_t seq_tail;
    char     _pad2[16];
} VLACommand;

/* Initialize watchdog state (call once at startup) */
void watchdog_init(void);

/* Take a torn-free snapshot of the SHM command. Returns true iff the copy
 * is consistent (even seq_head, seq_head == seq_tail, head stable across
 * the copy). Bounded: at most 2 attempts — never spins in the RT loop.
 * On false (torn read), callers must treat the cycle as no-new-data and
 * must NOT consume any payload fields. */
bool vla_shm_snapshot(const VLACommand* shm, VLACommand* out);

/* Check VLA command freshness against a torn-free snapshot from
 * vla_shm_snapshot() and update watchdog state. Returns true if command
 * is fresh and should be used. Must be called every control cycle with
 * current monotonic time (also drives the FRESH->STALE timeout). */
bool vla_watchdog_check(const VLACommand* snap, uint64_t now);

/* Get current watchdog state (for diagnostics) */
VLAState watchdog_state(void);

#endif /* AISP_WATCHDOG_H */