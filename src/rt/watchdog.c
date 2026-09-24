/*
 * VLA Stale-Command Watchdog Implementation
 * State machine: STARTUP -> FRESH -> STALE (with recovery to FRESH)
 */
#include "watchdog.h"

static VLAState vla_state = VLA_STATE_STARTUP;
static uint64_t last_vla_time = 0;
static uint64_t last_seq = 0;  /* Track last seen sequence number */

void watchdog_init(void) {
    vla_state = VLA_STATE_STARTUP;
    last_vla_time = 0;
    last_seq = 0;
}

bool vla_shm_snapshot(const VLACommand* shm, VLACommand* out) {
    /* Seqlock read side. mmap writes from the Python publisher are plain
     * userspace byte copies: with no synchronization, this reader could
     * see seq_head updated while vx/vy/vz are still old (torn frame).
     * Classic seqlock: reject odd heads (write in progress), reject
     * head/tail mismatch, re-check head stability after the payload copy.
     * Two attempts max — if torn twice, the writer is racing faster than
     * a control cycle; report no-new-data rather than consume garbage. */
    for (int attempt = 0; attempt < 2; attempt++) {
        uint64_t h1 = shm->seq_head;
        __sync_synchronize();   /* don't let the compiler fold/re-order reads */
        if (h1 & 1ULL) continue;                    /* write in progress */
        out->vx_nom      = shm->vx_nom;
        out->vy_nom      = shm->vy_nom;
        out->vz_nom      = shm->vz_nom;
        out->is_new_data = shm->is_new_data;
        __sync_synchronize();
        uint64_t t = shm->seq_tail;
        if (t == h1 && shm->seq_head == h1) {
            out->seq_head = h1;
            out->seq_tail = t;
            return true;
        }
    }
    return false;
}

bool vla_watchdog_check(const VLACommand* snap, uint64_t now) {
    /* In STARTUP or STALE: accept any new data (seq changed or first data).
     * In FRESH: require seq changed AND within 100ms window.
     * Caller guarantees *snap is a torn-free snapshot. */
    bool fresh = false;
    uint64_t seq = snap->seq_head;
    bool has_new_data = snap->is_new_data && (seq != last_seq);

    if (vla_state == VLA_STATE_FRESH) {
        fresh = has_new_data && (now - last_vla_time < VLA_STALE_NS);
    } else {
        fresh = has_new_data;  /* STARTUP or STALE: accept any new data */
    }

    if (fresh) {
        last_vla_time = now;
        last_seq = seq;
        if (vla_state == VLA_STATE_STARTUP || vla_state == VLA_STATE_STALE) {
            vla_state = VLA_STATE_FRESH;
        }
        return true;
    }

    /* No fresh data: check for FRESH->STALE timeout */
    if (vla_state == VLA_STATE_FRESH && (now - last_vla_time >= VLA_STALE_NS)) {
        vla_state = VLA_STATE_STALE;
    }
    /* STARTUP and STALE states remain unchanged without fresh data */
    return false;
}

VLAState watchdog_state(void) {
    return vla_state;
}