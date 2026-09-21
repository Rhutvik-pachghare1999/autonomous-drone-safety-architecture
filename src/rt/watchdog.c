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

bool vla_watchdog_check(const VLACommand* shm, uint64_t now) {
    /* In STARTUP or STALE: accept any new data (seq changed or first data).
     * In FRESH: require seq changed AND within 100ms window. */
    bool fresh = false;
    uint64_t seq = shm->sequence_number;
    bool has_new_data = shm->is_new_data && (seq != last_seq);

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