/*
 * VLA Stale-Command Watchdog Implementation
 * State machine: STARTUP -> FRESH -> STALE (with recovery to FRESH)
 */
#include "watchdog.h"

static VLAState vla_state = VLA_STATE_STARTUP;
static uint64_t last_vla_time = 0;

void watchdog_init(void) {
    vla_state = VLA_STATE_STARTUP;
    last_vla_time = 0;
}

bool vla_watchdog_check(const VLACommand* shm, uint64_t now) {
    /* In STARTUP or STALE: any new command (is_new_data=1) is accepted
     * as fresh regardless of timestamp. Only in FRESH state do we
     * enforce the 100ms staleness window. */
    bool fresh = false;
    if (vla_state == VLA_STATE_FRESH) {
        fresh = shm->is_new_data && (now - last_vla_time < VLA_STALE_NS);
    } else {
        fresh = shm->is_new_data;  /* STARTUP or STALE: accept any new data */
    }

    if (fresh) {
        last_vla_time = now;
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