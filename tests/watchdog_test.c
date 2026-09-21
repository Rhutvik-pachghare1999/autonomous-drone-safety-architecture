/*
 * VLA Watchdog Unit Test — exercises the REAL watchdog implementation
 * from src/rt/watchdog.c (single source of truth).
 *
 * Build (run from repo root):
 *   gcc -O3 -I src/rt -o /tmp/watchdog_test tests/watchdog_test.c src/rt/watchdog.c -lm -lrt
 *   /tmp/watchdog_test
 *
 * CI: Add as CMake target to run on every build.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include "watchdog.h"

static const char* state_name(VLAState s) {
    switch (s) {
        case VLA_STATE_STARTUP: return "STARTUP";
        case VLA_STATE_FRESH:   return "FRESH";
        case VLA_STATE_STALE:   return "STALE";
        default: return "UNKNOWN";
    }
}

static void assert_true(const char* test, bool cond, const char* msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s — %s (state=%s)\n", test, msg, state_name(watchdog_state()));
        exit(1);
    }
    printf("PASS: %s\n", test);
}

static void test_startup_fresh_first_command(void) {
    watchdog_init();
    VLACommand cmd = {0};
    cmd.is_new_data = 1;
    cmd.sequence_number = 1;  // First sequence number
    uint64_t now = 500000000ULL;  // large monotonic value
    bool r = vla_watchdog_check(&cmd, now);
    assert_true("STARTUP→FRESH on first fresh command", r && watchdog_state() == VLA_STATE_FRESH,
                "first command should be FRESH even with large now");
}

static void test_stale_after_timeout(void) {
    watchdog_init();
    VLACommand cmd = {0};
    cmd.is_new_data = 1;
    cmd.sequence_number = 1;
    uint64_t now = 1000000000ULL;
    vla_watchdog_check(&cmd, now);  // first fresh
    assert_true("initial FRESH", watchdog_state() == VLA_STATE_FRESH, "");

    /* Advance time by 150ms (no new data - same sequence) */
    now += 150000000ULL;
    cmd.is_new_data = 0;  // seq still 1
    vla_watchdog_check(&cmd, now);
    assert_true("FRESH→STALE after 100ms timeout", watchdog_state() == VLA_STATE_STALE,
                "should transition to STALE");
}

static void test_recover_on_new_fresh(void) {
    watchdog_init();
    VLACommand cmd = {0};
    cmd.is_new_data = 1;
    cmd.sequence_number = 1;
    uint64_t now = 1000000000ULL;
    vla_watchdog_check(&cmd, now);  // FRESH

    /* Go stale */
    now += 150000000ULL;
    cmd.is_new_data = 0;
    vla_watchdog_check(&cmd, now);
    assert_true("went STALE", watchdog_state() == VLA_STATE_STALE, "");

    /* New fresh command arrives with NEW sequence number */
    now += 50000000ULL;
    cmd.is_new_data = 1;
    cmd.sequence_number = 2;  // NEW sequence
    bool r = vla_watchdog_check(&cmd, now);
    assert_true("STALE→FRESH on new data", r && watchdog_state() == VLA_STATE_FRESH,
                "should recover to FRESH");
}

static void test_no_recover_on_stale_data(void) {
    watchdog_init();
    VLACommand cmd = {0};
    cmd.is_new_data = 1;
    cmd.sequence_number = 1;
    uint64_t now = 1000000000ULL;
    vla_watchdog_check(&cmd, now);  // FRESH

    /* Go stale */
    now += 150000000ULL;
    cmd.is_new_data = 0;
    vla_watchdog_check(&cmd, now);
    assert_true("went STALE", watchdog_state() == VLA_STATE_STALE, "");

    /* Old command arrives (same seq=1, is_new_data=0) — should stay STALE */
    now += 10000000ULL;
    cmd.is_new_data = 0;
    cmd.sequence_number = 1;
    bool r = vla_watchdog_check(&cmd, now);
    assert_true("STALE stays STALE on old data", !r && watchdog_state() == VLA_STATE_STALE,
                "should stay STALE");
}

static void test_fresh_timeout_exact_boundary(void) {
    watchdog_init();
    VLACommand cmd = {0};
    cmd.is_new_data = 1;
    cmd.sequence_number = 1;
    uint64_t now = 1000000000ULL;
    vla_watchdog_check(&cmd, now);  // FRESH

    /* Just before timeout boundary (99,999,999 ns) — should be fresh */
    now += 99999999ULL;
    cmd.is_new_data = 0;
    vla_watchdog_check(&cmd, now);
    assert_true("FRESH just before 100ms boundary", watchdog_state() == VLA_STATE_FRESH,
                "should still be FRESH before boundary");

    /* At exactly 100ms — should be stale (>= check) */
    now += 1;
    vla_watchdog_check(&cmd, now);
    assert_true("STALE at exactly 100ms boundary", watchdog_state() == VLA_STATE_STALE,
                "should be STALE at boundary (>= check)");
}

int main(void) {
    printf("=== Watchdog Real Function Tests (src/rt/watchdog.c) ===\n");
    test_startup_fresh_first_command();
    test_stale_after_timeout();
    test_recover_on_new_fresh();
    test_no_recover_on_stale_data();
    test_fresh_timeout_exact_boundary();
    printf("\nAll watchdog tests PASSED\n");
    return 0;
}