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

/* Build a clean (un-torn) seqlock frame: even seq, head == tail.
 * This mirrors what vla_shm_snapshot() accepts from the Python writer. */
static void set_cmd(VLACommand* c, uint64_t seq, uint8_t is_new) {
    c->seq_head = seq;
    c->seq_tail = seq;
    c->is_new_data = is_new;
}

/* Take snapshot then run watchdog — mirrors the real RT loop call site. */
static bool snap_check(VLACommand* c, uint64_t now) {
    VLACommand snap = {0};
    bool clean = vla_shm_snapshot(c, &snap);
    if (!clean) memset(&snap, 0, sizeof(snap));  /* torn -> no new data */
    return vla_watchdog_check(&snap, now);
}

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
    set_cmd(&cmd, 2, 1);  // first command (even seq: seqlock-stable)
    uint64_t now = 500000000ULL;  // large monotonic value
    bool r = snap_check(&cmd, now);
    assert_true("STARTUP→FRESH on first fresh command", r && watchdog_state() == VLA_STATE_FRESH,
                "first command should be FRESH even with large now");
}

static void test_stale_after_timeout(void) {
    watchdog_init();
    VLACommand cmd = {0};
    set_cmd(&cmd, 2, 1);
    uint64_t now = 1000000000ULL;
    snap_check(&cmd, now);  // first fresh
    assert_true("initial FRESH", watchdog_state() == VLA_STATE_FRESH, "");

    /* Advance time by 150ms (no new data - same sequence) */
    now += 150000000ULL;
    set_cmd(&cmd, 2, 0);  // seq still 2, no new data
    snap_check(&cmd, now);
    assert_true("FRESH→STALE after 100ms timeout", watchdog_state() == VLA_STATE_STALE,
                "should transition to STALE");
}

static void test_recover_on_new_fresh(void) {
    watchdog_init();
    VLACommand cmd = {0};
    set_cmd(&cmd, 2, 1);
    uint64_t now = 1000000000ULL;
    snap_check(&cmd, now);  // FRESH

    /* Go stale */
    now += 150000000ULL;
    set_cmd(&cmd, 2, 0);
    snap_check(&cmd, now);
    assert_true("went STALE", watchdog_state() == VLA_STATE_STALE, "");

    /* New fresh command arrives with NEW sequence number */
    now += 50000000ULL;
    set_cmd(&cmd, 4, 1);  // NEW sequence
    bool r = snap_check(&cmd, now);
    assert_true("STALE→FRESH on new data", r && watchdog_state() == VLA_STATE_FRESH,
                "should recover to FRESH");
}

static void test_no_recover_on_stale_data(void) {
    watchdog_init();
    VLACommand cmd = {0};
    set_cmd(&cmd, 2, 1);
    uint64_t now = 1000000000ULL;
    snap_check(&cmd, now);  // FRESH

    /* Go stale */
    now += 150000000ULL;
    set_cmd(&cmd, 2, 0);
    snap_check(&cmd, now);
    assert_true("went STALE", watchdog_state() == VLA_STATE_STALE, "");

    /* Old command arrives (same seq=1, is_new_data=0) — should stay STALE */
    now += 10000000ULL;
    set_cmd(&cmd, 2, 0);  // old data, original seq
    bool r = snap_check(&cmd, now);
    assert_true("STALE stays STALE on old data", !r && watchdog_state() == VLA_STATE_STALE,
                "should stay STALE");
}

static void test_fresh_timeout_exact_boundary(void) {
    watchdog_init();
    VLACommand cmd = {0};
    set_cmd(&cmd, 2, 1);
    uint64_t now = 1000000000ULL;
    snap_check(&cmd, now);  // FRESH

    /* Just before timeout boundary (99,999,999 ns) — should be fresh */
    now += 99999999ULL;
    set_cmd(&cmd, 2, 0);
    snap_check(&cmd, now);
    assert_true("FRESH just before 100ms boundary", watchdog_state() == VLA_STATE_FRESH,
                "should still be FRESH before boundary");

    /* At exactly 100ms — should be stale (>= check) */
    now += 1;
    snap_check(&cmd, now);
    assert_true("STALE at exactly 100ms boundary", watchdog_state() == VLA_STATE_STALE,
                "should be STALE at boundary (>= check)");
}


static void test_torn_read_rejected(void) {
    /* Seqlock: odd seq_head = write in progress; head != tail = raced a
     * write. Both must be rejected by vla_shm_snapshot (no new data,
     * never a mixed old/new velocity vector). */
    watchdog_init();
    VLACommand cmd = {0};

    /* Clean frame seq=2: fresh, payload intact. */
    set_cmd(&cmd, 2, 1);
    cmd.vx_nom = 1.0; cmd.vy_nom = 2.0; cmd.vz_nom = 3.0;
    uint64_t now = 1000000000ULL;
    assert_true("clean frame is fresh", snap_check(&cmd, now), "");
    VLACommand snap = {0};
    assert_true("snapshot clean", vla_shm_snapshot(&cmd, &snap), "");
    assert_true("payload intact",
                snap.vx_nom == 1.0 && snap.vy_nom == 2.0 && snap.vz_nom == 3.0, "");

    /* Torn: odd head (writer mid-copy). */
    cmd.seq_head = 3; cmd.seq_tail = 3;
    snap = (VLACommand){0};
    assert_true("odd head rejected", !vla_shm_snapshot(&cmd, &snap), "");

    /* Torn: even head but stale tail (raced a full write pass). */
    cmd.seq_head = 4; cmd.seq_tail = 2;
    snap = (VLACommand){0};
    assert_true("head/tail mismatch rejected", !vla_shm_snapshot(&cmd, &snap), "");

    /* After the two torn rejects, seq=2 is still last_seq: a clean
     * re-present of seq=4 is accepted and marks fresh. */
    set_cmd(&cmd, 4, 1);
    assert_true("clean seq=4 fresh after torn rejects",
                snap_check(&cmd, now), "");
}

int main(void) {
    printf("=== Watchdog Real Function Tests (src/rt/watchdog.c) ===\n");
    test_startup_fresh_first_command();
    test_stale_after_timeout();
    test_recover_on_new_fresh();
    test_no_recover_on_stale_data();
    test_fresh_timeout_exact_boundary();
    test_torn_read_rejected();
    printf("\nAll watchdog tests PASSED\n");
    return 0;
}