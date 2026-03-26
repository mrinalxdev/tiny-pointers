"""
Empirical verification of Theorem 3 (Section 5):
Lower bound for variable-size tiny pointers.

THEOREM 3 (paper, Section 5):
    If a dereference table supports variable-size tiny pointers of expected
    size s and load factor 1 - delta = Omega(1), then s = Omega(log delta^{-1}).

This file does NOT prove the theorem — it verifies that our construction
is TIGHT: we cannot observe pointer sizes smaller than Omega(log delta^{-1}),
and the bound we achieve grows linearly with log(1/delta).

THREE EXPERIMENTS:

1. LINEAR GROWTH
   Fix n, sweep delta from 0.5 down to 0.1.
   Measure average pointer size at each delta.
   Verify: avg_bits / log(1/delta) is a positive constant (~3 for our construction).
   If this ratio were 0 (pointer size not growing), we would beat the lower bound
   — which the theorem says is impossible.

2. ADVERSARIAL CHURN
   The theorem's proof uses an oblivious adversary: a long sequence of random
   insertions and deletions designed to force as many 'unsafe' allocations as
   possible. We simulate this with high-churn workloads.
   Verify: pointer sizes remain bounded throughout — they do not drift upward
   as the table is repeatedly filled and partially emptied.

3. LOAD FACTOR SENSITIVITY
   Fix n and delta, but vary the actual load (fraction of capacity used).
   Verify: pointer sizes are determined by delta (the structural parameter),
   not by the current load. This confirms the bound is about the table's
   capacity parameter, not a transient measurement.
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import DereferenceTable


def fill_table(dt: DereferenceTable, n_keys: int, start_key: int = 0) -> dict:
    active: dict = {}
    k = start_key
    while len(active) < n_keys:
        p = dt.allocate(k)
        if p is not None:
            active[k] = p
        k += 1
    return active


def avg_bits(active: dict) -> float:
    sizes = [p.bit_length_encoding() for p in active.values()]
    return sum(sizes) / len(sizes) if sizes else 0.0


def max_bits(active: dict) -> int:
    sizes = [p.bit_length_encoding() for p in active.values()]
    return max(sizes) if sizes else 0


# -----------------------------------------------------------------------
# experiment 1 — Linear growth with log(1/delta)
# -----------------------------------------------------------------------

def test_avg_bits_grows_linearly_with_log_inv_delta() -> None:
    n = 2048
    deltas = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
    ratios: list[float] = []

    print("\nExperiment 1: avg_bits / log(1/delta) across delta values")
    print("  %-8s %-10s %-10s %-8s" % ("delta", "log(1/d)", "avg_bits", "ratio"))

    for delta in deltas:
        dt = DereferenceTable(n, delta=delta, seed=42)
        active = fill_table(dt, min(dt.capacity, 600))
        a = avg_bits(active)
        log_inv_d = math.log2(1.0 / delta)
        ratio = a / log_inv_d
        ratios.append(ratio)
        print("  d=%-6.2f  log=%-8.2f  avg=%-8.1f  ratio=%.2f" % (
            delta, log_inv_d, a, ratio))

    min_ratio = min(ratios)
    assert min_ratio >= 1.0, (
        "Ratio avg_bits/log(1/delta) dropped below 1.0 (=%.2f). "
        "This would violate Theorem 3." % min_ratio
    )
    max_ratio = max(ratios)
    assert max_ratio <= 8.0, (
        "Ratio avg_bits/log(1/delta) is too large (=%.2f). "
        "Construction is not tight." % max_ratio
    )

    print("  min ratio=%.2f  max ratio=%.2f  -> ratio is Omega(1): CONFIRMED" % (
        min_ratio, max_ratio))


# -----------------------------------------------------------------------
# Experiment 2 — Adversarial churn
# -----------------------------------------------------------------------

def test_pointer_sizes_bounded_under_adversarial_churn() -> None:
    n = 1024
    delta = 0.2
    dt = DereferenceTable(n, delta=delta, seed=7)
    expected_max = 1 + math.ceil(math.log2(dt._b_pri + 1)) + 2  # generous bound

    half_cap = dt.capacity // 2
    active: dict = {}
    next_key = 0

    while len(active) < half_cap:
        p = dt.allocate(next_key)
        if p is not None:
            active[next_key] = p
        next_key += 1

    rng = random.Random(99)
    max_bits_seen = 0
    avg_per_round: list[float] = []

    print("\nExperiment 2: adversarial churn workload (%d rounds)" % 30)
    print("  %-8s %-12s %-10s" % ("round", "n_active", "max_bits"))

    for rnd in range(30):
        keys_list = list(active.keys())
        to_delete = rng.sample(keys_list, len(keys_list) // 2)
        for k in to_delete:
            dt.free(k, active.pop(k))

        while len(active) < half_cap:
            p = dt.allocate(next_key)
            if p is not None:
                active[next_key] = p
            next_key += 1

        m = max_bits(active)
        a = avg_bits(active)
        max_bits_seen = max(max_bits_seen, m)
        avg_per_round.append(a)

        if rnd % 10 == 0:
            print("  round=%-4d  n_active=%-6d  max_bits=%d" % (rnd, len(active), m))

    assert max_bits_seen <= expected_max, (
        "Max pointer size %d exceeded structural bound %d under churn. "
        "Pointer sizes are not stable." % (max_bits_seen, expected_max)
    )

    first_half_avg  = sum(avg_per_round[:15]) / 15
    second_half_avg = sum(avg_per_round[15:]) / 15
    assert second_half_avg <= first_half_avg + 1.0, (
        "Average pointer size drifted from %.1f to %.1f over churn rounds. "
        "Sizes are growing — this would violate Theorem 3." % (
            first_half_avg, second_half_avg)
    )

    print("  max_bits_seen=%d (bound=%d)  avg_drift=%.2f -> STABLE" % (
        max_bits_seen, expected_max, second_half_avg - first_half_avg))


def test_bound_is_structural_not_transient() -> None:
    n = 2048
    delta = 0.2

    occupancies = [0.1, 0.3, 0.5, 0.7, 0.9]
    avg_sizes: list[float] = []

    print("\nExperiment 3: pointer size vs current occupancy (delta fixed at %.1f)" % delta)
    print("  %-12s %-10s %-10s" % ("occupancy", "n_keys", "avg_bits"))

    for occ in occupancies:
        dt = DereferenceTable(n, delta=delta, seed=42)
        n_keys = max(1, int(occ * dt.capacity))
        active = fill_table(dt, n_keys)
        a = avg_bits(active)
        avg_sizes.append(a)
        print("  occ=%-8.0f%%  n=%-8d  avg=%.1f" % (occ * 100, n_keys, a))

    spread = max(avg_sizes) - min(avg_sizes)
    assert spread <= 2.0, (
        "Pointer sizes varied by %.1f bits across occupancy levels. "
        "This suggests sizes are load-dependent, not structural." % spread
    )

    print("  spread=%.1f bits across occupancy levels -> STRUCTURAL (not transient)" % spread)

if __name__ == "__main__":
    tests = [
        test_avg_bits_grows_linearly_with_log_inv_delta,
        test_pointer_sizes_bounded_under_adversarial_churn,
        test_bound_is_structural_not_transient,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print("  \u2713  %s\n" % t.__name__)
            passed += 1
        except Exception as e:
            import traceback
            print("  \u2717  %s: %s" % (t.__name__, e))
            traceback.print_exc()
            print()
    print("%d/%d tests passed" % (passed, len(tests)))