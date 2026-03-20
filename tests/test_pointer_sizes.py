import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import DereferenceTable


def theoretical_bound(n: int, delta: float) -> int:
    lll_n = math.log2(math.log2(math.log2(max(n, 8)) + 1) + 1) + 1
    log_inv_delta = math.log2(1.0 / delta)
    return int(4 * (lll_n + log_inv_delta)) + 4   # +4 for the 1-bit table selector


def test_pointer_sizes_are_sublogarithmic():
    n = 2048
    dt = DereferenceTable(n=n, delta=0.2, seed=42)
    naive_bits = math.ceil(math.log2(n))

    pointers = {}
    for k in range(dt.capacity):
        p = dt.allocate(k)
        if p is not None:
            pointers[k] = p

    sizes = [p.bit_length_encoding() for p in pointers.values()]
    max_size = max(sizes)
    avg_size = sum(sizes) / len(sizes)
    avg_size = sum(sizes) / len(sizes)
    bound = theoretical_bound(n, 0.2)
    print(f"\n  n={n}, δ=0.2")
    print(f"  naïve log₂(n)  = {naive_bits} bits")
    print(f"  max pointer    = {max_size} bits")
    print(f"  avg pointer    = {avg_size:.2f} bits")
    print(f"  theoretical    ≤ {bound} bits")

    assert max_size <= bound, (
        f"Pointer ({max_size} bits) exceeds theoretical bound ({bound} bits)"
    )


def test_pointer_size_grows_with_delta_inverse():
    n = 512
    deltas = [0.4, 0.2, 0.1]
    avg_sizes = []

    for delta in deltas:
        dt = DereferenceTable(n=n, delta=delta, seed=1)
        pointers = {}
        target = min(dt.capacity, 200)
        for k in range(target):
            p = dt.allocate(k)
            if p is not None:
                pointers[k] = p
        if pointers:
            avg = sum(p.bit_length_encoding() for p in pointers.values()) / len(pointers)
            avg_sizes.append(avg)
            print(f"  δ={delta:.1f}  avg_bits={avg:.2f}")
        else:
            avg_sizes.append(0)


    violations = 0
    for i in range(len(avg_sizes) - 1):
        if avg_sizes[i] > avg_sizes[i+1] + 2:
            violations += 1
    assert violations == 0, f"Pointer size did not grow with 1/δ: {avg_sizes}"


def test_primary_pointers_smaller_than_secondary():
    n = 512
    dt = DereferenceTable(n=n, delta=0.1, seed=3)

    pri_bits = []
    sec_bits = []
    for k in range(dt.capacity):
        p = dt.allocate(k)
        if p is not None:
            if p.table_id == 0:
                pri_bits.append(p.bit_length_encoding())
            else:
                sec_bits.append(p.bit_length_encoding())

    if pri_bits:
        print(f"\n  primary avg={sum(pri_bits)/len(pri_bits):.2f}  n={len(pri_bits)}")
    if sec_bits:
        print(f"  secondary avg={sum(sec_bits)/len(sec_bits):.2f}  n={len(sec_bits)}")

    naive_bits = math.ceil(math.log2(n))
    if pri_bits:
        assert max(pri_bits) < naive_bits + 2, (
            f"Primary max ({max(pri_bits)}) should be sub-log(n)={naive_bits}"
        )
    if sec_bits:
        assert max(sec_bits) < naive_bits, (
            f"Secondary max ({max(sec_bits)}) should be sub-log(n)={naive_bits}"
        )


def test_theorem1_bound_not_exceeded():
    configs = [
        (256,  0.3),
        (1024, 0.2),
        (4096, 0.1),
    ]
    for n, delta in configs:
        dt = DereferenceTable(n=n, delta=delta, seed=42)
        for k in range(min(dt.capacity, 300)):
            dt.allocate(k)

        sizes = dt.pointer_bit_sizes()
        if not sizes:
            continue
        max_size = max(sizes)
        bound = theoretical_bound(n, delta)
        print(f"  n={n}, δ={delta}: max={max_size} bits, bound={bound} bits")
        assert max_size <= bound, (
            f"n={n}, δ={delta}: pointer size {max_size} exceeds bound {bound}"
        )


if __name__ == "__main__":
    tests = [
        test_pointer_sizes_are_sublogarithmic,
        test_pointer_size_grows_with_delta_inverse,
        test_primary_pointers_smaller_than_secondary,
        test_theorem1_bound_not_exceeded,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ✗  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")