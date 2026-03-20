import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import DereferenceTable, TinyPointer

def test_dereference_uniqueness_small() -> None:
    dt = DereferenceTable(n=64, delta=0.3, seed=42)
    pointers: dict[int, TinyPointer] = {}
    for k in range(40):
        p = dt.allocate(k)
        if p is not None:
            pointers[k] = p

    slots = [dt.dereference(k, p) for k, p in pointers.items()]
    assert len(slots) == len(set(slots)), (
        "Dereference collided — uniqueness violated!"
    )


def test_dereference_uniqueness_large() -> None:
    n = 1024
    dt = DereferenceTable(n=n, delta=0.2, seed=7)
    pointers: dict[int, TinyPointer] = {}
    for k in range(int(0.75 * n)):
        p = dt.allocate(k)
        if p is not None:
            pointers[k] = p

    slots = [dt.dereference(k, p) for k, p in pointers.items()]
    assert len(slots) == len(set(slots)), (
        "Uniqueness violated at large scale!"
    )

def test_dereference_is_stable() -> None:
    dt = DereferenceTable(n=128, delta=0.25, seed=99)

    p_target = dt.allocate(999)
    assert p_target is not None
    slot_before = dt.dereference(999, p_target)

    others: dict[int, TinyPointer] = {}
    for k in range(50):
        p = dt.allocate(k)
        if p is not None:
            others[k] = p

    slot_during = dt.dereference(999, p_target)
    assert slot_before == slot_during, (
        "Stability violated: slot changed while others active!"
    )

    for k, p in others.items():
        dt.free(k, p)

    slot_after = dt.dereference(999, p_target)
    assert slot_before == slot_after, (
        "Stability violated: slot changed after others freed!"
    )


def test_slot_reuse_after_free() -> None:
    dt = DereferenceTable(n=64, delta=0.3, seed=5)

    p1 = dt.allocate(42)
    assert p1 is not None
    dt.free(42, p1)

    p2 = dt.allocate(42)
    assert p2 is not None
    s2 = dt.dereference(42, p2)
    assert 0 <= s2 < dt._n_total


def test_free_and_reuse_cycle() -> None:
    dt = DereferenceTable(n=256, delta=0.2, seed=13)
    for cycle in range(100):
        p = dt.allocate(0)
        assert p is not None, f"Allocation failed at cycle {cycle}"
        dt.free(0, p)


def test_capacity_respected() -> None:
    n = 128
    dt = DereferenceTable(n=n, delta=0.3, seed=77)
    capacity = dt.capacity

    allocated: dict[int, TinyPointer] = {}
    for k in range(capacity + 20):
        p = dt.allocate(k)
        if p is not None:
            allocated[k] = p

    assert dt.n_allocated <= capacity


def test_double_allocate_raises() -> None:
    dt = DereferenceTable(n=64, delta=0.3, seed=1)
    p = dt.allocate(7)
    assert p is not None
    raised = False
    try:
        dt.allocate(7)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError on double allocate"


def test_free_unallocated_raises() -> None:
    dt = DereferenceTable(n=64, delta=0.3, seed=2)
    p = dt.allocate(1)
    assert p is not None
    fake = TinyPointer.encode(0, 0, 0, 4, 2)
    raised = False
    try:
        dt.free(999, fake)
    except KeyError:
        raised = True
    assert raised, "Expected KeyError on free of unallocated key"


if __name__ == "__main__":
    tests = [
        test_dereference_uniqueness_small,
        test_dereference_uniqueness_large,
        test_dereference_is_stable,
        test_slot_reuse_after_free,
        test_free_and_reuse_cycle,
        test_capacity_respected,
        test_double_allocate_raises,
        test_free_unallocated_raises,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  \u2713  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  \u2717  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")