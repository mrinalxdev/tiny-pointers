from __future__ import annotations
import math

from .hash_functions import HashFamily
from .load_balancing_table import LoadBalancingTable
from .power_of_two_table import PowerOfTwoTable
from .tiny_pointer import TinyPointer


def _bucket_size_primary(delta: float) -> int:
    inv_delta = 1.0 / delta
    b = max(2, math.ceil(inv_delta ** 2 * math.log2(inv_delta)))
    return b


def _bucket_size_secondary(n_secondary: int) -> int:
    if n_secondary < 4:
        return 2
    ll = math.log2(math.log2(max(n_secondary, 4)))
    return max(2, math.ceil(ll))


def _round_up(n: int, divisor: int) -> int:
    return ((n + divisor - 1) // divisor) * divisor

class DereferenceTable:
    def __init__(self, n: int, delta: float = 0.1, seed: int | None = None) -> None:
        if n < 8:
            raise ValueError("n must be at least 8")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")

        self._n = n
        self._delta = delta
        self._capacity = int((1 - delta) * n)
        n_sec_raw = max(4, int(delta / 2 * n))
        n_pri_raw = n - n_sec_raw
        b_pri = _bucket_size_primary(delta)
        n_pri = _round_up(n_pri_raw, b_pri)
        if n_pri + 4 > n:
            n_pri = _round_up(max(b_pri, n // 2), b_pri)
        n_sec = n - n_pri
        if n_sec < 2:
            n_sec = 2

        b_sec = _bucket_size_secondary(n_sec)
        n_sec = _round_up(n_sec, b_sec)

        n_pri = n - n_sec
        n_pri = _round_up(n_pri, b_pri)
        self._n_total = n_pri + n_sec
        self._n_pri = n_pri
        self._n_sec = n_sec

        family = HashFamily(n, master_seed=seed)
        hf_pri  = HashFamily(n_pri // b_pri, master_seed=seed)
        hf_sec1 = HashFamily(max(1, n_sec // b_sec), master_seed=(seed or 0) ^ 0xDEADBEEF)
        hf_sec2 = HashFamily(max(1, n_sec // b_sec), master_seed=(seed or 0) ^ 0xCAFEBABE)

        self._primary   = LoadBalancingTable(n_pri, b_pri,   hf_pri.get(0))
        self._secondary = PowerOfTwoTable   (n_sec, b_sec,   hf_sec1.get(0), hf_sec2.get(0))

        self._b_pri = b_pri
        self._b_sec = b_sec

        self._pri_bucket_bits = max(1, math.ceil(math.log2(n_pri // b_pri + 1)))
        self._pri_slot_bits   = max(1, math.ceil(math.log2(b_pri + 1)))
        self._sec_slot_bits   = max(1, math.ceil(math.log2(b_sec + 1)))

        self._active: dict[int, TinyPointer] = {}

    def allocate(self, key: int) -> TinyPointer | None:
        if key in self._active:
            raise ValueError(f"Key {key!r} is already allocated. Free it first.")
        if len(self._active) >= self._capacity:
            return None   # table at capacity

        slot_pri = self._primary.allocate(key)
        if slot_pri is not None:
            p = TinyPointer.encode(
                table_id=0,
                bucket_index=self._primary._hash_fn(key),
                slot_index=slot_pri,
                bucket_bits=self._pri_bucket_bits,
                slot_bits=self._pri_slot_bits,
            )
            self._active[key] = p
            return p

        result = self._secondary.allocate(key)
        if result is None:
            return None

        choice, slot_sec = result
        p = TinyPointer.encode(
            table_id=1,
            bucket_index=choice,
            slot_index=slot_sec,
            bucket_bits=1,
            slot_bits=self._sec_slot_bits,
        )
        self._active[key] = p
        return p

    def dereference(self, key: int, p: TinyPointer) -> int:
        if p.table_id == 0:
            idx_in_primary = self._primary.dereference(key, p.slot_index)
            return idx_in_primary
        else:
            idx_in_secondary = self._secondary.dereference(key, p.bucket_index, p.slot_index)
            return self._n_pri + idx_in_secondary

    def free(self, key: int, p: TinyPointer) -> None:
        if key not in self._active:
            raise KeyError(f"Key {key!r} is not currently allocated.")
        if p.table_id == 0:
            self._primary.free(key, p.slot_index)
        else:
            self._secondary.free(key, p.bucket_index, p.slot_index)
        del self._active[key]

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def n_allocated(self) -> int:
        return len(self._active)

    @property
    def load_factor(self) -> float:
        return self.n_allocated / self._n_total if self._n_total > 0 else 0.0

    def pointer_bit_sizes(self) -> list[int]:
        return [p.bit_length_encoding() for p in self._active.values()]

    def stats(self) -> dict:
        bits = self.pointer_bit_sizes()
        avg_bits = sum(bits) / len(bits) if bits else 0.0
        max_bits = max(bits) if bits else 0
        theoretical_max = 1 + self._pri_bucket_bits + self._pri_slot_bits
        return {
            "n": self._n_total,
            "delta": self._delta,
            "capacity": self._capacity,
            "n_allocated": self.n_allocated,
            "load_factor": round(self.load_factor, 4),
            "primary_slots": self._n_pri,
            "secondary_slots": self._n_sec,
            "bucket_size_primary": self._b_pri,
            "bucket_size_secondary": self._b_sec,
            "avg_pointer_bits": round(avg_bits, 2),
            "max_pointer_bits": max_bits,
            "theoretical_max_bits": theoretical_max,
            "primary_load_factor": round(self._primary.load_factor, 4),
            "secondary_load_factor": round(self._secondary.load_factor, 4),
        }
