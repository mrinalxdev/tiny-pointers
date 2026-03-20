from __future__ import annotations
import math
from .hash_functions import HashFunction


_FULL = -1


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _lowest_free_slot(bitmap: int, b: int) -> int:
    full_mask = (1 << b) - 1
    free_bits = (~bitmap) & full_mask
    if free_bits == 0:
        return _FULL
    return (free_bits & -free_bits).bit_length() - 1


class PowerOfTwoTable:
    __slots__ = ("_n_slots", "_b", "_n_buckets",
                 "_h1", "_h2", "_bitmaps", "_slot_bits")

    def __init__(self, n_slots: int, b: int,
                 hash_fn_1: HashFunction, hash_fn_2: HashFunction) -> None:
        if n_slots % b != 0:
            raise ValueError(f"n_slots ({n_slots}) must be divisible by b ({b})")
        self._n_slots   = n_slots
        self._b         = b
        self._n_buckets = n_slots // b
        self._h1        = hash_fn_1
        self._h2        = hash_fn_2
        self._bitmaps: list[int] = [0] * self._n_buckets
        self._slot_bits = max(1, math.ceil(math.log2(b + 1)))

    def allocate(self, key: int) -> tuple[int, int] | None:
        b1 = self._h1(key)
        b2 = self._h2(key)

        free1 = self._b - _popcount(self._bitmaps[b1])
        free2 = self._b - _popcount(self._bitmaps[b2])

        if free1 == 0 and free2 == 0:
            return None   # both full

        if free1 >= free2:
            chosen_bucket = b1
            choice = 0
        else:
            chosen_bucket = b2
            choice = 1

        slot = _lowest_free_slot(self._bitmaps[chosen_bucket], self._b)
        self._bitmaps[chosen_bucket] |= (1 << slot)
        return (choice, slot)

    def free(self, key: int, choice: int, slot: int) -> None:
        bucket = self._h1(key) if choice == 0 else self._h2(key)
        self._bitmaps[bucket] &= ~(1 << slot)

    def dereference(self, key: int, choice: int, slot: int) -> int:
        bucket = self._h1(key) if choice == 0 else self._h2(key)
        return bucket * self._b + slot

    @property
    def n_slots(self) -> int:
        return self._n_slots

    @property
    def slot_bits(self) -> int:
        return self._slot_bits

    @property
    def load(self) -> int:
        return sum(_popcount(bm) for bm in self._bitmaps)

    @property
    def load_factor(self) -> float:
        return self.load / self._n_slots if self._n_slots > 0 else 0.0