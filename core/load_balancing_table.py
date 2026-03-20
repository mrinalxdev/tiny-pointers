from __future__ import annotations
import math
from .hash_functions import HashFunction


_FULL = -1 


def _lowest_free_slot(bitmap: int, b: int) -> int:
    full_mask = (1 << b) - 1
    free_bits = (~bitmap) & full_mask
    if free_bits == 0:
        return _FULL

    return (free_bits & -free_bits).bit_length() - 1


class LoadBalancingTable:
    __slots__ = ("_n_slots", "_b", "_n_buckets", "_hash_fn",
                 "_bitmaps", "_bucket_bits", "_slot_bits")

    def __init__(self, n_slots: int, b: int, hash_fn: HashFunction) -> None:
        if n_slots % b != 0:
            raise ValueError(f"n_slots ({n_slots}) must be divisible by b ({b})")
        self._n_slots = n_slots
        self._b = b
        self._n_buckets = n_slots // b
        self._hash_fn = hash_fn

        self._bitmaps: list[int] = [0] * self._n_buckets
        self._bucket_bits = max(1, math.ceil(math.log2(self._n_buckets + 1)))
        self._slot_bits   = max(1, math.ceil(math.log2(b + 1)))

    def allocate(self, key: int) -> int | None:
        bucket = self._hash_fn(key)
        slot = _lowest_free_slot(self._bitmaps[bucket], self._b)
        if slot == _FULL:
            return None  
        self._bitmaps[bucket] |= (1 << slot)
        return slot

    def free(self, key: int, slot: int) -> None:
        bucket = self._hash_fn(key)
        self._bitmaps[bucket] &= ~(1 << slot)

    def dereference(self, key: int, slot: int) -> int:
        bucket = self._hash_fn(key)
        return bucket * self._b + slot


    @property
    def n_slots(self) -> int:
        return self._n_slots

    @property
    def bucket_bits(self) -> int:
        return self._bucket_bits

    @property
    def slot_bits(self) -> int:
        return self._slot_bits

    @property
    def load(self) -> int:
        return sum(bin(bm).count("1") for bm in self._bitmaps)

    @property
    def load_factor(self) -> float:
        return self.load / self._n_slots if self._n_slots > 0 else 0.0