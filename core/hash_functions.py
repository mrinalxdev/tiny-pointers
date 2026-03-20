import random
import sys

_WORD = 64
_MOD = 1 << _WORD


class HashFunction:
    __slots__ = ("_a", "_b", "_m", "_shift")

    def __init__(self, m: int, seed: int | None = None):
        if m <= 0:
            raise ValueError("m must be positive")
        rng = random.Random(seed)
        self._a = rng.randrange(1, _MOD) | 1   # must be odd
        self._b = rng.randrange(0, _MOD)
        self._m = m

    def __call__(self, key: int) -> int:
        if not isinstance(key, int):
            key = hash(key)
        key = key % _MOD
        h = (self._a * key + self._b) % _MOD
        return h % self._m


class HashFamily:
    def __init__(self, m: int, master_seed: int | None = None) -> None:
        self._m: int = m
        self._master_seed: int = master_seed if master_seed is not None else random.randrange(0, _MOD)
        self._cache: dict[int, HashFunction] = {}

    def get(self, index: int) -> HashFunction:
        if index not in self._cache:
            seed = (self._master_seed ^ (index * 0x9E3779B97F4A7C15)) % _MOD
            self._cache[index] = HashFunction(self._m, seed=seed)
        return self._cache[index]