from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TinyPointer:

    table_id:     int
    bucket_index: int
    slot_index:   int
    bucket_bits:  int
    slot_bits:    int

    @staticmethod
    def encode(
        table_id: int,
        bucket_index: int,
        slot_index: int,
        bucket_bits: int,
        slot_bits: int,
    ) -> "TinyPointer":
        return TinyPointer(
            table_id=table_id,
            bucket_index=bucket_index,
            slot_index=slot_index,
            bucket_bits=bucket_bits,
            slot_bits=slot_bits,
        )

    def bit_length_encoding(self) -> int:
        if self.table_id == 0:
            return 1 + self.slot_bits
        else:
            return 1 + self.bucket_bits + self.slot_bits

    def __repr__(self) -> str:
        return (
            f"TinyPointer(table={self.table_id}, "
            f"bucket={self.bucket_index}, slot={self.slot_index}, "
            f"bits={self.bit_length_encoding()})"
        )