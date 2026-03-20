from __future__ import annotations
ALLOC_FAILED: "TinyPointer | None" = None


class TinyPointer(int):

    # We use a factory so we can keep the int subclass simple
    @staticmethod
    def encode(table_id: int, bucket_index: int, slot_index: int,
               bucket_bits: int, slot_bits: int) -> "TinyPointer":
        value = (table_id << (bucket_bits + slot_bits)) | (bucket_index << slot_bits) | slot_index
        p = TinyPointer(value)
        # stash metadata as attributes (CPython allows this on int subclasses via __dict__)
        p._table_id = table_id
        p._bucket_index = bucket_index
        p._slot_index = slot_index
        p._bucket_bits = bucket_bits
        p._slot_bits = slot_bits
        return p

    @property
    def table_id(self) -> int:
        return self._table_id

    @property
    def bucket_index(self) -> int:
        return self._bucket_index

    @property
    def slot_index(self) -> int:
        return self._slot_index

    def bit_length_encoding(self) -> int:
        """
        Returns the number of bits actually stored in the pointer.

        The KEY INSIGHT from the paper: the bucket index is NOT stored in
        the pointer — it is recomputed from the key via h(key).  So the
        primary-table pointer only needs to store:
          - 1 bit for table_id  (primary vs secondary)
          - slot_bits for the intra-bucket slot index

        The secondary-table pointer stores:
          - 1 bit for table_id
          - 1 bit for which hash function (h1 or h2) was chosen  (bucket_bits=1)
          - slot_bits for the intra-bucket slot index

        This is what makes pointers "tiny": the bucket information comes
        for free from the owner's identity (Section 2, paragraph 3).
        """
        if self._table_id == 0:
            # Primary: only the slot index + 1 table-selector bit
            return 1 + self._slot_bits
        else:
            # Secondary: choice bit (1) + slot index + 1 table-selector bit
            return 1 + self._bucket_bits + self._slot_bits

    def __repr__(self) -> str:
        return (f"TinyPointer(table={self._table_id}, "
                f"bucket={self._bucket_index}, slot={self._slot_index}, "
                f"bits={self.bit_length_encoding()})")