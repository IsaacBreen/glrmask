from __future__ import annotations
from typing import Iterable, Optional, Tuple, List

from .range_set_abc import RangeSet
from .py_range_set import PyRangeSet

BIT_COUNTS = tuple(bin(i).count('1') for i in range(256))

class BitsetRangeSet(RangeSet[int]):
    """
    Represents a set of non-negative integers using a bitset.
    Implements the generic RangeSet[int] interface.
    This implementation is efficient for dense sets within a small range of integers.
    """
    __slots__ = ('_bitset', '_len')

    def __init__(self, intervals: Optional[Iterable[Tuple[int, int]]] = None):
        if not intervals:
            self._bitset = bytearray()
            self._len = 0
            return

        # Using PyRangeSet's logic to handle unsorted/overlapping intervals
        normalized_intervals = PyRangeSet._normalize(intervals)

        if not normalized_intervals:
            self._bitset = bytearray()
            self._len = 0
            return

        max_val = normalized_intervals[-1][1]
        if max_val < 0:
            raise ValueError("BitsetRangeSet only supports non-negative integers.")

        size = (max_val // 8) + 1
        self._bitset = bytearray(size)

        for start, end in normalized_intervals:
            if start < 0:
                raise ValueError("BitsetRangeSet only supports non-negative integers.")
            
            start_byte = start // 8
            start_bit = start % 8
            end_byte = end // 8
            end_bit = end % 8

            if start_byte == end_byte:
                mask = (0xff >> (8 - (end_bit - start_bit + 1))) << start_bit
                self._bitset[start_byte] |= mask
            else:
                # First byte
                self._bitset[start_byte] |= (0xff << start_bit) & 0xff
                # Middle bytes
                for i in range(start_byte + 1, end_byte):
                    self._bitset[i] = 0xff
                # Last byte
                self._bitset[end_byte] |= (0xff >> (7 - end_bit))
        
        self._len = sum(BIT_COUNTS[byte] for byte in self._bitset)

    @property
    def intervals(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(self.iter_ranges())

    def iter_ranges(self) -> Iterable[Tuple[int, int]]:
        """Iterates over all [start, end] intervals in the set."""
        if self.is_empty():
            return

        in_range = False
        start_range = 0
        
        # We need to check one past the last possible element to close the last range
        for i in range(len(self._bitset) * 8 + 1):
            is_set = self.contains(i) # contains handles out of bounds
            if is_set and not in_range:
                start_range = i
                in_range = True
            elif not is_set and in_range:
                yield (start_range, i - 1)
                in_range = False

    def to_ranges(self) -> List[Tuple[int]]:
        return list(self.intervals)

    def to_indices(self) -> List[int]:
        indices = []
        for i in range(len(self._bitset) * 8):
            byte_idx = i // 8
            bit_idx = i % 8
            if (self._bitset[byte_idx] >> bit_idx) & 1:
                indices.append(i)
        return indices

    def iter_indices(self) -> Iterable[int]:
        """Iterates over all individual indices in the set."""
        for byte_idx, byte in enumerate(self._bitset):
            if byte == 0:
                continue
            for bit_idx in range(8):
                if (byte >> bit_idx) & 1:
                    yield (byte_idx * 8) + bit_idx

    def contains(self, x: int) -> bool:
        if x < 0:
            return False
        byte_idx = x // 8
        if byte_idx >= len(self._bitset):
            return False
        bit_idx = x % 8
        return (self._bitset[byte_idx] >> bit_idx) & 1 != 0

    def union(self, other: RangeSet[int]) -> "BitsetRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented
        
        if isinstance(other, BitsetRangeSet):
            len_a = len(self._bitset)
            len_b = len(other._bitset)
            new_len = max(len_a, len_b)
            new_bitset = bytearray(new_len)
            
            new_bitset[:len_a] = self._bitset
            
            for i in range(len_b):
                new_bitset[i] |= other._bitset[i]
                
            res = BitsetRangeSet()
            res._bitset = new_bitset
            res._len = sum(BIT_COUNTS[byte] for byte in new_bitset)
            return res
        else:
            # Generic path for other RangeSet types
            return BitsetRangeSet(self.intervals + other.intervals)

    def intersection(self, other: RangeSet[int]) -> "BitsetRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented
        
        if isinstance(other, BitsetRangeSet):
            len_a = len(self._bitset)
            len_b = len(other._bitset)
            new_len = min(len_a, len_b)
            new_bitset = bytearray(new_len)
            
            for i in range(new_len):
                new_bitset[i] = self._bitset[i] & other._bitset[i]
                
            res = BitsetRangeSet()
            res._bitset = new_bitset
            res._len = sum(BIT_COUNTS[byte] for byte in new_bitset)
            return res
        else:
            # Generic path
            py_self = PyRangeSet(self.intervals)
            intersected = py_self.intersection(other)
            return BitsetRangeSet(intersected.intervals)

    def difference(self, other: RangeSet[int]) -> "BitsetRangeSet":
        if not isinstance(other, RangeSet):
            return NotImplemented
        
        if isinstance(other, BitsetRangeSet):
            len_a = len(self._bitset)
            len_b = len(other._bitset)
            new_bitset = self._bitset[:]
            
            limit = min(len_a, len_b)
            for i in range(limit):
                new_bitset[i] &= ~other._bitset[i]
                
            res = BitsetRangeSet()
            res._bitset = new_bitset
            res._len = sum(BIT_COUNTS[byte] for byte in new_bitset)
            return res
        else:
            py_self = PyRangeSet(self.intervals)
            differenced = py_self.difference(other)
            return BitsetRangeSet(differenced.intervals)

    def issuperset(self, other: RangeSet[int]) -> bool:
        """Return True if self is a superset of other."""
        if not isinstance(other, RangeSet):
            return NotImplemented
        
        if isinstance(other, BitsetRangeSet):
            len_a = len(self._bitset)
            len_b = len(other._bitset)
            if len_b > len_a:
                # Check if the extra part of other's bitset is all zeros
                for i in range(len_a, len_b):
                    if other._bitset[i] != 0:
                        return False
            
            limit = min(len_a, len_b)
            for i in range(limit):
                if (self._bitset[i] & other._bitset[i]) != other._bitset[i]:
                    return False
            return True
        else:
            # Generic path
            return other.issubset(self)

    def issubset(self, other: RangeSet[int]) -> bool:
        """Return True if self is a subset of other."""
        if not isinstance(other, RangeSet):
            return NotImplemented
        
        if isinstance(other, BitsetRangeSet):
            len_a = len(self._bitset)
            len_b = len(other._bitset)
            if len_a > len_b:
                # Check if the extra part of self's bitset is all zeros
                for i in range(len_b, len_a):
                    if self._bitset[i] != 0:
                        return False
            
            limit = min(len_a, len_b)
            for i in range(limit):
                # Check if all bits set in self are also set in other
                if (self._bitset[i] & other._bitset[i]) != self._bitset[i]:
                    return False
            return True
        else:
            return other.issuperset(self)

    def isdisjoint(self, other: RangeSet[int]) -> bool:
        """Return True if self has no elements in common with other."""
        if not isinstance(other, RangeSet):
            return NotImplemented
        
        if isinstance(other, BitsetRangeSet):
            limit = min(len(self._bitset), len(other._bitset))
            for i in range(limit):
                if (self._bitset[i] & other._bitset[i]) != 0:
                    return False
            return True
        else:
            # Generic path
            return self.intersection(other).is_empty()

    def union_update(self, other: RangeSet[int]) -> None:
        """Update self with the union of self and other."""
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")

        if isinstance(other, BitsetRangeSet):
            len_a = len(self._bitset)
            len_b = len(other._bitset)
            if len_b > len_a:
                self._bitset.extend(bytearray(len_b - len_a))

            for i in range(len_b):
                self._bitset[i] |= other._bitset[i]

            self._len = sum(BIT_COUNTS[byte] for byte in self._bitset)
        else:
            # Generic path
            new_set = self.union(other)
            self._bitset = new_set._bitset
            self._len = new_set._len

    def intersection_update(self, other: RangeSet[int]) -> None:
        """Update self with the intersection of self and other."""
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")

        if isinstance(other, BitsetRangeSet):
            len_a = len(self._bitset)
            len_b = len(other._bitset)
            new_len = min(len_a, len_b)

            if len_a > new_len:
                self._bitset = self._bitset[:new_len]

            for i in range(new_len):
                self._bitset[i] &= other._bitset[i]

            self._len = sum(BIT_COUNTS[byte] for byte in self._bitset)
        else:
            new_set = self.intersection(other)
            self._bitset = new_set._bitset
            self._len = new_set._len

    def difference_update(self, other: RangeSet[int]) -> None:
        """Update self with the set difference self \\ other."""
        if not isinstance(other, RangeSet):
            raise TypeError("other must be a RangeSet")

        if isinstance(other, BitsetRangeSet):
            limit = min(len(self._bitset), len(other._bitset))
            for i in range(limit):
                self._bitset[i] &= ~other._bitset[i]
            self._len = sum(BIT_COUNTS[byte] for byte in self._bitset)
        else:
            new_set = self.difference(other)
            self._bitset = new_set._bitset
            self._len = new_set._len

    def is_empty(self) -> bool:
        return self._len == 0

    def __len__(self) -> int:
        return self._len

    def __repr__(self) -> str:
        return f"BitsetRangeSet({self.intervals!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, BitsetRangeSet):
            a, b = self._bitset, other._bitset
            if len(a) > len(b):
                a, b = b, a
            
            if a != b[:len(a)]:
                return False
            
            for i in range(len(a), len(b)):
                if b[i] != 0:
                    return False
            return True

        if isinstance(other, RangeSet):
            return self.intervals == other.intervals
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.intervals)

    @classmethod
    def from_ranges(cls, ranges: List[List[int]]) -> 'BitsetRangeSet':
        return cls(iter(map(tuple, ranges)))

    @classmethod
    def from_indices(cls, indices: Iterable[int]) -> 'BitsetRangeSet':
        indices = list(indices)
        if not indices:
            return cls()
        
        if any(i < 0 for i in indices):
            raise ValueError("BitsetRangeSet only supports non-negative integers.")
        
        max_val = max(indices)
        
        size = (max_val // 8) + 1
        bitset = bytearray(size)
        
        unique_indices = set(indices)
        for i in unique_indices:
            byte_idx = i // 8
            bit_idx = i % 8
            bitset[byte_idx] |= (1 << bit_idx)
            
        res = cls()
        res._bitset = bitset
        res._len = len(unique_indices)
        return res

    @classmethod
    def empty(cls) -> 'BitsetRangeSet':
        return cls()

    @classmethod
    def from_json(cls, data: List[List[int]]) -> 'BitsetRangeSet':
        return cls.from_ranges(data)

    def to_json(self) -> List[List[int]]:
        return [list(r) for r in self.intervals]
