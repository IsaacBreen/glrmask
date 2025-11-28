"""
A model that wraps the Rust-native `GrammarConstraintState` using the optimized fill method.
This preallocates a numpy array and fills it directly rather than allocating a new Bitset each time.
"""
from typing import Dict, Optional, List, Tuple, Iterable
import numpy as np
import _sep1 as ffi
from ..range_set.ffi_bitset import FFIBitset as Bitset


class I32ArrayMask:
    """
    A lightweight wrapper around an i32 numpy array to provide the to_ranges() interface
    expected by the benchmark runner.
    """
    __slots__ = ('_buffer',)
    
    def __init__(self, buffer: np.ndarray):
        self._buffer = buffer
    
    def to_ranges(self) -> List[Tuple[int, int]]:
        """Convert the bitmask to a list of (start, end) ranges."""
        ranges = []
        indices = []
        
        # Extract set bits from the i32 array
        for word_idx, word in enumerate(self._buffer):
            if word == 0:
                continue
            base = word_idx * 32
            # Check each bit
            w = int(word) & 0xFFFFFFFF  # Ensure unsigned
            while w:
                bit_pos = (w & -w).bit_length() - 1  # Find lowest set bit
                indices.append(base + bit_pos)
                w &= w - 1  # Clear lowest set bit
        
        if not indices:
            return []
        
        # Convert sorted indices to ranges
        indices.sort()
        start = indices[0]
        end = start
        for i in range(1, len(indices)):
            if indices[i] == end + 1:
                end = indices[i]
            else:
                ranges.append((start, end))
                start = indices[i]
                end = start
        ranges.append((start, end))
        
        return ranges
    
    def iter_indices(self) -> Iterable[int]:
        """Iterate over all set bit indices."""
        for word_idx, word in enumerate(self._buffer):
            if word == 0:
                continue
            base = word_idx * 32
            w = int(word) & 0xFFFFFFFF
            while w:
                bit_pos = (w & -w).bit_length() - 1
                yield base + bit_pos
                w &= w - 1


class Model:
    """
    A model that wraps the Rust-native `GrammarConstraintState` using the fill method.
    
    This is similar to rust_model.py but uses fill_next_token_bitmask to fill a 
    preallocated numpy array rather than get_mask_bv which allocates a new Bitset.
    """
    def __init__(self, constraint: ffi.GrammarConstraint, constraint_state: ffi.GrammarConstraintState):
        self.constraint = constraint
        self.constraint_state = constraint_state
        # Preallocate the buffer once
        self.buffer_size = constraint_state.mask_buffer_size_i32()
        self.buffer = np.zeros(self.buffer_size, dtype=np.int32)
        # For compatibility with statistics printer
        self.arena: Dict = {}
        self.roots_map: Dict = {}

    @staticmethod
    def from_json_string(s: str) -> "Model":
        constraint = ffi.GrammarConstraint.from_json_string(s)
        constraint_state = ffi.GrammarConstraintState(constraint)
        return Model(constraint, constraint_state)

    def get_mask(self) -> I32ArrayMask:
        """
        Calls the underlying Rust implementation using fill_next_token_bitmask.
        
        This fills the preallocated numpy array rather than allocating a new Bitset.
        Returns a lightweight wrapper that provides the to_ranges() interface.
        """
        # Clear and fill the preallocated buffer
        self.buffer.fill(0)
        self.constraint_state.fill_next_token_bitmask(self.buffer)
        
        # Return a lightweight wrapper that provides to_ranges()
        return I32ArrayMask(self.buffer.copy())  # Copy so caller can't modify our buffer

    def commit(self, token_id: int):
        """Commits a token to the underlying Rust state."""
        self.constraint_state.commit(token_id)

    def reset(self):
        """Resets the model state to its initial condition."""
        self.constraint_state = ffi.GrammarConstraintState(self.constraint)
        # Buffer can be reused

    def is_end(self, node: int) -> bool:
        # Dummy implementation, not used.
        return False
