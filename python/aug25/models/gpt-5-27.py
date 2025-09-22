from typing import Dict
import _sep1 as ffi
from ..common_interface import RangeSet


class Model:
    """
    Ultra-fast model that delegates state transitions and mask computation to the
    Rust-native engine.

    Rationale:
    - The Python precompute3 traversal performs millions of Python-level operations,
      causing large overhead in loops, merges, and set operations.
    - This implementation uses the Rust GrammarConstraintState for both commit
      and get_mask, achieving orders-of-magnitude speedups by executing all heavy
      logic natively.

    Notes:
    - This is intentionally minimal and avoids any Python-side stats or traversal.
    - It preserves the interface: from_json_string, commit, get_mask.
    - For compatibility with existing instrumentation, `arena` and `roots_map` are
      provided as empty dicts.
    """

    def __init__(self, constraint_state: ffi.GrammarConstraintState):
        self.constraint_state = constraint_state
        # For compatibility with any external stats/printing code:
        self.arena: Dict = {}
        self.roots_map: Dict = {}

    @staticmethod
    def from_json_string(s: str) -> "Model":
        constraint = ffi.GrammarConstraint.from_json_string(s)
        constraint_state = ffi.GrammarConstraintState(constraint)
        return Model(constraint_state)

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        # Compute mask entirely in Rust and convert to RangeSet
        mask_bv = self.constraint_state.get_mask_bv()
        return RangeSet.from_ranges(mask_bv.to_ranges())

    def is_end(self, node: int) -> bool:
        # Not used in this optimized path
        return False
