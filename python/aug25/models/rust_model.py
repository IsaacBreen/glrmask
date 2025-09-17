from typing import Dict, Optional
import _sep1 as ffi
from ..common_interface import RangeSet

class Model:
    """
    A model that wraps the Rust-native `GrammarConstraintState` to serve as a baseline.
    """
    def __init__(self, constraint_state: ffi.GrammarConstraintState):
        self.constraint_state = constraint_state
        # For compatibility with statistics printer
        self.arena: Dict = {}
        self.roots_map: Dict = {}

    @staticmethod
    def from_json_string(s: str) -> "Model":
        constraint = ffi.GrammarConstraint.from_json_string(s)
        constraint_state = ffi.GrammarConstraintState(constraint)
        return Model(constraint_state)

    def get_mask(self) -> RangeSet:
        """Calls the underlying Rust implementation for get_mask."""
        print("\n--- get_mask START (rust_model.py) ---")

        mask_bv = self.constraint_state.get_mask_bv()

        print("\n--- get_mask END (rust_model.py) ---")
        return RangeSet.from_ranges(mask_bv.to_ranges())

    def commit(self, token_id: int):
        """Commits a token to the underlying Rust state."""
        self.constraint_state.commit(token_id)

    def is_end(self, node: int) -> bool:
        # Dummy implementation, not used.
        return False
