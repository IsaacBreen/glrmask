from typing import Dict
import _sep1 as ffi
from ..common_interface import RangeSet

class Model:
    """
    A quasi-model that acts as a thin wrapper to call the Rust-native
    `GrammarConstraintState.get_mask()` method. This serves as a baseline
    to measure the overhead of the Python-based scheduler and data structures
    in other models.
    """
    IS_RUST_WRAPPER = True

    # The arena and roots_map are not used by this model.
    arena: Dict = {}
    roots_map: Dict = {}

    @staticmethod
    def from_json_string(s: str) -> "Model":
        # This model doesn't use any precomputed data from the JSON.
        # It relies on the GrammarConstraint object directly.
        return Model()

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> RangeSet:
        # This method should not be called. The benchmark runner has a special
        # case for models with IS_RUST_WRAPPER=True.
        raise NotImplementedError(
            "rust_model.get_mask should not be called directly. "
            "The benchmark runner handles it."
        )

    def is_end(self, node: int) -> bool:
        # Dummy implementation, not used.
        return False
