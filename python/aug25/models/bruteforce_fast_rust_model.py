from typing import Dict, List
import json
import _sep1 as ffi
from ..common_interface import RangeSet

class BruteForceFastRustModel:
    """
    A model that wraps the Rust-native `GrammarConstraintState` but uses a
    brute-force get_mask implementation. It checks each *internal* token individually
    by cloning the current state for efficiency.
    """
    def __init__(self,
                 constraint: ffi.GrammarConstraint,
                 constraint_state: ffi.GrammarConstraintState,
                 internal_to_original_map: Dict[int, RangeSet],
                 internal_to_representative_map: Dict[int, int],
                 internal_max_llm_token: int):
        self.constraint = constraint
        self.constraint_state = constraint_state
        self.internal_to_original_map = internal_to_original_map
        self.internal_to_representative_map = internal_to_representative_map
        self.internal_max_llm_token = internal_max_llm_token
        # For compatibility with statistics printer
        self.arena: Dict = {}
        self.roots_map: Dict = {}

    @staticmethod
    def from_json_string(s: str) -> "BruteForceFastRustModel":
        constraint = ffi.GrammarConstraint.from_json_string(s)
        constraint_state = ffi.GrammarConstraintState(constraint)
        
        data = json.loads(s)
        vocab = data['vocab']

        internal_to_original_map = {
            int(k): RangeSet.from_indices(v) for k, v in dict(vocab['internal_to_original']).items()
        }

        internal_to_representative_map = {}
        for original, internal in vocab['original_to_internal']:
            if internal not in internal_to_representative_map:
                internal_to_representative_map[internal] = original
        
        internal_max_llm_token = vocab['internal_max_llm_token']

        return BruteForceFastRustModel(
            constraint,
            constraint_state,
            internal_to_original_map,
            internal_to_representative_map,
            internal_max_llm_token
        )

    def get_mask(self) -> RangeSet:
        """
        Determines the allowed token mask by iterating through all internal tokens
        and checking if committing a representative original token leads to a valid state.
        """
        allowed_mask = RangeSet.empty()
        
        for internal_token_id in range(self.internal_max_llm_token + 1):
            representative_token_id = self.internal_to_representative_map.get(internal_token_id)
            if representative_token_id is None:
                continue

            # Create a temporary state by cloning the current state
            temp_state = self.constraint_state.clone()
            # Check if the next token is valid
            temp_state.commit(representative_token_id)
            
            if temp_state.is_valid():
                original_tokens = self.internal_to_original_map.get(internal_token_id)
                if original_tokens:
                    allowed_mask = allowed_mask.union(original_tokens)

        return allowed_mask

    def commit(self, token_id: int):
        """Commits a token to the underlying Rust state and records it."""
        self.constraint_state.commit(token_id)

    def is_end(self, node: int) -> bool:
        # Dummy implementation, not used.
        return False

Model = BruteForceFastRustModel
