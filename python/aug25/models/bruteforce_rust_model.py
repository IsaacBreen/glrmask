from typing import Dict, List
import json
import _sep1 as ffi
from ..common_interface import RangeSet
from tqdm import tqdm

class BruteForceRustModel:
    """
    A model that wraps the Rust-native `GrammarConstraintState` but uses a
    brute-force get_mask implementation. It checks each token individually by
    replaying the commit history.
    """
    def __init__(self, constraint: ffi.GrammarConstraint, constraint_state: ffi.GrammarConstraintState, id_to_token: Dict[int, bytes]):
        self.constraint = constraint
        self.constraint_state = constraint_state
        self.id_to_token = id_to_token
        self.committed_tokens: List[int] = []
        # For compatibility with statistics printer
        self.arena: Dict = {}
        self.roots_map: Dict = {}

    @staticmethod
    def from_json_string(s: str) -> "BruteForceRustModel":
        constraint = ffi.GrammarConstraint.from_json_string(s)
        constraint_state = ffi.GrammarConstraintState(constraint)
        
        data = json.loads(s)
        id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}

        return BruteForceRustModel(constraint, constraint_state, id_to_token)

    def get_mask(self) -> RangeSet:
        """
        Determines the allowed token mask by iterating through all possible tokens
        and checking if committing them leads to a valid state.
        """
        allowed_tokens = []
        
        for token_id in tqdm(self.id_to_token.keys(), desc="get_mask (bruteforce_rust)"):
            # Create a temporary state by replaying history
            temp_state = ffi.GrammarConstraintState(self.constraint)
            for committed_token in self.committed_tokens:
                temp_state.commit(committed_token)
            
            # Check if the next token is valid
            temp_state.commit(token_id)
            mask_bv = temp_state.get_mask_bv()
            if mask_bv.to_ranges(): # Non-empty mask means the token is valid
                allowed_tokens.append(token_id)

        return RangeSet.from_indices(allowed_tokens)

    def commit(self, token_id: int):
        """Commits a token to the underlying Rust state and records it."""
        self.constraint_state.commit(token_id)
        self.committed_tokens.append(token_id)

    def is_end(self, node: int) -> bool:
        # Dummy implementation, not used.
        return False

Model = BruteForceRustModel
