import json
from typing import Dict, List, Tuple, Optional

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module

class Model(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        print("deepseek-1.py Model is a placeholder and not implemented.")
        self.roots_map = {}
        self.arena = {}
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None

    @staticmethod
    def from_json_string(s: str) -> "Model":
        data = json.loads(s)
        roots_map = data.get('precomputed3', [])
        arena_json = data.get('trie3_god', {})
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map.get(state_id, -1)

    def is_end(self, node: int) -> bool:
        return False

    def iter_edges(self, node: int, token: int):
        yield from ()

    def commit(self, token_id: int):
        if self.constraint_state:
            self.constraint_state.commit(token_id)


    def get_mask(self) -> RangeSet:
        print("\n--- get_mask START (deepseek-1.py) ---")
        print(self.constraint_state)
        state_to_gss = self.constraint_state.filtered_state_gss_map()
        print(f"Filtered state_to_gss: { {k: v.ptr() for k, v in state_to_gss.items()} }")
        print("deepseek-1.py get_mask is not implemented.")
        # Return an empty mask as a fallback
        return RangeSet.empty()

