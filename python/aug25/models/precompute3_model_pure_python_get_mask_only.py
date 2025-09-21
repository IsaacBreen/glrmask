import heapq
from typing import Dict, List, Tuple, Optional

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS

# Import the original model to wrap it, and PyAcc which is used in get_mask's logic
from .precompute3_model_pure_python import Model as InnerModel
from .precompute3_model_pure_python import PyAcc


# Add a dummy profiler for when not running under kernprof
try:
    # This will be injected by the kernprof script.
    profile
except NameError:
    # If not running under kernprof, create a dummy decorator.
    def profile(func): return func


class Model(GraphProvider):
    """
    A wrapper model focusing on the get_mask implementation, delegating
    state management and updates to an inner model instance.
    This is designed to isolate get_mask for performance analysis and
    potential optimization, while reusing the complex state logic from
    the main model.
    """

    def __init__(self, inner_model: InnerModel):
        self.inner_model: InnerModel = inner_model

        # Copy necessary fields for get_mask to have direct access
        self.arena: Dict[int, dict] = self.inner_model.arena
        self.roots_map: Dict[int, int] = self.inner_model.roots_map
        self.max_depth: Dict[int, int] = self.inner_model.max_depth
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = self.inner_model.possible_matches_cache
        self.tokenizer_max_state: int = self.inner_model.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = self.inner_model.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = self.inner_model.internal_to_original_map

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        """Creates the model from a JSON string by first creating the inner model."""
        inner_model = InnerModel.from_json_string(s)
        return Model(inner_model)

    def commit(self, token_id: int):
        """Passes through the commit operation to the inner model."""
        self.inner_model.commit(token_id)

    @property
    def state(self) -> Dict[int, GSS]:
        """Provides access to the state from the inner model."""
        return self.inner_model.state

    def is_end(self, node: int) -> bool:
        """
        Checks if a node in the trie is a terminal node.
        This is a local implementation needed by get_mask.
        """
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    @profile
    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.
        This implementation is copied from the original model and adapted to this wrapper class.
        """
        state_map = self.state
        all_ones_mask = self.all_internal_llm_tokens_bitset
        final_mask = ffi.Bitset.zeros()

        values: Dict[int, Tuple[GSS, ffi.Bitset]] = {}
        stopped: set[int] = set()
        todo: Dict[int, set[int]] = {}
        depth_heap: List[int] = []

        heappush = heapq.heappush
        heappop = heapq.heappop
        roots_map = self.roots_map
        max_depth = self.max_depth
        arena = self.arena
        is_end = self.is_end

        # Seed
        for sid, gss in state_map.items():
            new_mask = all_ones_mask
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss, existing_mask = existing
                merged_gss = existing_gss.merge(gss)
                values[root_idx] = (merged_gss, new_mask)
            else:
                values[root_idx] = (gss, new_mask)

            depth = max_depth[root_idx]
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        def get_disallowed_terminals_py(gss: GSS) -> ffi.HybridL2Bitset:
            merged_acc = gss.reduce_acc()
            if merged_acc is None:
                return ffi.HybridL2Bitset.all()
            return merged_acc.terminals_union.complement()

        # Main loop
        while True:
            node_indices: Optional[set[int]] = None
            current_depth = -1
            while depth_heap:
                current_depth = heappop(depth_heap)
                node_indices = todo.pop(current_depth, None)
                if node_indices:
                    break
            if not node_indices:
                break

            for node_idx in node_indices:
                if node_idx in stopped:
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    continue
                gss_node, llm_mask = item

                # End-node handling
                if is_end(node_idx):
                    forbidden_llm_tokens = ffi.Bitset.zeros()
                    disallowed_terminals_l2 = get_disallowed_terminals_py(gss_node)
                    for (start, end), disallowed_bv in disallowed_terminals_l2.range_values():
                        if disallowed_bv.is_empty():
                            continue
                        end = min(end, self.tokenizer_max_state)
                        for tsid in range(start, end + 1):
                            possible_matches_for_state = self.possible_matches_cache.get(tsid)
                            if not possible_matches_for_state:
                                continue
                            for terminal_id_str, llm_tokens_for_terminal in possible_matches_for_state.items():
                                terminal_id = int(terminal_id_str)
                                if disallowed_bv.contains(terminal_id):
                                    forbidden_llm_tokens = forbidden_llm_tokens.union(llm_tokens_for_terminal)

                    gss_active_tokens = all_ones_mask
                    glr_active_tokens = llm_mask.intersection(gss_active_tokens)
                    final_allowed_tokens = glr_active_tokens.difference(forbidden_llm_tokens)
                    if not final_allowed_tokens.is_empty():
                        final_mask = final_mask.union(final_allowed_tokens)

                if llm_mask.is_empty():
                    stopped.add(node_idx)
                    continue

                # Transitions grouped by (pop, llm_bv)
                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                for (pop, llm_bv), dests in children:
                    popped = gss_node.popn(pop)
                    llm_empty = llm_bv.is_empty()

                    for dest_idx, state_bv in dests:
                        matched: List[GSS] = []
                        if not state_bv.is_empty():
                            for sid_val in popped.peek():
                                if state_bv.contains(sid_val):
                                    matched.append(popped.isolate(sid_val))
                        if not matched:
                            continue

                        child_gss_node = GSS.merge_many(matched)
                        child_llm_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)
                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss, existing_mask = existing
                            merged_gss = existing_gss.merge(child_gss_node)
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (merged_gss, combined_mask)
                        else:
                            values[d] = (child_gss_node, child_llm_mask)

                        enqueue(max_depth[d], d)

        # Convert internal mask back to original IDs
        original_mask = ffi.Bitset.zeros()
        for internal_id in final_mask.to_indices():
            if internal_id in self.internal_to_original_map:
                original_mask.insert(self.internal_to_original_map[internal_id])
        return RangeSet.from_ranges(original_mask.to_ranges())
