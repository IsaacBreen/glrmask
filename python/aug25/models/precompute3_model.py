import json
import time
import heapq
from typing import Dict, List, Tuple, Optional, Any

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from gss_tester.fast_impl import FastGSS
from tqdm.auto import tqdm


# --- GSS Bridging Helpers ---
# These helpers are used to convert the GSS state from the Rust FFI (`GSSNode`)
# into the pure-Python `FastGSS` implementation. This is computationally
# expensive but necessary to use the Python GSS logic.

_path_cache: Dict[int, List[List[Any]]] = {}

def _get_paths(node: ffi.GSSNode) -> List[List[Any]]:
    node_ptr = node.ptr()
    if node_ptr in _path_cache:
        return _path_cache[node_ptr]
    if node.max_depth() == 0:
        return [[]]
    paths = []
    for state_id, pred_node in node.popn_fast(1):
        for p_path in _get_paths(pred_node):
            paths.append(p_path + [state_id])
    _path_cache[node_ptr] = paths
    return paths

def from_gss_node(gss_node: ffi.GSSNode, acc_factory) -> FastGSS:
    paths = _get_paths(gss_node)
    gss = FastGSS.initial(acc_factory)
    # This is inefficient: builds many intermediate GSS objects.
    # A more optimized from_paths constructor would be better.
    merged_gss_list = []
    for path in paths:
        path_gss = FastGSS.initial(acc_factory)
        for item in path:
            path_gss = path_gss.push(item)
        merged_gss_list.append(path_gss)
    
    if not merged_gss_list:
        return FastGSS.initial(acc_factory)
        
    return FastGSS.merge(merged_gss_list, lambda a, b: a)

class Model(GraphProvider):
    """
    Precomputed trie model (third-generation).
    Normalizes input arena by converting JSON bitsets into ffi.Bitset instances
    and provides graph traversal and mask computation interfaces.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Map tokenizer state -> trie root node
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.max_depth: Dict[int, int] = {}

        # Normalize arena children bitsets and cache max_depth
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        for uid, node in tqdm(
            self.arena.items(),
            desc="Normalizing precompute3 BVs",
            total=len(self.arena),
        ):
            uid_int = int(uid)
            try:
                md = node.get("max_depth", 0)
                self.max_depth[uid_int] = int(md)
            except Exception:
                self.max_depth[uid_int] = 0

            children = node.get("children") or []
            if not children:
                node["children"] = []
                continue

            new_children = []
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv = bs_from_json(dumps(llm_bv_json))

                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv = bs_from_json(dumps(state_bv_json))
                    new_dest_map.append((int(dest_idx), state_bv))

                new_children.append(((int(pop), llm_bv), new_dest_map))
            node["children"] = new_children

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)
        model.constraint_state = ffi.GrammarConstraintState.from_json_string(s)
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("end", False))

    def iter_edges(self, node: int, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        Only used by equivalence checking; not performance-critical.
        """
        children = self.arena.get(node, {}).get("children") or []
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():  # Epsilon on GSS stack
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end):
                                yield (int(pop), sid, int(dest_idx))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """
        state_to_gss_nodes = self.constraint_state.get_state_gss()
        acc_factory = lambda: None # Accumulators are not used in this model
        
        # Convert Rust GSSNodes to Python FastGSS instances.
        # This is a slow operation required to bridge the two implementations.
        state_to_gss = { sid: from_gss_node(gss_node, acc_factory) for sid, gss_node in state_to_gss_nodes.items() }
        t0 = time.time()

        final_mask = ffi.Bitset.zeros()

        # node_idx -> (set(GSSNode), Bitset)
        values: Dict[int, Tuple[FastGSS, ffi.Bitset]] = {}

        stopped: set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths (may contain duplicates)


        # Seed: map tokenizer states and their filtered GSS to trie roots
        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth = self.max_depth

        for sid, gss_node in state_to_gss_nodes.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)
            
            gss = state_to_gss[sid]
            new_mask = gss_node.allowed_llm_tokens()

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss, existing_mask = existing
                merged_gss = FastGSS.merge([existing_gss, gss], lambda a, b: a)
                values[root_idx] = (merged_gss, existing_mask.union(new_mask))
            else:
                values[root_idx] = (gss, new_mask)

            depth = max_depth[root_idx]
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        # Main scheduler

        # Helper to enqueue a node at a given depth
        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        heappop = heapq.heappop
        arena = self.arena
        is_end = self.is_end

        while True:
            # Pop the smallest depth bucket (skip stale heap entries)
            node_indices: Optional[set[int]] = None
            while depth_heap:
                current_depth = heappop(depth_heap)
                node_indices = todo.pop(current_depth, None)
                if node_indices:
                    break
            if not node_indices:
                break  # nothing left to process

            # Process all nodes in this depth bucket
            for node_idx in node_indices:
                if node_idx in stopped:
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    continue
                gss, llm_mask = item

                # End-node handling
                if is_end(node_idx):
                    final_mask = final_mask.union(llm_mask)

                if gss.is_empty():
                    stopped.add(node_idx)
                    continue

                # Transitions grouped by (pop, llm_bv)
                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                for (pop, llm_bv), dests in children:
                    # Collect all pops from GSS parents
                    peeks = gss.popn_fast(pop)
                    if not peeks:
                        continue

                    llm_empty = llm_bv.is_empty()

                    for dest_idx, state_bv in dests:
                        # Filter peeks by destination state bitset
                        matched = []
                        if not state_bv.is_empty():
                            contains = state_bv.contains
                            for sid_val, parent_node in peeks:
                                if contains(sid_val):
                                    matched.append(parent_node)
                        if not matched:
                            continue
                        
                        # Create a new GSS from the matched parent nodes from the previous GSS
                        child_gss = FastGSS.from_heads(set(matched), gss)

                        # Compute child mask (intersection with llm_bv when present)
                        child_llm_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)

                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss, existing_mask = existing
                            new_gss = FastGSS.merge([existing_gss, child_gss], lambda a, b: a)
                            # Only re-enqueue if effectively changed
                            if new_gss._heads == existing_gss._heads:
                                continue
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (new_gss, combined_mask)
                        else:
                            values[d] = (child_gss, child_llm_mask)

                        enqueue(max_depth[d], d)

        return RangeSet.from_ranges(final_mask.to_ranges())
