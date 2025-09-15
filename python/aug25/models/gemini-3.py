import json
import time
import heapq
from typing import Dict, List, Tuple, Optional, Set

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm


class Model(GraphProvider):
    """
    An optimized precomputed trie model (Gemini-3 generation).

    This model introduces significant optimizations to the graph structure during
    initialization to accelerate the `get_mask` operation.

    Optimizations:
    1. Graph Restructuring: Transitions are pre-processed into a mapping from
       state ID -> destination nodes, eliminating a major bottleneck.
    2. Hybrid Edge Representation: A sparse map is used for most transitions,
       while dense bitsets are kept for transitions involving many states to
       conserve memory.
    3. Edge Merging: Parallel edges are merged during initialization to create a
       more compact and efficient graph representation.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Map tokenizer state -> trie root node
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.max_depth: Dict[int, int] = {}

        # --- Optimization: Pre-process and restructure the graph ---
        self._optimize_arena()

    def _optimize_arena(self):
        """
        Restructures the arena for optimal `get_mask` performance.
        This is the core of the optimization strategy.
        """
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string
        DENSE_THRESHOLD = 256  # Tunable parameter

        for uid, node in tqdm(
            self.arena.items(),
            desc="Optimizing Gemini-3 Graph",
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

            # 1. Group original children by (pop, llm_bv) to merge them
            grouped_children: Dict[Tuple[int, str], List] = {}
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                key = (int(pop), dumps(llm_bv_json))
                if key not in grouped_children:
                    grouped_children[key] = []
                grouped_children[key].extend(dest_map)

            new_children = []
            for (pop, llm_bv_json_str), dest_map_list in grouped_children.items():
                llm_bv = bs_from_json(llm_bv_json_str)

                # 2. Merge state bitsets for the same destination index
                merged_dest_map: Dict[int, ffi.Bitset] = {}
                for dest_idx, state_bv_json in dest_map_list:
                    dest_idx_int = int(dest_idx)
                    state_bv = bs_from_json(dumps(state_bv_json))
                    if dest_idx_int in merged_dest_map:
                        merged_dest_map[dest_idx_int] = merged_dest_map[dest_idx_int].union(state_bv)
                    else:
                        merged_dest_map[dest_idx_int] = state_bv

                # 3. Partition into sparse, dense, and epsilon transitions
                sid_to_dest_map: Dict[int, List[int]] = {}
                dense_dests: List[Tuple[int, ffi.Bitset]] = []
                epsilon_dests: List[int] = []

                for dest_idx_int, state_bv in merged_dest_map.items():
                    if state_bv.is_empty():
                        epsilon_dests.append(dest_idx_int)
                        continue

                    num_states = sum(end - start for start, end in state_bv.to_ranges())

                    if num_states > DENSE_THRESHOLD:
                        dense_dests.append((dest_idx_int, state_bv))
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end):
                                if sid not in sid_to_dest_map:
                                    sid_to_dest_map[sid] = []
                                sid_to_dest_map[sid].append(dest_idx_int)

                new_children.append(
                    ((pop, llm_bv), (sid_to_dest_map, dense_dests, epsilon_dests))
                )
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
        Adapted to work with the optimized graph structure.
        """
        children = self.arena.get(node, {}).get("children") or []
        for (pop, llm_bv), (sid_to_dest_map, dense_dests, epsilon_dests) in children:
            if llm_bv.contains(token):
                for dest_idx in epsilon_dests:
                    yield (pop, None, dest_idx)

                for sid, dest_indices in sid_to_dest_map.items():
                    for dest_idx in dest_indices:
                        yield (pop, sid, dest_idx)

                for dest_idx, state_bv in dense_dests:
                    for start, end in state_bv.to_ranges():
                        for sid in range(start, end):
                            yield (pop, sid, dest_idx)

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """
        state_to_gss = self.constraint_state.get_state_to_gss_map()

        final_mask = ffi.Bitset.zeros()
        values: Dict[int, Tuple[Set[ffi.GSSNode], ffi.Bitset]] = {}
        stopped: Set[int] = set()
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        # Seed: map tokenizer states and their filtered GSS to trie roots
        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth = self.max_depth

        for sid, gss in state_to_gss.items():
            root_idx = roots_map.get(sid)
            if root_idx is None:
                continue

            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()

            existing = values.get(root_idx)
            if existing is not None:
                gss_set, existing_mask = existing
                gss_set.add(gss_clone)
                values[root_idx] = (gss_set, existing_mask.union(new_mask))
            else:
                values[root_idx] = ({gss_clone}, new_mask)

            depth = max_depth[root_idx]
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        # Main scheduler
        heappop = heapq.heappop
        arena = self.arena
        is_end = self.is_end

        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        while depth_heap:
            current_depth = heappop(depth_heap)
            node_indices = todo.pop(current_depth, None)
            if not node_indices:
                continue

            for node_idx in node_indices:
                if node_idx in stopped:
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    continue
                gss_set, llm_mask = item

                if is_end(node_idx):
                    final_mask = final_mask.union(llm_mask)

                if not gss_set:
                    stopped.add(node_idx)
                    continue

                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                for (pop, llm_bv), (sid_to_dest_map, dense_dests, _) in children:
                    peeks = []
                    for g in gss_set:
                        peeks.extend(g.popn_fast(pop))
                    if not peeks:
                        continue

                    dest_to_gss: Dict[int, List[ffi.GSSNode]] = {}

                    # --- OPTIMIZED PEEK FILTERING ---
                    # Process sparse transitions using the precomputed map
                    if sid_to_dest_map:
                        for sid, gss_node in peeks:
                            dest_indices = sid_to_dest_map.get(sid)
                            if dest_indices:
                                for dest_idx in dest_indices:
                                    dest_to_gss.setdefault(dest_idx, []).append(gss_node)

                    # Process dense transitions with the original loop
                    if dense_dests:
                        for dest_idx, state_bv in dense_dests:
                            contains = state_bv.contains
                            for sid, gss_node in peeks:
                                if contains(sid):
                                    dest_to_gss.setdefault(dest_idx, []).append(gss_node)
                    # --- END OPTIMIZED PEEK FILTERING ---

                    if not dest_to_gss:
                        continue

                    child_llm_mask = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)

                    for d, child_gss_nodes in dest_to_gss.items():
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss_set, existing_mask = existing
                            old_len = len(existing_gss_set)
                            existing_gss_set.update(child_gss_nodes)

                            # Replicating original model's logic: only update and re-queue
                            # if the GSS set has grown. This may be a bug in the original
                            # as it ignores mask updates, but we preserve it for compatibility.
                            if len(existing_gss_set) == old_len:
                                continue
                            
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (existing_gss_set, combined_mask)
                        else:
                            values[d] = (set(child_gss_nodes), child_llm_mask)

                        enqueue(max_depth[d], d)

        return RangeSet.from_ranges(final_mask.to_ranges())
