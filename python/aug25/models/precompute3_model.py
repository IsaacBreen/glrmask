import json
import time
import heapq
import collections
from typing import Dict, List, Tuple, Optional

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm


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
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.id_to_token: Dict[int, bytes] = {}
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
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
        model.id_to_token = {v: k.encode('utf-8') for k, v in data['llm_token_map'].items()}
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
        token_bytes = self.id_to_token.get(token_id)
        if token_bytes:
            self.commit_bytes(token_bytes)

    def python_step(self, gss_node: ffi.GSSNode, terminal_id: int) -> ffi.GSSNode:
        """
        This should implement the GLR parser step logic in Python.
        It takes a GSS node and a terminal ID, and returns a new GSS node
        representing the state after processing the token.
        """
        raise NotImplementedError("Python GLR parser step is not implemented.")

    def commit_bytes(self, llm_token_bytes: bytes):
        if not llm_token_bytes:
            return

        state = self.constraint_state
        tokenizer = self.constraint.tokenizer()
        
        current_state_map = state.get_state_map()

        for gss in current_state_map.values():
            ffi.gss_reset_llm_tokens(gss)

        state_map = {}
        terminals_map = {}
        for tokenizer_state_id in current_state_map.keys():
            end_state, matches = tokenizer.execute_from_state(llm_token_bytes, tokenizer_state_id)
            if end_state is not None:
                state_map[tokenizer_state_id] = end_state
            
            terminals = ffi.Bitset.zeros()
            for term_id, _ in matches:
                terminals.insert(term_id)
            terminals_map[tokenizer_state_id] = terminals

        for gss in current_state_map.values():
            ffi.gss_prune_disallowed_terminals(gss, terminals_map)
            ffi.gss_map_allowed_terminals_tokenizer_states(gss, state_map)

        new_overall_state = {}
        
        processing_queue = collections.defaultdict(dict)
        processing_queue[0] = current_state_map

        sorted_offsets = sorted(processing_queue.keys())
        
        while sorted_offsets:
            offset = sorted_offsets.pop(0)
            states_to_process = processing_queue.pop(offset)

            for tokenizer_s_id_at_offset, gss_at_offset in states_to_process.items():
                if offset >= len(llm_token_bytes):
                    continue

                end_state, matches = tokenizer.execute_from_state(llm_token_bytes[offset:], tokenizer_s_id_at_offset)

                # Placeholder for Python parser logic

                if end_state is not None:
                    final_tokenizer_state = end_state
                    if final_tokenizer_state in new_overall_state:
                        new_overall_state[final_tokenizer_state] = ffi.gss_merge_many_with_depth([new_overall_state[final_tokenizer_state], gss_at_offset], 1)
                    else:
                        new_overall_state[final_tokenizer_state] = gss_at_offset
        
        for gss in new_overall_state.values():
            ffi.gss_reset_llm_tokens(gss)
        
        final_state = {sid: gss for sid, gss in new_overall_state.items() if gss.is_ok()}

        for gss in final_state.values():
            ffi.gss_fuse_predecessors(gss, 1)

        state.set_state_map(final_state)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """
        state_to_gss = self.constraint_state.filtered_state_gss_map()
        t0 = time.time()

        final_mask = ffi.Bitset.zeros()

        # node_idx -> (set(GSSNode), Bitset)
        values: Dict[int, Tuple[set, ffi.Bitset]] = {}

        stopped: set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths (may contain duplicates)


        # Seed: map tokenizer states and their filtered GSS to trie roots
        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth = self.max_depth

        for sid, gss in state_to_gss.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

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
                gss_set, llm_mask = item

                # End-node handling
                if is_end(node_idx):
                    final_mask = final_mask.union(llm_mask)

                if not gss_set:
                    stopped.add(node_idx)
                    continue

                # Transitions grouped by (pop, llm_bv)
                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                for (pop, llm_bv), dests in children:
                    # Collect all pops from GSS parents
                    peeks = []
                    for g in gss_set:
                        peeks.extend(g.popn_fast(pop))
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

                        # Merge matched parents
                        child_gss_nodes = matched  # already a list of parent nodes

                        # Compute child mask (intersection with llm_bv when present)
                        child_llm_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)

                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss_set, existing_mask = existing
                            old_len = len(existing_gss_set)
                            existing_gss_set.update(child_gss_nodes)
                            # Only re-enqueue if effectively changed
                            if len(existing_gss_set) == old_len:
                                continue
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (existing_gss_set, combined_mask)
                        else:
                            values[d] = (set(child_gss_nodes), child_llm_mask)

                        enqueue(max_depth[d], d)

        return RangeSet.from_ranges(final_mask.to_ranges())
