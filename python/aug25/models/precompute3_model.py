import json
import time
import heapq
from typing import Dict, List, Tuple, Optional, Type

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm
from gss_tester.interface import GSS
from gss_tester.rust_impl import RustGSS


class Model(GraphProvider):
    """
    Precomputed trie model (third-generation).
    Normalizes input arena by converting JSON bitsets into ffi.Bitset instances
    and provides graph traversal and mask computation interfaces.
    """
    def __init__(self, tokenizer, parser, roots_map, arena: Dict[int, dict]):
        self.tokenizer = tokenizer
        self.parser = parser
        self.roots_map = roots_map
        self.arena = arena

        self.gss_class: Type[GSS] = RustGSS
        self.acc_factory = lambda: 0
        self.merge_func = lambda a, b: a + b
        
        # The model's state is a Python dictionary of tokenizer_state -> GSS object
        initial_gss = self.gss_class.initial(self.acc_factory)
        self.state: Dict[int, GSS] = {
            self.tokenizer.initial_state_id(): initial_gss
        }
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
        constraint = ffi.GrammarConstraint.from_json_string(s)
        tokenizer = constraint.tokenizer()
        parser = constraint.get_parser()
        roots_map = {int(k): int(v) for k, v in constraint.precompute3_json_string().rsplit('],', 1)[0].split('[', 2)[-1].replace('],[', '],[').split('],[')}

        data = json.loads(s)
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        
        return Model(tokenizer, parser, roots_map, arena)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        node_data = self.arena.get(node)
        if not node_data: return False
        return bool((node_data.get("value") or {}).get("end", False))

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

    def step(self, gss_state: GSS, terminal_id: int) -> GSS:
        """
        A pure-Python implementation of a GLR parser step.
        This function takes a GSS node and a terminal ID, and applies the
        shifts and reduces defined in the parse table to produce a new GSS.
        """
        raise NotImplementedError("Python-side GLR step is not implemented yet.")

    def commit(self, token_id: int):
        # This method is complex and currently relies on Rust-side GSS logic.
        # We adapt it to use the Python GSS wrapper (`RustGSS`), which in turn
        # calls the necessary methods on the underlying `PyGSSNode` objects.
        token_bytes = self.tokenizer.id_to_token(token_id) # Assuming this method exists
        if not token_bytes:
            return

        # 1. Reset LLM tokens on all current GSS nodes
        for gss in self.state.values():
            for node in gss.stacks.keys():
                node.reset_llm_tokens()

        # 3. Map tokenizer states based on token consumption
        state_map = {}
        terminals_map = {}
        for tokenizer_state_id in self.state.keys():
            exec_result = self.tokenizer.execute_from_state(token_bytes, tokenizer_state_id)
            if exec_result.end_state is not None:
                state_map[tokenizer_state_id] = exec_result.end_state
            
            terminals = ffi.Bitset.zeros()
            for match in exec_result.matches:
                terminals.insert(match.id)
            terminals_map[tokenizer_state_id] = terminals

        for gss in self.state.values():
            for node in gss.stacks.keys():
                node.prune_disallowed_terminals(terminals_map)
                node.map_allowed_terminals_tokenizer_states(state_map)

        # 4. Main processing loop
        new_overall_state: Dict[int, GSS] = {}
        processing_queue = list(self.state.items())
        self.state = {} # Clear old state

        offset = 0
        while processing_queue:
            tokenizer_s_id, gss_object = processing_queue.pop(0)
            
            # This simplified loop only processes from the start of the token bytes.
            # A full implementation would handle offsets correctly.
            exec_result = self.tokenizer.execute_from_state(token_bytes, tokenizer_s_id)

            for match in exec_result.matches:
                try:
                    # This is where the unimplemented step function would be called
                    next_gss = self.step(gss_object, match.id)
                    
                    if next_gss.is_ok():
                        # In a full implementation, this would go into a queue for the next offset
                        # For now, we just add it to the final state at the initial tokenizer state
                        next_tokenizer_id = self.tokenizer.initial_state_id()
                        if next_tokenizer_id in new_overall_state:
                            new_overall_state[next_tokenizer_id] = self.gss_class.merge(
                                [new_overall_state[next_tokenizer_id], next_gss], self.merge_func
                            )
                        else:
                            new_overall_state[next_tokenizer_id] = next_gss

                except NotImplementedError:
                    # Since step is not implemented, we can't proceed with parsing.
                    # We will just carry over the state to the end.
                    pass

            if exec_result.end_state is not None:
                final_tokenizer_state = exec_result.end_state
                if final_tokenizer_state in new_overall_state:
                    existing_gss = new_overall_state[final_tokenizer_state]
                    new_overall_state[final_tokenizer_state] = self.gss_class.merge(
                        [existing_gss, gss_object], self.merge_func
                    )
                else:
                    new_overall_state[final_tokenizer_state] = gss_object
        
        self.state = new_overall_state

        # 5. Fuse predecessors
        for gss in self.state.values():
            for node in gss.stacks.keys():
                node.fuse_predecessors(1)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """
        # The get_mask logic is already in Python, but it needs to get its
        # initial state from `self.state`, which contains GSS objects.
        state_to_gss = self.state

        final_mask = ffi.Bitset.zeros()

        # node_idx -> (set(PyGSSNode), Bitset)
        values: Dict[int, Tuple[set, ffi.Bitset]] = {}

        stopped: set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths (may contain duplicates)


        # Seed: map tokenizer states and their filtered GSS to trie roots
        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth = self.max_depth

        for sid, gss_obj in state_to_gss.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            # Unpack the GSS object into its constituent head nodes
            gss_nodes = gss_obj.stacks.keys()
            if not gss_nodes:
                continue

            # Compute the combined mask from all head nodes
            new_mask = ffi.Bitset.zeros()
            for node in gss_nodes:
                new_mask = new_mask.union(node.allowed_llm_tokens())

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss_set, existing_mask = existing
                existing_gss_set.update(gss_nodes)
                values[root_idx] = (existing_gss_set, existing_mask.union(new_mask))
            else:
                values[root_idx] = (set(gss_nodes), new_mask)

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
