import json
import time
import heapq
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
        self.max_depth: Dict[int, int] = {}
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None

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
    def from_json_string(s: str) -> "Model":
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
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

    def get_mask(self) -> ffi.Bitset:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """
        t0 = time.time()
        print(f"[{time.time() - t0:.4f}] get_mask: start")

        state_to_gss: Dict[int, ffi.GSSNode] = self.constraint_state.filtered_state_gss_map()

        final_mask = ffi.Bitset.zeros()

        # node_idx -> (set(GSSNode), Bitset)
        values: Dict[int, Tuple[set, ffi.Bitset]] = {}

        stopped: set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths (may contain duplicates)

        print(f"[{time.time() - t0:.4f}] get_mask: after init")

        # Seed: map tokenizer states and their filtered GSS to trie roots
        t_seed_start = time.time()
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
        t_seed_end = time.time()
        print(f"[{time.time() - t0:.4f}] get_mask: seed loop took {t_seed_end - t_seed_start:.4f}s")

        # Main scheduler
        t_loop_start = time.time()

        # Timing accumulators
        time_pop_bucket = 0.0
        time_node_setup = 0.0
        time_end_check = 0.0
        time_popn_collect = 0.0
        time_filter_peeks = 0.0
        time_merge_matched = 0.0
        time_merge_values = 0.0
        hits_pop_bucket = 0
        hits_node_setup = 0
        hits_end_check = 0
        hits_popn_collect = 0
        hits_filter_peeks = 0
        hits_merge_matched = 0
        hits_merge_values = 0

        loop_count = 0

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
            t1 = time.time()
            node_indices: Optional[set[int]] = None
            while depth_heap:
                current_depth = heappop(depth_heap)
                node_indices = todo.pop(current_depth, None)
                if node_indices:
                    break
            time_pop_bucket += time.time() - t1
            if not node_indices:
                break  # nothing left to process
            hits_pop_bucket += 1
            loop_count += 1

            # Process all nodes in this depth bucket
            for node_idx in node_indices:
                t1 = time.time()
                if node_idx in stopped:
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    continue
                gss_set, llm_mask = item
                time_node_setup += time.time() - t1
                hits_node_setup += 1

                # End-node handling
                t1 = time.time()
                if is_end(node_idx):
                    final_mask = final_mask.union(llm_mask)
                time_end_check += time.time() - t1
                hits_end_check += 1

                if not gss_set:
                    stopped.add(node_idx)
                    continue

                # Transitions grouped by (pop, llm_bv)
                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                for (pop, llm_bv), dests in children:
                    # Collect all pops from GSS parents
                    t1 = time.time()
                    peeks = []
                    for g in gss_set:
                        peeks.extend(g.popn_fast(pop))
                    time_popn_collect += time.time() - t1
                    hits_popn_collect += 1
                    if not peeks:
                        continue

                    llm_empty = llm_bv.is_empty()

                    for dest_idx, state_bv in dests:
                        # Filter peeks by destination state bitset
                        t1 = time.time()
                        matched = []
                        if not state_bv.is_empty():
                            contains = state_bv.contains
                            for sid_val, parent_node in peeks:
                                if contains(sid_val):
                                    matched.append(parent_node)
                        time_filter_peeks += time.time() - t1
                        hits_filter_peeks += 1
                        if not matched:
                            continue

                        # Merge matched parents
                        t1 = time.time()
                        child_gss_nodes = matched  # already a list of parent nodes
                        time_merge_matched += time.time() - t1
                        hits_merge_matched += 1

                        # Compute child mask (intersection with llm_bv when present)
                        child_llm_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)

                        d = dest_idx
                        t1 = time.time()
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss_set, existing_mask = existing
                            old_len = len(existing_gss_set)
                            existing_gss_set.update(child_gss_nodes)
                            # Only re-enqueue if effectively changed
                            if len(existing_gss_set) == old_len:
                                time_merge_values += time.time() - t1
                                hits_merge_values += 1
                                continue
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (existing_gss_set, combined_mask)
                        else:
                            values[d] = (set(child_gss_nodes), child_llm_mask)
                        time_merge_values += time.time() - t1
                        hits_merge_values += 1

                        enqueue(max_depth[d], d)

        t_loop_end = time.time()
        print(f"[{time.time() - t0:.4f}] get_mask: scheduler loop finished in {t_loop_end - t_loop_start:.4f}s ({loop_count} iterations)")
        print(f"    - 1. Pop bucket:        {time_pop_bucket:9.4f}s ({hits_pop_bucket:8d} hits)")
        print(f"    - 2. Node setup:        {time_node_setup:9.4f}s ({hits_node_setup:8d} hits)")
        print(f"    - 3. End check:         {time_end_check:9.4f}s ({hits_end_check:8d} hits)")
        print(f"    - 4. Pop'n'collect:     {time_popn_collect:9.4f}s ({hits_popn_collect:8d} hits)")
        print(f"    - 5. Filter peeks:      {time_filter_peeks:9.4f}s ({hits_filter_peeks:8d} hits)")
        print(f"    - 6. Merge matched:     {time_merge_matched:9.4f}s ({hits_merge_matched:8d} hits)")
        print(f"    - 7. Merge into values: {time_merge_values:9.4f}s ({hits_merge_values:8d} hits)")

        print(f"[{time.time() - t0:.4f}] get_mask: returning")

        original_mask = self.constraint.internal_bv_to_original(final_mask)
        temp = RangeSet.from_ranges(original_mask.to_ranges())
        ref = self.constraint_state.get_mask()
        print("Final computed mask:", temp)
        print("Reference mask from Rust state:", ref)
        assert (temp == ref).all()
        return temp


