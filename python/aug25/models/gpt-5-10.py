import json
import heapq
import time
from bisect import bisect_left
from typing import Dict, List, Tuple, Optional, Iterable, Set

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # compiled module
from tqdm.auto import tqdm


class Model(GraphProvider):
    """
    Optimized trie model (gpt-5-10).
    Key ideas:
      - Normalize and merge parallel edges per node by (pop, llm_bv) and dest.
      - Process transitions grouped by pop; for each pop:
          - Pop the GSS once, group peeks by state-id, and pre-sort sids.
          - For each dest's state-bv, enumerate matching sids via range-vs-sorted-sids merge
            (O(#ranges + #matches)) rather than N*contains checks.
          - Accumulate parent nodes per dest and union LLM masks per dest lazily.
      - Avoid scanning irrelevant (sid, parent) pairs repeatedly across dests and llm groups.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Map tokenizer state -> trie root node
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None

        # Node -> max trie depth (for scheduling)
        self.max_depth: Dict[int, int] = {}

        # Node -> end flag
        self._is_end: Dict[int, bool] = {}

        # Node -> pop -> list[(llm_bv, [(dest_idx, state_bv), ...])]
        # All bitsets are ffi.Bitset; children are merged (same pop,llm_bv,dest merged).
        self.by_pop: Dict[int, Dict[int, List[Tuple[ffi.Bitset, List[Tuple[int, ffi.Bitset]]]]]] = {}

        # Normalize arena: merge edges with identical (pop, llm_bv, dest)
        # Use a per-node bitset cache to deduplicate bitset instances from JSON strings.
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        for uid, node in tqdm(
            arena.items(),
            desc="Normalizing gpt-5-10 model",
            total=len(arena),
        ):
            node_id = int(uid)

            # Cache end flag and max depth
            try:
                self._is_end[node_id] = bool((node.get("value") or {}).get("clean_end", False))
            except Exception:
                self._is_end[node_id] = False

            try:
                self.max_depth[node_id] = int(node.get("max_depth", 0))
            except Exception:
                self.max_depth[node_id] = 0

            children = node.get("children") or []
            if not children:
                self.by_pop[node_id] = {}
                continue

            # Build pop -> llm_key -> (llm_bv, dest_idx -> state_bv union)
            pop_groups: Dict[int, Dict[str, Tuple[ffi.Bitset, Dict[int, ffi.Bitset]]]] = {}
            bitset_cache: Dict[str, ffi.Bitset] = {}

            def get_bitset_from_json_string(sjson: str) -> ffi.Bitset:
                b = bitset_cache.get(sjson)
                if b is None:
                    b = bs_from_json(sjson)
                    bitset_cache[sjson] = b
                return b

            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                pop = int(pop)
                llm_bv_json_str = dumps(llm_bv_json)
                llm_bv = get_bitset_from_json_string(llm_bv_json_str)

                # pop -> group
                group = pop_groups.get(pop)
                if group is None:
                    group = {}
                    pop_groups[pop] = group

                entry = group.get(llm_bv_json_str)
                if entry is None:
                    entry = (llm_bv, {})  # (llm_bv_obj, dest_idx -> state_bv)
                    group[llm_bv_json_str] = entry

                _, dest_accum = entry

                # Accumulate state bitsets per destination
                for dest_idx, state_bv_json in dest_map:
                    dest_idx = int(dest_idx)
                    state_bv_json_str = dumps(state_bv_json)
                    state_bv = get_bitset_from_json_string(state_bv_json_str)
                    existing = dest_accum.get(dest_idx)
                    if existing is None:
                        dest_accum[dest_idx] = state_bv
                    else:
                        # Merge parallel edges: union state sets for same (pop, llm_bv, dest)
                        dest_accum[dest_idx] = existing.union(state_bv)

            # Convert to final by_pop structure with lists
            node_by_pop: Dict[int, List[Tuple[ffi.Bitset, List[Tuple[int, ffi.Bitset]]]]] = {}
            for pop, llm_groups in pop_groups.items():
                groups_list: List[Tuple[ffi.Bitset, List[Tuple[int, ffi.Bitset]]]] = []
                for llm_key, (llm_bv, dest_accum) in llm_groups.items():
                    # Convert dict to list for each group
                    dest_list = [(int(d), sbv) for d, sbv in dest_accum.items()]
                    groups_list.append((llm_bv, dest_list))
                node_by_pop[int(pop)] = groups_list

            self.by_pop[node_id] = node_by_pop

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
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return self._is_end.get(int(node), False)

    def iter_edges(self, node: int, token: int) -> Iterable[Tuple[int, Optional[int], int]]:
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        Only used by equivalence checking; not performance-critical.

        Note: If a state's bitset is empty, treat as epsilon on GSS stack (no state filter).
        """
        node = int(node)
        by_pop = self.by_pop.get(node)
        if not by_pop:
            return
        for pop, groups in by_pop.items():
            for llm_bv, dests in groups:
                if llm_bv.contains(token):
                    for dest_idx, state_bv in dests:
                        if state_bv.is_empty():  # Epsilon on GSS stack
                            yield (int(pop), None, int(dest_idx))
                        else:
                            for start, end in state_bv.to_ranges():
                                # to_ranges uses [start, end) semantics
                                s = int(start)
                                e = int(end)
                                for sid in range(s, e):
                                    yield (int(pop), sid, int(dest_idx))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. Highly optimized scheduler that minimizes per-dest membership checks
        by using: pop-grouped peeks, sid-indexed parent lists, and range-vs-sorted-sids scanning.
        """
        print("\n--- get_mask START ---")
        state_to_gss = self.constraint_state.filtered_state_gss_map()

        t0 = time.time()
        final_mask = ffi.Bitset.zeros()

        # node_idx -> (set(GSSNode), Bitset)
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}

        stopped: Set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, Set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths

        # Seed: map tokenizer states and their filtered GSS to trie roots
        heappush = heapq.heappush
        heappop = heapq.heappop
        roots_map = self.roots_map
        max_depth = self.max_depth

        print("\n--- Seeding work queue ---")
        for sid, gss in state_to_gss.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)


            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss, existing_mask = existing
                print(f"    - MERGE: gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={gss_clone.ptr()}, mask2={new_mask.to_ranges()}")
                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, gss_clone], 1)
                # union masks for same root if multiple start states map here
                merged_mask = existing_mask.union(new_mask)
                values[root_idx] = (merged_gss, merged_mask)
                print(f"      - Merged result: gss_ptr={merged_gss.ptr()}, mask={merged_mask.to_ranges()}")
            else:
                values[root_idx] = (gss_clone, new_mask)

            depth = max_depth.get(root_idx, 0)
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        # Helper to enqueue a node at a given depth
        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        by_pop = self.by_pop
        is_end = self.is_end

        # Core scheduler
        print("\n--- Main loop ---")
        iter_count = 0
        while True:
            iter_count += 1
            # Pop the smallest depth bucket (skip stale heap entries)
            node_indices: Optional[Set[int]] = None
            while depth_heap:
                current_depth = heappop(depth_heap)
                print(f"\n[{iter_count}] Popping depth={current_depth}")
                node_indices = todo.pop(current_depth, None)
                if node_indices:
                    break
            if not node_indices:
                break  # nothing left to process

            # Process all nodes in this depth bucket

            # Process all nodes in this depth bucket
            for node_idx in node_indices:
                if node_idx in stopped:
                    print(f"  - Node {node_idx}: SKIPPING (already stopped)")
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    print(f"  - Node {node_idx}: SKIPPING (no value)")
                    continue
                gss_node, llm_mask = item
                print(f"  - PROCESS: node_ptr={node_idx}, gss_ptr={gss_node.ptr()}, mask={llm_mask.to_ranges()}")

                # End-node handling
                if is_end(node_idx):
                    print(f"    - END NODE found. Updating final_mask.")
                    print(f"      - final_mask before: {final_mask.to_ranges()}")
                    gss_active_tokens = gss_node.allowed_llm_tokens()
                    tokens_to_add = llm_mask.intersection(gss_active_tokens)
                    print(f"      - glr_active_tokens to union: {tokens_to_add.to_ranges()}")
                    final_mask = final_mask.union(tokens_to_add)
                    print(f"      - final_mask after:  {final_mask.to_ranges()}")

                if not gss_node.is_alive():
                    stopped.add(node_idx)
                    print(f"    - STOPPING node {node_idx} (GSS not alive)")
                    continue

                node_by_pop = by_pop.get(node_idx)
                if not node_by_pop:
                    continue

                # For each pop value at this node:

                # For each pop value at this node:
                for pop, groups in node_by_pop.items():
                    print(f"    - Edge group: pop={pop}")
                    # Collect all pops from GSS parents exactly once per pop
                    peeks = gss_node.popn_fast(pop)
                    print(f"      - Found {len(peeks)} peeks from GSS")
                    if not peeks:
                        continue


                    # Group peeks by state id and build sorted sid list
                    # Also build a set of all parent nodes (for epsilon state filters)
                    peeks_by_sid: Dict[int, List[ffi.GSSNode]] = {}
                    parents_all_set: Set[ffi.GSSNode] = set()
                    parents_all_list: List[ffi.GSSNode] = []
                    for sid_val, parent_node in peeks:
                        lst = peeks_by_sid.get(sid_val)
                        if lst is None:
                            peeks_by_sid[sid_val] = [parent_node]
                        else:
                            lst.append(parent_node)
                        parents_all_list.append(parent_node)

                    if not peeks_by_sid:
                        continue

                    sorted_sids = sorted(peeks_by_sid.keys())
                    nsids = len(sorted_sids)

                    # Utility: iterate sids within a Bitset's ranges efficiently against sorted_sids
                    def sids_in_statebv(state_bv: ffi.Bitset) -> Iterable[int]:
                        # Merge-scan using bisect to skip to each range's start
                        idx = 0
                        for start, end in state_bv.to_ranges():
                            s = int(start)
                            e = int(end)
                            if idx < nsids:
                                idx = bisect_left(sorted_sids, s, idx, nsids)
                            else:
                                return
                            while idx < nsids:
                                sid = sorted_sids[idx]
                                if sid >= e:
                                    break
                                yield sid
                                idx += 1

                    # Process all (llm_bv, dests) groups under this pop

                    # Process all (llm_bv, dests) groups under this pop
                    for llm_bv, dests in groups:
                        print(f"    - Edge: llm_bv={llm_bv.to_ranges()}")
                        # Compute the child LLM mask for this group once
                        if llm_bv.is_empty():
                            group_child_mask = llm_mask
                        else:
                            group_child_mask = llm_mask.intersection(llm_bv)
                        print(f"      - Child mask: {group_child_mask.to_ranges()}")

                        # Iterate dests; for each, collect matching parent nodes via sids_in_statebv
                        for dest_idx, state_bv in dests:
                            # Determine matched parent nodes set
                            if state_bv.is_empty():
                                child_gss_nodes_list = parents_all_list
                                if not child_gss_nodes_list:
                                    continue
                            else:
                                # Collect parents for the sids matched by state_bv
                                child_gss_nodes_list = []
                                for sid in sids_in_statebv(state_bv):
                                    lst = peeks_by_sid.get(sid)
                                if lst:
                                    child_gss_nodes_list.extend(lst)
                            if not child_gss_nodes_list:
                                continue
                            print(f"      - Dest: idx={dest_idx}, state_bv={state_bv.to_ranges()}")
                            print(f"        - Matched {len(child_gss_nodes_list)} parent GSS nodes")

                            child_gss = ffi.gss_merge_many_with_depth(child_gss_nodes_list, 1)
                            if not child_gss.is_alive():
                                continue

                            d = dest_idx
                            existing = values.get(d)
                            if existing is not None:
                                existing_gss, existing_mask = existing
                                print(f"        - Enqueue {d}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={child_gss.ptr()}, mask2={group_child_mask.to_ranges()}")
                                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, child_gss], 1)

                                if merged_gss.ptr() == existing_gss.ptr():
                                    continue

                                new_mask = existing_mask.union(group_child_mask)
                                values[d] = (merged_gss, new_mask)
                                print(f"          - Merged result: gss_ptr={merged_gss.ptr()}, mask={new_mask.to_ranges()}")
                            else:
                                values[d] = (child_gss, group_child_mask)
                                print(f"        - Enqueue {d}: CREATING gss_ptr={child_gss.ptr()}, mask={group_child_mask.to_ranges()}")

                            enqueue(max_depth.get(d, 0), d)

        print(f"\n--- get_mask END (took {time.time() - t0:.4f}s) ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())
