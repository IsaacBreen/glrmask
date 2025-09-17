import heapq
import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from tqdm.auto import tqdm

import _sep1 as ffi
from ..common_interface import GraphProvider, RangeSet


class Model(GraphProvider):
    """
    An optimized precomputed trie model.

    This model applies several optimizations during initialization to accelerate
    the get_mask computation:
    1.  Parallel Edge Merging: Edges between the same two nodes with the same
        conditions (pop count, LLM token bitset) are merged by unioning their
        state bitsets. Similarly, edges with identical state conditions are
        merged by unioning their LLM token bitsets. This reduces the total
        number of edges to process.
    2.  Restructured Children for Fast Lookups: The node's children data
        structure is transformed from a list of edges to a dictionary mapping
        pop counts to a list of transitions. Each transition contains a
        precomputed map from GSS state ID to destination node index (`sid_map`).
        This eliminates the most expensive loop in `get_mask` (filtering peeks),
        replacing it with a direct dictionary lookup.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict], max_state_id: int):
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.max_depth: Dict[int, int] = {}

        # 1. Initial normalization of arena from JSON and max_depth caching
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        nodes_to_process = list(self.arena.items())
        for uid, node in tqdm(
            nodes_to_process,
            desc="Normalizing and Optimizing Graph",
            total=len(nodes_to_process),
        ):
            uid_int = int(uid)
            self.max_depth[uid_int] = int(node.get("max_depth", 0))

            children = node.get("children") or []
            if not children:
                node["children"] = {}
                continue

            # Convert JSON bitsets to ffi.Bitset instances
            normalized_children = []
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv = bs_from_json(dumps(llm_bv_json))
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv = bs_from_json(dumps(state_bv_json))
                    new_dest_map.append((int(dest_idx), state_bv))
                normalized_children.append(((int(pop), llm_bv), new_dest_map))

            # 2. Parallel Edge Merging
            # Merge state_bvs for identical (pop, llm_bv, dest)
            merged_edges = defaultdict(ffi.Bitset.zeros)
            for (pop, llm_bv), dests in normalized_children:
                for dest, state_bv in dests:
                    key = (pop, llm_bv, dest)
                    merged_edges[key] = merged_edges[key].union(state_bv)

            # Re-group by (pop, llm_bv)
            grouped_by_pop_llm = defaultdict(list)
            for (pop, llm_bv, dest), state_bv in merged_edges.items():
                grouped_by_pop_llm[(pop, llm_bv)].append((dest, state_bv))

            # Merge llm_bvs for identical (pop, dests)
            merged_by_dests = defaultdict(ffi.Bitset.zeros)
            for (pop, llm_bv), dests in grouped_by_pop_llm.items():
                dests.sort()
                dests_key = tuple((d, s.to_json_string()) for d, s in dests)
                key = (pop, dests_key)
                merged_by_dests[key] = merged_by_dests[key].union(llm_bv)

            # Rebuild children list
            rebuilt_children = []
            for (pop, dests_key), llm_bv in merged_by_dests.items():
                dests = [(d, bs_from_json(s_json)) for d, s_json in dests_key]
                rebuilt_children.append(((pop, llm_bv), dests))

            # 3. Restructure for fast lookup in get_mask
            final_children = defaultdict(list)
            for (pop, llm_bv), dests in rebuilt_children:
                sid_map = {}
                epsilon_dests = []
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():
                        epsilon_dests.append(dest_idx)
                    else:
                        for start, end in state_bv.to_ranges():
                            end = min(end, max_state_id)
                            for sid in range(start, end):
                                sid_map[sid] = dest_idx
                final_children[pop].append((llm_bv, (sid_map, epsilon_dests)))

            node["children"] = final_children

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        max_state_id = int(max(dict(data['parser']['stage_7_table']).keys()))
        model = Model(roots_map, arena, max_state_id)
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int):
        """
        Explode packed transitions from the optimized graph representation.
        Used for validation.
        """
        node_data = self.arena.get(node, {})
        children_by_pop = node_data.get("children") or {}
        for pop, transitions in children_by_pop.items():
            for llm_bv, (sid_map, epsilon_dests) in transitions:
                if llm_bv.contains(token):
                    for sid, dest_idx in sid_map.items():
                        yield (pop, sid, dest_idx)
                    for dest_idx in epsilon_dests:
                        yield (pop, None, dest_idx)

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """
        print("\n--- get_mask START ---")
        print(self.constraint_state)
        state_to_gss = self.constraint_state.filtered_state_gss_map()
        print(f"Filtered state_to_gss: { {k: v.ptr() for k, v in state_to_gss.items()} }")

        final_mask = ffi.Bitset.zeros()
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}
        todo: Dict[int, set[int]] = defaultdict(set)
        depth_heap: List[int] = []

        # Seed: map tokenizer states and their GSS nodes to trie roots
        print("\n--- Seeding work queue ---")

        # Seed: map tokenizer states and their GSS nodes to trie roots
        print("\n--- Seeding work queue ---")
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(sid)
            if root_idx is None:
                continue


            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            existing = values.get(root_idx)
            if existing:
                existing_gss, existing_mask = existing
                print(f"    - MERGE: gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={gss_clone.ptr()}, mask2={new_mask.to_ranges()}")
                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, gss_clone], 1)
                merged_mask = existing_mask.union(new_mask)
                values[root_idx] = (merged_gss, merged_mask)
                print(f"      - Merged result: gss_ptr={merged_gss.ptr()}, mask={merged_mask.to_ranges()}")
            else:
                values[root_idx] = (gss_clone, new_mask)
                depth = self.max_depth[root_idx]

                if not todo[depth]:
                    heapq.heappush(depth_heap, depth)
                todo[depth].add(root_idx)

        # Main scheduler loop
        print("\n--- Main loop ---")
        # Main scheduler loop
        print("\n--- Main loop ---")
        iter_count = 0
        while depth_heap:
            iter_count += 1
            current_depth = heapq.heappop(depth_heap)
            node_indices = todo.pop(current_depth, set())
            print(f"\n[{iter_count}] Processing depth={current_depth}, nodes={node_indices}")
            if not node_indices:
                continue


            for node_idx in node_indices:
                item = values.pop(node_idx, None)
                if item is None:
                    print(f"  - Node {node_idx}: SKIPPING (no value)")
                    continue
                gss_node, llm_mask = item
                print(f"  - PROCESS: node_ptr={node_idx}, gss_ptr={gss_node.ptr()}, mask={llm_mask.to_ranges()}")

                if self.is_end(node_idx):
                    print(f"    - END NODE found. Updating final_mask.")
                    print(f"      - final_mask before: {final_mask.to_ranges()}")
                    gss_active_tokens = gss_node.allowed_llm_tokens()
                    tokens_to_add = llm_mask.intersection(gss_active_tokens)
                    print(f"      - glr_active_tokens to union: {tokens_to_add.to_ranges()}")
                    final_mask = final_mask.union(tokens_to_add)
                    print(f"      - final_mask after:  {final_mask.to_ranges()}")

                if not gss_node.is_alive():
                    print(f"    - STOPPING node {node_idx} (GSS not alive)")
                    continue

                children_by_pop = self.arena.get(node_idx, {}).get("children", {})
                for pop, transitions in children_by_pop.items():
                    print(f"    - Edge group: pop={pop}")
                    peeks = gss_node.popn_fast(pop)
                    print(f"      - Found {len(peeks)} peeks from GSS")
                    if not peeks:
                        continue

                    for llm_bv, (sid_map, epsilon_dests) in transitions:
                        child_llm_mask = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)
                        if child_llm_mask.is_empty():
                            continue
                        print(f"    - Edge: llm_bv={llm_bv.to_ranges()}")
                        print(f"      - Child mask: {child_llm_mask.to_ranges()}")

                        dest_to_parents = defaultdict(list)
                        if sid_map:
                            for sid, parent_gss in peeks:
                                # print(f"      - Peek: sid={sid}, parent_gss={parent_gss.ptr()}")
                                dest_idx = sid_map.get(sid)
                                if dest_idx is not None:
                                    dest_to_parents[dest_idx].append(parent_gss)
                        
                        if epsilon_dests:
                            all_parents = [p for _, p in peeks]
                            if all_parents:
                                for dest_idx in epsilon_dests:
                                    dest_to_parents[dest_idx].extend(all_parents)

                        for dest_idx, parents_list in dest_to_parents.items():
                            print(f"      - Dest: idx={dest_idx}")
                            print(f"        - Matched {len(parents_list)} parent GSS nodes")
                            if not parents_list:
                                continue
                            child_gss = ffi.gss_merge_many_with_depth(parents_list, 1)
                            if not child_gss.is_alive():
                                continue

                            existing = values.get(dest_idx)
                            if existing:
                                existing_gss, existing_mask = existing
                                print(f"        - Enqueue {dest_idx}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={child_gss.ptr()}, mask2={child_llm_mask.to_ranges()}")
                                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, child_gss], 1)
                                new_mask = existing_mask.union(child_llm_mask)
                                if merged_gss.ptr() == existing_gss.ptr() and new_mask == existing_mask:
                                    continue
                                values[dest_idx] = (merged_gss, new_mask)
                                print(f"          - Merged result: gss_ptr={merged_gss.ptr()}, mask={new_mask.to_ranges()}")
                            else:
                                values[dest_idx] = (child_gss, child_llm_mask)
                                print(f"        - Enqueue {dest_idx}: CREATING gss_ptr={child_gss.ptr()}, mask={child_llm_mask.to_ranges()}")

                            depth = self.max_depth[dest_idx]
                            if not todo[depth]:
                                heapq.heappush(depth_heap, depth)
                            todo[depth].add(dest_idx)
                            
        print("\n--- get_mask END ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())
