import json
import time
from typing import Dict, List, Tuple, Optional, Iterable
from collections import defaultdict

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi


class Model(GraphProvider):
    """
    GPT-5-2 optimized graph provider.

    Design:
    - Input is the "precompute3" JSON graph (most detailed). We normalize up-front:
      * Convert all token/state JSON bitsets into ffi.Bitset once.
      * For each trie node, group children by `pop` value into compact groups:
          group = (pop, epsilon_edges, state_edges)
          - epsilon_edges: list[(dest_idx, llm_bv)] where the state filter is empty (i.e., no state constraint)
          - state_edges: list[(dest_idx, state_bv, llm_bv)] with a non-empty parser-state filter
    - get_mask() optimizations:
      * Single pop for each pop-group per node (no redundant gss_popn_collect).
      * Epsilon edges use all popped parents directly.
      * For state-filtered edges, we group popped parents once by sid to avoid scanning peeks repeatedly.
      * Minimize Python-level overhead (no tqdm/prints, reuse locals, avoid unnecessary conversions).

    The external FFI types (Bitset, GSSNode) are assumed to be well-optimized.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Roots: tokenizer_state_id -> trie_root_id
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        # Internal arena representation:
        # arena[node_id] = {
        #   "value": dict or None,
        #   "groups": List[Tuple[pop: int, epsilon_edges: List[(dest:int, llm_bv:Bitset)], state_edges: List[(dest:int, state_bv:Bitset, llm_bv:Bitset)]]],
        # }
        self.arena: Dict[int, dict] = {}
        # Precomputed node max_depth
        self.max_depth: Dict[int, int] = {}

        # Normalize nodes
        for uid_raw, node_raw in arena.items():
            uid = int(uid_raw)
            value = (node_raw.get("value") or {}) if isinstance(node_raw, dict) else {}
            self.max_depth[uid] = int((node_raw or {}).get("max_depth", 0) or 0)

            children = (node_raw or {}).get("children") or []
            groups_by_pop: Dict[int, Tuple[List[Tuple[int, ffi.Bitset]], List[Tuple[int, ffi.Bitset, ffi.Bitset]]]] = {}

            # Consume precompute3 children: [((pop, llm_bv_json), [(dest_idx, state_bv_json), ...]), ...]
            for edge_key, dest_map in children:
                pop_raw, llm_bv_json = edge_key
                pop = int(pop_raw)
                # Convert token bitset JSON -> ffi.Bitset once
                llm_bv = ffi.Bitset.from_json_string(json.dumps(llm_bv_json))

                eps_list, state_list = groups_by_pop.get(pop, (None, None))
                if eps_list is None:
                    eps_list = []
                    state_list = []
                    groups_by_pop[pop] = (eps_list, state_list)

                # Convert state_bv_json to ffi.Bitset once and bucket
                for dest_idx_raw, state_bv_json in dest_map:
                    dest_idx = int(dest_idx_raw)
                    state_bv = ffi.Bitset.from_json_string(json.dumps(state_bv_json))
                    if state_bv.is_empty():
                        # No parser-state constraint (epsilon on GSS stack)
                        eps_list.append((dest_idx, llm_bv))
                    else:
                        state_list.append((dest_idx, state_bv, llm_bv))

            # Finalize groups list
            groups: List[Tuple[int, List[Tuple[int, ffi.Bitset]], List[Tuple[int, ffi.Bitset, ffi.Bitset]]]] = []
            for pop, (eps_list, state_list) in groups_by_pop.items():
                groups.append((pop, eps_list, state_list))

            self.arena[uid] = {
                "value": value,
                "groups": groups,
            }

    @staticmethod
    def from_json_string(s: str) -> "Model":
        data = json.loads(s)
        # Graph comes in "precompute3" format
        roots_map = data['precomputed3']
        arena_json = data['trie3_god']
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int) -> Iterable[Tuple[int, Optional[int], int]]:
        """
        For equivalence checking only.
        Explodes state bitsets into individual SIDs to match the checker interface.
        """
        node_data = self.arena.get(node, {})
        groups = node_data.get("groups") or []
        for pop, eps_edges, state_edges in groups:
            # Epsilon (no state constraint)
            for dest_idx, llm_bv in eps_edges:
                if llm_bv.contains(token):
                    yield (int(pop), None, int(dest_idx))

            # State-filtered edges
            for dest_idx, state_bv, llm_bv in state_edges:
                if not llm_bv.contains(token):
                    continue
                # state_bv.to_ranges() yields (start, end) and precompute3 iterated range(start, end)
                for start, end in state_bv.to_ranges():
                    for sid in range(int(start), int(end)):
                        yield (int(pop), int(sid), int(dest_idx))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        print("\n--- get_mask START ---")
        state_to_gss = self.constraint_state.filtered_state_gss_map()

        # Local aliases for speed
        gss_merge_many = ffi.gss_merge_many_with_depth
        gss_popn_collect = ffi.gss_popn_collect
        Bitset = ffi.Bitset

        final_mask = Bitset.zeros()
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}
        stopped: set[int] = set()
        todo: Dict[int, set[int]] = defaultdict(set)

        # Seed: map tokenizer states to trie roots; merge into values and schedule
        print("\n--- Seeding work queue ---")
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(int(sid))
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
                merged_gss = gss_merge_many([existing_gss, gss_clone], 1)
                merged_mask = existing_mask.union(new_mask)
                values[root_idx] = (merged_gss, merged_mask)
                print(f"      - Merged result: gss_ptr={merged_gss.ptr()}, mask={merged_mask.to_ranges()}")
            else:
                values[root_idx] = (gss_clone, new_mask)
            depth = self.max_depth.get(root_idx, 0)
            todo[depth].add(root_idx)

        # Main loop; depth-ascending scheduler
        print("\n--- Main loop ---")

        # Main loop; depth-ascending scheduler
        print("\n--- Main loop ---")
        iter_count = 0
        while todo:
            iter_count += 1
            current_depth = min(todo.keys())
            node_indices = todo.pop(current_depth)
            print(f"\n[{iter_count}] Processing depth={current_depth}, nodes={node_indices}")


            for node_idx in list(node_indices):
                if node_idx in stopped:
                    print(f"  - Node {node_idx}: SKIPPING (already stopped)")
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    print(f"  - Node {node_idx}: SKIPPING (no value)")
                    continue
                gss_node, llm_mask = item
                print(f"  - PROCESS: node_ptr={node_idx}, gss_ptr={gss_node.ptr()}, mask={llm_mask.to_ranges()}")

                # If node is an end-node, collect allowed tokens from agg
                if self.is_end(node_idx):
                    print(f"    - END NODE found. Updating final_mask.")
                    print(f"      - final_mask before: {final_mask.to_ranges()}")
                    gss_active_tokens = gss_node.allowed_llm_tokens()
                    tokens_to_add = llm_mask.intersection(gss_active_tokens)
                    print(f"      - glr_active_tokens to union: {tokens_to_add.to_ranges()}")
                    final_mask = final_mask.union(tokens_to_add)
                    print(f"      - final_mask after:  {final_mask.to_ranges()}")

                # If GSS exhausted, stop exploring this node
                if not gss_node.is_ok():
                    stopped.add(node_idx)
                    print(f"    - STOPPING node {node_idx} (GSS not alive)")
                    continue

                node = self.arena.get(node_idx, {})
                groups = node.get("groups") or []

                # Process each pop-group once
                # Process each pop-group once
                for pop, eps_edges, state_edges in groups:
                    print(f"    - Edge group: pop={pop}")
                    # Pop parents from the GSS
                    peeks = gss_popn_collect(gss_node, int(pop))
                    print(f"      - Found {len(peeks)} peeks from GSS")
                    if not peeks:
                        continue


                    # 1) Epsilon edges: no state filter -> all parents match
                    if eps_edges:
                        # Merge all popped parents into a single child GSS
                        only_parents = [parent for _, parent in peeks]
                        if only_parents:
                            eps_child = gss_merge_many(only_parents, 1)
                            if eps_child.is_ok():
                                # Apply llm_bv restrictions per dest and enqueue
                                for dest_idx, llm_bv in eps_edges:
                                    print(f"    - Epsilon Edge: dest={dest_idx}, llm_bv={llm_bv.to_ranges()}")
                                    child_llm_mask = llm_mask.intersection(llm_bv)
                                    print(f"      - Child mask: {child_llm_mask.to_ranges()}")
                                    if child_llm_mask.is_empty():
                                        continue

                                    # Clone and restrict by token bitset if not empty
                                    g = eps_child
                                    if not g.is_alive():
                                        continue

                                    d = int(dest_idx)
                                    existing = values.get(d)
                                    if existing is not None:
                                        existing_gss, existing_mask = existing
                                        print(f"        - Enqueue {d}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={g.ptr()}, mask2={child_llm_mask.to_ranges()}")
                                        combined_gss = gss_merge_many([existing_gss, g], 1)
                                        combined_mask = existing_mask.union(child_llm_mask)
                                        values[d] = (combined_gss, combined_mask)
                                        print(f"          - Merged result: gss_ptr={combined_gss.ptr()}, mask={combined_mask.to_ranges()}")
                                    else:
                                        values[d] = (g, child_llm_mask)
                                        print(f"        - Enqueue {d}: CREATING gss_ptr={g.ptr()}, mask={child_llm_mask.to_ranges()}")

                                    child_depth = self.max_depth.get(d, 0)
                                    todo[child_depth].add(d)

                    # 2) State-filtered edges
                    if state_edges:
                        # Bucket popped parents by parser-state id (sid)
                        sid_to_parents: Dict[int, List[ffi.GSSNode]] = {}
                        for sid_val, parent_node in peeks:
                            s = int(sid_val)
                            bucket = sid_to_parents.get(s)
                            if bucket is None:
                                sid_to_parents[s] = [parent_node]
                            else:
                                bucket.append(parent_node)

                        if not sid_to_parents:
                            continue

                        # For each state-filtered edge, collect matched parents efficiently

                        # For each state-filtered edge, collect matched parents efficiently
                        for dest_idx, state_bv, llm_bv in state_edges:
                            print(f"    - Edge: dest={dest_idx}, state_bv={state_bv.to_ranges()}, llm_bv={llm_bv.to_ranges()}")
                            child_llm_mask = llm_mask.intersection(llm_bv)
                            print(f"      - Child mask: {child_llm_mask.to_ranges()}")
                            if child_llm_mask.is_empty():
                                continue

                            matched_parents: List[ffi.GSSNode] = []
                            # Iterate only sids present among popped parents
                            for s, parents in sid_to_parents.items():
                                if state_bv.contains(s):
                                    matched_parents.extend(parents)

                            print(f"        - Matched {len(matched_parents)} parent GSS nodes")
                            if not matched_parents:
                                continue


                            child = gss_merge_many(matched_parents, 1)
                            if not child.is_alive():
                                continue

                            # Restrict by LLM tokens and enqueue
                            g = child

                            d = int(dest_idx)
                            existing = values.get(d)
                            if existing is not None:
                                existing_gss, existing_mask = existing
                                print(f"        - Enqueue {d}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={g.ptr()}, mask2={child_llm_mask.to_ranges()}")
                                combined_gss = gss_merge_many([existing_gss, g], 1)
                                combined_mask = existing_mask.union(child_llm_mask)
                                values[d] = (combined_gss, combined_mask)
                                print(f"          - Merged result: gss_ptr={combined_gss.ptr()}, mask={combined_mask.to_ranges()}")
                            else:
                                values[d] = (g, child_llm_mask)
                                print(f"        - Enqueue {d}: CREATING gss_ptr={g.ptr()}, mask={child_llm_mask.to_ranges()}")

                            child_depth = self.max_depth.get(d, 0)
                            todo[child_depth].add(d)
        
        print("\n--- get_mask END ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())
                                            child_depth = self.max_depth.get(d, 0)
                                            todo[child_depth].add(d)
                                    else:
                                        values[d] = g
                                        child_depth = self.max_depth.get(d, 0)
                                        todo[child_depth].add(d)

                    # 2) State-filtered edges
                    if state_edges:
                        # Bucket popped parents by parser-state id (sid)
                        sid_to_parents: Dict[int, List[ffi.GSSNode]] = {}
                        for sid_val, parent_node in peeks:
                            s = int(sid_val)
                            bucket = sid_to_parents.get(s)
                            if bucket is None:
                                sid_to_parents[s] = [parent_node]
                            else:
                                bucket.append(parent_node)

                        if not sid_to_parents:
                            continue

                        # For each state-filtered edge, collect matched parents efficiently
                        for dest_idx, state_bv, llm_bv in state_edges:
                            matched_parents: List[ffi.GSSNode] = []
                            # Iterate only sids present among popped parents
                            for s, parents in sid_to_parents.items():
                                if state_bv.contains(s):
                                    matched_parents.extend(parents)

                            if not matched_parents:
                                continue

                            child = gss_merge_many(matched_parents, 1)
                            if not child.is_ok():
                                continue

                            # Restrict by LLM tokens and enqueue
                            g = child.clone_node()
                            if not llm_bv.is_empty():
                                gss_allow_only(g, llm_bv)
                            if not g.is_ok():
                                continue

                            d = int(dest_idx)
                            existing = values.get(d)
                            if existing is not None:
                                combined = gss_merge_many([existing, g], 1)
                                if combined.ptr() != existing.ptr():
                                    values[d] = combined
                                    child_depth = self.max_depth.get(d, 0)
                                    todo[child_depth].add(d)
                            else:
                                values[d] = g
                                child_depth = self.max_depth.get(d, 0)
                                todo[child_depth].add(d)
        
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        return RangeSet.from_ranges(original_mask.to_ranges())
