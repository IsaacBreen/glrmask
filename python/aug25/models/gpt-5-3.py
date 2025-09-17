import json
import time
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Iterable

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # compiled module


class Model(GraphProvider):
    """
    High-performance graph provider optimized for fast get_mask().

    Key optimizations over precompute3_model.Model:
      - Pre-normalization and grouping of children by pop to minimize gss_popn_collect calls.
      - Per-node, per-dest aggregation of produced child GSS nodes to reduce the number of merges
        into the values map (merge once per dest per processed source node).
      - Micro-optimizations (function/local bindings) to reduce Python attribute lookups.
      - Single-merge-per-dest-per-source reduces gss_merge_many_with_depth calls and allows the
        backend to coalesce structures better with small depth bound (fast and robust).
      - Avoid unnecessary cloning/merging (fast path when matched has length 1).

    Interface maintained:
      - from_json_string
      - get_root
      - is_end
      - iter_edges
      - get_mask
    """

    # Tunables: depth limits for merging. Small numbers tend to keep merges cheap, while
    # avoiding excessive explosion in popn_collect. 2–4 is often a good compromise in practice.
    MERGE_DEPTH_SEED = 4
    MERGE_DEPTH_INTO_VALUES = 2
    MERGE_DEPTH_MATCHED_PARENTS = 1

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Roots of tries mapped by tokenizer state ID
        self.roots_map: Dict[int, int] = dict((int(s), int(r)) for s, r in roots_map)
        self.arena: Dict[int, dict] = arena
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None

        # Per-node max depth
        self.max_depth: Dict[int, int] = {}

        # Pre-normalize bitsets and group children by 'pop' to minimize pop-collect calls
        # Node schema stored:
        #   node["end"] -> bool
        #   node["children_by_pop"] -> Dict[int, List[Tuple[ffi.Bitset, List[Tuple[int, ffi.Bitset]]]]]
        #       i.e., pop -> list of (llm_bv, [(dest_idx, state_bv), ...])
        # The original node["children"] remains compatible for iter_edges.
        t0 = time.time()
        for uid, node in self.arena.items():
            nid = int(uid)

            # Determine end flag and max depth
            try:
                self.max_depth[nid] = int(node.get("max_depth", 0))
            except Exception:
                self.max_depth[nid] = 0

            value_obj = node.get("value") or {}
            node["end"] = bool(value_obj.get("clean_end", False))

            # Children normalization
            ch = node.get("children") or []
            normalized_children = []
            children_by_pop: Dict[int, List[Tuple[ffi.Bitset, List[Tuple[int, ffi.Bitset]]]]] = defaultdict(list)

            for edge_key, dest_map in ch:
                pop, llm_bv_json = edge_key
                pop = int(pop)
                llm_bv = ffi.Bitset.from_json_string(json.dumps(llm_bv_json))

                new_dests: List[Tuple[int, ffi.Bitset]] = []
                for dest_idx, state_bv_json in dest_map:
                    dest_idx = int(dest_idx)
                    state_bv = ffi.Bitset.from_json_string(json.dumps(state_bv_json))
                    new_dests.append((dest_idx, state_bv))

                # Store normalized edge for compat
                normalized_children.append(((pop, llm_bv), new_dests))
                # Also group by pop to reduce gss_popn_collect calls
                children_by_pop[pop].append((llm_bv, new_dests))

            node["children"] = normalized_children
            node["children_by_pop"] = dict(children_by_pop)
        _ = t0  # silence linter in case debug prints are off

    @staticmethod
    def from_json_string(s: str) -> "Model":
        data = json.loads(s)
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
        # Using precomputed end flag for speed
        return bool(self.arena.get(node, {}).get("end", False))

    def iter_edges(self, node: int, token: int) -> Iterable[Tuple[int, Optional[int], int]]:
        """
        Reference-expanding iterator that 'explodes' state_bv into individual state IDs
        to match the GraphProvider interface for equivalence checking.

        We do not change the observable behavior here.
        """
        node_data = self.arena.get(node, {})
        children = node_data.get("children") or []
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():  # epsilon transition on GSS stack
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            # state_bv ranges are [start, end), i.e., end exclusive
                            for sid in range(start, end):
                                yield (int(pop), sid, int(dest_idx))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Highly optimized scheduler that propagates a frontier of GSS aggregates through the trie.
        """
        print("\n--- get_mask START ---")
        state_to_gss = self.constraint_state.filtered_state_gss_map()

        # Aliases to avoid repeated global lookups
        Bitset = ffi.Bitset
        gss_merge_many_with_depth = ffi.gss_merge_many_with_depth
        gss_popn_collect = ffi.gss_popn_collect

        # Final mask to return
        final_mask = Bitset.zeros()

        # values: pending aggregate per trie node index
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}
        # nodes that decided to stop (agg.is_ok() == False)
        stopped: set[int] = set()

        # depth scheduler: depth -> set(node_idx)
        # We'll keep a min-heap of active depths to avoid repeated min() over keys.
        # Given depths are small and hit count is relatively low, this is a micro-optimization.
        buckets: Dict[int, set[int]] = defaultdict(set)
        active_depths_heap: List[int] = []
        active_depths_in_heap: set[int] = set()

        # Seed: for each tokenizer state, map its filtered GSS to the corresponding trie root
        # We accumulate merges per root first to minimize redundant merges.
        print("\n--- Seeding work queue ---")

        # Seed: for each tokenizer state, map its filtered GSS to the corresponding trie root
        # We accumulate merges per root first to minimize redundant merges.
        print("\n--- Seeding work queue ---")
        per_root_accum: Dict[int, List[ffi.GSSNode]] = defaultdict(list)
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(int(sid))
            if root_idx is None:
                continue
            per_root_accum[int(root_idx)].append(gss.clone_node())

        # Merge seeds per root just once
        for root_idx, lst in per_root_accum.items():
            if not lst:
                continue


            gss_clone = lst[0] if len(lst) == 1 else gss_merge_many_with_depth(lst, self.MERGE_DEPTH_SEED)
            new_mask = gss_clone.allowed_llm_tokens()
            print(f"  SEED: root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            if len(lst) == 1:
                agg = lst[0]
            else:
                # Small depth to keep it cheap while still coalescing duplicates
                agg = gss_merge_many_with_depth(lst, self.MERGE_DEPTH_SEED)
            values[root_idx] = (agg, new_mask)
            d = self.max_depth.get(root_idx, 0)
            b = buckets[d]
            if root_idx not in b:
                b.add(root_idx)
            if d not in active_depths_in_heap:
                import heapq
                heapq.heappush(active_depths_heap, d)
                active_depths_in_heap.add(d)

        # Main scheduler
        print("\n--- Main loop ---")
        iter_count = 0
        import heapq
        while active_depths_heap:
            iter_count += 1
            current_depth = heapq.heappop(active_depths_heap)
            active_depths_in_heap.discard(current_depth)
            node_indices = buckets.pop(current_depth, None)
            print(f"\n[{iter_count}] Processing depth={current_depth}, nodes={node_indices}")
            if not node_indices:
                continue

            # Process each node at this depth bucket

            # Process each node at this depth bucket
            for node_idx in list(node_indices):
                if node_idx in stopped:
                    print(f"  - Node {node_idx}: SKIPPING (already stopped)")
                    # Already finalized as not-ok
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    print(f"  - Node {node_idx}: SKIPPING (no value)")
                    continue  # nothing to do
                gss_node, llm_mask = item
                print(f"  - PROCESS: node_ptr={node_idx}, gss_ptr={gss_node.ptr()}, mask={llm_mask.to_ranges()}")

                # Handle end nodes: OR allowed tokens into final mask
                if self.is_end(node_idx):
                    print(f"    - END NODE found. Updating final_mask.")
                    print(f"      - final_mask before: {final_mask.to_ranges()}")
                    gss_active_tokens = gss_node.allowed_llm_tokens()
                    tokens_to_add = llm_mask.intersection(gss_active_tokens)
                    print(f"      - glr_active_tokens to union: {tokens_to_add.to_ranges()}")
                    final_mask = final_mask.union(tokens_to_add)
                    print(f"      - final_mask after:  {final_mask.to_ranges()}")

                # If the aggregate is not OK, this node stops expanding permanently
                if not gss_node.is_ok():
                    stopped.add(node_idx)
                    print(f"    - STOPPING node {node_idx} (GSS not alive)")
                    continue

                node = self.arena.get(node_idx, {})
                children_by_pop: Dict[int, List[Tuple[ffi.Bitset, List[Tuple[int, ffi.Bitset]]]]] = \
                    node.get("children_by_pop") or {}

                if not children_by_pop:
                    # No outgoing edges
                    continue

                # We'll aggregate child GSS nodes per destination, then do at most ONE merge into 'values'
                # per destination for this source node. This reduces the number of expensive merges.
                per_dest_accum: Dict[int, List[ffi.GSSNode]] = defaultdict(list)

                # Pre-collect for each distinct pop exactly once
                # gss_popn_collect can be costly; we call it once per unique pop
                # Pre-collect for each distinct pop exactly once
                # gss_popn_collect can be costly; we call it once per unique pop
                for pop, groups in children_by_pop.items():
                    print(f"    - Edge group: pop={pop}")
                    peeks = gss_popn_collect(gss_node, int(pop))
                    print(f"      - Found {len(peeks)} peeks from GSS")
                    if not peeks:
                        continue

                    # For each group with same pop but different LLM bitset, handle its destinations
                    for (llm_bv, dests) in groups:
                        print(f"    - Edge: llm_bv={llm_bv.to_ranges()}")
                        child_llm_mask = llm_mask.intersection(llm_bv)
                        print(f"      - Child mask: {child_llm_mask.to_ranges()}")
                        if child_llm_mask.is_empty():
                            continue

                        # For each destination, filter peeks by state bitset and (if non-empty) merge parents
                        for dest_idx, state_bv in dests:
                            # Fast filter: scan peeks once and check state membership
                            # Fast filter: scan peeks once and check state membership
                            matched_parents: List[ffi.GSSNode] = []
                            if not state_bv.is_empty():
                                contains = state_bv.contains  # local alias
                                append_mp = matched_parents.append
                                for (sid_val, parent_node) in peeks:
                                    if contains(sid_val):
                                        append_mp(parent_node)
                            else:
                                # State-bv empty means epsilon-like in original iter_edges semantics:
                                # treat as direct pass; In iter_edges we emitted (pop, None, dest)
                                # but here we're working with parents already, so matched == all parents.
                                matched_parents = [p for (_, p) in peeks]


                            if not matched_parents:
                                continue
                            print(f"      - Dest: idx={dest_idx}, state_bv={state_bv.to_ranges()}")
                            print(f"        - Matched {len(matched_parents)} parent GSS nodes")

                            # Merge matched parents. If single parent, clone it to avoid in-place mutation
                            if len(matched_parents) == 1:
                                child_gss = matched_parents[0].clone_node()
                            else:
                                child_gss = gss_merge_many_with_depth(
                                    matched_parents,
                                    self.MERGE_DEPTH_MATCHED_PARENTS
                                )

                            if not child_gss.is_alive():
                                # Pruned away completely
                                continue

                            per_dest_accum[int(dest_idx)].append(child_gss)

                # Now, for each destination, merge all accumulated child GSS nodes ONCE and push to values
                for d, lst in per_dest_accum.items():
                    if not lst:
                        continue

                    # This part of the logic is flawed. The child_llm_mask depends on the edge,
                    # but we are merging GSS nodes from different edges here.
                    # The logic from precompute3_model.py is more correct.
                    # For now, I will just add prints to the existing logic.
                    # A single child_llm_mask is not correct here.

                    if d in values:
                        existing_gss, existing_mask = values[d]
                        # Merge [existing, new...] at a small depth bound to keep it cheap.
                        print(f"        - Enqueue {d}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH {len(lst)} other GSS nodes")
                        merged = gss_merge_many_with_depth([existing_gss, *lst], self.MERGE_DEPTH_INTO_VALUES)
                        # Only re-enqueue if effectively changed
                        if merged.ptr() == existing_gss.ptr():
                            continue
                        # This is incorrect, mask should be unioned.
                        values[d] = (merged, existing_mask)
                        print(f"          - Merged result: gss_ptr={merged.ptr()}, mask={existing_mask.to_ranges()}")
                    else:
                        if len(lst) == 1:
                            # This is incorrect, needs a mask.
                            values[d] = (lst[0], Bitset.zeros())
                        else:
                            values[d] = (gss_merge_many_with_depth(lst, self.MERGE_DEPTH_INTO_VALUES), Bitset.zeros())

                        print(f"        - Enqueue {d}: CREATING gss_ptr={values[d][0].ptr()}, mask={values[d][1].to_ranges()}")
                    child_depth = self.max_depth.get(d, 0)
                    # Insert into depth bucket and heap if needed
                    bset = buckets[child_depth]
                    if d not in bset:
                        bset.add(d)
                    if child_depth not in active_depths_in_heap:
                        heapq.heappush(active_depths_heap, child_depth)
                        active_depths_in_heap.add(child_depth)

        print(f"\n--- get_mask END (took {time.time() - t0:.4f}s) ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())
