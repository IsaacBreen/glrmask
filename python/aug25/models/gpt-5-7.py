import json
import heapq
from typing import Dict, List, Tuple, Optional, DefaultDict

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # compiled module with Bitset and GSSNode
from tqdm.auto import tqdm


class Model(GraphProvider):
    """
    Optimized precomputed trie model (generation 5.7).

    Key changes vs precompute3_model:
    - Build a per-node, per-pop index mapping tokenizer state -> [(dest_idx, llm_bv or None)]
      so we can jump directly from a GSS pop to the relevant destinations without scanning
      all dest bitsets.
    - Aggregate contributions over all matched states per (node, pop, dest) before updating
      the scheduler's working set to minimize repeated unions and enqueue operations.
    - Preserve the external interface and iter_edges semantics for equivalence checking.

    Semantics around llm_bv:
    - If an edge group carries an empty llm_bv (bv.is_empty() == True) it is treated as "no filter"
      (i.e., child mask = current llm mask). When multiple groups contribute to a dest/state, the
      resulting behavior is: if any group is "no filter", the union is "no filter"; otherwise we
      intersect with union of all contributing llm_bv.
    """

    # Entry type for fast pop index:
    # For a given node and pop, for a specific tokenizer state (sid),
    # the value is a list of (dest_idx, llm_bv_or_none) where:
    #   - llm_bv_or_none is ffi.Bitset if constrained, or None if unconstrained.
    _StateEntries = Dict[int, List[Tuple[int, Optional[ffi.Bitset]]]]

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict], max_state_id: int):
        # Map tokenizer state -> trie root node
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.max_depth: Dict[int, int] = {}

        # children normalized as in precompute3_model, plus build pop-index for fast lookup
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        # node_id -> (pop -> state -> list[(dest_idx, bv or None)])
        self._pop_index: Dict[int, Dict[int, Model._StateEntries]] = {}

        for uid, node in tqdm(
            self.arena.items(),
            desc="Normalizing gpt-5-7 arena and building pop index",
            total=len(self.arena),
        ):
            uid_int = int(uid)

            # Cache max_depth
            try:
                md = node.get("max_depth", 0)
                self.max_depth[uid_int] = int(md)
            except Exception:
                self.max_depth[uid_int] = 0

            # Normalize children into ffi bitsets (keep original structure for iter_edges)
            children = node.get("children") or []
            if not children:
                node["children"] = []
                self._pop_index[uid_int] = {}
                continue

            new_children = []
            # Build fast index holder for this node
            pop_to_state_entries: Dict[int, Model._StateEntries] = {}

            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                pop = int(pop)
                llm_bv = bs_from_json(dumps(llm_bv_json))

                # "Unconstrained" llm filter is represented by empty bitset as per existing impl
                llm_unconstrained = llm_bv.is_empty()

                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    dest_idx = int(dest_idx)
                    state_bv = bs_from_json(dumps(state_bv_json))
                    new_dest_map.append((dest_idx, state_bv))

                    # Build fast pop index for this (node, pop)
                    # Skip epsilon-on-GSS transitions (empty state_bv) here: the existing get_mask
                    # ignores them (handled only by iter_edges in precompute3_model).
                    if not state_bv.is_empty():
                        # lazily allocate per-pop map
                        s_map = pop_to_state_entries.get(pop)
                        if s_map is None:
                            s_map = {}
                            pop_to_state_entries[pop] = s_map

                        # Fan-out to states in this bitset
                        # to_ranges() yields disjoint intervals [start, end)
                        for start, end in state_bv.to_ranges():
                            # Note: states are small (~1000); this is a fast loop and allows
                            # direct sid lookup during get_mask without per-dest contains() calls.
                            end = min(end, max_state_id + 1)
                            for sid in range(start, end):
                                entries = s_map.get(sid)
                                if entries is None:
                                    entries = []
                                    s_map[sid] = entries
                                # Store None for unconstrained (means "no filter")
                                entries.append((dest_idx, None if llm_unconstrained else llm_bv))

                new_children.append(((pop, llm_bv), new_dest_map))

            node["children"] = new_children
            self._pop_index[uid_int] = pop_to_state_entries

    @staticmethod
    def from_json_string(s: str) -> "Model":
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
        Explode packed transitions into (pop, state_id or None, dest_idx).
        Only used by equivalence checking; not performance-critical.

        This uses the normalized arena children (same as precompute3_model).
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
        GSS nodes. This version uses the per-pop, per-state fast index to avoid
        scanning all destination bitsets on every transition.
        """
        print("\n--- get_mask START ---")
        print(self.constraint_state)
        state_to_gss = self.constraint_state.filtered_state_gss_map()
        print(f"Filtered state_to_gss: { {k: v.ptr() for k, v in state_to_gss.items()} }")

        state_to_gss = self.constraint_state.get_state_map()
        Bitset = ffi.Bitset

        final_mask: ffi.Bitset = Bitset.zeros()

        # node_idx -> (set(GSSNode), Bitset)
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}

        stopped: set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths (may contain duplicates)

        # Seed: map tokenizer states and their filtered GSS to trie roots
        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth = self.max_depth

        print("\n--- Seeding work queue ---")
        for sid, gss in state_to_gss.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            # Note: clone_node and allowed_llm_tokens provided by ffi; fast operations

            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss, existing_mask = existing
                print(f"    - MERGE: gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={gss_clone.ptr()}, mask2={new_mask.to_ranges()}")
                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, gss_clone], 1)
                # Union allowed tokens into the node-level mask
                values[root_idx] = (merged_gss, existing_mask.union(new_mask))
                print(f"      - Merged result: gss_ptr={merged_gss.ptr()}, mask={values[root_idx][1].to_ranges()}")
            else:
                values[root_idx] = (gss_clone, new_mask)


            depth = max_depth[root_idx]
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        heappop = heapq.heappop
        arena = self.arena
        is_end = self.is_end
        pop_index_all = self._pop_index

        print("\n--- Main loop ---")
        iter_count = 0
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

            print(f"\n[{iter_count}] Processing depth={current_depth}, nodes={node_indices}")

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

                # Fast dispatch via precomputed pop-index for this node
                pop_index = pop_index_all.get(node_idx)
                if not pop_index:
                    continue  # no outgoing transitions

                # For each pop group, we collect peeks once and map sids directly to dests
                for pop, state_to_entries in pop_index.items():
                    print(f"    - Edge group: pop={pop}")
                    # Collect all pops from GSS parents
                    peeks = gss_node.popn_fast(pop)
                    print(f"      - Found {len(peeks)} peeks from GSS")
                    if not peeks:
                        continue

                    # Bucket peeks by sid to dedupe membership lookups and aggregate parents
                    sid_to_parents: Dict[int, List] = {}
                    for sid_val, parent_node in peeks:
                        lst = sid_to_parents.get(sid_val)
                        if lst is None:
                            lst = [parent_node]
                            sid_to_parents[sid_val] = lst
                        else:
                            lst.append(parent_node)

                    # Aggregate per-destination contributions across all matched sids
                    # dest_idx -> (parents_list, any_unconstrained, union_llm_bv or None if none seen yet)
                    dest_parents: Dict[int, List] = {}
                    dest_any_unconstrained: Dict[int, bool] = {}
                    dest_llm_union: Dict[int, ffi.Bitset] = {}

                    for sid_val, parents in sid_to_parents.items():
                        entries = state_to_entries.get(sid_val)
                        if not entries:
                            continue
                        # entries: List[(dest_idx, bv_or_none)]
                        for dest_idx, bv in entries:
                            # Accumulate parents per dest
                            plist = dest_parents.get(dest_idx)
                            if plist is None:
                                dest_parents[dest_idx] = parents.copy()
                            else:
                                plist.extend(parents)

                            if bv is None:
                                # Unconstrained filter present for this dest at this state
                                dest_any_unconstrained[dest_idx] = True
                            else:
                                # Union llm_bv across matched states for this dest
                                existing_bv = dest_llm_union.get(dest_idx)
                                if existing_bv is None:
                                    dest_llm_union[dest_idx] = bv
                                else:
                                    dest_llm_union[dest_idx] = existing_bv.union(bv)

                    if not dest_parents:
                        continue

                    # Merge into scheduler state per dest
                    enqueue = self._enqueue_helper(todo, depth_heap)

                    # Merge into scheduler state per dest
                    for d, child_nodes_list in dest_parents.items():
                        print(f"      - Dest: idx={d}")
                        print(f"        - Matched {len(child_nodes_list)} parent GSS nodes")
                        if not child_nodes_list:
                            continue
                        child_gss = ffi.gss_merge_many_with_depth(child_nodes_list, 1)
                        if not child_gss.is_alive():
                            continue

                        # Compute child mask for this dest
                        if dest_any_unconstrained.get(d, False):
                            child_llm_mask = llm_mask
                        else:
                            bv_union = dest_llm_union.get(d)
                            if bv_union is None:
                                child_llm_mask = Bitset.zeros()
                            else:
                                child_llm_mask = llm_mask.intersection(bv_union)

                        existing = values.get(d)
                        if existing is not None:
                            existing_gss, existing_mask = existing
                            print(f"        - Enqueue {d}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={child_gss.ptr()}, mask2={child_llm_mask.to_ranges()}")
                            merged_gss = ffi.gss_merge_many_with_depth([existing_gss, child_gss], 1)

                            # Merge masks unconditionally; correctness requires new tokens to propagate
                            new_mask = existing_mask.union(child_llm_mask)
                            values[d] = (merged_gss, new_mask)

                            # Re-enqueue if either set size grew or new_mask is different (not just identity)
                            if merged_gss.ptr() != existing_gss.ptr() or new_mask != existing_mask:
                                enqueue(max_depth[d], d)
                        else:
                            # Initialize from scratch
                            values[d] = (child_gss, child_llm_mask)
                            print(f"        - Enqueue {d}: CREATING gss_ptr={child_gss.ptr()}, mask={child_llm_mask.to_ranges()}")
                            enqueue(max_depth[d], d)

        print("\n--- get_mask END ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())

    @staticmethod
    def _enqueue_helper(todo: Dict[int, set], depth_heap: List[int]):
        heappush = heapq.heappush

        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        return enqueue

