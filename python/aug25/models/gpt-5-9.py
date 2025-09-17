import json
import heapq
import time
from typing import Dict, List, Tuple, Optional, Iterable, Set

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
from tqdm import tqdm

class _PopGroup:
    """
    Transitions of a node grouped by a single pop value.

    - sid_to_arcs: maps a tokenizer state id (sid) to a list of arcs:
        [(dest_idx, llm_bv_or_None), ...]
      where llm_bv_or_None is None meaning "no restriction (all tokens)".
    - eps_arcs: list of arcs that do not restrict by SID (epsilon on state filter):
        [(dest_idx, llm_bv_or_None), ...]
    """

    __slots__ = ("pop", "sid_to_arcs", "eps_arcs")

    def __init__(
        self,
        pop: int,
        sid_to_arcs: Dict[int, List[Tuple[int, Optional[ffi.Bitset]]]],
        eps_arcs: List[Tuple[int, Optional[ffi.Bitset]]],
    ):
        self.pop = int(pop)
        self.sid_to_arcs = sid_to_arcs
        self.eps_arcs = eps_arcs


class _NodeData:
    """
    Preprocessed node for fast traversal.
    """

    __slots__ = ("end_flag", "groups", "max_depth")

    def __init__(self, end_flag: bool, groups: Dict[int, _PopGroup], max_depth: int):
        self.end_flag = bool(end_flag)
        self.groups = groups  # pop -> _PopGroup
        self.max_depth = int(max_depth)


class Model(GraphProvider):
    """
    High-performance model (gpt-5-9):
    - Preprocess children transitions into per-pop lookup from SID -> [(dest, LLM-BV or None)]
      plus epsilon-on-state arcs.
    - At runtime, for each (node, pop), group GSS peeks by SID and map them directly to dests.
    - Significantly reduces the "filter-by-dest" inner loop work.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict], max_state_id: int):
        # Map tokenizer state -> trie root node
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None

        # Normalize arena and build fast structures
        self.nodes: Dict[int, _NodeData] = {}
        self.max_depth: Dict[int, int] = {}

        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        for uid, node in tqdm(arena.items()):
            uid_int = int(uid)

            # Depth cache
            try:
                md = node.get("max_depth", 0)
                md_int = int(md)
            except Exception:
                md_int = 0
            self.max_depth[uid_int] = md_int

            # End flag
            end_flag = bool((node.get("value") or {}).get("clean_end", False))

            children = node.get("children") or []
            if not children:
                self.nodes[uid_int] = _NodeData(end_flag=end_flag, groups={}, max_depth=md_int)
                continue

            # Aggregate by pop
            # pop -> {
            #     "sid_map": Dict[int, Dict[int, Optional[ffi.Bitset]]],  # sid -> dest_idx -> llm_bv_or_None
            #     "eps_map": Dict[int, Optional[ffi.Bitset]],              # dest_idx -> llm_bv_or_None
            # }
            pop_acc: Dict[int, Dict[str, dict]] = {}

            for edge_key, dest_map in children:
                pop_val, llm_bv_json = edge_key
                pop_val = int(pop_val)

                llm_bv = bs_from_json(dumps(llm_bv_json))
                # Convention: None means "no restriction" (all tokens allowed)
                llm_mask: Optional[ffi.Bitset] = None if llm_bv.is_empty() else llm_bv

                entry = pop_acc.get(pop_val)
                if entry is None:
                    entry = {"sid_map": {}, "eps_map": {}}
                    pop_acc[pop_val] = entry
                sid_map: Dict[int, Dict[int, Optional[ffi.Bitset]]] = entry["sid_map"]
                eps_map: Dict[int, Optional[ffi.Bitset]] = entry["eps_map"]

                # For each destination, update its SID constraints
                for dest_idx, state_bv_json in dest_map:
                    dest_idx = int(dest_idx)
                    state_bv = bs_from_json(dumps(state_bv_json))

                    # Empty state BV: treat as "epsilon on state filter" edge
                    # It applies regardless of SID membership (after popping).
                    if state_bv.is_empty():
                        existing_eps = eps_map.get(dest_idx)
                        if existing_eps is None:
                            # No prior restriction or prior was "unrestricted"
                            eps_map[dest_idx] = None if llm_mask is None else llm_mask
                        else:
                            # Merge llm masks
                            if existing_eps is None or llm_mask is None:
                                eps_map[dest_idx] = None
                            else:
                                eps_map[dest_idx] = existing_eps.union(llm_mask)
                        continue

                    # Non-empty state BV: expand its ranges to SIDs and store per SID
                    to_ranges = state_bv.to_ranges
                    for start, end in to_ranges():
                        # [start, end)
                        # For each SID in the range, union the llm mask into that dest
                        end = min(int(end), max_state_id + 1)
                        for sid_val in range(start, end):
                            by_dest = sid_map.get(sid_val)
                            if by_dest is None:
                                by_dest = {}
                                sid_map[sid_val] = by_dest
                            prev = by_dest.get(dest_idx)
                            if prev is None:
                                # None means "unrestricted"
                                by_dest[dest_idx] = None if llm_mask is None else llm_mask
                            else:
                                # Merge restrictions
                                if prev is None or llm_mask is None:
                                    by_dest[dest_idx] = None
                                else:
                                    by_dest[dest_idx] = prev.union(llm_mask)

            # Convert accumulators into compact runtime structures
            groups: Dict[int, _PopGroup] = {}
            for pop_val, entry in pop_acc.items():
                sid_map: Dict[int, Dict[int, Optional[ffi.Bitset]]] = entry["sid_map"]
                eps_map: Dict[int, Optional[ffi.Bitset]] = entry["eps_map"]

                # Convert sid_map values to lists for fast iteration
                sid_to_arcs: Dict[int, List[Tuple[int, Optional[ffi.Bitset]]]] = {}
                for sid_val, dest_map in sid_map.items():
                    # dest_map: dest_idx -> llm_bv_or_None
                    # Convert to list of (dest_idx, llm_bv_or_None)
                    # We keep as-is; sorted order is not necessary at runtime.
                    arcs_list = [(d, bv) for d, bv in dest_map.items()]
                    sid_to_arcs[int(sid_val)] = arcs_list

                # Epsilon arcs to list
                eps_arcs: List[Tuple[int, Optional[ffi.Bitset]]] = [(d, bv) for d, bv in eps_map.items()]

                groups[int(pop_val)] = _PopGroup(pop=int(pop_val), sid_to_arcs=sid_to_arcs, eps_arcs=eps_arcs)

            self.nodes[uid_int] = _NodeData(end_flag=end_flag, groups=groups, max_depth=md_int)

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
        nd = self.nodes.get(int(node))
        if nd is None:
            return False
        return nd.end_flag

    def iter_edges(self, node: int, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx), filtered by token.
        Only used by equivalence checking; not performance-critical.
        """
        nd = self.nodes.get(int(node))
        if not nd or not nd.groups:
            return
        t = int(token)
        for pop_val, group in nd.groups.items():
            # Epsilon arcs
            for dest_idx, llm_bv in group.eps_arcs:
                if (llm_bv is None) or llm_bv.contains(t):
                    yield (int(pop_val), None, int(dest_idx))
            # SID-filtered arcs
            for sid_val, arcs in group.sid_to_arcs.items():
                for dest_idx, llm_bv in arcs:
                    if (llm_bv is None) or llm_bv.contains(t):
                        yield (int(pop_val), int(sid_val), int(dest_idx))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. Optimized to avoid per-destination filtering by using
        precomputed SID->arcs mapping per pop group.
        """
        print("\n--- get_mask START ---")
        state_to_gss = self.constraint_state.filtered_state_gss_map()

        t0 = time.time()
        final_mask = ffi.Bitset.zeros()

        # node_idx -> (set(GSSNode), Bitset)
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}

        stopped: Set[int] = set()               # nodes that stopped (no gss parents)
        todo: Dict[int, Set[int]] = {}          # depth -> set(node_idx)
        depth_heap: List[int] = []              # min-heap of depths (may contain duplicates)

        # Helper to enqueue a node at a given depth
        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heapq.heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        # Seed: map tokenizer states and their filtered GSS to trie roots
        print("\n--- Seeding work queue ---")
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
            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss, existing_mask = existing
                print(f"    - MERGE: gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={gss_clone.ptr()}, mask2={new_mask.to_ranges()}")
                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, gss_clone], 1)
                merged_mask = existing_mask.union(new_mask)
                values[root_idx] = (merged_gss, merged_mask)
                print(f"      - Merged result: gss_ptr={merged_gss.ptr()}, mask={merged_mask.to_ranges()}")
            else:
                values[root_idx] = (gss_clone, new_mask)

            depth = max_depth.get(root_idx, 0)
            if todo.get(depth) is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                todo[depth].add(root_idx)

        heappop = heapq.heappop
        nodes = self.nodes

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

                # End-node handling
                if nodes.get(node_idx, None) and nodes[node_idx].end_flag:
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

                nd = nodes.get(node_idx)
                if nd is None or not nd.groups:
                    continue

                # For every pop group of this node, do one GSS pop and route by SID using precomputed arcs
                for pop_val, group in nd.groups.items():
                    print(f"    - Edge group: pop={pop_val}")
                    # Collect all parents from GSS after popping 'pop_val'
                    peeks = gss_node.popn_fast(pop_val)
                    print(f"      - Found {len(peeks)} peeks from GSS")
                    if not peeks:
                        continue

                    sid_to_parents: Dict[int, List[ffi.GSSNode]] = {}
                    eps_parents: Optional[List[ffi.GSSNode]] = None

                    # If epsilon arcs exist, we need the set of all parent nodes
                    if group.eps_arcs:
                        # Construct set of parent nodes from peeks
                        eps_parents = []
                        for sid_val, parent_node in peeks:
                            eps_parents.add(parent_node)

                    # Group peeks by SID (parents kept as lists; sets are built only at merge time)
                    for sid_val, parent_node in peeks:
                        lst = sid_to_parents.get(sid_val)
                        if lst is None:
                            lst = [parent_node]
                            sid_to_parents[sid_val] = lst
                        else:
                            lst.append(parent_node)

                    # Prepare accumulators for next nodes: dest -> (gss_set, child_llm_mask)
                    next_gss: Dict[int, List[ffi.GSSNode]] = {}
                    next_mask: Dict[int, ffi.Bitset] = {}

                    # Cache intersections of llm_mask with edge llm_bv
                    # Key: id(llm_bv) or None; Value: Bitset
                    inter_cache: Dict[Optional[int], ffi.Bitset] = {}
                    inter_cache[None] = llm_mask  # None means "no restriction"

                    # Process epsilon arcs (no SID filter)
                    if eps_parents:
                        parents_all = eps_parents
                        for dest_idx, llm_bv in group.eps_arcs:
                            # Child mask = llm_mask (no restriction) OR intersection with llm_bv
                            # print(f"        - Epsilon Edge: dest={dest_idx}, llm_bv={'None' if llm_bv is None else llm_bv.to_ranges()}")

                            if llm_bv is None:
                                child_mask = inter_cache[None]
                            else:
                                key = id(llm_bv)
                                child_mask = inter_cache.get(key)
                                if child_mask is None:
                                    child_mask = llm_mask.intersection(llm_bv)
                                    inter_cache[key] = child_mask
                            # print(f"          - Child mask: {child_mask.to_ranges()}")

                            # Merge parents
                            s = next_gss.get(dest_idx)
                            if s is None:
                                s = []
                                next_gss[dest_idx] = s
                            s.extend(parents_all)

                            # Merge mask
                            m = next_mask.get(dest_idx)
                            if m is None:
                                next_mask[dest_idx] = child_mask
                            else:
                                next_mask[dest_idx] = m.union(child_mask)

                    # Process SID-filtered arcs
                    sid_to_arcs = group.sid_to_arcs
                    for sid_val, parents_list in sid_to_parents.items():
                        # print(f"      - Peek group: sid={sid_val}, num_parents={len(parents_list)}")
                        arcs = sid_to_arcs.get(sid_val)
                        if not arcs:
                            continue

                        for dest_idx, llm_bv in arcs:
                            # Child mask
                            if llm_bv is None:
                                child_mask = inter_cache[None]
                            else:
                                key = id(llm_bv)
                                child_mask = inter_cache.get(key)
                                if child_mask is None:
                                    child_mask = llm_mask.intersection(llm_bv)
                                    inter_cache[key] = child_mask
                                # print(f"        - Edge: dest={dest_idx}, llm_bv={'None' if llm_bv is None else llm_bv.to_ranges()}")
                                # print(f"          - Child mask: {child_mask.to_ranges()}")

                            # Merge parents
                            s = next_gss.get(dest_idx)
                            if s is None:
                                s = []
                                next_gss[dest_idx] = s
                            s.extend(parents_list)  # set dedups automatically

                            # Merge mask
                            m = next_mask.get(dest_idx)
                            if m is None:
                                next_mask[dest_idx] = child_mask
                            else:
                                next_mask[dest_idx] = m.union(child_mask)

                    # Flush to values and enqueue
                    for d, parent_list in next_gss.items():
                        print(f"      - Dest: idx={d}")
                        print(f"        - Matched {len(parent_list)} parent GSS nodes")
                        if not parent_list:
                            continue
                        child_gss = ffi.gss_merge_many_with_depth(parent_list, 1)
                        if not child_gss.is_alive():
                            continue

                        existing = values.get(d)
                        if existing is not None:
                            existing_gss, existing_mask = existing
                            print(f"        - Enqueue {d}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={child_gss.ptr()}, mask2={next_mask[d].to_ranges()}")
                            merged_gss = ffi.gss_merge_many_with_depth([existing_gss, child_gss], 1)
                            combined_mask = existing_mask.union(next_mask[d])
                            values[d] = (merged_gss, combined_mask)
                            print(f"          - Merged result: gss_ptr={merged_gss.ptr()}, mask={combined_mask.to_ranges()}")
                            # Only re-enqueue if gss_set actually changed
                            if merged_gss.ptr() != existing_gss.ptr():
                                enqueue(self.max_depth.get(d, 0), d)
                        else:
                            values[d] = (child_gss, next_mask[d])
                            print(f"        - Enqueue {d}: CREATING gss_ptr={child_gss.ptr()}, mask={next_mask[d].to_ranges()}")
                            enqueue(self.max_depth.get(d, 0), d)

        print(f"\n--- get_mask END (took {time.time() - t0:.4f}s) ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())
