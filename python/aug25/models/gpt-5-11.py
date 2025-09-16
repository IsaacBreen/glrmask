import json
import heapq
from typing import Dict, List, Tuple, Optional, Set

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
from tqdm import tqdm


class _ArcBundle:
    """
    Compact storage for a set of arcs: parallel arrays for dests and masks.
    masks[i] is either an ffi.Bitset or None meaning "no restriction".
    """

    __slots__ = ("dests", "masks", "length")

    def __init__(self, dests: List[int], masks: List[Optional[ffi.Bitset]]):
        self.dests = dests
        self.masks = masks
        self.length = len(dests)


class _PopGroup:
    """
    Transitions of a node grouped by a single pop value.
    sid_to_arcs: maps a tokenizer state id (sid) to an _ArcBundle
    eps_arcs: _ArcBundle of arcs that do not restrict by SID (epsilon on state filter)
    """

    __slots__ = ("pop", "sid_to_arcs", "eps_arcs")

    def __init__(
        self,
        pop: int,
        sid_to_arcs: Dict[int, _ArcBundle],
        eps_arcs: _ArcBundle,
    ):
        self.pop = int(pop)
        self.sid_to_arcs = sid_to_arcs
        self.eps_arcs = eps_arcs


class _NodeData:
    """
    Preprocessed node for fast traversal.
    groups: list of _PopGroup
    """

    __slots__ = ("end_flag", "groups", "max_depth")

    def __init__(self, end_flag: bool, groups: List[_PopGroup], max_depth: int):
        self.end_flag = bool(end_flag)
        self.groups = groups
        self.max_depth = int(max_depth)


class Model(GraphProvider):
    """
    High-performance model (gpt-5-11):

    Key ideas:
    - Preprocess children transitions into per-pop lookup from SID -> ArcBundle
      plus epsilon-on-state arcs, like gpt-5-9, but using cache-friendly parallel
      arrays and storing groups in lists (not dicts) for faster iteration.
    - Aggressively merge parallel edges during init: per (pop, SID/dest) we union
      LLM masks so runtime work is purely routing and set/mask unions.
    - Runtime scheduler mirrors prior fast models but with fewer dict/list
      allocations and faster inner loops.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict], max_state_id: int):
        # Map tokenizer state -> trie root node
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None

        # Normalize arena and build fast structures
        self.nodes: Dict[int, _NodeData] = {}
        self.max_depth: Dict[int, int] = {}

        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        for uid, node in tqdm(arena.items(), desc="Building GPT-5-11 model"):
            uid_int = int(uid)

            # Depth cache
            try:
                md = node.get("max_depth", 0)
                md_int = int(md)
            except Exception:
                md_int = 0
            self.max_depth[uid_int] = md_int

            # End flag
            end_flag = bool((node.get("value") or {}).get("end", False))

            children = node.get("children") or []
            if not children:
                self.nodes[uid_int] = _NodeData(end_flag=end_flag, groups=[], max_depth=md_int)
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
            groups: List[_PopGroup] = []
            for pop_val, entry in pop_acc.items():
                sid_map: Dict[int, Dict[int, Optional[ffi.Bitset]]] = entry["sid_map"]
                eps_map: Dict[int, Optional[ffi.Bitset]] = entry["eps_map"]

                # Convert sid_map values to ArcBundles
                sid_to_arcs: Dict[int, _ArcBundle] = {}
                for sid_val, dest_map in sid_map.items():
                    # dest_map: dest_idx -> llm_bv_or_None
                    # Convert to ArcBundle of (dests, masks)
                    # We keep unsorted; order not required at runtime.
                    dests: List[int] = []
                    masks: List[Optional[ffi.Bitset]] = []
                    for d, bv in dest_map.items():
                        dests.append(int(d))
                        masks.append(bv)
                    sid_to_arcs[int(sid_val)] = _ArcBundle(dests=dests, masks=masks)

                # Epsilon arcs to ArcBundle
                if eps_map:
                    eps_dests: List[int] = []
                    eps_masks: List[Optional[ffi.Bitset]] = []
                    for d, bv in eps_map.items():
                        eps_dests.append(int(d))
                        eps_masks.append(bv)
                    eps_bundle = _ArcBundle(dests=eps_dests, masks=eps_masks)
                else:
                    eps_bundle = _ArcBundle(dests=[], masks=[])

                groups.append(_PopGroup(pop=int(pop_val), sid_to_arcs=sid_to_arcs, eps_arcs=eps_bundle))

            # Store
            self.nodes[uid_int] = _NodeData(end_flag=end_flag, groups=groups, max_depth=md_int)

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        max_state_id = int(max(dict(data["parser"]["stage_7_table"]).keys()))
        model = Model(roots_map, arena, max_state_id)
        constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(constraint)
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
        for group in nd.groups:
            # Epsilon arcs
            eps = group.eps_arcs
            for i in range(eps.length):
                dest_idx = eps.dests[i]
                llm_bv = eps.masks[i]
                if (llm_bv is None) or llm_bv.contains(t):
                    yield (group.pop, None, int(dest_idx))
            # SID-filtered arcs
            for sid_val, bundle in group.sid_to_arcs.items():
                dests = bundle.dests
                masks = bundle.masks
                for i in range(bundle.length):
                    dest_idx = dests[i]
                    llm_bv = masks[i]
                    if (llm_bv is None) or llm_bv.contains(t):
                        yield (group.pop, int(sid_val), int(dest_idx))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. Optimized to avoid per-destination filtering by using
        precomputed SID->ArcBundle mapping per pop group and cache-friendly data layout.
        """
        state_to_gss = self.constraint_state.get_state_map()
        final_mask = ffi.Bitset.zeros()

        # node_idx -> (set(GSSNode), Bitset)
        values: Dict[int, Tuple[Set[ffi.GSSNode], ffi.Bitset]] = {}

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
        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth_map = self.max_depth

        for sid, gss in state_to_gss.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            # If there are no allowed tokens at the seed, no path can produce tokens later.
            if new_mask.is_empty():
                continue

            existing = values.get(root_idx)
            if existing is not None:
                gss_set, existing_mask = existing
                gss_set.add(gss_clone)
                values[root_idx] = (gss_set, existing_mask.union(new_mask))
            else:
                values[root_idx] = ({gss_clone}, new_mask)

            depth = max_depth_map.get(root_idx, 0)
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        heappop = heapq.heappop
        nodes = self.nodes

        while True:
            # Pop the smallest depth bucket (skip stale heap entries)
            node_indices: Optional[Set[int]] = None
            while depth_heap:
                current_depth = heappop(depth_heap)
                node_indices = todo.pop(current_depth, None)
                if node_indices:
                    break
            if not node_indices:
                break  # nothing left to process

            # Process all nodes in this depth bucket
            for node_idx in node_indices:
                item = values.pop(node_idx, None)
                if item is None:
                    continue
                gss_set, llm_mask = item

                # End-node handling
                nd = nodes.get(node_idx)
                if nd is None:
                    continue
                if nd.end_flag:
                    final_mask = final_mask.union(llm_mask)

                if not gss_set:
                    continue

                groups = nd.groups
                if not groups:
                    continue

                # For every pop group of this node, do one GSS pop and route by SID using precomputed arcs
                for group in groups:
                    pop_val = group.pop

                    # Collect all parents from GSS after popping 'pop_val'
                    # Note: We build a simple list, then group by SID only once.
                    peeks: List[Tuple[int, ffi.GSSNode]] = []
                    for g in gss_set:
                        peeks.extend(g.popn_fast(pop_val))
                    if not peeks:
                        continue

                    # Prepare accumulators for next nodes: dest -> (gss_set, child_llm_mask)
                    next_gss: Dict[int, Set[ffi.GSSNode]] = {}
                    next_mask: Dict[int, ffi.Bitset] = {}

                    # Cache intersections of llm_mask with edge llm_bv
                    # Key: id(llm_bv) or None; Value: Bitset
                    inter_cache: Dict[Optional[int], ffi.Bitset] = {}
                    inter_cache[None] = llm_mask  # None means "no restriction"

                    # If epsilon arcs exist, we need the set of all parent nodes
                    eps = group.eps_arcs
                    if eps.length:
                        parents_all = set()
                        for _, parent_node in peeks:
                            parents_all.add(parent_node)

                        # Process epsilon arcs (no SID filter)
                        eps_dests = eps.dests
                        eps_masks = eps.masks
                        for i in range(eps.length):
                            dest_idx = eps_dests[i]
                            llm_bv = eps_masks[i]
                            if llm_bv is None:
                                child_mask = inter_cache[None]
                            else:
                                key = id(llm_bv)
                                child_mask = inter_cache.get(key)
                                if child_mask is None:
                                    child_mask = llm_mask.intersection(llm_bv)
                                    inter_cache[key] = child_mask

                            # Merge parents
                            s = next_gss.get(dest_idx)
                            if s is None:
                                s = set()
                                next_gss[dest_idx] = s
                            s.update(parents_all)

                            # Merge mask
                            m = next_mask.get(dest_idx)
                            if m is None:
                                next_mask[dest_idx] = child_mask
                            else:
                                next_mask[dest_idx] = m.union(child_mask)

                    # Group peeks by SID (parents kept as lists; sets are built only at merge time)
                    sid_to_parents: Dict[int, List[ffi.GSSNode]] = {}
                    for sid_val, parent_node in peeks:
                        lst = sid_to_parents.get(sid_val)
                        if lst is None:
                            sid_to_parents[sid_val] = [parent_node]
                        else:
                            lst.append(parent_node)

                    # Process SID-filtered arcs
                    sid_to_arcs = group.sid_to_arcs
                    for sid_val, parents_list in sid_to_parents.items():
                        bundle = sid_to_arcs.get(sid_val)
                        if not bundle:
                            continue

                        dests = bundle.dests
                        masks = bundle.masks
                        for i in range(bundle.length):
                            dest_idx = dests[i]
                            llm_bv = masks[i]

                            # Child mask
                            if llm_bv is None:
                                child_mask = inter_cache[None]
                            else:
                                key = id(llm_bv)
                                child_mask = inter_cache.get(key)
                                if child_mask is None:
                                    child_mask = llm_mask.intersection(llm_bv)
                                    inter_cache[key] = child_mask

                            # Merge parents
                            s = next_gss.get(dest_idx)
                            if s is None:
                                s = set()
                                next_gss[dest_idx] = s
                            s.update(parents_list)  # set dedups automatically

                            # Merge mask
                            m = next_mask.get(dest_idx)
                            if m is None:
                                next_mask[dest_idx] = child_mask
                            else:
                                next_mask[dest_idx] = m.union(child_mask)

                    # Flush to values and enqueue
                    for d, parent_set in next_gss.items():
                        existing = values.get(d)
                        depth_d = nodes.get(d).max_depth if d in nodes else 0
                        if existing is not None:
                            existing_gss_set, existing_mask = existing
                            old_len = len(existing_gss_set)
                            existing_gss_set.update(parent_set)
                            combined_mask = existing_mask.union(next_mask[d])
                            values[d] = (existing_gss_set, combined_mask)
                            # Only re-enqueue if gss_set actually changed
                            if len(existing_gss_set) != old_len:
                                enqueue(depth_d, d)
                        else:
                            values[d] = (parent_set, next_mask[d])
                            enqueue(depth_d, d)

        return RangeSet.from_ranges(final_mask.to_ranges())
