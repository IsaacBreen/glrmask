import json
import heapq
from typing import Dict, List, Tuple, Optional, Set, Iterable

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

    Improvements over GPT-5-11:
    - sid_to_bundle: maps a tokenizer state id (sid) to a deduplicated _ArcBundle.
      Many SIDs often share the exact same arcs; by canonicalizing and reusing
      bundles we reduce memory and dramatically cut runtime work by grouping peeks
      by bundle (not only by SID).
    - eps_bundle: _ArcBundle of arcs that do not restrict by SID (epsilon on state filter).
    """

    __slots__ = ("pop", "sid_to_bundle", "eps_bundle")

    def __init__(
        self,
        pop: int,
        sid_to_bundle: Dict[int, _ArcBundle],
        eps_bundle: _ArcBundle,
    ):
        self.pop = int(pop)
        self.sid_to_bundle = sid_to_bundle
        self.eps_bundle = eps_bundle


class _NodeData:
    """
    Preprocessed node for fast traversal.
    groups: list of _PopGroup (list for faster iteration order and locality).
    """

    __slots__ = ("end_flag", "groups", "max_depth")

    def __init__(self, end_flag: bool, groups: List[_PopGroup], max_depth: int):
        self.end_flag = bool(end_flag)
        self.groups = groups
        self.max_depth = int(max_depth)


class Model(GraphProvider):
    """
    High-performance model (gpt-5-12):

    Builds on gpt-5-11 with key enhancements to reduce 99th percentile and max times:
    - Deduplicate per-SID arc lists into shared ArcBundles, then during get_mask
      group peeks by ArcBundle instead of by SID. This eliminates repeated
      per-destination work across many SIDs that share the same transitions.
    - Retains cache-friendly parallel arrays for arcs (dests, masks) and a compact
      group list per node.
    - Uses intersection cache keyed by edge llm mask object identity to minimize
      repeated llm_mask intersections within a step.

    Interface remains the same and iter_edges is faithful to the original graph.
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

        for uid, node in tqdm(arena.items(), desc="Building GPT-5-12 model"):
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
                            # Merge llm masks: any None makes it unrestricted
                            if existing_eps is None or llm_mask is None:
                                eps_map[dest_idx] = None
                            else:
                                eps_map[dest_idx] = existing_eps.union(llm_mask)
                        continue

                    # Non-empty state BV: expand its ranges to SIDs and store per SID
                    to_ranges = state_bv.to_ranges
                    for start, end in to_ranges():
                        # [start, end)
                        end = min(int(end), max_state_id + 1)
                        for sid_val in range(int(start), end):
                            by_dest = sid_map.get(sid_val)
                            if by_dest is None:
                                by_dest = {}
                                sid_map[sid_val] = by_dest
                            prev = by_dest.get(dest_idx)
                            if prev is None:
                                # None means "unrestricted"
                                by_dest[dest_idx] = None if llm_mask is None else llm_mask
                            else:
                                # Merge restrictions: any None becomes unrestricted
                                if prev is None or llm_mask is None:
                                    by_dest[dest_idx] = None
                                else:
                                    by_dest[dest_idx] = prev.union(llm_mask)

            # Convert accumulators into compact runtime structures with bundle deduplication
            groups: List[_PopGroup] = []
            for pop_val, entry in pop_acc.items():
                sid_map: Dict[int, Dict[int, Optional[ffi.Bitset]]] = entry["sid_map"]
                eps_map: Dict[int, Optional[ffi.Bitset]] = entry["eps_map"]

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

                # Deduplicate sid -> ArcBundle by canonicalizing dest->mask mappings
                # Canonical key: tuple(sorted((dest, mask_key))), mask_key is None or bitset.to_json_string()
                mask_json_cache: Dict[int, str] = {}
                bundle_cache: Dict[Tuple[Tuple[int, Optional[str]], ...], _ArcBundle] = {}

                def mask_key(b: Optional[ffi.Bitset]) -> Optional[str]:
                    if b is None:
                        return None
                    bid = id(b)
                    s = mask_json_cache.get(bid)
                    if s is None:
                        # Note: This runs only at build time; using json string ensures content-dedup
                        s = b.to_json_string()
                        mask_json_cache[bid] = s
                    return s

                sid_to_bundle: Dict[int, _ArcBundle] = {}
                for sid_val, dest_map in sid_map.items():
                    # Build canonical key
                    items: List[Tuple[int, Optional[str]]] = []
                    for d, bv in dest_map.items():
                        items.append((int(d), mask_key(bv)))
                    items.sort(key=lambda x: x[0])
                    key = tuple(items)

                    bundle = bundle_cache.get(key)
                    if bundle is None:
                        # Materialize ArcBundle with actual objects
                        dests: List[int] = []
                        masks: List[Optional[ffi.Bitset]] = []
                        for d, mk in items:
                            dests.append(d)
                            # Rebuild bv by reverse lookup: we have mk (string) but need original object
                            # We can find bv by reading from dest_map again since items were created from it
                            bv = dest_map[d]
                            masks.append(bv)
                        bundle = _ArcBundle(dests=dests, masks=masks)
                        bundle_cache[key] = bundle

                    sid_to_bundle[int(sid_val)] = bundle

                groups.append(_PopGroup(pop=int(pop_val), sid_to_bundle=sid_to_bundle, eps_bundle=eps_bundle))

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

    def iter_edges(self, node: int, token: int) -> Iterable[Tuple[int, Optional[int], int]]:
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
            eps = group.eps_bundle
            for i in range(eps.length):
                dest_idx = eps.dests[i]
                llm_bv = eps.masks[i]
                if (llm_bv is None) or llm_bv.contains(t):
                    yield (group.pop, None, int(dest_idx))
            # SID-filtered arcs
            for sid_val, bundle in group.sid_to_bundle.items():
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
        precomputed SID->ArcBundle mapping per pop group and grouping peeks by
        ArcBundle to minimize repeated work.
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
        heappop = heapq.heappop
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

                # For every pop group of this node, do one GSS pop and route by SID using precomputed bundles
                for group in groups:
                    pop_val = group.pop

                    # Collect all parents from GSS after popping 'pop_val'
                    # peeks: list of (sid_val, parent_node)
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

                    # Process epsilon arcs (no SID filter) first if any
                    eps = group.eps_bundle
                    if eps.length:
                        # Build set of all parent nodes once
                        parents_all = set()
                        for _, parent_node in peeks:
                            parents_all.add(parent_node)

                        eps_dests = eps.dests
                        eps_masks = eps.masks
                        # Merge epsilon arcs
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

                    # Group peeks by ArcBundle (dedup across SIDs sharing identical arcs)
                    sid_to_bundle = group.sid_to_bundle
                    bundle_to_parents: Dict[_ArcBundle, List[ffi.GSSNode]] = {}
                    for sid_val, parent_node in peeks:
                        bundle = sid_to_bundle.get(sid_val)
                        if bundle is None:
                            continue
                        lst = bundle_to_parents.get(bundle)
                        if lst is None:
                            bundle_to_parents[bundle] = [parent_node]
                        else:
                            lst.append(parent_node)

                    if not bundle_to_parents:
                        # No SID-specific arcs matched
                        pass
                    else:
                        # Process each unique bundle once
                        for bundle, parents_list in bundle_to_parents.items():
                            # Build set of parents for this bundle (dedups)
                            parents_set = set(parents_list)

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
                                s.update(parents_set)  # set dedups automatically

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
                            # Only re-enqueue if gss_set actually changed (fastest policy observed)
                            if len(existing_gss_set) != old_len:
                                enqueue(depth_d, d)
                        else:
                            values[d] = (parent_set, next_mask[d])
                            enqueue(depth_d, d)

        return RangeSet.from_ranges(final_mask.to_ranges())
