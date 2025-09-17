import json
from bisect import bisect_left, bisect_right
from operator import itemgetter
from typing import Dict, List, Tuple, Optional, Iterable

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # compiled module


def _merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # Canonicalize state intervals: sort and merge overlaps; ensure (a <= b) and ints
    sorted_intervals = sorted([(int(a), int(b)) for a, b in intervals if a is not None and b is not None])
    if not sorted_intervals:
        return []
    merged: List[Tuple[int, int]] = []
    ca, cb = sorted_intervals[0]
    for a, b in sorted_intervals[1:]:
        if a <= cb + 1:  # overlapping or touching
            if b > cb:
                cb = b
        else:
            merged.append((ca, cb))
            ca, cb = a, b
    merged.append((ca, cb))
    return merged


class _Dest:
    __slots__ = ("dest", "starts", "ends")

    def __init__(self, dest: int, intervals: List[Tuple[int, int]]):
        self.dest = int(dest)
        if intervals:
            starts = [int(a) for a, _ in intervals]
            ends = [int(b) for _, b in intervals]
            self.starts = starts  # type: List[int]
            self.ends = ends      # type: List[int]
        else:
            self.starts = []      # empty => special epsilon case (ignored in get_mask)
            self.ends = []


class _Group:
    __slots__ = ("llm_rs", "llm_bv", "dests")

    def __init__(self, llm_rs: RangeSet, llm_bv: Optional[ffi.Bitset], dests: List[_Dest]):
        self.llm_rs = llm_rs
        self.llm_bv = llm_bv
        self.dests = dests      # type: List[_Dest]


class _PopEntry:
    __slots__ = ("groups", "has_state")

    def __init__(self, groups: List[_Group], has_state: bool):
        self.groups = groups     # type: List[_Group]
        self.has_state = bool(has_state)


class Model(GraphProvider):
    """
    Highly optimized precompute3 model.

    Key optimizations:
    - Precompiles and deduplicates edges per node by (pop, llm_range_set).
    - Prebuilds ffi.Bitset for each unique LLM token RangeSet to avoid per-step constructions.
    - Groups transitions by 'pop' and calls gss_popn_collect once per pop per node expansion.
    - Filters popped parents to dest's state intervals via bisect on a single sorted sid list.
    - Priority-queue scheduler with de-duplication; avoids min() over dict of buckets.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # roots_map: (state_id -> trie_root_node)
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.depths: Dict[int, int] = {}
        self.is_end_node: Dict[int, bool] = {}
        # Node -> pop -> _PopEntry
        self.node_pop_map: Dict[int, Dict[int, _PopEntry]] = {}

        # Build nodes
        for uid, node in arena.items():
            node_id = int(uid)
            self.depths[node_id] = int(node.get("max_depth", 0) or 0)
            self.is_end_node[node_id] = bool((node.get("value") or {}).get("end", False))

            children = node.get("children") or []
            if not children:
                self.node_pop_map[node_id] = {}
                continue

            # Group edges by pop and LLM interval key to deduplicate
            # Structure: pop -> llm_key -> {"llm_rs": RangeSet, "dests": List[_Dest]}
            by_pop: Dict[int, Dict[Tuple[Tuple[int, int], ...], dict]] = {}

            for edge_key, dest_map in children:
                pop_raw, llm_bv_json = edge_key
                pop = int(pop_raw)
                llm_rs = RangeSet.from_ranges(llm_bv_json)
                # Create a hashable key for grouping identical LLM token BVs
                llm_key = tuple((int(a), int(b)) for a, b in (llm_rs.intervals or []))

                dests_list = []
                for dest_idx, state_bv in (dest_map or []):
                    # Canonicalize intervals
                    merged_intervals = _merge_intervals(state_bv or [])
                    dests_list.append(_Dest(int(dest_idx), merged_intervals))

                pop_map = by_pop.get(pop)
                if pop_map is None:
                    pop_map = {}
                    by_pop[pop] = pop_map

                entry = pop_map.get(llm_key)
                if entry is None:
                    entry = {"llm_rs": llm_rs, "dests": dests_list}
                    pop_map[llm_key] = entry
                else:
                    # Extend existing group with more dests
                    entry["dests"].extend(dests_list)

            # Finalize pop entries with prebuilt bitsets and has_state flag
            pop_entries: Dict[int, _PopEntry] = {}
            for pop, groups_dict in by_pop.items():
                groups: List[_Group] = []
                has_state = False
                for llm_key, gdata in groups_dict.items():
                    llm_rs: RangeSet = gdata["llm_rs"]
                    intervals = llm_rs.intervals or []
                    llm_bv = ffi.Bitset.from_ranges(intervals) if intervals else None

                    # dests: keep in the order they were encountered
                    dests: List[_Dest] = gdata["dests"]

                    # Track if any dest has actual state filter (non-empty intervals)
                    for d in dests:
                        if d.starts:
                            has_state = True
                            break

                    groups.append(_Group(llm_rs, llm_bv, dests))

                pop_entries[pop] = _PopEntry(groups, has_state)

            self.node_pop_map[node_id] = pop_entries

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
        return bool(self.is_end_node.get(int(node), False))

    def iter_edges(self, node: int, token: int):
        """
        Explodes grouped representation for equivalence checking:
        For each matching (pop, llm_rs), expand each dest's state intervals sid-by-sid.
        """
        node = int(node)
        pop_entries = self.node_pop_map.get(node) or {}
        for pop, pop_entry in pop_entries.items():
            for group in pop_entry.groups:
                if group.llm_rs.contains(token):
                    for dest in group.dests:
                        if not dest.starts:
                            # Epsilon transition on GSS stack (no state filter)
                            yield (int(pop), None, int(dest.dest))
                        else:
                            # Expand all states in the BV
                            for a, b in zip(dest.starts, dest.ends):
                                # inclusive range
                                for sid in range(a, b + 1):
                                    yield (int(pop), int(sid), int(dest.dest))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        state_to_gss = self.constraint_state.get_state_map()
        # Local bindings for speed
        depths = self.depths
        node_pop_map = self.node_pop_map
        is_end_node = self.is_end_node

        Bitset = ffi.Bitset
        zeros = Bitset.zeros
        gss_merge_many = ffi.gss_merge_many_with_depth
        gss_popn_collect = ffi.gss_popn_collect
        gss_allow_only = ffi.gss_allow_only_llm_tokens_and_prune

        final_mask = zeros()

        # Aggregated value per trie node waiting to be processed
        values: Dict[int, ffi.GSSNode] = {}
        # Nodes that decided to stop (agg.is_ok() == False)
        stopped: set[int] = set()

        # Priority queue scheduler (depth ascending)
        import heapq
        heap: List[Tuple[int, int]] = []
        in_queue: set[int] = set()

        # Seed: map tokenizer state -> trie root by merging cloned GSS nodes
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(int(sid))
            if root_idx is None:
                continue
            r = int(root_idx)
            gss_clone = gss.clone_node()
            if r in values:
                merged = gss_merge_many([values[r], gss_clone], 9999999)
                if merged.ptr() != values[r].ptr():
                    values[r] = merged
            else:
                values[r] = gss_clone
            if r not in in_queue:
                heapq.heappush(heap, (int(depths.get(r, 0)), r))
                in_queue.add(r)

        # Main scheduler loop
        while heap:
            cur_depth, node_idx = heapq.heappop(heap)
            if node_idx in stopped:
                continue
            agg = values.pop(node_idx, None)
            if agg is None:
                # stale entry in heap: node was already processed/merged and requeued
                in_queue.discard(node_idx)
                continue
            # mark as no longer in queue
            in_queue.discard(node_idx)

            # End-node handling
            if is_end_node.get(node_idx, False):
                allowed_tokens = agg.allowed_llm_tokens()
                final_mask = final_mask.union(allowed_tokens)

            # Stop condition
            if not agg.is_ok():
                stopped.add(node_idx)
                continue

            # Process grouped edges
            pop_entries = node_pop_map.get(node_idx)
            if not pop_entries:
                continue

            # Iterate over pops; compute peeks once per pop if there exists at least one dest with state filter.
            for pop, pop_entry in pop_entries.items():
                if not pop_entry.has_state:
                    # No destinations have state filters; per legacy get_mask semantics we ignore epsilon-only edges.
                    continue

                peeks = gss_popn_collect(agg, int(pop))
                if not peeks:
                    continue

                # Sort peeks by sid once and build arrays for fast slicing
                # peeks elements: (sid_val, parent_node)
                peeks.sort(key=itemgetter(0))
                sid_vals = [p[0] for p in peeks]
                parent_nodes = [p[1] for p in peeks]

                # For each group (same LLM token BV), filter per dest by state intervals
                for group in pop_entry.groups:
                    llm_bv = group.llm_bv  # Optional[ffi.Bitset]

                    for dest in group.dests:
                        starts = dest.starts
                        if not starts:
                            # As per original get_mask: ignore epsilon transitions on GSS stack here.
                            continue

                        # Build matched parent nodes by slicing parent_nodes within each [a,b]
                        ends = dest.ends
                        matched: List[ffi.GSSNode] = []

                        # Efficiently collect parents in ranges
                        # Using bisect on sid_vals for each interval [a, b]
                        for a, b in zip(starts, ends):
                            lo = bisect_left(sid_vals, a)
                            if lo >= len(sid_vals):
                                break
                            hi = bisect_right(sid_vals, b, lo)
                            if hi > lo:
                                # extend matched with the slice of parent nodes
                                matched.extend(parent_nodes[lo:hi])

                        if not matched:
                            continue

                        child_gss = gss_merge_many(matched, 1)
                        if llm_bv is not None:
                            gss_allow_only(child_gss, llm_bv)
                        if not child_gss.is_ok():
                            continue

                        d = dest.dest
                        existing = values.get(d)
                        if existing is not None:
                            combined = gss_merge_many([existing, child_gss], 1)
                            # Only re-enqueue if structurally changed
                            if combined.ptr() != existing.ptr():
                                values[d] = combined
                                if d not in in_queue:
                                    heapq.heappush(heap, (int(depths.get(d, 0)), d))
                                    in_queue.add(d)
                        else:
                            values[d] = child_gss
                            if d not in in_queue:
                                    heapq.heappush(heap, (int(depths.get(d, 0)), d))
                                    in_queue.add(d)

        original_mask = self.constraint.internal_bv_to_original(final_mask)
        return RangeSet.from_ranges(original_mask.to_ranges())


