import json
from bisect import bisect_left, bisect_right
import time
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
            self.is_end_node[node_id] = bool((node.get("value") or {}).get("clean_end", False))

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
        print("\n--- get_mask START ---")
        print(self.constraint_state)
        state_to_gss = self.constraint_state.filtered_state_gss_map()
        print(f"Filtered state_to_gss: { {k: v.ptr() for k, v in state_to_gss.items()} }")

        # Local bindings for speed
        depths = self.depths
        node_pop_map = self.node_pop_map
        is_end_node = self.is_end_node

        Bitset = ffi.Bitset
        zeros = Bitset.zeros
        gss_merge_many = ffi.gss_merge_many_with_depth
        gss_popn_collect = ffi.gss_popn_collect

        final_mask = zeros()

        # Aggregated value per trie node waiting to be processed
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}
        # Nodes that decided to stop (agg.is_ok() == False)
        stopped: set[int] = set()

        # Priority queue scheduler (depth ascending)
        import heapq
        heap: List[Tuple[int, int]] = []
        in_queue: set[int] = set()

        # Seed: map tokenizer state -> trie root by merging cloned GSS nodes
        print("\n--- Seeding work queue ---")
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(int(sid))
            if root_idx is None:
                continue
            r = int(root_idx)
            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            print(f"  SEED: sid={sid}, root_idx={r}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            if r in values:
                existing_gss, existing_mask = values[r]
                print(f"    - MERGE: gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={gss_clone.ptr()}, mask2={new_mask.to_ranges()}")
                merged_gss = gss_merge_many([existing_gss, gss_clone], 1)
                merged_mask = existing_mask.union(new_mask)
                values[r] = (merged_gss, merged_mask)
                print(f"      - Merged result: gss_ptr={merged_gss.ptr()}, mask={merged_mask.to_ranges()}")
            else:
                values[r] = (gss_clone, new_mask)
            if r not in in_queue:
                heapq.heappush(heap, (int(depths.get(r, 0)), r))
                in_queue.add(r)

        # Main scheduler loop
        while heap:
            cur_depth, node_idx = heapq.heappop(heap)
            if node_idx in stopped:
                print(f"  - Node {node_idx}: SKIPPING (already stopped)")
                continue
            item = values.pop(node_idx, None)
            if item is None:
                print(f"  - Node {node_idx}: SKIPPING (no value)")
                # stale entry in heap: node was already processed/merged and requeued
                in_queue.discard(node_idx)
                continue
            gss_node, llm_mask = item
            print(f"  - PROCESS: node_ptr={node_idx}, gss_ptr={gss_node.ptr()}, mask={llm_mask.to_ranges()}")

            # mark as no longer in queue
            in_queue.discard(node_idx)

            # End-node handling
            if is_end_node.get(node_idx, False):
                print(f"    - END NODE found. Updating final_mask.")
                print(f"      - final_mask before: {final_mask.to_ranges()}")
                gss_active_tokens = gss_node.allowed_llm_tokens()
                tokens_to_add = llm_mask.intersection(gss_active_tokens)
                print(f"      - glr_active_tokens to union: {tokens_to_add.to_ranges()}")
                final_mask = final_mask.union(tokens_to_add)
                print(f"      - final_mask after:  {final_mask.to_ranges()}")

            # Stop condition
            if not gss_node.is_ok():
                stopped.add(node_idx)
                print(f"    - STOPPING node {node_idx} (GSS not alive)")
                continue

            # Process grouped edges
            pop_entries = node_pop_map.get(node_idx)
            if not pop_entries:
                continue

            # Iterate over pops; compute peeks once per pop if there exists at least one dest with state filter.
            for pop, pop_entry in pop_entries.items():
                print(f"    - Edge group: pop={pop}")
                if not pop_entry.has_state:
                    # No destinations have state filters; per legacy get_mask semantics we ignore epsilon-only edges.
                    continue

                peeks = gss_popn_collect(gss_node, int(pop))
                print(f"      - Found {len(peeks)} peeks from GSS")
                if not peeks:
                    continue

                # Sort peeks by sid once and build arrays for fast slicing
                # peeks elements: (sid_val, parent_node)
                peeks.sort(key=itemgetter(0))
                sid_vals = [p[0] for p in peeks]
                parent_nodes = [p[1] for p in peeks]

                # For each group (same LLM token BV), filter per dest by state intervals
                for group in pop_entry.groups:
                    print(f"    - Edge: llm_bv={'None' if group.llm_bv is None else group.llm_bv.to_ranges()}")
                    llm_bv = group.llm_bv  # Optional[ffi.Bitset]
                    child_llm_mask = llm_mask if llm_bv is None else llm_mask.intersection(llm_bv)
                    print(f"      - Child mask: {child_llm_mask.to_ranges()}")
                    if child_llm_mask.is_empty():
                        continue

                    for dest in group.dests:
                        print(f"      - Dest: idx={dest.dest}, state_intervals={list(zip(dest.starts, dest.ends))}")
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

                        print(f"        - Matched {len(matched)} parent GSS nodes")
                        if not matched:
                            continue

                        child_gss = gss_merge_many(matched, 1)
                        if not child_gss.is_alive():
                            continue

                        d = dest.dest
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss, existing_mask = existing
                            print(f"        - Enqueue {d}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={child_gss.ptr()}, mask2={child_llm_mask.to_ranges()}")
                            combined = gss_merge_many([existing_gss, child_gss], 1)
                            combined_mask = existing_mask.union(child_llm_mask)
                            # Only re-enqueue if structurally changed
                            if combined.ptr() != existing_gss.ptr():
                                values[d] = (combined, combined_mask)
                                print(f"          - Merged result: gss_ptr={combined.ptr()}, mask={combined_mask.to_ranges()}")
                                if d not in in_queue:
                                    heapq.heappush(heap, (int(depths.get(d, 0)), d))
                                    in_queue.add(d)
                        else:
                            values[d] = (child_gss, child_llm_mask)
                            print(f"        - Enqueue {d}: CREATING gss_ptr={child_gss.ptr()}, mask={child_llm_mask.to_ranges()}")
                            if d not in in_queue:
                                    heapq.heappush(heap, (int(depths.get(d, 0)), d))
                                    in_queue.add(d)

        print("\n--- get_mask END ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())


