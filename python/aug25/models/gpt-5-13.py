import heapq
from bisect import bisect_left, bisect_right
from typing import Dict, List, Set, Tuple, Optional

import _sep1 as ffi
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel
from ..common_interface import GraphProvider, RangeSet

try:
    profile
except NameError:
    def profile(func): return func


class Model(GraphProvider):
    """
    Optimized GraphProvider Model focusing on accelerating get_mask.

    Key optimizations:
    - Preindex possible_matches_cache by terminal_id -> sorted lists of (tsid, bitset).
      This converts the per-state lookup at end nodes from:
         for tsid in range: for each terminal in pmc[tsid] if bv.contains(terminal): union
      to:
         for terminal in bv: for tsid in terminal_index where start<=tsid<=end: union
      which is typically significantly faster because the number of disallowed terminals
      (bv) is far smaller than iterating over every terminal in every tsid.

    Assumptions based on the problem statement:
    - _sep1 (ffi) primitives are not a bottleneck.
    - LeveledGSS is not a bottleneck.
    - possible_matches_cache is static per model (safe to preindex once).
    """

    def __init__(self, inner_model: InnerModel):
        self.inner_model: InnerModel = inner_model

        # Local aliases to speed up attribute access in hot paths
        im: InnerModel = self.inner_model
        self.arena: Dict[int, dict] = im.arena
        self.roots_map: Dict[int, int] = im.roots_map
        self.max_depth: Dict[int, int] = im.max_depth
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = im.possible_matches_cache
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map

        # Preindex possible_matches_cache by terminal id for fast range queries.
        # Structure: terminal_id -> (tsid_list_sorted, bitset_list_aligned)
        self._pmc_by_terminal: Dict[int, Tuple[List[int], List[ffi.Bitset]]] = self._build_pmc_by_terminal_index(
            self.possible_matches_cache
        )

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        return Model(InnerModel.from_json_string(s))

    def commit(self, token_id: int) -> None:
        self.inner_model.commit(token_id)

    @property
    def state(self) -> Dict[int, GSS]:
        return self.inner_model.state

    def is_end(self, node: int) -> bool:
        return bool(((self.arena.get(node) or {}).get("value") or {}).get("clean_end", False))

    def _build_pmc_by_terminal_index(
        self,
        pmc: Optional[Dict[int, Dict[int, ffi.Bitset]]],
    ) -> Dict[int, Tuple[List[int], List[ffi.Bitset]]]:
        """
        Build an index: for each terminal_id, store sorted (tsid, bitset) pairs.
        This allows fast union of bitsets over a tsid interval for a fixed terminal.
        """
        index: Dict[int, Tuple[List[int], List[ffi.Bitset]]] = {}
        if not pmc:
            return index

        # Collect unsorted lists first
        temp: Dict[int, List[Tuple[int, ffi.Bitset]]] = {}
        for tsid, term_map in pmc.items():
            if not term_map:
                continue
            # terminal ids may be ints or str-like; ensure int
            for terminal_key, llm_tokens in term_map.items():
                terminal_id = int(terminal_key)
                lst = temp.get(terminal_id)
                if lst is None:
                    temp[terminal_id] = [(int(tsid), llm_tokens)]
                else:
                    lst.append((int(tsid), llm_tokens))

        # Sort and separate into aligned parallel arrays for bisect and fast iteration
        for terminal_id, pairs in temp.items():
            pairs.sort(key=lambda x: x[0])
            tsids: List[int] = [p[0] for p in pairs]
            bitsets: List[ffi.Bitset] = [p[1] for p in pairs]
            index[terminal_id] = (tsids, bitsets)

        return index

    def _union_llm_tokens_for_disallowed_range(
        self,
        start: int,
        end: int,
        terminals_bv: ffi.Bitset,
    ) -> ffi.Bitset:
        """
        Efficiently compute the union of LLM token bitsets for a given tsid interval [start, end]
        and a set of disallowed terminals (terminals_bv).

        Uses the preindexed mapping: terminal -> sorted (tsid_list, bitset_list).
        For each terminal in terminals_bv, only the tsids present for that terminal are iterated,
        and only those within [start, end] are included using bisect to slice.
        """
        if terminals_bv.is_empty():
            return ffi.Bitset.zeros()

        result: ffi.Bitset = ffi.Bitset.zeros()
        pmc_by_terminal = self._pmc_by_terminal

        # Iterate terminals in the bitset; expected to be relatively sparse
        for terminal_id in terminals_bv.to_indices():
            entry = pmc_by_terminal.get(int(terminal_id))
            if not entry:
                continue
            tsids, bitsets = entry
            # Locate the indices covering tsids within [start, end]
            left = bisect_left(tsids, start)
            if left >= len(tsids):
                continue
            right = bisect_right(tsids, end, lo=left)
            if right <= left:
                continue
            # Union over the slice
            for i in range(left, right):
                result = result.union(bitsets[i])

        return result

    @profile
    def get_mask(self) -> RangeSet:
        state_map: Dict[int, GSS] = self.state
        all_ones: Optional[ffi.Bitset] = self.all_internal_llm_tokens_bitset
        if all_ones is None:
            # Fallback: if not provided, treat as empty (no tokens allowed).
            all_ones = ffi.Bitset.zeros()

        final_mask: ffi.Bitset = ffi.Bitset.zeros()

        values: Dict[int, Tuple[GSS, ffi.Bitset]] = {}
        stopped: Set[int] = set()
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        max_state: int = self.tokenizer_max_state

        # Initialize frontier per depth
        for sid, gss in state_map.items():
            r: Optional[int] = roots_map.get(int(sid))
            if r is None:
                continue
            r = int(r)
            if r in values:
                # Merge GSS at the same root and propagate unconstrained llm mask
                values[r] = (values[r][0].merge(gss), all_ones)
            else:
                values[r] = (gss, all_ones)

            d: int = max_depth[r]
            bucket = todo.get(d)
            if bucket is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                bucket.add(r)

        def enqueue(d: int, n: int) -> None:
            bucket = todo.get(d)
            if bucket is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                bucket.add(n)

        def disallowed_terminals(g: GSS) -> ffi.HybridL2Bitset:
            acc = g.reduce_acc()
            return ffi.HybridL2Bitset.all() if acc is None else acc.terminals_union.complement()

        pmc_by_terminal = self._pmc_by_terminal  # local alias

        while depth_heap:
            depth: int = hpop(depth_heap)
            nodes: Optional[Set[int]] = todo.pop(depth, None)
            if not nodes:
                continue

            for node in nodes:
                if node in stopped:
                    continue

                item: Optional[Tuple[GSS, ffi.Bitset]] = values.pop(node, None)
                if item is None:
                    continue
                gss_node, llm_mask = item

                # Process acceptance at end nodes
                if is_end(node):
                    # Compute forbidden LLM tokens by scanning disallowed terminals more efficiently.
                    forbid: ffi.Bitset = ffi.Bitset.zeros()
                    if pmc_by_terminal:
                        disallowed = disallowed_terminals(gss_node)
                        for (start, end), terminals_bv in disallowed.range_values():
                            if terminals_bv.is_empty():
                                continue
                            if start > max_state:
                                continue
                            if end < 0:
                                continue
                            a = max(0, int(start))
                            b = min(int(end), max_state)
                            if a > b:
                                continue
                            forbid_range = self._union_llm_tokens_for_disallowed_range(a, b, terminals_bv)
                            if not forbid_range.is_empty():
                                forbid = forbid.union(forbid_range)

                    # Compute allowed tokens and update final mask
                    allowed: ffi.Bitset = llm_mask.difference(forbid)
                    if not allowed.is_empty():
                        final_mask = final_mask.union(allowed)

                # If the current reachable mask is empty, prune further traversal
                if llm_mask.is_empty():
                    stopped.add(node)
                    continue

                # Explore transitions
                node_rec = arena.get(node, None)
                if not node_rec:
                    continue
                children = node_rec.get("children") or []
                if not children:
                    continue

                for (pop, llm_bv), dests in children:
                    popped: GSS = gss_node.popn(pop)
                    llm_bv_empty = llm_bv.is_empty()

                    # For each destination, filter by state bitset and propagate masks
                    for dest_idx, state_bv in dests:
                        if state_bv.is_empty():
                            continue

                        # Gather matched GSS states
                        peeks = popped.peek()
                        matched: List[GSS] = [popped.isolate(s) for s in peeks if state_bv.contains(s)]
                        if not matched:
                            continue

                        child_gss: GSS = GSS.merge_many(matched)
                        child_mask: ffi.Bitset = llm_mask if llm_bv_empty else llm_mask.intersection(llm_bv)

                        d: int = int(dest_idx)
                        prev = values.get(d)
                        if prev is None:
                            values[d] = (child_gss, child_mask)
                        else:
                            g0, m0 = prev
                            values[d] = (g0.merge(child_gss), m0.union(child_mask))

                        enqueue(max_depth[d], d)

        # Convert internal-token mask back to original tokenizer ids
        original_mask: ffi.Bitset = ffi.Bitset.zeros()
        internal_to_original = self.internal_to_original_map
        for i in final_mask.to_indices():
            j = internal_to_original.get(i)
            if j is not None:
                original_mask.insert(j)

        return RangeSet.from_ranges(original_mask.to_ranges())
