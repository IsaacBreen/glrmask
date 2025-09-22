import heapq
from typing import Dict, List, Set, Optional
import portion as P

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel, PyAcc
from ..common_interface import GraphProvider, RangeSet


def ints_to_interval(indices) -> P.Interval:
    iv = P.empty()
    for i in indices:
        iv = iv | P.singleton(int(i))
    return iv


def interval_to_int_ranges(iv: P.Interval):
    ranges = []
    for atom in iv:
        lower = atom.lower
        upper = atom.upper
        if atom.left is P.OPEN:
            lower += 1
        if atom.right is P.OPEN:
            upper -= 1
        if lower <= upper:
            ranges.append((int(lower), int(upper)))
    return ranges


def iterate_interval_ints(iv: P.Interval):
    for atom in iv:
        lower = atom.lower
        upper = atom.upper
        if atom.left is P.OPEN:
            lower += 1
        if atom.right is P.OPEN:
            upper -= 1
        if lower <= upper:
            for i in range(int(lower), int(upper) + 1):
                yield i


class Model(GraphProvider):
    def __init__(self, inner_model: InnerModel):
        self.inner_model: InnerModel = inner_model
        im: InnerModel = self.inner_model
        self.arena: Dict[int, dict] = im.arena
        self.roots_map: Dict[int, int] = im.roots_map
        self.max_depth: Dict[int, int] = im.max_depth
        self.possible_matches_cache: Optional[Dict[int, Dict[int, P.Interval]]] = im.possible_matches_cache
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[P.Interval] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map
        self.all_terminals_bitset: Optional[P.Interval] = im.all_terminals_bitset

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

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.

        Changes for get_mask_only:
        - Initialize a per-accumulator LLM mask (PyAcc.llm_mask) BEFORE traversal by computing
          the forbidden terminals -> forbidden LLM tokens and taking the complement.
        - Consume terminals_union (set to HybridL2Bitset.all()) after initialization.
        - As we traverse edges, intersect llm_mask with the edge's LLM bitset using apply.
        - At end nodes, simply reduce acc over the GSS and union the llm_mask into the final.
        """
        state_map: Dict[int, GSS] = self.state

        all_ones: P.Interval = self.all_internal_llm_tokens_bitset or P.empty()
        final_mask: P.Interval = P.empty()

        # We carry only GSS per node; the per-path LLM mask lives inside PyAcc.llm_mask
        values: Dict[int, GSS] = {}
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        pmc: Dict[int, Dict[int, P.Interval]] = self.possible_matches_cache or {}
        max_state: int = self.tokenizer_max_state

        # Seed: Initialize llm_mask in each GSS, consume terminals_union, and enqueue roots.
        def initialize_acc(acc: PyAcc) -> PyAcc:
            # Compute allowed LLM tokens from disallowed terminals for this accumulator
            disallowed_llm_mask: P.Interval = P.empty()
            disallowed_map = dict(acc.terminals_union)

            for tsid, disallowed_terminals in disallowed_map.items():
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm: Dict[int, P.Interval] = pmc[tsid]
                for atom in disallowed_terminals:
                    low, up = atom.lower, atom.upper
                    if atom.left is P.OPEN: low += 1
                    if atom.right is P.OPEN: up -= 1
                    for terminal_id in range(int(low), int(up) + 1):
                        if terminal_id in terminals_to_llm:
                            disallowed_llm_mask = disallowed_llm_mask | terminals_to_llm[terminal_id]

            allowed_mask = all_ones - disallowed_llm_mask
            return PyAcc(
                terminals_union=tuple(),  # consume
                llm_mask=allowed_mask,
            )

        apply_memo: Dict[PyAcc, PyAcc] = {}
        for sid, gss in state_map.items():
            r: int = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc, apply_memo)
            if r in values:
                values[r] = values[r].merge(gss_initialized)
            else:
                values[r] = gss_initialized

            d: int = max_depth[r]
            bucket: Optional[Set[int]] = todo.get(d)
            if bucket is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                bucket.add(r)

        def enqueue(d: int, n: int) -> None:
            bucket: Optional[Set[int]] = todo.get(d)
            if bucket is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                bucket.add(n)

        # Main loop
        while depth_heap:
            depth: int = hpop(depth_heap)
            while todo[depth]:
                node: int = todo[depth].pop()
                gss_node: GSS = values.pop(node)

                # End-node handling: just union the allowed LLM tokens
                if is_end(node):
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    if reduced_acc:
                        final_mask = final_mask | reduced_acc.llm_mask

                # Traverse edges and propagate masks
                for (pop, llm_iv), dests in (arena.get(node, {}).get("children") or []):
                    popped: GSS = gss_node.popn(pop)
                    if popped.is_empty():
                        continue

                    for dest_idx, state_bv in dests:
                        if state_bv.empty:
                            continue

                        values_to_keep = [s for s in popped.peek() if s in state_bv]
                        if not values_to_keep:
                            continue

                        child_gss: GSS = popped.isolate_many(values_to_keep)
                        if child_gss.is_empty():
                            continue

                        # Apply edge LLM mask by intersecting per-acc llm_mask with llm_bv
                        if not llm_iv.empty:
                            acc_memo: Dict[PyAcc, Optional[PyAcc]] = {}

                            def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                                if acc in acc_memo:
                                    return acc_memo[acc]
                                new_mask = acc.llm_mask & llm_iv
                                if new_mask.empty:
                                    result = None
                                else:
                                    result = PyAcc(
                                        terminals_union=acc.terminals_union,
                                        llm_mask=new_mask
                                    )
                                acc_memo[acc] = result
                                return result

                            child_gss = child_gss.apply_and_prune(intersect_and_prune)
                            if child_gss.is_empty():
                                continue

                        d: int = int(dest_idx)
                        if d in values:
                            existing_gss = values[d]
                            new_gss = child_gss
                            merged_gss = existing_gss.merge(new_gss)
                            values[d] = merged_gss
                        else:
                            values[d] = child_gss
                        enqueue(max_depth[d], d)

            todo.pop(depth)

        # Convert internal mask back to original IDs and return as a RangeSet
        original_iv: P.Interval = P.empty()
        for i in iterate_interval_ints(final_mask):
            if i in self.internal_to_original_map:
                original_iv = original_iv | P.singleton(self.internal_to_original_map[i])

        return RangeSet.from_ranges(interval_to_int_ranges(original_iv))

