import heapq
import _sep1 as ffi
from time import perf_counter
from typing import Dict, List, Set, Tuple, Optional

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel
from ..common_interface import GraphProvider, RangeSet

try:
    profile
except NameError:
    def profile(func): return func


class Model(GraphProvider):
    def __init__(self, inner_model: InnerModel):
        self.inner_model: InnerModel = inner_model
        im: InnerModel = self.inner_model
        self.arena: Dict[int, dict] = im.arena
        self.roots_map: Dict[int, int] = im.roots_map
        self.max_depth: Dict[int, int] = im.max_depth
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = im.possible_matches_cache
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map

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

    @profile
    def get_mask(self) -> RangeSet:
        # Profiling counters
        loops_while_depth_heap: int = 0
        nodes_visited: int = 0
        items_processed: int = 0
        end_nodes_reached_items: int = 0
        end_nodes_unique: Set[int] = set()
        bitset_union_calls: int = 0
        bitset_difference_calls: int = 0
        bitset_intersection_calls: int = 0
        hybrid_complement_calls: int = 0
        hybrid_range_values_calls: int = 0
        tsid_iterations: int = 0
        pm_get_calls: int = 0
        pm_items_iterated: int = 0
        children_groups_processed: int = 0
        transitions_processed: int = 0
        t_start: float = perf_counter()

        def bs_union(a: ffi.Bitset, b: ffi.Bitset) -> ffi.Bitset:
            nonlocal bitset_union_calls
            bitset_union_calls += 1
            return a.union(b)

        def bs_diff(a: ffi.Bitset, b: ffi.Bitset) -> ffi.Bitset:
            nonlocal bitset_difference_calls
            bitset_difference_calls += 1
            return a.difference(b)

        def bs_inter(a: ffi.Bitset, b: ffi.Bitset) -> ffi.Bitset:
            nonlocal bitset_intersection_calls
            bitset_intersection_calls += 1
            return a.intersection(b)

        state_map: Dict[int, GSS] = self.state
        all_ones: Optional[ffi.Bitset] = self.all_internal_llm_tokens_bitset
        final_mask: ffi.Bitset = ffi.Bitset.zeros()

        values: Dict[int, Tuple[GSS, ffi.Bitset]] = {}
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        pmc: Optional[Dict[int, Dict[int, ffi.Bitset]]] = self.possible_matches_cache
        max_state: int = self.tokenizer_max_state

        # Seed with initial states; start with all LLM tokens allowed.
        for sid, gss in state_map.items():
            r: Optional[int] = roots_map.get(int(sid))
            if r is None:
                continue
            r = int(r)
            if r in values:
                existing_gss, existing_mask = values[r]
                merged_gss = existing_gss.merge(gss)
                # existing_mask and all_ones should both be the same "all" mask; union for completeness.
                values[r] = (merged_gss, bs_union(existing_mask, all_ones))
            else:
                values[r] = (gss, all_ones)
            d: int = max_depth[r]
            b: Optional[Set[int]] = todo.get(d)
            if b is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                b.add(r)

        def disallowed_terminals(g: GSS) -> ffi.HybridL2Bitset:
            acc = g.reduce_acc()
            if acc is None:
                return ffi.HybridL2Bitset.all()
            nonlocal hybrid_complement_calls
            hybrid_complement_calls += 1
            return acc.terminals_union.complement()

        while depth_heap:
            loops_while_depth_heap += 1
            depth: int = hpop(depth_heap)
            nodes: Optional[Set[int]] = todo.pop(depth, None)
            if not nodes:
                continue
            for node in nodes:
                nodes_visited += 1
                item: Optional[Tuple[GSS, ffi.Bitset]] = values.pop(node, None)
                if item is None:
                    continue

                gss_node, llm_mask = item

                end_node_flag: bool = is_end(node)
                if end_node_flag:
                    end_nodes_unique.add(node)
                    end_nodes_reached_items += 1
                    forbid: ffi.Bitset = ffi.Bitset.zeros()
                    hyb = disallowed_terminals(gss_node)
                    hybrid_range_values_calls += 1
                    for (start, end), bv in hyb.range_values():
                        if bv.is_empty():
                            continue
                        for tsid in range(start, min(end, max_state) + 1):
                            tsid_iterations += 1
                            pm_get_calls += 1
                            pm: Optional[Dict[int, ffi.Bitset]] = pmc.get(tsid) if pmc is not None else None
                            if not pm:
                                continue
                            for terminal_id_str, llm_tokens in pm.items():
                                pm_items_iterated += 1
                                if bv.contains(int(terminal_id_str)):
                                    forbid = bs_union(forbid, llm_tokens)
                    allowed: ffi.Bitset = bs_diff(llm_mask, forbid)
                    if not allowed.is_empty():
                        final_mask = bs_union(final_mask, allowed)

                items_processed += 1

                if llm_mask.is_empty():
                    continue

                for (pop, llm_bv), dests in (arena.get(node, {}).get("children") or []):
                    children_groups_processed += 1
                    popped: GSS = gss_node.popn(pop)
                    for dest_idx, state_bv in dests:
                        transitions_processed += 1
                        if not state_bv.is_empty():
                            matched: List[GSS] = [popped.isolate(s) for s in popped.peek() if state_bv.contains(s)]
                            if not matched:
                                continue
                        else:
                            continue
                        child_gss: GSS = GSS.merge_many(matched)
                        child_mask: ffi.Bitset
                        if llm_bv.is_empty():
                            child_mask = llm_mask
                        else:
                            child_mask = bs_inter(llm_mask, llm_bv)
                        d: int = int(dest_idx)
                        if d in values:
                            existing_gss, existing_mask = values[d]
                            merged_gss = existing_gss.merge(child_gss)
                            combined_mask = bs_union(existing_mask, child_mask)
                            values[d] = (merged_gss, combined_mask)
                        else:
                            values[d] = (child_gss, child_mask)
                        b: Optional[Set[int]] = todo.get(max_depth[d])
                        if b is None:
                            todo[max_depth[d]] = {d}
                            hp(depth_heap, max_depth[d])
                        else:
                            b.add(d)

        original_mask: ffi.Bitset = ffi.Bitset.zeros()
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_mask.insert(self.internal_to_original_map[i])
        t_end: float = perf_counter()
        # Print profiling summary
        print("[get_mask profile]")
        print(f"  time_sec: {t_end - t_start:.6f}")
        print(f"  while_iterations: {loops_while_depth_heap}")
        print(f"  nodes_visited: {nodes_visited}")
        print(f"  items_processed: {items_processed}")
        print(f"  end_nodes_items: {end_nodes_reached_items}")
        print(f"  end_nodes_unique: {len(end_nodes_unique)}")
        print(f"  bitset: union={bitset_union_calls}, difference={bitset_difference_calls}, intersection={bitset_intersection_calls}")
        print(f"  hybrid: complement={hybrid_complement_calls}, range_values={hybrid_range_values_calls}")
        print(f"  tsid_iterations: {tsid_iterations}")
        print(f"  pm_get_calls: {pm_get_calls}, pm_items_iterated: {pm_items_iterated}")
        print(f"  children_groups_processed: {children_groups_processed}, transitions_processed: {transitions_processed}")
        return RangeSet.from_ranges(original_mask.to_ranges())
