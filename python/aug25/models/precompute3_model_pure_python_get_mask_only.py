import heapq
import _sep1 as ffi

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel
from ..common_interface import GraphProvider, RangeSet

try:
    profile
except NameError:
    def profile(func): return func


class Model(GraphProvider):
    def __init__(self, inner_model: InnerModel):
        self.inner_model = inner_model
        im = self.inner_model
        self.arena = im.arena
        self.roots_map = im.roots_map
        self.max_depth = im.max_depth
        self.possible_matches_cache = im.possible_matches_cache
        self.tokenizer_max_state = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map = im.internal_to_original_map

    @staticmethod
    def from_json_string(s: str):
        return Model(InnerModel.from_json_string(s))

    def commit(self, token_id: int):
        self.inner_model.commit(token_id)

    @property
    def state(self):
        return self.inner_model.state

    def is_end(self, node: int) -> bool:
        return bool(((self.arena.get(node) or {}).get("value") or {}).get("clean_end", False))

    @profile
    def get_mask(self):
        state_map = self.state
        all_ones = self.all_internal_llm_tokens_bitset
        final_mask = ffi.Bitset.zeros()

        values = {}
        stopped = set()
        todo = {}
        depth_heap = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map, max_depth, arena = self.roots_map, self.max_depth, self.arena
        is_end = self.is_end
        pmc = self.possible_matches_cache
        max_state = self.tokenizer_max_state

        for sid, gss in state_map.items():
            r = roots_map.get(int(sid))
            if r is None:
                continue
            r = int(r)
            if r in values:
                values[r] = (values[r][0].merge(gss), all_ones)
            else:
                values[r] = (gss, all_ones)
            d = max_depth[r]
            b = todo.get(d)
            if b is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                b.add(r)

        def enqueue(d: int, n: int):
            b = todo.get(d)
            if b is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                b.add(n)

        def disallowed_terminals(g: GSS):
            acc = g.reduce_acc()
            return ffi.HybridL2Bitset.all() if acc is None else acc.terminals_union.complement()

        while depth_heap:
            depth = hpop(depth_heap)
            nodes = todo.pop(depth, None)
            if not nodes:
                continue
            for node in nodes:
                if node in stopped:
                    continue
                item = values.pop(node, None)
                if item is None:
                    continue
                gss_node, llm_mask = item

                if is_end(node):
                    forbid = ffi.Bitset.zeros()
                    for (start, end), bv in disallowed_terminals(gss_node).range_values():
                        if bv.is_empty():
                            continue
                        for tsid in range(start, min(end, max_state) + 1):
                            pm = pmc.get(tsid)
                            if not pm:
                                continue
                            for terminal_id_str, llm_tokens in pm.items():
                                if bv.contains(int(terminal_id_str)):
                                    forbid = forbid.union(llm_tokens)
                    allowed = llm_mask.difference(forbid)
                    if not allowed.is_empty():
                        final_mask = final_mask.union(allowed)

                if llm_mask.is_empty():
                    stopped.add(node)
                    continue

                for (pop, llm_bv), dests in (arena.get(node, {}).get("children") or []):
                    popped = gss_node.popn(pop)
                    for dest_idx, state_bv in dests:
                        if not state_bv.is_empty():
                            matched = [popped.isolate(s) for s in popped.peek() if state_bv.contains(s)]
                            if not matched:
                                continue
                        else:
                            continue
                        child_gss = GSS.merge_many(matched)
                        child_mask = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)
                        d = int(dest_idx)
                        if d in values:
                            g0, m0 = values[d]
                            values[d] = (g0.merge(child_gss), m0.union(child_mask))
                        else:
                            values[d] = (child_gss, child_mask)
                        enqueue(max_depth[d], d)

        original_mask = ffi.Bitset.zeros()
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_mask.insert(self.internal_to_original_map[i])
        return RangeSet.from_ranges(original_mask.to_ranges())
