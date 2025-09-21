import heapq
import _sep1 as ffi
from typing import Dict, List, Set, Tuple, Optional

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel
from ..common_interface import GraphProvider, RangeSet

try:
    profile
except NameError:
    def profile(func): return func

from collections import defaultdict, deque

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

        self.adj: Dict[int, List[int]] = defaultdict(list)
        for node in self.arena:
            for _, dests in self.arena[node].get("children", []):
                for dest_idx, _ in dests:
                    self.adj[node].append(dest_idx)

        in_deg: Dict[int, int] = defaultdict(int)
        for node in self.adj:
            for dest in self.adj[node]:
                in_deg[dest] += 1

        self.topo: List[int] = []
        q: deque = deque([n for n in self.arena if in_deg[n] == 0])
        while q:
            n = q.popleft()
            self.topo.append(n)
            for dest in self.adj[n]:
                in_deg[dest] -= 1
                if in_deg[dest] == 0:
                    q.append(dest)

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
        state_map: Dict[int, GSS] = self.state
        all_ones: Optional[ffi.Bitset] = self.all_internal_llm_tokens_bitset
        final_mask: ffi.Bitset = ffi.Bitset.zeros()

        accum_gss: Dict[int, GSS] = {}
        accum_mask: Dict[int, ffi.Bitset] = {}

        for sid, gss in state_map.items():
            r: Optional[int] = self.roots_map.get(int(sid))
            if r is None:
                continue
            r = int(r)
            if r in accum_gss:
                accum_gss[r] = accum_gss[r].merge(gss)
            else:
                accum_gss[r] = gss
                accum_mask[r] = all_ones

        is_end = self.is_end
        pmc: Optional[Dict[int, Dict[int, ffi.Bitset]]] = self.possible_matches_cache
        max_state: int = self.tokenizer_max_state
        arena: Dict[int, dict] = self.arena

        def disallowed_terminals(g: GSS) -> ffi.HybridL2Bitset:
            acc = g.reduce_acc()
            return ffi.HybridL2Bitset.all() if acc is None else acc.terminals_union.complement()

        for node in self.topo:
            if node not in accum_mask:
                continue
            llm_mask = accum_mask[node]
            if llm_mask.is_empty():
                continue
            gss_node = accum_gss[node]

            if is_end(node):
                forbid: ffi.Bitset = ffi.Bitset.zeros()
                for (start, end), bv in disallowed_terminals(gss_node).range_values():
                    if bv.is_empty():
                        continue
                    for tsid in range(start, min(end, max_state) + 1):
                        pm: Optional[Dict[int, ffi.Bitset]] = pmc.get(tsid)
                        if not pm:
                            continue
                        for terminal_id_str, llm_tokens in pm.items():
                            if bv.contains(int(terminal_id_str)):
                                forbid = forbid.union(llm_tokens)
                allowed: ffi.Bitset = llm_mask.difference(forbid)
                if not allowed.is_empty():
                    final_mask = final_mask.union(allowed)

            for (pop, llm_bv), dests in (arena.get(node, {}).get("children") or []):
                popped: GSS = gss_node.popn(pop)
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():
                        continue
                    matched: List[GSS] = [popped.isolate(s) for s in popped.peek() if state_bv.contains(s)]
                    if not matched:
                        continue
                    child_gss: GSS = GSS.merge_many(matched)
                    child_mask: ffi.Bitset = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)
                    if child_mask.is_empty():
                        continue
                    d: int = int(dest_idx)
                    if d in accum_gss:
                        accum_gss[d] = accum_gss[d].merge(child_gss)
                        accum_mask[d] = accum_mask[d].union(child_mask)
                    else:
                        accum_gss[d] = child_gss
                        accum_mask[d] = child_mask

        original_mask: ffi.Bitset = ffi.Bitset.zeros()
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_mask.insert(self.internal_to_original_map[i])
        return RangeSet.from_ranges(original_mask.to_ranges())