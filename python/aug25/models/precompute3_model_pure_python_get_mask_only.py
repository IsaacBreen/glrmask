import heapq
import _sep1 as ffi
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel
from ..common_interface import GraphProvider, RangeSet

try:
    profile
except NameError:
    def profile(f): return f


class Model(GraphProvider):
    def __init__(self, inner_model: InnerModel):
        self.inner_model = inner_model

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        return Model(InnerModel.from_json_string(s))

    def commit(self, token_id: int):
        self.inner_model.commit(token_id)

    @property
    def state(self):
        return self.inner_model.state

    @profile
    def get_mask(self) -> RangeSet:
        m = self.inner_model
        B, H = ffi.Bitset, ffi.HybridL2Bitset

        state_map = m.state
        ones = m.all_internal_llm_tokens_bitset
        final_mask = B.zeros()

        values = {}
        stopped = set()
        todo = {}
        heap = []
        heappush, heappop = heapq.heappush, heapq.heappop

        # Seed
        for sid, gss in state_map.items():
            root = m.roots_map.get(int(sid))
            if root is None:
                continue
            d = int(root)
            if d in values:
                values[d] = (values[d][0].merge(gss), ones)
            else:
                values[d] = (gss, ones)
            depth = m.max_depth[d]
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {d}
                heappush(heap, depth)
            else:
                bucket.add(d)

        def enqueue(depth, node):
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node}
                heappush(heap, depth)
            else:
                bucket.add(node)

        def disallowed_terminals(gss):
            acc = gss.reduce_acc()
            return H.all() if acc is None else acc.terminals_union.complement()

        pmc = m.possible_matches_cache or {}
        arena = m.arena
        max_depth = m.max_depth
        tmax = m.tokenizer.max_state()

        while True:
            node_indices = None
            while heap:
                depth = heappop(heap)
                node_indices = todo.pop(depth, None)
                if node_indices:
                    break
            if not node_indices:
                break

            for idx in node_indices:
                if idx in stopped:
                    continue
                item = values.pop(idx, None)
                if item is None:
                    continue
                gss_node, llm_mask = item
                nd = arena.get(idx, {})

                # End-node handling
                if (nd.get("value") or {}).get("clean_end"):
                    forb = B.zeros()
                    for (start, end), dis in disallowed_terminals(gss_node).range_values():
                        if dis.is_empty():
                            continue
                        end = min(end, tmax)
                        for tsid in range(start, end + 1):
                            pm = pmc.get(tsid)
                            if not pm:
                                continue
                            for term_id, token_bv in pm.items():
                                if dis.contains(int(term_id)):
                                    forb = forb.union(token_bv)
                    if not llm_mask.is_empty():
                        allowed = llm_mask.difference(forb)
                        if not allowed.is_empty():
                            final_mask = final_mask.union(allowed)

                if llm_mask.is_empty():
                    stopped.add(idx)
                    continue

                # Transitions
                for (pop, llm_bv), dests in (nd.get("children") or []):
                    popped = gss_node.popn(pop)
                    child_mask = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)
                    for dest_idx, state_bv in dests:
                        matched = [popped.isolate(s) for s in popped.peek() if state_bv.contains(s)]
                        if not matched:
                            continue
                        child_gss = GSS.merge_many(matched)
                        d = int(dest_idx)
                        if d in values:
                            eg, em = values[d]
                            values[d] = (eg.merge(child_gss), em.union(child_mask))
                        else:
                            values[d] = (child_gss, child_mask)
                        enqueue(max_depth[d], d)

        # Map internal to original IDs
        original_mask = B.zeros()
        ins = original_mask.insert
        mp = m.internal_to_original_map
        for internal_id in final_mask.to_indices():
            orig_id = mp.get(internal_id)
            if orig_id is not None:
                ins(orig_id)
        return RangeSet.from_ranges(original_mask.to_ranges())
