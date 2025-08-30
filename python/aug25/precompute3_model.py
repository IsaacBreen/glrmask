import json
from typing import Dict, List, Tuple
from .common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module

class Precompute3(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.arena = arena
        # Normalize BVs to RangeSet; store stateIDBV as list of (s,e).
        for n in self.arena.values():
            ch = n.get("children") or []
            newch = []
            for edge_key, dest_map in ch:
                pop, llm_bv_json = edge_key
                llm_rs = RangeSet.from_json(llm_bv_json)
                newdm = []
                for dest_idx, state_bv in dest_map:
                    newdm.append((int(dest_idx), [(int(a), int(b)) for a, b in state_bv]))
                newch.append(((int(pop), llm_rs), newdm))
            n["children"] = newch

    @staticmethod
    def from_json_string(s: str) -> 'Precompute3':
        arr = json.loads(s)
        roots_map, arena_json = arr
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        return Precompute3(roots_map, arena)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("end", False))

    def iter_edges(self, node: int, token: int):
        # For precompute3, llm BV filters at the edge label.
        for (pop, llm_rs), dests in self.arena.get(node, {}).get("children") or []:
            if llm_rs.contains(token):
                for dest_idx, _ in dests:
                    # The common interface doesn't handle state_bv, so we yield None for state_id.
                    yield (int(pop), None, int(dest_idx))

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        final_mask = ffi.Bitset.zeros()
        from collections import deque
        q = deque()
        visited: Dict[int, ffi.GSSNode] = {}

        # Seed from roots
        for sid, gss in state_to_gss.items():
            if sid in self.roots_map:
                root_idx = self.roots_map[sid]
                if root_idx in visited:
                    existing_gss = visited[root_idx]
                    merged = ffi.gss_merge_many_with_depth([existing_gss, gss.clone_node()], 1)
                    if id(merged) != id(existing_gss):
                        visited[root_idx] = merged
                        q.append((root_idx, merged))
                else:
                    cloned_gss = gss.clone_node()
                    visited[root_idx] = cloned_gss
                    q.append((root_idx, cloned_gss))

        while q:
            u, gss = q.popleft()

            if self.is_end(u):
                final_mask = final_mask.union(gss.allowed_llm_tokens())

            for (pop, llm_rs), dests in self.arena.get(u, {}).get("children") or []:
                peeks = ffi.gss_popn_collect(gss, int(pop))
                if not peeks: continue

                for dest_idx, state_bv in dests:
                    matched = []
                    for (sid_val, parent_node) in peeks:
                        ok = False
                        for (a, b) in state_bv:
                            if a <= sid_val <= b: ok = True; break
                        if ok:
                            matched.append(parent_node)
                    if not matched: continue

                    merged = ffi.gss_merge_many_with_depth(matched, 1)
                    if llm_rs.intervals:
                        bv = ffi.Bitset.from_ranges(llm_rs.intervals)
                        ffi.gss_allow_only_llm_tokens_and_prune(merged, bv)

                    if merged.is_ok():
                        dest_idx_int = int(dest_idx)
                        if dest_idx_int in visited:
                            existing_gss = visited[dest_idx_int]
                            newly_merged = ffi.gss_merge_many_with_depth([existing_gss, merged], 1)
                            if id(newly_merged) != id(existing_gss):
                                visited[dest_idx_int] = newly_merged
                                q.append((dest_idx_int, newly_merged))
                        else:
                            visited[dest_idx_int] = merged
                            q.append((dest_idx_int, merged))
        return final_mask
