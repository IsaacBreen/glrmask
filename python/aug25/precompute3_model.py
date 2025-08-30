import json
from typing import Dict, List, Tuple
from .common_interface import GraphProvider, RangeSet
import _sep1 as ffi
import heapq
import itertools

class Model(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.arena = arena
        # Add max_depth and normalize BVs to RangeSet; store stateIDBV as list of (s,e).
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
            n['max_depth'] = n.get('max_depth', float('inf'))

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        arr = json.loads(s)
        roots_map, arena_json = arr
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        return Model(roots_map, arena)

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

        values: Dict[int, ffi.GSSNode] = {}
        todo: List[Tuple[int, int, int]] = []  # (depth, counter, node_idx)
        counter = itertools.count()

        # 1. Seed
        initial_nodes: Dict[int, ffi.GSSNode] = {}
        for sid, gss in state_to_gss.items():
            if sid in self.roots_map:
                root_idx = self.roots_map[sid]
                if root_idx in initial_nodes:
                    existing = initial_nodes[root_idx]
                    initial_nodes[root_idx] = ffi.gss_merge_many_with_depth([existing, gss.clone_node()], 1)
                else:
                    initial_nodes[root_idx] = gss.clone_node()

        for node_idx, gss in initial_nodes.items():
            values[node_idx] = gss
            depth = self.arena.get(node_idx, {}).get("max_depth", float('inf'))
            heapq.heappush(todo, (depth, next(counter), node_idx))

        stopped_nodes: set[int] = set()

        while todo:
            _depth, _, u_idx = heapq.heappop(todo)

            if u_idx in stopped_nodes:
                continue

            if u_idx not in values:
                continue

            gss = values.pop(u_idx)

            # process_fn
            u_node = self.arena.get(u_idx, {})
            is_end = (u_node.get("value") or {}).get("end", False)

            keep_going = gss.is_ok()
            if is_end:
                final_mask = final_mask.union(gss.allowed_llm_tokens())

            if not keep_going:
                stopped_nodes.add(u_idx)
                continue

            # step_fn (grouped)
            for (pop, llm_rs), dests in u_node.get("children", []):
                peeks = ffi.gss_popn_collect(gss, int(pop))
                if not peeks: continue

                for dest_idx, state_bv_ranges in dests:
                    matched = []
                    for (sid_val, parent_node) in peeks:
                        ok = False
                        for (a, b) in state_bv_ranges:
                            if a <= sid_val <= b: ok = True; break
                        if ok:
                            matched.append(parent_node)
                    if not matched: continue

                    merged = ffi.gss_merge_many_with_depth(matched, 1)
                    if not llm_rs.is_empty():
                        bv = ffi.Bitset.from_ranges(llm_rs.intervals)
                        ffi.gss_allow_only_llm_tokens_and_prune(merged, bv)

                    if merged.is_ok():
                        dest_idx_int = int(dest_idx)
                        if dest_idx_int in values:
                            existing_gss = values[dest_idx_int]
                            values[dest_idx_int] = ffi.gss_merge_many_with_depth([existing_gss, merged], 1)
                        else:
                            values[dest_idx_int] = merged

                        child_depth = self.arena.get(dest_idx_int, {}).get("max_depth", float('inf'))
                        heapq.heappush(todo, (child_depth, next(counter), dest_idx_int))
        return final_mask
