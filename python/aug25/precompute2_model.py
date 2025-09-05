import json
from typing import Dict, List, Tuple
from collections import defaultdict
from .common_interface import GraphProvider, RangeSet
import _sep1 as ffi

class Model(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.arena = arena
        self.max_depth: Dict[int, int] = {}
        # Convert BVs in-place to RangeSet
        for uid, n in self.arena.items():
            try:
                self.max_depth[int(uid)] = int(n.get("max_depth", 0))
            except Exception:
                self.max_depth[int(uid)] = 0

            val = n.get("value") or {}
            if "live_tokens" in val and val["live_tokens"] is not None:
                val["live_tokens"] = RangeSet.from_json(val["live_tokens"])
            else:
                val["live_tokens"] = RangeSet.empty()
            n["value"] = val

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
        for (pop, llm_rs), dests in self.arena.get(node, {}).get("children") or []:
            if llm_rs.contains(token):
                for dest_idx, state_bv_ranges in dests:
                    if not state_bv_ranges:
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv_ranges:
                            for sid in range(start, end + 1):
                                yield (int(pop), sid, int(dest_idx))

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        final_mask = ffi.Bitset.zeros()
        values: Dict[int, ffi.GSSNode] = {}
        stopped: set[int] = set()
        todo: Dict[int, set[int]] = defaultdict(set)

        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)
            if root_idx in values:
                merged = ffi.gss_merge_many_with_depth([values[root_idx], gss.clone_node()], 1)
                if merged.ptr() != values[root_idx].ptr():
                    values[root_idx] = merged
            else:
                values[root_idx] = gss.clone_node()
            depth = self.max_depth.get(root_idx, 0)
            todo[depth].add(root_idx)

        while todo:
            current_depth = min(todo.keys())
            node_indices = todo.pop(current_depth)

            for node_idx in list(node_indices):
                if node_idx in stopped:
                    continue

                agg = values.pop(node_idx, None)
                if agg is None:
                    continue

                if self.is_end(node_idx):
                    final_mask = final_mask.union(agg.allowed_llm_tokens())

                if not agg.is_ok():
                    stopped.add(node_idx)
                    continue

                node = self.arena.get(node_idx, {})
                children = node.get("children") or []
                for (pop, llm_rs), dests in children:
                    peeks = ffi.gss_popn_collect(agg, int(pop))
                    if not peeks:
                        continue

                    for dest_idx, state_bv in dests:
                        matched = []
                        if not state_bv: # Empty state_bv means match all (like Option<StateID>::None)
                            matched = [p for _, p in peeks]
                        else:
                            for (sid_val, parent_node) in peeks:
                                for (a, b) in state_bv:
                                    if a <= sid_val <= b:
                                        matched.append(parent_node)
                                        break
                        if not matched:
                            continue

                        child_gss = ffi.gss_merge_many_with_depth(matched, 1)

                        if llm_rs.intervals:
                            edge_bv = ffi.Bitset.from_ranges(llm_rs.intervals)
                            ffi.gss_allow_only_llm_tokens_and_prune(child_gss, edge_bv)
                        if not child_gss.is_ok():
                            continue

                        d = int(dest_idx)
                        if d in values:
                            combined = ffi.gss_merge_many_with_depth([values[d], child_gss], 1)
                            if combined.ptr() == values[d].ptr():
                                continue
                            values[d] = combined
                        else:
                            values[d] = child_gss

                        child_depth = self.max_depth.get(d, 0)
                        todo[child_depth].add(d)

        return final_mask
