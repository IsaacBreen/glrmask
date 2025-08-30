import json
from typing import Dict, List, Tuple
from collections import defaultdict
from common_interface import GraphProvider, RangeSet
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
                newdm = []
                for di, bv in dest_map:
                    rs = RangeSet.from_json(bv)
                    newdm.append((int(di), rs))
                pk, sid = edge_key
                sidp = None if sid is None else int(sid)
                newch.append(((int(pk), sidp), newdm))
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
        # Reference edges are token-gated on their BVs. This provider yields only matching edges.
        for (pop, sid), dests in self.arena.get(node, {}).get("children") or []:
            for dest, rs in dests:
                if rs.contains(token):
                    yield (int(pop), sid, int(dest))

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        final_mask = ffi.Bitset.zeros()
        values: Dict[int, ffi.GSSNode] = {}
        stopped: set[int] = set()
        todo: Dict[int, set[int]] = defaultdict(set)

        # Seed
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

        # Main loop
        while todo:
            current_depth = min(todo.keys())
            node_indices = todo.pop(current_depth)

            for node_idx in list(node_indices):
                if node_idx in stopped:
                    continue

                agg = values.pop(node_idx, None)
                if agg is None:
                    continue

                # Process
                if self.is_end(node_idx):
                    final_mask = final_mask.union(agg.allowed_llm_tokens())

                if not agg.is_ok():
                    stopped.add(node_idx)
                    continue

                # Step
                node = self.arena.get(node_idx, {})
                children = node.get("children") or []
                for (pop, sid_opt), dests in children:
                    peeks = ffi.gss_popn_collect(agg, int(pop))
                    if not peeks:
                        continue

                    # Filter peeks by state_id
                    matched_parents = []
                    if sid_opt is None:
                        matched_parents = [p for _, p in peeks]
                    else:
                        sid_val = int(sid_opt)
                        matched_parents = [p for sid, p in peeks if sid == sid_val]

                    if not matched_parents:
                        continue

                    child_gss = ffi.gss_merge_many_with_depth(matched_parents, 1)
                    if not child_gss.is_ok():
                        continue

                    for dest_idx, llm_rs in dests:
                        gss_for_dest = child_gss.clone_node()

                        if llm_rs.intervals:
                            edge_bv = ffi.Bitset.from_ranges(llm_rs.intervals)
                            ffi.gss_allow_only_llm_tokens_and_prune(gss_for_dest, edge_bv)

                        if not gss_for_dest.is_ok():
                            continue

                        d = int(dest_idx)
                        if d in values:
                            combined = ffi.gss_merge_many_with_depth([values[d], gss_for_dest], 1)
                            if combined.ptr() == values[d].ptr():
                                continue
                            values[d] = combined
                        else:
                            values[d] = gss_for_dest

                        child_depth = self.max_depth.get(d, 0)
                        todo[child_depth].add(d)

        return final_mask
