import json
from typing import Dict, List, Tuple
from .common_interface import GraphProvider, RangeSet
from .precompute3_model import Model as Precompute3Model
import _sep1 as ffi

class Model(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.arena = arena
        # Convert BVs in-place to RangeSet
        for n in self.arena.values():
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

    def to_precompute3(self) -> 'Precompute3Model':
        arena3: Dict[int, dict] = {}
        for u, node in self.arena.items():
            out_children = []
            groups: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], Dict[int, set]] = {}
            for (pop, sid), dests in node.get("children") or []:
                for dest, llm_rs in dests:
                    key = (pop, llm_rs.intervals)
                    dmap = groups.setdefault(key, {})
                    dmap.setdefault(dest, set()).add(sid if sid is not None else -1)
            for (pop, llm_key), dmap in groups.items():
                llm_bv = list(list(p) for p in llm_key)
                dest_map = []
                for dest, sids in dmap.items():
                    if -1 in sids:
                        state_bv = [[0, 2**31 - 1]]
                    else:
                        sids_list = sorted([s for s in sids if s is not None])
                        rngs = []
                        if sids_list:
                            cs = ce = sids_list[0]
                            for v in sids_list[1:]:
                                if v == ce + 1:
                                    ce = v
                                else:
                                    rngs.append([cs, ce])
                                    cs = ce = v
                            rngs.append([cs, ce])
                        state_bv = rngs
                    dest_map.append([int(dest), state_bv])
                out_children.append([[int(pop), llm_bv], dest_map])
            arena3[u] = {
                "value": {"end": bool((node.get("value") or {}).get("end", False))},
                "children": out_children,
            }
        roots = list(self.roots_map.items())
        return Precompute3Model(roots, arena3)

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        # This is inefficient as it converts on every call.
        # A real competitor would implement this directly.
        p3_model = self.to_precompute3()
        return p3_model.get_mask(state_to_gss)
