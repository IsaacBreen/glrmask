import json
from typing import Dict, List, Tuple, Optional
from .common_interface import GraphProvider
from .precompute3_model import Precompute3

# Reuse the RangeSet approach from tree-minimization-competition/trie_stuff.py
# Copy minimal RangeSet here (or import from a shared util if you prefer).

class RangeSet:
    def __init__(self, intervals: List[Tuple[int,int]]):
        self.intervals = self._merge_unsorted(intervals)
    @staticmethod
    def _merge_unsorted(ranges):
        items = sorted(((int(s),int(e)) for s,e in ranges), key=lambda x: x[0])
        if not items: return []
        out = []
        cs, ce = items[0]
        for ns, ne in items[1:]:
            if ns <= ce + 1:
                if ne > ce: ce = ne
            else:
                out.append((cs,ce)); cs,ce = ns,ne
        out.append((cs,ce))
        return out
    def contains(self, x: int) -> bool:
        import bisect
        starts = [s for s,_ in self.intervals]
        i = bisect.bisect_right(starts, x) - 1
        if i < 0: return False
        s,e = self.intervals[i]
        return s <= x <= e
    def union(self, other:'RangeSet')->'RangeSet':
        return RangeSet(self.intersection(other).intervals + self.difference(other).intervals + other.difference(self).intervals)  # fallback; not perf critical
    def intersection(self, other:'RangeSet')->'RangeSet':
        a, b = self.intervals, other.intervals
        i=j=0; out=[]
        while i<len(a) and j<len(b):
            s1,e1=a[i]; s2,e2=b[j]
            s=max(s1,s2); e=min(e1,e2)
            if s<=e: out.append((s,e))
            if e1<e2: i+=1
            else: j+=1
        return RangeSet(out)
    def is_empty(self)->bool: return len(self.intervals)==0

class Precompute2(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int,int]], arena: Dict[int,dict]):
        self.roots_map = dict((int(s),int(r)) for s,r in roots_map)
        self.arena = arena
        # Convert BVs in-place to RangeSet
        for n in self.arena.values():
            val = n.get("value") or {}
            if "live_tokens" in val:
                lv = [(int(a),int(b)) for a,b in val["live_tokens"]]
                val["live_tokens"] = RangeSet(lv)
                n["value"] = val
            ch = n.get("children") or []
            newch = []
            for edge_key, dest_map in ch:
                newdm = []
                for di, bv in dest_map:
                    rs = RangeSet([(int(a),int(b)) for a,b in bv])
                    newdm.append((int(di), rs))
                pk, sid = edge_key
                sidp = None if sid is None else int(sid)
                newch.append(((int(pk), sidp), newdm))
            n["children"] = newch

    @staticmethod
    def from_json_string(s: str) -> 'Precompute2':
        arr = json.loads(s)
        roots_map, arena_json = arr
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k,v in arena_values}
        return Precompute2(roots_map, arena)

    def get_root(self, state_id:int)->int:
        return self.roots_map[int(state_id)]

    def is_end(self, node:int)->bool:
        return bool((self.arena[node].get("value") or {}).get("end", False))

    def iter_edges(self, node:int, token:int):
        # Reference edges are token-gated on their BVs. This provider yields only matching edges.
        for (pop, sid), dests in self.arena[node].get("children") or []:
            for dest, rs in dests:
                if rs.contains(token):
                    yield (int(pop), sid, int(dest))

    def to_precompute3(self) -> 'Precompute3':
        """
        Convert to the normalized precompute3 format in-memory:
           - group edges for each node by (pop, llm_bv, dest) -> collect state_ids bitset per dest
           - But per spec, precompute3 groups by (pop, llm_bv) then dest->stateIDBV.
        Minimal faithful transformation.
        """
        arena3: Dict[int, dict] = {}
        for u, node in self.arena.items():
            out_children = []
            # group: (pop, llm_bv serialized key) -> dict(dest -> set(state_ids))
            groups: Dict[Tuple[int, Tuple[Tuple[int,int],...]], Dict[int, set]] = {}
            for (pop, sid), dests in node.get("children") or []:
                for dest, llm_rs in dests:
                    key = (pop, tuple(llm_rs.intervals))
                    dmap = groups.setdefault(key, {})
                    if sid is not None:
                        dmap.setdefault(dest, set()).add(int(sid))
                    else:
                        dmap.setdefault(dest, set()).add(-1)  # use -1 to mark Any
            for (pop, llm_key), dmap in groups.items():
                llm_bv = list(list(p) for p in llm_key)
                # dest_map: dest -> StateIDBV (ranges)
                dest_map = []
                for dest, sids in dmap.items():
                    if -1 in sids:
                        state_bv = [[0, (1<<63)-1]]  # symbolic “all”; your consumer should treat this specially
                    else:
                        sids_list = sorted(sids)
                        # convert discrete ids to disjoint ranges
                        rngs=[]
                        if sids_list:
                            cs = ce = sids_list[0]
                            for v in sids_list[1:]:
                                if v == ce+1: ce=v
                                else: rngs.append([cs,ce]); cs=ce=v
                            rngs.append([cs,ce])
                        state_bv = rngs
                    dest_map.append([int(dest), state_bv])
                out_children.append([[int(pop), llm_bv], dest_map])
            arena3[u] = {
                "value": {"end": bool((node.get("value") or {}).get("end", False))},
                "children": out_children,
            }
        roots = list(self.roots_map.items())
        return Precompute3(roots, arena3)
import json
from typing import Dict, List, Tuple, Optional
from .common_interface import GraphProvider
from .precompute3_model import Precompute3


# Reuse the RangeSet approach from tree-minimization-competition/trie_stuff.py
# Copy minimal RangeSet here (or import from a shared util if you prefer).

class RangeSet:
    def __init__(self, intervals: List[Tuple[int, int]]):
        self.intervals = self._merge_unsorted(intervals)

    @staticmethod
    def _merge_unsorted(ranges):
        items = sorted(((int(s), int(e)) for s, e in ranges), key=lambda x: x[0])
        if not items: return []
        out = []
        cs, ce = items[0]
        for ns, ne in items[1:]:
            if ns <= ce + 1:
                if ne > ce: ce = ne
            else:
                out.append((cs, ce));
                cs, ce = ns, ne
        out.append((cs, ce))
        return out

    def contains(self, x: int) -> bool:
        import bisect
        starts = [s for s, _ in self.intervals]
        i = bisect.bisect_right(starts, x) - 1
        if i < 0: return False
        s, e = self.intervals[i]
        return s <= x <= e

    def union(self, other: 'RangeSet') -> 'RangeSet':
        return RangeSet(self.intervals + other.intervals)

    def intersection(self, other: 'RangeSet') -> 'RangeSet':
        a, b = self.intervals, other.intervals
        i = j = 0;
        out = []
        while i < len(a) and j < len(b):
            s1, e1 = a[i];
            s2, e2 = b[j]
            s = max(s1, s2);
            e = min(e1, e2)
            if s <= e: out.append((s, e))
            if e1 < e2:
                i += 1
            else:
                j += 1
        return RangeSet(out)

    def is_empty(self) -> bool: return len(self.intervals) == 0


class Precompute2(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.arena = arena
        # Convert BVs in-place to RangeSet
        for n in self.arena.values():
            val = n.get("value") or {}
            if "live_tokens" in val:
                lv = [(int(a), int(b)) for a, b in val["live_tokens"]]
                val["live_tokens"] = RangeSet(lv)
                n["value"] = val
            ch = n.get("children") or []
            newch = []
            for edge_key, dest_map in ch:
                newdm = []
                for di, bv in dest_map:
                    rs = RangeSet([(int(a), int(b)) for a, b in bv])
                    newdm.append((int(di), rs))
                pk, sid = edge_key
                sidp = None if sid is None else int(sid)
                newch.append(((int(pk), sidp), newdm))
            n["children"] = newch

    @staticmethod
    def from_json_string(s: str) -> 'Precompute2':
        arr = json.loads(s)
        roots_map, arena_json = arr
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        return Precompute2(roots_map, arena)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena[node].get("value") or {}).get("end", False))

    def iter_edges(self, node: int, token: int):
        # Reference edges are token-gated on their BVs. This provider yields only matching edges.
        for (pop, sid), dests in self.arena[node].get("children") or []:
            for dest, rs in dests:
                if rs.contains(token):
                    yield (int(pop), sid, int(dest))

    def to_precompute3(self) -> 'Precompute3':
        arena3: Dict[int, dict] = {}
        for u, node in self.arena.items():
            out_children = []
            groups: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], Dict[int, set]] = {}
            for (pop, sid), dests in node.get("children") or []:
                for dest, llm_rs in dests:
                    key = (pop, tuple(llm_rs.intervals))
                    dmap = groups.setdefault(key, {})
                    dmap.setdefault(dest, set()).add(sid)
            for (pop, llm_key), dmap in groups.items():
                llm_bv = list(list(p) for p in llm_key)
                dest_map = []
                for dest, sids in dmap.items():
                    sids_list = sorted([s for s in sids if s is not None])
                    rngs = []
                    if sids_list:
                        cs = ce = sids_list[0]
                        for v in sids_list[1:]:
                            if v == ce + 1: ce = v
                            else: rngs.append([cs, ce]); cs = ce = v
                        rngs.append([cs, ce])
                    dest_map.append([int(dest), rngs])
                out_children.append([[int(pop), llm_bv], dest_map])
            arena3[u] = {
                "value": {"end": bool((node.get("value") or {}).get("end", False))},
                "children": out_children,
            }
        roots = list(self.roots_map.items())
        return Precompute3(roots, arena3)
