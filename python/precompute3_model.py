import json
from typing import Dict, List, Tuple
from .common_interface import GraphProvider
import _sep1 as ffi  # the compiled module

class RangeSet:
    def __init__(self, intervals: List[Tuple[int,int]]):
        self.intervals = self._merge_unsorted(intervals)
    @staticmethod
    def _merge_unsorted(ranges):
        items = sorted(((int(s),int(e)) for s,e in ranges), key=lambda x: x[0])
        if not items: return []
        out=[]; cs,ce=items[0]
        for ns,ne in items[1:]:
            if ns<=ce+1:
                if ne>ce: ce=ne
            else:
                out.append((cs,ce)); cs,ce=ns,ne
        out.append((cs,ce)); return out
    def contains(self,x:int)->bool:
        import bisect
        starts=[s for s,_ in self.intervals]
        i=bisect.bisect_right(starts,x)-1
        if i<0: return False
        s,e=self.intervals[i]; return s<=x<=e

class Precompute3(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int,int]], arena: Dict[int,dict]):
        self.roots_map = dict((int(s),int(r)) for s,r in roots_map)
        self.arena = arena
        # Normalize BVs to RangeSet; store stateIDBV as list of (s,e).
        for n in self.arena.values():
            ch = n.get("children") or []
            newch=[]
            for edge_key, dest_map in ch:
                pop, llm_bv_json = edge_key
                llm_rs = RangeSet([(int(a),int(b)) for a,b in llm_bv_json])
                newdm=[]
                for dest_idx, state_bv in dest_map:
                    # state_bv: list of [s,e], or sentinel for ALL
                    # We'll keep as list; lookup uses contains() against a small helper.
                    newdm.append((int(dest_idx), [(int(a),int(b)) for a,b in state_bv]))
                newch.append(((int(pop), llm_rs), newdm))
            n["children"] = newch

    @staticmethod
    def from_json_string(s: str) -> 'Precompute3':
        arr = json.loads(s)
        roots_map, arena_json = arr
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k,v in arena_values}
        return Precompute3(roots_map, arena)

    def get_root(self, state_id:int)->int:
        return self.roots_map[int(state_id)]

    def is_end(self, node:int)->bool:
        return bool((self.arena[node].get("value") or {}).get("end", False))

    def iter_edges(self, node:int, token:int):
        # For precompute3, llm BV filters at the edge label; we leave filtering to caller when needed.
        for (pop, llm_rs), dests in self.arena[node].get("children") or []:
            yield (int(pop), llm_rs, dests)

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        final_mask = ffi.Bitset.zeros()
        # Simple queue: list of (trie_node_index, GSSNode)
        from collections import deque
        q = deque()
        visited = set()
        # Seed from roots
        for sid, gss in state_to_gss.items():
            if sid in self.roots_map:
                q.append((self.roots_map[sid], gss.clone_node()))
        while q:
            u, gss = q.popleft()
            key = (u, id(gss))
            if key in visited:
                continue
            visited.add(key)
            if self.is_end(u):
                final_mask = final_mask.union(gss.allowed_llm_tokens())
            for (pop, llm_rs), dests in self.arena[u].get("children") or []:
                # Pop in GSS and filter by state id BV
                peeks = ffi.gss_popn_collect(gss, int(pop))
                # Build dest expansions per dest node
                for dest_idx, state_bv in dests:
                    # Expand peeks that match state id
                    matched = []
                    for (sid_val, parent_node) in peeks:
                        # Treat "all" by a convention: state_bv empty means none; if large range (0..huge) treat as all
                        ok = False
                        for (a,b) in state_bv:
                            if a <= sid_val <= b:
                                ok = True
                                break
                        if ok:
                            matched.append(parent_node)
                    if not matched:
                        continue
                    merged = ffi.gss_merge_many_with_depth(matched, 1)
                    # Intersect allowed LLM tokens for this edge
                    if llm_rs.intervals:
                        bv = ffi.Bitset.from_ranges(llm_rs.intervals)
                        ffi.gss_allow_only_llm_tokens_and_prune(merged, bv)
                    if merged.is_ok():
                        q.append((int(dest_idx), merged))
        return final_mask
import json
from typing import Dict, List, Tuple
from .common_interface import GraphProvider
import _sep1 as ffi  # the compiled module


class RangeSet:
    def __init__(self, intervals: List[Tuple[int, int]]):
        self.intervals = self._merge_unsorted(intervals)

    @staticmethod
    def _merge_unsorted(ranges):
        items = sorted(((int(s), int(e)) for s, e in ranges), key=lambda x: x[0])
        if not items: return []
        out = [];
        cs, ce = items[0]
        for ns, ne in items[1:]:
            if ns <= ce + 1:
                if ne > ce: ce = ne
            else:
                out.append((cs, ce));
                cs, ce = ns, ne
        out.append((cs, ce));
        return out

    def contains(self, x: int) -> bool:
        import bisect
        starts = [s for s, _ in self.intervals]
        i = bisect.bisect_right(starts, x) - 1
        if i < 0: return False
        s, e = self.intervals[i];
        return s <= x <= e


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
                llm_rs = RangeSet([(int(a), int(b)) for a, b in llm_bv_json])
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
        return bool((self.arena[node].get("value") or {}).get("end", False))

    def iter_edges(self, node: int, token: int):
        # For precompute3, llm BV filters at the edge label; we leave filtering to caller when needed.
        for (pop, llm_rs), dests in self.arena[node].get("children") or []:
            yield (int(pop), llm_rs, dests)

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        final_mask = ffi.Bitset.zeros()
        # Simple queue: list of (trie_node_index, GSSNode)
        from collections import deque
        q = deque()
        visited = {}  # (u, gss_ptr) -> gss_node to merge into
        # Seed from roots
        for sid, gss in state_to_gss.items():
            if sid in self.roots_map:
                q.append((self.roots_map[sid], gss.clone_node()))

        while q:
            u, gss = q.popleft()

            if self.is_end(u):
                final_mask = final_mask.union(gss.allowed_llm_tokens())

            for (pop, llm_rs), dests in self.arena[u].get("children") or []:
                peeks = ffi.gss_popn_collect(gss, int(pop))
                for dest_idx, state_bv in dests:
                    matched = []
                    for (sid_val, parent_node) in peeks:
                        ok = False
                        for (a, b) in state_bv:
                            if a <= sid_val <= b:
                                ok = True
                                break
                        if ok:
                            matched.append(parent_node)
                    if not matched:
                        continue

                    merged = ffi.gss_merge_many_with_depth(matched, 1)

                    if llm_rs.intervals:
                        bv = ffi.Bitset.from_ranges(llm_rs.intervals)
                        ffi.gss_allow_only_llm_tokens_and_prune(merged, bv)

                    if merged.is_ok():
                        # Use a tuple of sorted node pointers as a key for the GSS state to avoid hashing the GSSNode itself
                        # This is a simplification; a proper hash would be better.
                        # For now, we just re-queue. A more advanced version would merge states.
                        q.append((int(dest_idx), merged))
        return final_mask
