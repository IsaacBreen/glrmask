import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
from tqdm.auto import tqdm

class Model(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict], max_state_id: int):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.arena = arena
        self.max_depth: Dict[int, int] = {}
        # Convert precompute3 graph structure to precompute2-like structure
        for uid, n in tqdm(self.arena.items(), desc="Converting precompute3->precompute2", total=len(self.arena)):
            try:
                self.max_depth[int(uid)] = int(n.get("max_depth", 0))
            except Exception:
                self.max_depth[int(uid)] = 0

            p3_children = n.get("children") or []
            
            # Aggregate into precompute2 format: (pop, sid) -> {dest -> llm_bv}
            p2_children_agg = defaultdict(lambda: defaultdict(RangeSet.empty))

            for edge_key, dest_map in tqdm(p3_children, desc="Aggregating children", leave=False, disable=True):
                pop, llm_bv_json = edge_key
                llm_rs = RangeSet.from_ranges(llm_bv_json)
                if llm_rs.is_empty():
                    continue

                for dest_idx, state_bv_ranges in tqdm(dest_map, desc="Aggregating dests", leave=False, disable=True):
                    if not state_bv_ranges: # Corresponds to Option<StateID> == None
                        p2_key = (int(pop), None)
                        p2_children_agg[p2_key][int(dest_idx)] = p2_children_agg[p2_key][int(dest_idx)].union(llm_rs)
                    else:
                        for start, end in tqdm(state_bv_ranges, desc="Aggregating ranges", leave=False, disable=True):
                            end = min(int(end), start)
                            for sid in tqdm(list(range(int(start), end + 1)), desc="Aggregating ranges", leave=False, disable=True):
                                p2_key = (int(pop), sid)
                                p2_children_agg[p2_key][int(dest_idx)] = p2_children_agg[p2_key][int(dest_idx)].union(llm_rs)
            
            # Convert aggregated map to final list format
            new_children = []
            for (pop, sid), dests in tqdm(p2_children_agg.items(), desc="Converting to list", leave=False, disable=True):
                dest_list = list(dests.items())
                new_children.append(((pop, sid), dest_list))
            
            n["children"] = new_children

    @staticmethod
    def from_json_string(s: str) -> "Model":
        data = json.loads(s)
        # This model uses the precompute3 graph, as it's the most detailed representation
        roots_map = data['precomputed3']
        arena_json = data['trie3_god']
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        max_state_id = int(max(dict(data['parser']['stage_7_table']).keys()))
        model = Model(roots_map, arena, max_state_id)
        constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(constraint)
        return model

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

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        state_to_gss = self.constraint_state.get_state_to_gss_map()
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

                agg: Optional[int] = values.pop(node_idx, None)
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

        return RangeSet.from_ranges(final_mask.to_ranges())
