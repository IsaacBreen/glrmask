import json
from typing import Dict, List, Tuple
from collections import defaultdict
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm

class Model(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.arena = arena
        self.max_depth: Dict[int, int] = {}
        # Normalize BVs to RangeSet; store stateIDBV as list of (s,e). Record max_depth.
        for uid, node in tqdm(self.arena.items(), desc="Normalizing precompute3 BVs", total=len(self.arena)):
            # Record node max_depth (if absent, assume 0)
            try:
                self.max_depth[int(uid)] = int(node.get("max_depth", 0))
            except Exception:
                self.max_depth[int(uid)] = 0

            ch = node.get("children") or []
            newch = []
            for edge_key, dest_map in ch:
                pop, llm_bv_json = edge_key
                llm_rs = RangeSet.from_json(llm_bv_json)
                newdm = []
                for dest_idx, state_bv in dest_map:
                    newdm.append((int(dest_idx), [(int(a), int(b)) for a, b in state_bv]))
                newch.append(((int(pop), llm_rs), newdm))
            node["children"] = newch

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data['precomputed3']
        arena_json = data['trie3_god']
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        return Model(roots_map, arena)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("end", False))

    def iter_edges(self, node: int, token: int):
        # For equivalence checking, we must "explode" the state_bv into individual
        # state IDs to match the GraphProvider interface expected by the checker.
        # This is not used by the performance-critical get_mask() method.
        for (pop, llm_rs), dests in self.arena.get(node, {}).get("children") or []:
            if llm_rs.contains(token):
                for dest_idx, state_bv_ranges in dests:
                    if not state_bv_ranges: # Epsilon transition on GSS stack
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv_ranges:
                            for sid in range(start, end + 1):
                                yield (int(pop), sid, int(dest_idx))

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        # Final mask to return
        final_mask = ffi.Bitset.zeros()

        # values: pending value per trie node (accumulated GSS after merges)
        values: Dict[int, ffi.GSSNode] = {}
        # nodes that decided to stop (GSS not ok)
        stopped: set[int] = set()
        # depth scheduler: depth -> set(node_idx)
        todo: Dict[int, set[int]] = defaultdict(set)

        # Seed: for each tokenizer state, map its filtered GSS to the corresponding trie root
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)
            if root_idx in values:
                merged = ffi.gss_merge_many_with_depth([values[root_idx], gss.clone_node()], 1)
                # Re-enqueue only if the merge structurally changes the node
                if merged.ptr() != values[root_idx].ptr():
                    values[root_idx] = merged
            else:
                values[root_idx] = gss.clone_node()
            depth = self.max_depth.get(root_idx, 0)
            todo[depth].add(root_idx)

        # Main scheduler loop (depth-ascending)
        while todo:
            # Pop the smallest depth bucket
            current_depth = min(todo.keys())
            node_indices = todo.pop(current_depth)

            for node_idx in list(node_indices):
                if node_idx in stopped:
                    continue

                agg = values.pop(node_idx, None)
                if agg is None:
                    continue

                # Process callback (end-node handling + stop condition)
                if self.is_end(node_idx):
                    final_mask = final_mask.union(agg.allowed_llm_tokens())
                keep_going = agg.is_ok()
                if not keep_going:
                    stopped.add(node_idx)
                    continue

                # Grouped step over (pop, llm_rs)
                node = self.arena.get(node_idx, {})
                children = node.get("children") or []
                for (pop, llm_rs), dests in children:
                    peeks = ffi.gss_popn_collect(agg, int(pop))
                    if not peeks:
                        continue

                    for dest_idx, state_bv in dests:
                        # Filter popped parents by state bitset
                        matched = []
                        if state_bv:
                            for (sid_val, parent_node) in peeks:
                                for (a, b) in state_bv:
                                    if a <= sid_val <= b:
                                        matched.append(parent_node)
                                        break
                        if not matched:
                            continue

                        # Merge matched parents
                        child_gss = ffi.gss_merge_many_with_depth(matched, 1)
                        # Restrict by this edge's LLM token BV
                        if llm_rs.intervals:
                            edge_bv = ffi.Bitset.from_ranges(llm_rs.intervals)
                            ffi.gss_allow_only_llm_tokens_and_prune(child_gss, edge_bv)
                        if not child_gss.is_ok():
                            continue

                        d = int(dest_idx)
                        if d in values:
                            combined = ffi.gss_merge_many_with_depth([values[d], child_gss], 1)
                            # Only re-enqueue if effectively changed
                            if combined.ptr() == values[d].ptr():
                                continue
                            values[d] = combined
                        else:
                            values[d] = child_gss

                        child_depth = self.max_depth.get(d, 0)
                        todo[child_depth].add(d)

        return final_mask