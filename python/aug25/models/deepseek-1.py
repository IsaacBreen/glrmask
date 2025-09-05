import json
from typing import Dict, List, Tuple, Optional, Set
import time
from collections import defaultdict
from ..common_interface import GraphProvider
import _sep1 as ffi
from tqdm.auto import tqdm

class OptimizedModel(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map = {int(s): int(r) for s, r in roots_map}
        self.arena = arena
        self.max_depth: Dict[int, int] = {}
        self.children_map: Dict[int, List[Tuple[Tuple[int, ffi.Bitset], List[Tuple[int, ffi.Bitset]]]]] = {}

        for uid, node in tqdm(arena.items(), desc="Optimizing precompute3 BVs", total=len(arena)):
            uid_int = int(uid)
            self.max_depth[uid_int] = int(node.get("max_depth", 0))

            children = node.get("children") or []
            optimized_children = []
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv = ffi.Bitset.from_json_string(json.dumps(llm_bv_json))
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv = ffi.Bitset.from_json_string(json.dumps(state_bv_json))
                    new_dest_map.append((int(dest_idx), state_bv))
                optimized_children.append(((int(pop), llm_bv), new_dest_map))
            self.children_map[uid_int] = optimized_children

    @staticmethod
    def from_json_string(s: str) -> 'OptimizedModel':
        data = json.loads(s)
        roots_map = data['precomputed3']
        arena_json = data['trie3_god']
        arena = {int(k): v for k, v in arena_json.get("values", [])}
        return OptimizedModel(roots_map, arena)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        node_data = self.arena.get(node, {})
        return bool((node_data.get("value") or {}).get("end", False))

    def iter_edges(self, node: int, token: int):
        children = self.children_map.get(node, [])
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():
                        yield (pop, None, dest_idx)
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end):
                                yield (pop, sid, dest_idx)

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        t_start = time.time()
        final_mask = ffi.Bitset.zeros()
        values: Dict[int, Tuple[Set[ffi.GSSNode], ffi.Bitset]] = {}
        stopped: Set[int] = set()
        todo = defaultdict(set)

        # Initialize with roots
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(int(sid))
            if root_idx is None:
                continue
            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            if root_idx in values:
                gss_set, existing_mask = values[root_idx]
                gss_set.add(gss_clone)
                values[root_idx] = (gss_set, existing_mask.union(new_mask))
            else:
                values[root_idx] = ({gss_clone}, new_mask)
            todo[self.max_depth[root_idx]].add(root_idx)

        # Process nodes by depth
        while todo:
            current_depth = min(todo.keys())
            nodes = todo.pop(current_depth)
            for node_idx in nodes:
                if node_idx in stopped:
                    continue
                item = values.pop(node_idx, None)
                if item is None:
                    continue

                gss_set, llm_mask = item
                # Check end condition and filter OK states
                if self.is_end(node_idx):
                    final_mask = final_mask.union(llm_mask)
                gss_set = {g for g in gss_set if g}
                if not gss_set:
                    stopped.add(node_idx)
                    continue

                children = self.children_map.get(node_idx, [])
                # Group edges by pop count
                pop_edges = defaultdict(list)
                for (pop, llm_bv), dests in children:
                    pop_edges[pop].append((llm_bv, dests))

                # Precompute pop results for each unique pop count
                pop_results = {}
                for pop, edges in pop_edges.items():
                    peeks = []
                    for gss_node in gss_set:
                        peeks.extend(gss_node.popn_fast(pop))
                    pop_results[pop] = peeks

                # Process each edge group
                for pop, edges in pop_edges.items():
                    peeks = pop_results[pop]
                    if not peeks:
                        continue
                    for llm_bv, dests in edges:
                        edge_mask = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)
                        for dest_idx, state_bv in dests:
                            if state_bv.is_empty():
                                matched = [p for _, p in peeks]
                            else:
                                matched = [p for s, p in peeks if state_bv.contains(s)]
                            if not matched:
                                continue
                            child_gss = {p for p in matched if p}
                            if not child_gss:
                                continue

                            dest_node = dest_idx
                            if dest_node in values:
                                exist_set, exist_mask = values[dest_node]
                                exist_set.update(child_gss)
                                values[dest_node] = (exist_set, exist_mask.union(edge_mask))
                            else:
                                values[dest_node] = (child_gss, edge_mask)
                            todo[self.max_depth[dest_node]].add(dest_node)

        return final_mask