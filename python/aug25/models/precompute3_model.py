import json
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
from ..common_interface import GraphProvider
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm

class Model(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.arena = arena
        self.max_depth: Dict[int, int] = {}
        # Normalize BVs to ffi.Bitset; store stateIDBV as list of (s,e). Record max_depth.
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
                llm_bv = ffi.Bitset.from_json_string(json.dumps(llm_bv_json))
                newdm = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv = ffi.Bitset.from_json_string(json.dumps(state_bv_json))
                    newdm.append((int(dest_idx), state_bv))
                newch.append(((int(pop), llm_bv), newdm))
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
        for (pop, llm_bv), dests in self.arena.get(node, {}).get("children") or []:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty(): # Epsilon transition on GSS stack
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end):
                                yield (int(pop), sid, int(dest_idx))

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        t0 = time.time()
        print(f"[{time.time() - t0:.4f}] get_mask: start")

        # Final mask to return
        final_mask = ffi.Bitset.zeros()

        # values: pending value per trie node (accumulated GSS after merges)
        values: Dict[int, ffi.GSSNode] = {}
        # nodes that decided to stop (GSS not ok)
        stopped: set[int] = set()
        # depth scheduler: depth -> set(node_idx)
        todo: Dict[int, set[int]] = defaultdict(set)
        print(f"[{time.time() - t0:.4f}] get_mask: after init")

        # Seed: for each tokenizer state, map its filtered GSS to the corresponding trie root
        t_seed_start = time.time()
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)
            if root_idx in values:
                merged = ffi.gss_merge_many_with_depth([values[root_idx], gss.clone_node()], 9999999)
                # Re-enqueue only if the merge structurally changes the node
                if merged.ptr() != values[root_idx].ptr():
                    values[root_idx] = merged
            else:
                values[root_idx] = gss.clone_node()
            depth = self.max_depth[root_idx]
            todo[depth].add(root_idx)
        t_seed_end = time.time()
        print(f"[{time.time() - t0:.4f}] get_mask: seed loop took {t_seed_end - t_seed_start:.4f}s")

        # Main scheduler loop (depth-ascending)
        t_loop_start = time.time()

        # Timing accumulators
        time_pop_bucket = 0.0
        time_node_setup = 0.0
        time_end_check = 0.0
        time_popn_collect = 0.0
        time_filter_peeks = 0.0
        time_merge_matched = 0.0
        time_prune = 0.0
        time_merge_values = 0.0
        hits_pop_bucket = 0
        hits_node_setup = 0
        hits_end_check = 0
        hits_popn_collect = 0
        hits_filter_peeks = 0
        hits_merge_matched = 0
        hits_prune = 0
        hits_merge_values = 0

        loop_count = 0
        while todo:
            loop_count += 1

            # Pop the smallest depth bucket
            t1 = time.time()
            current_depth = min(todo.keys())
            node_indices = todo.pop(current_depth)
            time_pop_bucket += time.time() - t1
            hits_pop_bucket += 1

            for node_idx in list(node_indices):
                t1 = time.time()
                if node_idx in stopped:
                    continue

                agg: Optional[int] = values.pop(node_idx, None)
                if agg is None:
                    continue
                time_node_setup += time.time() - t1
                hits_node_setup += 1

                # Process callback (end-node handling + stop condition)
                t1 = time.time()
                if self.is_end(node_idx):
                    final_mask = final_mask.union(agg.allowed_llm_tokens())
                keep_going = agg.is_ok()
                time_end_check += time.time() - t1
                hits_end_check += 1
                if not keep_going:
                    stopped.add(node_idx)
                    continue

                # Grouped step over (pop, llm_bv)
                node = self.arena.get(node_idx, {})
                children = node.get("children") or []
                for (pop, llm_bv), dests in children:
                    t1 = time.time()
                    peeks = ffi.gss_popn_collect(agg, int(pop))
                    time_popn_collect += time.time() - t1
                    hits_popn_collect += 1
                    if not peeks:
                        continue

                    for dest_idx, state_bv in dests:
                        # Filter popped parents by state bitset
                        t1 = time.time()
                        matched = []
                        if not state_bv.is_empty():
                            for (sid_val, parent_node) in peeks:
                                if state_bv.contains(sid_val):
                                    matched.append(parent_node)
                        time_filter_peeks += time.time() - t1
                        hits_filter_peeks += 1
                        if not matched:
                            continue

                        # Merge matched parents
                        t1 = time.time()
                        child_gss = ffi.gss_merge_many_with_depth(matched, 999999999)
                        time_merge_matched += time.time() - t1
                        hits_merge_matched += 1

                        # Restrict by this edge's LLM token BV
                        t1 = time.time()
                        if not llm_bv.is_empty():
                            ffi.gss_allow_only_llm_tokens_and_prune(child_gss, llm_bv)
                        time_prune += time.time() - t1
                        hits_prune += 1
                        if not child_gss.is_ok():
                            continue

                        d = int(dest_idx)
                        t1 = time.time()
                        if d in values:
                            combined = ffi.gss_merge_many_with_depth([values[d], child_gss], 999999999)
                            # Only re-enqueue if effectively changed
                            if combined.ptr() == values[d].ptr():
                                continue
                            values[d] = combined
                        else:
                            values[d] = child_gss
                        time_merge_values += time.time() - t1
                        hits_merge_values += 1

                        child_depth = self.max_depth[d]
                        todo[child_depth].add(d)

        t_loop_end = time.time()
        print(f"[{time.time() - t0:.4f}] get_mask: scheduler loop finished in {t_loop_end - t_loop_start:.4f}s ({loop_count} iterations)")
        print(f"    - 1. Pop bucket:        {time_pop_bucket:9.4f}s ({hits_pop_bucket:8d} hits)")
        print(f"    - 2. Node setup:        {time_node_setup:9.4f}s ({hits_node_setup:8d} hits)")
        print(f"    - 3. End check:         {time_end_check:9.4f}s ({hits_end_check:8d} hits)")
        print(f"    - 4. Pop'n'collect:     {time_popn_collect:9.4f}s ({hits_popn_collect:8d} hits)")
        print(f"    - 5. Filter peeks:      {time_filter_peeks:9.4f}s ({hits_filter_peeks:8d} hits)")
        print(f"    - 6. Merge matched:     {time_merge_matched:9.4f}s ({hits_merge_matched:8d} hits)")
        print(f"    - 7. Prune:             {time_prune:9.4f}s ({hits_prune:8d} hits)")
        print(f"    - 8. Merge into values: {time_merge_values:9.4f}s ({hits_merge_values:8d} hits)")

        print(f"[{time.time() - t0:.4f}] get_mask: returning")
        return final_mask
