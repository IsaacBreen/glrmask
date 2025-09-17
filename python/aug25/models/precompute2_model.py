import json
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
from tqdm.auto import tqdm

class Model(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict], max_state_id: int):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.constraint: Optional[ffi.GrammarConstraint] = None
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
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int):
        # Reference edges are token-gated on their BVs. This provider yields only matching edges.
        for (pop, sid), dests in self.arena.get(node, {}).get("children") or []:
            for dest, rs in dests:
                if rs.contains(token):
                    yield (int(pop), sid, int(dest))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        print("\n--- get_mask START ---")
        state_to_gss = self.constraint_state.filtered_state_gss_map()

        final_mask = ffi.Bitset.zeros()
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}
        stopped: set[int] = set()
        todo: Dict[int, set[int]] = defaultdict(set)

        # Seed
        print("\n--- Seeding work queue ---")
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)
            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            if root_idx in values:
                existing_gss, existing_mask = values[root_idx]
                print(f"    - MERGE: gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={gss_clone.ptr()}, mask2={new_mask.to_ranges()}")
                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, gss_clone], 1)
                merged_mask = existing_mask.union(new_mask)
                values[root_idx] = (merged_gss, merged_mask)
                print(f"      - Merged result: gss_ptr={merged_gss.ptr()}, mask={merged_mask.to_ranges()}")
            else:
                values[root_idx] = (gss_clone, new_mask)
            depth = self.max_depth.get(root_idx, 0)
            todo[depth].add(root_idx)

        # Main loop
        print("\n--- Main loop ---")
        iter_count = 0
        while todo:
            iter_count += 1
            current_depth = min(todo.keys())
            node_indices = todo.pop(current_depth)
            print(f"\n[{iter_count}] Processing depth={current_depth}, nodes={node_indices}")


            for node_idx in list(node_indices):
                if node_idx in stopped:
                    print(f"  - Node {node_idx}: SKIPPING (already stopped)")
                    continue

                item: Optional[Tuple[ffi.GSSNode, ffi.Bitset]] = values.pop(node_idx, None)
                if item is None:
                    print(f"  - Node {node_idx}: SKIPPING (no value)")
                    continue
                gss_node, llm_mask = item
                print(f"  - PROCESS: node_ptr={node_idx}, gss_ptr={gss_node.ptr()}, mask={llm_mask.to_ranges()}")

                # Process
                if self.is_end(node_idx):
                    print(f"    - END NODE found. Updating final_mask.")
                    print(f"      - final_mask before: {final_mask.to_ranges()}")
                    gss_active_tokens = gss_node.allowed_llm_tokens()
                    tokens_to_add = llm_mask.intersection(gss_active_tokens)
                    print(f"      - glr_active_tokens to union: {tokens_to_add.to_ranges()}")
                    final_mask = final_mask.union(tokens_to_add)
                    print(f"      - final_mask after:  {final_mask.to_ranges()}")

                if not gss_node.is_ok():
                    stopped.add(node_idx)
                    print(f"    - STOPPING node {node_idx} (GSS not alive)")
                    continue

                # Step

                # Step
                node = self.arena.get(node_idx, {})
                children = node.get("children") or []
                for (pop, sid_opt), dests in children:
                    print(f"    - Edge: pop={pop}, sid_opt={sid_opt}")
                    peeks = ffi.gss_popn_collect(gss_node, int(pop))
                    print(f"      - Found {len(peeks)} peeks from GSS")
                    if not peeks:
                        continue


                    # Filter peeks by state_id
                    if sid_opt is None:
                        matched_parents = [p for _, p in peeks]
                    else:
                        sid_val = int(sid_opt)
                        matched_parents = [p for sid, p in peeks if sid == sid_val]
                    print(f"        - Matched {len(matched_parents)} parent GSS nodes")

                    if not matched_parents:
                        continue

                    for dest_idx, llm_rs in dests:
                        print(f"      - Dest: idx={dest_idx}, llm_rs={llm_rs.intervals}")
                        edge_bv = ffi.Bitset.from_ranges(llm_rs.intervals)
                        child_llm_mask = llm_mask.intersection(edge_bv)
                        print(f"        - Child mask: {child_llm_mask.to_ranges()}")
                        if child_llm_mask.is_empty():
                            continue


                        if not child_gss.is_alive():
                            continue

                        d = int(dest_idx)
                        if d in values:
                            existing_gss, existing_mask = values[d]
                            print(f"        - Enqueue {d}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={child_gss.ptr()}, mask2={child_llm_mask.to_ranges()}")
                            combined_gss = ffi.gss_merge_many_with_depth([existing_gss, child_gss], 1)
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (combined_gss, combined_mask)
                            print(f"          - Merged result: gss_ptr={combined_gss.ptr()}, mask={combined_mask.to_ranges()}")
                        else:
                            values[d] = (child_gss, child_llm_mask)
                            print(f"        - Enqueue {d}: CREATING gss_ptr={child_gss.ptr()}, mask={child_llm_mask.to_ranges()}")

                        child_depth = self.max_depth.get(d, 0)
                        todo[child_depth].add(d)

        print("\n--- get_mask END ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())
