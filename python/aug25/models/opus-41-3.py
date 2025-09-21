import heapq
import _sep1 as ffi
from typing import Dict, List, Set, Tuple, Optional, DefaultDict
from collections import defaultdict

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel
from ..common_interface import GraphProvider, RangeSet

try:
    profile
except NameError:
    def profile(func): return func


class Model(GraphProvider):
    def __init__(self, inner_model: InnerModel):
        self.inner_model: InnerModel = inner_model
        im: InnerModel = self.inner_model
        self.arena: Dict[int, dict] = im.arena
        self.roots_map: Dict[int, int] = im.roots_map
        self.max_depth: Dict[int, int] = im.max_depth
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = im.possible_matches_cache
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map

        # Precompute optimized graph structures
        self._precompute_graph_structures()

    def _precompute_graph_structures(self):
        """Precompute efficient graph structures for traversal."""
        # 1. Build adjacency lists with direct access
        self.forward_edges: Dict[int, List[Tuple]] = {}
        self.end_nodes: Set[int] = set()

        for node, data in self.arena.items():
            if self.is_end(node):
                self.end_nodes.add(node)

            children = (data or {}).get("children")
            if children:
                edges = []
                for (pop, llm_bv), dests in children:
                    for dest_idx, state_bv in dests:
                        edges.append((pop, llm_bv, dest_idx, state_bv))
                if edges:
                    self.forward_edges[node] = edges

        # 2. Compute reachability to end nodes (for early pruning)
        self.can_reach_end = self._compute_end_reachability()

        # 3. Group nodes by depth for batch processing
        self.nodes_by_depth: DefaultDict[int, List[int]] = defaultdict(list)
        for node, depth in self.max_depth.items():
            self.nodes_by_depth[depth].append(node)

    def _compute_end_reachability(self) -> Set[int]:
        """Compute which nodes can reach an end node."""
        can_reach = set(self.end_nodes)
        changed = True

        while changed:
            changed = False
            for node in list(can_reach):
                # Find all predecessors
                for src, edges in self.forward_edges.items():
                    if src not in can_reach:
                        for _, _, dest, _ in edges:
                            if dest == node:
                                can_reach.add(src)
                                changed = True
                                break

        return can_reach

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        return Model(InnerModel.from_json_string(s))

    def commit(self, token_id: int) -> None:
        self.inner_model.commit(token_id)

    @property
    def state(self) -> Dict[int, GSS]:
        return self.inner_model.state

    def is_end(self, node: int) -> bool:
        return bool(((self.arena.get(node) or {}).get("value") or {}).get("clean_end", False))

    @profile
    def get_mask(self) -> RangeSet:
        state_map: Dict[int, GSS] = self.state
        all_ones: Optional[ffi.Bitset] = self.all_internal_llm_tokens_bitset
        final_mask: ffi.Bitset = ffi.Bitset.zeros()

        # Early exit if no valid states
        if not state_map:
            return RangeSet.from_ranges([])

        # State tracking with aggressive pruning
        values: Dict[int, Tuple[GSS, ffi.Bitset]] = {}
        processed: Set[int] = set()  # Nodes we've fully processed

        # Initialize with root states
        for sid, gss in state_map.items():
            r: Optional[int] = self.roots_map.get(int(sid))
            if r is None:
                continue

            # Skip nodes that can't reach end
            if r not in self.can_reach_end:
                continue

            r = int(r)
            if r in values:
                values[r] = (values[r][0].merge(gss), all_ones)
            else:
                values[r] = (gss, all_ones)

        # Process by depth order (more efficient than heap)
        max_d = max(self.max_depth.values()) if self.max_depth else 0

        def disallowed_terminals(g: GSS) -> ffi.HybridL2Bitset:
            acc = g.reduce_acc()
            return ffi.HybridL2Bitset.all() if acc is None else acc.terminals_union.complement()

        # Cache for GSS operations to avoid recomputation
        gss_cache: Dict[Tuple[int, int, frozenset], GSS] = {}

        for depth in range(max_d + 1):
            nodes = self.nodes_by_depth.get(depth, [])

            # Batch process all nodes at this depth
            next_values: Dict[int, List[Tuple[GSS, ffi.Bitset]]] = defaultdict(list)

            for node in nodes:
                if node not in values or node in processed:
                    continue

                # Skip if can't reach end
                if node not in self.can_reach_end:
                    continue

                processed.add(node)
                gss_node, llm_mask = values.pop(node)

                # Early exit if mask is empty
                if llm_mask.is_empty():
                    continue

                # Process end nodes
                if node in self.end_nodes:
                    forbid: ffi.Bitset = ffi.Bitset.zeros()
                    for (start, end), bv in disallowed_terminals(gss_node).range_values():
                        if bv.is_empty():
                            continue
                        for tsid in range(start, min(end, self.tokenizer_max_state) + 1):
                            pm: Optional[Dict[int, ffi.Bitset]] = self.possible_matches_cache.get(tsid)
                            if not pm:
                                continue
                            for terminal_id_str, llm_tokens in pm.items():
                                if bv.contains(int(terminal_id_str)):
                                    forbid = forbid.union(llm_tokens)

                    allowed: ffi.Bitset = llm_mask.difference(forbid)
                    if not allowed.is_empty():
                        final_mask = final_mask.union(allowed)

                # Process edges
                edges = self.forward_edges.get(node, [])
                for pop, llm_bv, dest_idx, state_bv in edges:
                    # Skip if destination can't reach end
                    if dest_idx not in self.can_reach_end:
                        continue

                    # Cache key for GSS operation
                    cache_key = (id(gss_node), pop, frozenset(gss_node.peek()) if not state_bv.is_empty() else frozenset())

                    if cache_key in gss_cache:
                        child_gss = gss_cache[cache_key]
                    else:
                        popped: GSS = gss_node.popn(pop)

                        if not state_bv.is_empty():
                            matched: List[GSS] = [popped.isolate(s) for s in popped.peek() if state_bv.contains(s)]
                            if not matched:
                                continue
                            child_gss = GSS.merge_many(matched)
                        else:
                            child_gss = popped

                        gss_cache[cache_key] = child_gss

                    child_mask: ffi.Bitset = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)

                    # Early exit if child mask is empty
                    if child_mask.is_empty():
                        continue

                    next_values[dest_idx].append((child_gss, child_mask))

            # Merge all values for next iteration
            for dest_idx, items in next_values.items():
                if not items:
                    continue

                # Batch merge GSS states
                if len(items) == 1:
                    values[dest_idx] = items[0]
                else:
                    # Merge all GSS states and union all masks
                    gss_list = [g for g, _ in items]
                    mask_list = [m for _, m in items]

                    merged_gss = GSS.merge_many(gss_list)
                    merged_mask = mask_list[0]
                    for m in mask_list[1:]:
                        merged_mask = merged_mask.union(m)

                    values[dest_idx] = (merged_gss, merged_mask)

        # Convert to original token IDs
        original_mask: ffi.Bitset = ffi.Bitset.zeros()
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_mask.insert(self.internal_to_original_map[i])

        return RangeSet.from_ranges(original_mask.to_ranges())