"""
Ultra-fast get_mask engine (100x-focused) for the precompute3 model.

Key idea:
- Replace per-edge GSS operations (popn, isolate, merge, apply) with a precomputed
  projection of "top-of-stack state transitions" relative to the original stacks.
- Traverse the trie using purely:
  - Python set intersections for state gating (much faster than RangeSet.contains per item)
  - RangeSet bitset ops only for token masks (union/intersection) per edge block
- Compute per-acc allowed-token masks once (via GSS.apply) and union them up-front,
  then treat path LLM-token masks as: (allowed_union) ∩ (structural_union_along_edges).
  This is correct because intersection distributes over union across paths/accs.

No stats, minimal Python-level overhead, and avoids per-edge GSS structural work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Set, FrozenSet, DefaultDict
from collections import defaultdict, deque

import _sep1 as ffi

# Import the same primitives used by the existing models
from python.aug25.common_interface import RangeSet
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS


# ------------------------------
# Utilities
# ------------------------------

def rangeset_to_frozenset(rs: RangeSet) -> FrozenSet[int]:
    """
    Convert a RangeSet of contiguous [start, end] ranges into a frozenset of ints.
    This is used for extremely fast Python-side membership and intersections.
    """
    result: Set[int] = set()
    for start, end in rs.to_ranges():
        if end >= start:
            result.update(range(start, end + 1))
    return frozenset(result)


# ------------------------------
# Data models
# ------------------------------

@dataclass(frozen=True)
class EdgeBlock:
    """
    One compressed "edge block" from the trie:
    - pop: how many stack frames to pop before testing state gating
    - llm_bv: RangeSet mask to intersect with path tokens
    - dests: list of (dest_idx, allowed_state_set) pairs
      where allowed_state_set is the set of top-of-stack states allowed for this transition
    """
    pop: int
    llm_bv: RangeSet
    dests: Tuple[Tuple[int, FrozenSet[int]], ...]


@dataclass
class OptimizedGraph:
    """
    Immutable compressed representation of the trie graph with Python-set
    state gating for very fast membership/intersection operations.
    """
    edges_by_node: Dict[int, Tuple[EdgeBlock, ...]] = field(default_factory=dict)
    end_nodes: FrozenSet[int] = field(default_factory=frozenset)
    max_depth: Dict[int, int] = field(default_factory=dict)
    roots_map: Dict[int, int] = field(default_factory=dict)
    pops_used: Tuple[int, ...] = field(default_factory=tuple)  # sorted unique pops


@dataclass
class GSSProjection:
    """
    Projection of a GSS that allows us to avoid per-edge GSS operations:
    - heads_by_pop[k] gives the set of heads after popping k frames
    - down[k][d][s] gives the set of heads after popping k first, then d more,
      for stacks that had top-of-stack state `s` after the first k pops.
    """
    pops: Tuple[int, ...]
    heads_by_pop: Dict[int, FrozenSet[int]]
    down: Dict[int, Dict[int, Dict[int, FrozenSet[int]]]]  # k -> d -> s -> set_of_states


# ------------------------------
# Builder functions
# ------------------------------

def build_optimized_graph(arena: Dict[int, dict]) -> OptimizedGraph:
    """
    Convert the arena structure into an optimized graph:
      - Convert each state_bv RangeSet into a Python frozenset for O(1) membership.
      - Fold into EdgeBlock objects per node.
      - Collect all distinct pop values used anywhere.
    """
    edges_by_node: Dict[int, List[EdgeBlock]] = {}
    end_nodes: Set[int] = set()
    max_depth: Dict[int, int] = {}
    pops_used_set: Set[int] = set()

    # Precompute end flags and max_depth
    for uid, node in arena.items():
        uid_int = int(uid)
        max_depth[uid_int] = int(node.get("max_depth", 0) or 0)
        val = node.get("value") or {}
        if bool(val.get("clean_end", False)):
            end_nodes.add(uid_int)

    # Build edges
    for uid, node in arena.items():
        uid_int = int(uid)
        children = node.get("children") or []
        edge_blocks: List[EdgeBlock] = []

        for (pop, llm_bv), dests in children:
            p = int(pop)
            pops_used_set.add(p)

            new_dests: List[Tuple[int, FrozenSet[int]]] = []
            for dest_idx, state_bv in dests:
                # Convert RangeSet => frozenset of ints once
                state_set = rangeset_to_frozenset(state_bv)
                new_dests.append((int(dest_idx), state_set))

            edge_blocks.append(
                EdgeBlock(
                    pop=p,
                    llm_bv=llm_bv,
                    dests=tuple(new_dests)
                )
            )

        edges_by_node[uid_int] = tuple(edge_blocks)

    pops_used = tuple(sorted(pops_used_set))
    return OptimizedGraph(
        edges_by_node=edges_by_node,
        end_nodes=frozenset(end_nodes),
        max_depth=max_depth,
        roots_map={},  # will be filled by the engine
        pops_used=pops_used,
    )


def compute_gss_projection(merged_gss: GSS, pops_used: Iterable[int]) -> GSSProjection:
    """
    For a merged GSS (for one root), compute:
      - heads_by_pop for all k in pops_used
      - down[k][d][s] for all k,d in pops_used and all heads s at pop k
    """
    pops = tuple(sorted(set(pops_used)))
    heads_by_pop: Dict[int, FrozenSet[int]] = {}
    popped_cache: Dict[int, GSS] = {}

    # Compute heads_by_pop and cache popped GSS per k
    for k in pops:
        popped = merged_gss.popn(k) if k != 0 else merged_gss
        popped_cache[k] = popped
        heads = set(popped.peek())
        heads_by_pop[k] = frozenset(heads)

    # Build down transitions
    down: Dict[int, Dict[int, Dict[int, FrozenSet[int]]]] = {}
    for k in pops:
        down[k] = {}
        popped_k = popped_cache[k]
        heads_k = heads_by_pop[k]
        # Cache isolate(s) for s in heads_k to reuse across d
        iso_cache: Dict[int, GSS] = {}
        for s in heads_k:
            iso_cache[s] = popped_k.isolate(s)

        for d in pops:
            d_map: Dict[int, FrozenSet[int]] = {}
            for s, iso in iso_cache.items():
                deeper = iso.popn(d) if d != 0 else iso
                d_map[int(s)] = frozenset(set(deeper.peek()))
            down[k][d] = d_map

    return GSSProjection(pops=pops, heads_by_pop=heads_by_pop, down=down)


# ------------------------------
# Engine
# ------------------------------

class FastMaskEngine:
    """
    Construct once per Model state, then call compute_mask() per step.

    This engine relies on model internals:
    - model.arena: Dict[node_id, dict] with "children" of form [((pop, llm_bv_json), dests), ...]
      already normalized to RangeSet by the existing model's __init__.
    - model.roots_map: Dict[tokenizer_state_id, root_node_id]
    - model.state: Dict[tokenizer_state_id, GSS]
    - model.possible_matches_cache: Dict[tokenizer_state_id, Dict[terminal_id, RangeSet]]
    - model.all_internal_llm_tokens_bitset: RangeSet (universe tokens)
    - model.internal_to_original_map: Dict[int, int]
    """

    def __init__(self, model):
        self.model = model

        # Build optimized graph once
        self.graph = build_optimized_graph(model.arena)
        self.graph.roots_map = dict(model.roots_map)  # copy

        # Universe tokens bitset
        self.universe: RangeSet = model.all_internal_llm_tokens_bitset if model.all_internal_llm_tokens_bitset is not None else RangeSet.empty()

        # Precompute GSS projections per root by merging states by root
        states_by_root: Dict[int, List[GSS]] = defaultdict(list)
        for tsid, gss in model.state.items():
            r = self.graph.roots_map[int(tsid)]
            states_by_root[r].append(gss)

        self.projections: Dict[int, GSSProjection] = {}
        for r, gss_list in states_by_root.items():
            if len(gss_list) == 1:
                merged = gss_list[0]
            else:
                merged = GSS.merge_many(gss_list)
            self.projections[r] = compute_gss_projection(merged, self.graph.pops_used)

        # Compute union of allowed-by-terminals across all accumulators in all initial GSS
        # We do this once and reuse for all roots; this matches union_i (Uni - forb_i).
        self.allowed_by_terminals_union: RangeSet = self._compute_allowed_by_terminals_union()

    def _compute_allowed_by_terminals_union(self) -> RangeSet:
        """
        Compute union across accumulators of (universe - llm_tokens_disallowed_by_terminals).

        Implementation: iterate over accumulators via gss.apply to avoid exposing GSS internals.
        """
        pmc: Dict[int, Dict[int, RangeSet]] = self.model.possible_matches_cache or {}
        allowed_union: RangeSet = RangeSet.empty()
        universe: RangeSet = self.universe

        def acc_allowed(acc) -> RangeSet:
            # acc.terminals_union: Dict[tokenizer_state_id, RangeSet_of_terminals]
            disallowed_llm_mask = RangeSet.empty()
            disallowed_map = getattr(acc, "terminals_union", {}) or {}
            if not disallowed_map:
                return universe

            for tsid, disallowed_terms in disallowed_map.items():
                tsid_int = int(tsid)
                mapping = pmc.get(tsid_int)
                if not mapping:
                    continue
                # Iterate terminals
                for terminal_id in disallowed_terms.to_indices():
                    rs = mapping.get(int(terminal_id))
                    if rs is not None:
                        disallowed_llm_mask = disallowed_llm_mask.union(rs)
            return universe.difference(disallowed_llm_mask)

        # Visit all GSS objects and union allowed masks across all accumulators
        for gss in self.model.state.values():
            def collector(acc):
                nonlocal allowed_union
                allowed_union = allowed_union.union(acc_allowed(acc))
                return acc  # identity; we don't want to modify the GSS
            # Enumerate accumulators; ignore the returned GSS
            gss.apply(collector)

        return allowed_union

    def compute_mask(self) -> RangeSet:
        """
        Main algorithm:
        - Seed each root node with:
            - state-summary S_by_pop from GSSProjection for that root
            - mask = allowed_by_terminals_union
        - Traverse the trie using state gating (Python set ops) and token-mask propagation
          (RangeSet intersections/unions), aggregating per-node until fixpoint (acyclic here).
        - At end nodes, union their masks, then map internal->original ids.
        """
        edges_by_node = self.graph.edges_by_node
        end_nodes = self.graph.end_nodes
        max_depth = self.graph.max_depth
        pops_used = self.graph.pops_used

        # Per-node data
        node_masks: Dict[int, RangeSet] = {}
        node_states_by_pop: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))

        # Seed from roots
        depth_buckets: Dict[int, Set[int]] = defaultdict(set)
        depths_heap: List[int] = []

        def enqueue(n: int):
            d = max_depth.get(n, 0)
            if n not in buckets_nodes[d]:
                buckets_nodes[d].add(n)
                # push depth if not already in heap
                if not depth_in_heap[d]:
                    depths_heap.append(d)
                    depth_in_heap[d] = True

        # We'll manage heap manually (depths small), to reduce overhead
        buckets_nodes: Dict[int, Set[int]] = defaultdict(set)
        depth_in_heap: DefaultDict[int, bool] = defaultdict(bool)

        # Roots: group tokenizer states by root and seed per-root states/masks
        states_by_root: Dict[int, List[int]] = defaultdict(list)
        for tsid in self.model.state.keys():
            r = self.graph.roots_map[int(tsid)]
            states_by_root[r].append(int(tsid))

        for root, _tsids in states_by_root.items():
            proj = self.projections[root]
            # Seed state summaries
            s_map: Dict[int, Set[int]] = defaultdict(set)
            for k in pops_used:
                s_map[k] = set(proj.heads_by_pop.get(k, frozenset()))
            node_states_by_pop[root] = s_map
            # Seed mask
            node_masks[root] = self.allowed_by_terminals_union
            # Enqueue
            d = max_depth.get(root, 0)
            buckets_nodes[d].add(root)
            if not depth_in_heap[d]:
                depths_heap.append(d)
                depth_in_heap[d] = True

        # We'll process depths in increasing order (similar to original)
        depths_heap.sort()

        # Memo: A function to compute child states contribution via a parent node's projection
        def push_states_through_edge(parent_node: int, parent_proj: GSSProjection, pop_k: int, allowed_states: FrozenSet[int]) -> Dict[int, Set[int]]:
            """
            Given:
              - parent_node's current states S[k] (we'll read from node_states_by_pop[parent_node][k])
              - pop count k for the edge
              - allowed_states set R (state gating from edge)
            Return:
              - child S_by_pop map: for each d in pops_used, union over s in (S[k] ∩ R) of down[k][d][s]
            """
            parent_S_map = node_states_by_pop[parent_node]
            present = parent_S_map.get(pop_k)
            if not present:
                return {}

            # T = S[k] ∩ R
            # Use Python set intersection (very fast)
            T = present.intersection(allowed_states)
            if not T:
                return {}

            res: Dict[int, Set[int]] = defaultdict(set)
            down_k = parent_proj.down.get(pop_k, {})
            # For each s in T, union transitions for all d in pops_used
            for s in T:
                s_map = down_k.get(0, {}).get(s) is None  # quick presence check
                # If s not in down_k[d], skip (shouldn't happen unless GSS empty)
                for d in pops_used:
                    s_targets = down_k.get(d, {}).get(s)
                    if s_targets:
                        # s_targets is a FrozenSet[int]
                        if res.get(d) is None:
                            res[d] = set(s_targets)
                        else:
                            res[d].update(s_targets)
            return res

        # For quick lookup of the projection per node: any node seeded by X roots
        # can receive contributions from multiple projections; we maintain a multimap
        node_projections_sources: Dict[int, Set[int]] = defaultdict(set)
        for root in states_by_root.keys():
            node_projections_sources[root].add(root)

        # BFS / DP over depths
        while depths_heap:
            depth = depths_heap.pop(0)
            nodes_here = list(buckets_nodes[depth])
            buckets_nodes[depth].clear()
            depth_in_heap[depth] = False

            for node in nodes_here:
                # Acquire node mask and states
                node_mask = node_masks.get(node)
                s_map = node_states_by_pop.get(node)
                if node_mask is None and not s_map:
                    # No meaningful contribution
                    continue

                # Quick short-circuit: If no outgoing edges, continue (end-node handled later)
                eblocks = edges_by_node.get(node, ())
                if not eblocks:
                    continue

                # Determine which projections to use for this node
                # Initially a node only has contributions from its root's projection,
                # but as graph branches, a node can aggregate from multiple roots.
                source_roots = node_projections_sources.get(node)
                if not source_roots:
                    # Fallback: derive root by reverse lookup (rare)
                    # In practice, roots seed this mapping; so skip otherwise
                    continue

                for block in eblocks:
                    # If node has mask, propagate token mask through this edge
                    new_mask: Optional[RangeSet] = None
                    if node_mask is not None:
                        new_mask = node_mask.intersection(block.llm_bv)
                        if new_mask.is_empty():
                            # No token survives through this edge -> skip mask propagation
                            new_mask = None

                    # For each dest, we need a combined states contribution from all projections
                    for dest_idx, allowed_states in block.dests:
                        # Combine states contributions from each projection that contributed to this node
                        combined_child_states: Dict[int, Set[int]] = defaultdict(set)
                        any_states = False
                        for root in source_roots:
                            proj = self.projections[root]
                            child_contrib = push_states_through_edge(node, proj, block.pop, allowed_states)
                            if child_contrib:
                                any_states = True
                                for d, s_set in child_contrib.items():
                                    if s_set:
                                        combined_child_states[d].update(s_set)
                                        # Track that 'dest_idx' is now influenced by this root's stacks
                                        node_projections_sources[dest_idx].add(root)

                        # If no states pass the gating, skip
                        if not any_states:
                            continue

                        # Update dest's states map; track if anything changed
                        dest_states_map = node_states_by_pop[dest_idx]
                        states_changed = False
                        for d, s_set in combined_child_states.items():
                            before = dest_states_map.get(d)
                            if before is None:
                                dest_states_map[d] = set(s_set)
                                states_changed = True
                            else:
                                old_len = len(before)
                                before.update(s_set)
                                if len(before) != old_len:
                                    states_changed = True

                        # Update dest's mask if any
                        mask_changed = False
                        if new_mask is not None:
                            current = node_masks.get(dest_idx)
                            if current is None:
                                node_masks[dest_idx] = new_mask
                                mask_changed = True
                            else:
                                # Union
                                updated = current.union(new_mask)
                                if updated != current:
                                    node_masks[dest_idx] = updated
                                    mask_changed = True

                        # Enqueue dest if either contribution changed
                        if states_changed or mask_changed:
                            d2 = max_depth.get(dest_idx, 0)
                            if dest_idx not in buckets_nodes[d2]:
                                buckets_nodes[d2].add(dest_idx)
                                if not depth_in_heap[d2]:
                                    depths_heap.append(d2)
                                    depth_in_heap[d2] = True
                # end for eblocks
        # end BFS

        # Collect final masks from end nodes and convert to original IDs
        final_mask_internal = RangeSet.empty()
        for n in end_nodes:
            m = node_masks.get(n)
            if m is not None and not m.is_empty():
                final_mask_internal = final_mask_internal.union(m)

        # Convert internal -> original ids
        mapping: Dict[int, int] = self.model.internal_to_original_map or {}
        original_indices: List[int] = []
        for i in final_mask_internal.to_indices():
            if i in mapping:
                original_indices.append(mapping[i])

        return RangeSet.from_indices(original_indices)


# ------------------------------
# Public API
# ------------------------------

def fast_get_mask(model) -> RangeSet:
    """
    Plug-and-play replacement for model.get_mask(): much faster.

    Usage:
      from gpt-5-26 import fast_get_mask
      mask = fast_get_mask(your_model_instance)

    The model instance must be a precompute3 model (pure Python version) with:
      - .arena, .roots_map, .state, .possible_matches_cache,
        .all_internal_llm_tokens_bitset, .internal_to_original_map
      as prepared by Model.from_json_string(...) + commit(...) calls.
    """
    engine = FastMaskEngine(model)
    return engine.compute_mask()


# Optional convenience wrapper class if you want a drop-in model with optimized get_mask.
class UltraFastModel:
    """
    A thin wrapper around an existing Model (e.g. precompute3_model_pure_python_with_stats.Model)
    that replaces get_mask with the optimized engine. commit() delegates to the wrapped model.
    """
    def __init__(self, base_model):
        self._base = base_model

    @staticmethod
    def from_json_string(s: str) -> "UltraFastModel":
        # Lazily import the slow model and wrap it
        from python.aug25.models.precompute3_model_pure_python_with_stats import Model as SlowModel
        bm = SlowModel.from_json_string(s)
        return UltraFastModel(bm)

    def get_mask(self) -> RangeSet:
        return fast_get_mask(self._base)

    def commit(self, token_id: int):
        self._base.commit(token_id)

    def is_end(self, node: int) -> bool:
        return self._base.is_end(node)

    # Expose properties for compatibility
    @property
    def arena(self): return self._base.arena

    @property
    def roots_map(self): return self._base.roots_map

    @property
    def state(self): return self._base.state

    @property
    def possible_matches_cache(self): return self._base.possible_matches_cache

    @property
    def all_internal_llm_tokens_bitset(self): return self._base.all_internal_llm_tokens_bitset

    @property
    def internal_to_original_map(self): return self._base.internal_to_original_map

    @property
    def tokenizer(self): return self._base.tokenizer

    @property
    def glr_parser(self): return self._base.glr_parser

    @property
    def ignore_terminal_id(self): return self._base.ignore_terminal_id

    @property
    def parser_table(self): return self._base.parser_table

    @property
    def tokenizer_initial_state(self): return self._base.tokenizer_initial_state

    @property
    def tokenizer_max_state(self): return self._base.tokenizer_max_state

    @property
    def id_to_token(self): return self._base.id_to_token

Model = UltraFastModel