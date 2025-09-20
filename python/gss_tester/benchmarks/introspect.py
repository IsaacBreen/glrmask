from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type

# We intentionally avoid importing LeveledGSS at import time to prevent import cycles.
# We'll import it lazily within the introspection functions.


@dataclass
class LeveledNodeStats:
    # Upper-level structure
    upper_nodes: int = 0
    upper_branch_nodes: int = 0
    upper_interfaces: int = 0
    upper_edges: int = 0

    # Lower-level structure
    lower_nodes: int = 0
    lower_branch_nodes: int = 0
    lower_leaf_nodes: int = 0
    lower_edges: int = 0


@dataclass
class StackTrieStats:
    # Derived purely from to_stacks()
    stack_count: int
    total_elements: int
    unique_prefix_nodes: int
    edges_in_trie: int
    max_depth: int
    avg_depth: float
    avg_branching_factor: float
    # A simple “sharing” indicator: how many repeated elements were collapsed by the trie.
    # 1.0 means no sharing; closer to 0 means heavy sharing.
    sharing_ratio: float


@dataclass
class StructuralSummary:
    # Always available
    trie: StackTrieStats
    # Implementation-specific (optional)
    leveled: Optional[LeveledNodeStats] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {"trie": asdict(self.trie)}
        if self.leveled is not None:
            data["leveled"] = asdict(self.leveled)
        return data


def _compute_trie_from_stacks(stacks: Sequence[Tuple[List[Any], Any]]) -> StackTrieStats:
    """
    Build a prefix trie from the explicit stacks and compute structural stats.
    This is an approximation of sharing behavior available for any implementation.
    """
    # Root node not counted in unique_prefix_nodes for readability.
    trie: Dict[Any, Dict] = {}
    unique_nodes = 0
    edges = 0
    total_elements = 0
    max_depth = 0

    # Insert paths into the trie
    for vals, _acc in stacks:
        total_elements += len(vals)
        if len(vals) > max_depth:
            max_depth = len(vals)
        node = trie
        for v in vals:
            if v not in node:
                node[v] = {}
                unique_nodes += 1
            node = node[v]
            edges += 1

    # Compute average depth among non-empty stacks
    stack_count = len(stacks)
    avg_depth = (total_elements / stack_count) if stack_count > 0 else 0.0

    # Compute avg branching factor across internal trie nodes.
    # We'll do a BFS to count internal nodes and sum their degrees.
    internal_nodes = 0
    degree_sum = 0
    queue: List[Dict] = [trie]
    while queue:
        cur = queue.pop()
        # A node is internal if it has children.
        if cur:
            internal_nodes += 1
            deg = len(cur)
            degree_sum += deg
            for child in cur.values():
                queue.append(child)

    avg_bf = (degree_sum / internal_nodes) if internal_nodes > 0 else 0.0
    sharing_ratio = (unique_nodes / total_elements) if total_elements > 0 else 0.0

    return StackTrieStats(
        stack_count=stack_count,
        total_elements=total_elements,
        unique_prefix_nodes=unique_nodes,
        edges_in_trie=edges,
        max_depth=max_depth,
        avg_depth=avg_depth,
        avg_branching_factor=avg_bf,
        sharing_ratio=sharing_ratio,
    )


def _try_introspect_leveled(gss_obj: Any) -> Optional[LeveledNodeStats]:
    """
    If the object is a LeveledGSS, traverse its internal representation and
    compute detailed node/edge counts. Returns None if not applicable.
    """
    try:
        # Lazy import to avoid circular references.
        from gss_tester.leveled_impl import LeveledGSS, Upper, UpperBranch, Interface, Lower, LowerBranch, Leaf
    except Exception:
        return None

    if not isinstance(gss_obj, LeveledGSS):
        return None

    stats = LeveledNodeStats()

    # We'll traverse both the upper-level graph (tree-like) and referenced lower-level nodes.
    visited_upper: Set[int] = set()
    visited_lower: Set[int] = set()

    def visit_upper(u: Any):
        oid = id(u)
        if oid in visited_upper:
            return
        visited_upper.add(oid)
        stats.upper_nodes += 1

        inner = u.inner
        # Interface -> link to lower
        if isinstance(inner, Interface):
            stats.upper_interfaces += 1
            # Visit lower node
            visit_lower(inner.node)
        elif isinstance(inner, UpperBranch):
            stats.upper_branch_nodes += 1
            # Count edges and visit children
            for _val, kids in inner.children.items():
                for child in kids.values():
                    stats.upper_edges += 1
                    visit_upper(child)
        else:
            # Unknown type; safeguard
            pass

    def visit_lower(l: Any):
        oid = id(l)
        if oid in visited_lower:
            return
        visited_lower.add(oid)
        stats.lower_nodes += 1

        inner = l.inner
        if isinstance(inner, LowerBranch):
            stats.lower_branch_nodes += 1
            for _val, kids in inner.children.items():
                for child in kids.values():
                    stats.lower_edges += 1
                    visit_lower(child)
        elif isinstance(inner, Leaf):
            stats.lower_leaf_nodes += 1
        else:
            # Unknown; ignore
            pass

    visit_upper(gss_obj.inner)
    return stats


def summarize_structure(gss_obj: Any) -> StructuralSummary:
    """
    Produce a StructuralSummary using to_stacks() and optional implementation-specific
    introspection.
    """
    try:
        stacks = gss_obj.to_stacks()
    except Exception:
        # If to_stacks fails, return a minimal summary
        stacks = []

    trie_stats = _compute_trie_from_stacks(stacks)
    leveled_stats = _try_introspect_leveled(gss_obj)
    return StructuralSummary(trie=trie_stats, leveled=leveled_stats)
