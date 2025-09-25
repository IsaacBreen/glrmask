from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, MutableMapping, Mapping, Union


@dataclass(frozen=True)
class UnconditionalEdge:
    """
    Edge that does not filter by parser state.
    """
    __slots__ = ()


@dataclass
class StateEdge:
    """
    Edge that filters by a set of parser states.
    """
    states: Set[int]


EdgeLike = Union[UnconditionalEdge, StateEdge]


@dataclass
class NodeOpt:
    """
    Optimizer-friendly node representation.
    children[token][pop][dest] = Edge
      - token: int (LLM token id)
      - pop: int
      - dest: int (node id)
      - Edge: UnconditionalEdge or StateEdge(states: Set[int])
    """
    children: Dict[int, Dict[int, Dict[int, EdgeLike]]] = field(default_factory=dict)
    is_end: bool = False


def _unconditionalize_guaranteed_transitions(
    nodes: MutableMapping[int, NodeOpt],
    alive: Mapping[int, Set[int]],
) -> int:
    """
    Fast unconditionalization pass: turn StateEdge into UnconditionalEdge
    when it's provably redundant given the current Alive set.

    Safety criterion (local fast path from the spec):
      Let S_src = Alive[src], S_edge = edge.states. If S_src ⊆ S_edge,
      then making this edge unconditional cannot introduce any new
      parser state on its first hop, so it’s safe immediately.

    Parameters:
      nodes: NodeOpt graph as a mapping of node_id -> NodeOpt
      alive: mapping node_id -> set of parser states Alive at that node

    Returns:
      The number of edges converted from StateEdge to UnconditionalEdge.

    Notes:
      - This pass does not modify the Alive map because the accepted
        transformations do not add newly reachable states at this stage.
      - This only handles the guaranteed (trivial) unconditionalizations.
        Non-trivial candidates (S_src \\ S_edge != ∅) require the tentative
        propagation check described in the main algorithm.
    """
    changed = 0
    # Reuse a single instance; it's immutable and fine to share.
    UNCOND = UnconditionalEdge()

    for src_id, node in nodes.items():
        s_src = alive.get(src_id, set())
        # If s_src is empty, upgrading is still safe under the current Alive.
        # It can also help expose passthrough nodes for removal.
        for token, pop_map in node.children.items():
            for pop, dest_map in pop_map.items():
                # We overwrite edge values without changing keys, so it's safe
                # to iterate items directly.
                for dest, edge in list(dest_map.items()):
                    if isinstance(edge, StateEdge):
                        if s_src.issubset(edge.states):
                            dest_map[dest] = UNCOND
                            changed += 1

    return changed
