from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Set, MutableMapping, Mapping, Union

# Optional profiling hook – harmless if profiling is disabled.
try:
    from .rangeset_stats import record_metric
except Exception:  # pragma: no cover
    # Fallback no‑op stub so the module can be imported without the profiling package.
    def record_metric(name: str, value: float = 1.0) -> None:  # type: ignore
        pass


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
    # Statistics
    total_state_edges = 0   # how many StateEdge objects we inspected
    changed = 0             # how many we turned into UnconditionalEdge

    # Reuse a single immutable UnconditionalEdge instance.
    UNCOND = UnconditionalEdge()

    start_time = time.time()

    for src_id, node in nodes.items():
        s_src = alive.get(src_id, set())
        for token, pop_map in node.children.items():
            for pop, dest_map in pop_map.items():
                # Iterate over a static list to safely mutate the dict while looping.
                for dest, edge in list(dest_map.items()):
                    if isinstance(edge, StateEdge):
                        total_state_edges += 1
                        if s_src.issubset(edge.states):
                            dest_map[dest] = UNCOND
                            changed += 1

    elapsed = time.time() - start_time

    # Human‑readable summary (always printed)
    print(
        f\"[NodeOpt] unconditionalize: {changed}/{total_state_edges} StateEdge(s) "
        f\"converted in {elapsed:.4f}s\"
    )

    # Optional profiling record (no‑op if profiling disabled)
    try:
        record_metric('NodeOpt.unconditionalize.changed', changed)
        record_metric('NodeOpt.unconditionalize.total_state_edges', total_state_edges)
        record_metric('NodeOpt.unconditionalize.time_sec', elapsed)
    except Exception:  # pragma: no cover
        pass

    return changed
