import os
import sys
from typing import Protocol, Iterable, Optional, Tuple

from .range_set import RangeSet
    print("Using FFI-backed RangeSet.", file=sys.stderr)
    try:
        from .range_set.ffi_range_set import RangeSet
    except ImportError:
        print("Warning: FFI RangeSet not found, falling back to Python implementation.", file=sys.stderr)
        from .range_set.py_range_set import RangeSet
else:
    from .range_set.py_range_set import RangeSet

class GraphProvider(Protocol):
    """
    A common interface for different graph-based model implementations to be
    used in equivalence checking and other analysis tools.
    """
    def get_root(self, state_id: int) -> int:
        """Get the root node index for a given tokenizer state ID."""
        ...

    def is_end(self, node: int) -> bool:
        """Check if a given node is an accepting/end state."""
        ...

    def iter_edges(self, node: int, token: int) -> Iterable[Tuple[int, Optional[int], int]]:
        """
        For a given node and token, iterate over all possible transitions.
        Yields tuples of (pop_count, state_id_or_none, destination_node_index).
        - pop_count: Number of items to pop from the GSS.
        - state_id_or_none: The state ID required at the top of the GSS for this
          transition, or None for an epsilon transition on the GSS.
        - destination_node_index: The index of the node to transition to.
        """
        ...


__all__ = ["GraphProvider", "RangeSet"]
