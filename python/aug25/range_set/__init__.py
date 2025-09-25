from .py_range_set import PyRangeSet
from .ffi_range_set import FFIRangeSet
from .bitset_range_set import BitsetRangeSet
from .nodeopt_optimizer import (
    NodeOpt,
    UnconditionalEdge,
    StateEdge,
    _unconditionalize_guaranteed_transitions,
)

# RangeSet = PyRangeSet
RangeSet = FFIRangeSet

__all__ = ["RangeSet", "PyRangeSet", "FFIRangeSet", "BitsetRangeSet", "NodeOpt", "UnconditionalEdge", "StateEdge", "_unconditionalize_guaranteed_transitions"]

