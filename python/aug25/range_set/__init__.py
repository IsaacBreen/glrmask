from .py_range_set import PyRangeSet
from .ffi_range_set import FFIRangeSet
from .bitset_range_set import BitsetRangeSet

# RangeSet = PyRangeSet
RangeSet = FFIRangeSet

__all__ = ["RangeSet", "PyRangeSet", "FFIRangeSet", "BitsetRangeSet"]

