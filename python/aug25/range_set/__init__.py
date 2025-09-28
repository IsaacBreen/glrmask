from .py_range_set import PyRangeSet
from .ffi_range_set import FFIRangeSet
from .bitset_range_set import BitsetRangeSet
from .roaring_range_set import RoaringRangeSet

# RangeSet = PyRangeSet
RangeSet = FFIRangeSet

__all__ = ["RangeSet", "PyRangeSet", "FFIRangeSet", "BitsetRangeSet", "RoaringRangeSet"]

