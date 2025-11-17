from .py_range_set import PyRangeSet
from .ffi_range_set import FFIRangeSet
from .roaring_range_set import RoaringRangeSet
from .set_range_set import SetRangeSet
from .bitset_range_set import BitsetRangeSet

# RangeSet = PyRangeSet
RangeSet = FFIRangeSet

__all__ = ["RangeSet", "PyRangeSet", "FFIRangeSet", "RoaringRangeSet", "SetRangeSet", "BitsetRangeSet"]

