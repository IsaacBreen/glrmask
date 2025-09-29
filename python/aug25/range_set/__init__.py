from .py_range_set import PyRangeSet
from .ffi_range_set import FFIRangeSet
from .bitset_range_set import BitsetRangeSet
from .roaring_range_set import RoaringRangeSet
from .set_range_set import SetRangeSet

# RangeSet = PyRangeSet
RangeSet = FFIRangeSet

__all__ = ["RangeSet", "PyRangeSet", "FFIRangeSet", "BitsetRangeSet", "RoaringRangeSet", "SetRangeSet"]

