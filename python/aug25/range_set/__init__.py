from .py_range_set import PyRangeSet
from .ffi_range_set import FFIRangeSet
from .bitset_range_set import BitsetRangeSet
from .fast_py_range_set import FastPyRangeSet

# RangeSet = PyRangeSet
# RangeSet = FFIRangeSet
RangeSet = FastPyRangeSet

__all__ = ["RangeSet", "PyRangeSet", "FFIRangeSet", "BitsetRangeSet", "FastPyRangeSet"]

