from .py_range_set import PyRangeSet
from .ffi_range_set import FFIRangeSet

# RangeSet = PyRangeSet
RangeSet = FFIRangeSet

__all__ = ["RangeSet"]
