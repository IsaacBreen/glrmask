from __future__ import annotations
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Iterable

from ..interface import GSS, T, Acc, NewAcc

try:
    # The native module is built and placed in the python/ directory or installed in the venv
    from leveled_gss_rs import LeveledGSS as _LeveledRSGSS
except ImportError as e:
    raise ImportError(
        "Could not import the Rust-based LeveledGSS implementation. "
        "Please build the native module by running `maturin develop` in `python/leveled_rs/`"
    ) from e


class LeveledRSGSS(GSS[T, Acc], Generic[T, Acc]):
    """A Python wrapper for the Rust LeveledGSS implementation."""

    _inner: _LeveledRSGSS

    def __init__(self, inner: _LeveledRSGSS):
        self._inner = inner

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledRSGSS[T, Acc]:
        return cls(_LeveledRSGSS.from_stacks(stacks))

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        return self._inner.to_stacks()

    def push(self, value: T) -> LeveledRSGSS[T, Acc]:
        return LeveledRSGSS(self._inner.push(value))

    @classmethod
    def push_many(cls, items: Iterable[Tuple[GSS[T, Acc], T]]) -> LeveledRSGSS[T, Acc]:
        # Assuming items are LeveledRSGSS instances for unwrapping
        inner_items = [(item[0]._inner, item[1]) for item in items]  # type: ignore
        return cls(_LeveledRSGSS.push_many(inner_items))

    def pop(self) -> LeveledRSGSS[T, Acc]:
        return LeveledRSGSS(self._inner.pop())

    def popn(self, n: int) -> LeveledRSGSS[T, Acc]:
        return LeveledRSGSS(self._inner.popn(n))

    def is_empty(self) -> bool:
        return self._inner.is_empty()

    def isolate(self, value: Optional[T]) -> LeveledRSGSS[T, Acc]:
        return LeveledRSGSS(self._inner.isolate(value))

    def isolate_many(self, values: Iterable[Optional[T]]) -> LeveledRSGSS[T, Acc]:
        # The Rust implementation expects a set.
        return LeveledRSGSS(self._inner.isolate_many(set(values)))

    def apply(self, func: Callable[[Acc], NewAcc], memo: Optional[Dict[int, Any]] = None) -> GSS[T, NewAcc]:
        # This wrapper handles the memo argument correctly for the Rust implementation.
        return LeveledRSGSS(self._inner.apply(func, memo=memo))

    def prune(self, predicate: Callable[[Acc], bool], memo: Optional[Dict[int, Any]] = None) -> LeveledRSGSS[T, Acc]:
        return LeveledRSGSS(self._inner.prune(predicate, memo=memo))

    def apply_and_prune(self, mutator: Callable[[Acc], Optional[NewAcc]], memo: Optional[Dict[int, Any]] = None) -> GSS[T, NewAcc]:
        return LeveledRSGSS(self._inner.apply_and_prune(mutator, memo=memo))

    def merge(self, other: GSS[T, Acc]) -> LeveledRSGSS[T, Acc]:
        if not isinstance(other, LeveledRSGSS):
            raise TypeError(f"Can only merge LeveledRSGSS with another LeveledRSGSS, not {type(other)}")
        return LeveledRSGSS(self._inner.merge(other._inner))

    @classmethod
    def merge_many(cls, gss_list: Iterable[GSS[T, Acc]]) -> LeveledRSGSS[T, Acc]:
        # We need to unwrap the inner objects for the Rust implementation.
        inner_list = [gss._inner for gss in gss_list]  # type: ignore
        return cls(_LeveledRSGSS.merge_many(inner_list))

    def peek(self) -> Set[T]:
        return self._inner.peek()

    def reduce_acc(self) -> Optional[Acc]:
        return self._inner.reduce_acc()

    def to_reference_impl(self) -> GSS[T, Acc]:
        return self._inner.to_reference_impl()

    def __str__(self) -> str:
        return str(self._inner).replace("LeveledGSS", "LeveledRSGSS", 1)

    def __repr__(self) -> str:
        return repr(self._inner).replace("LeveledGSS", "LeveledRSGSS", 1)


# Alias for test runner discovery
Leveled_rsGSS = LeveledRSGSS
