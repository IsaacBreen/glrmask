from __future__ import annotations
from typing import List, Tuple, Callable, Optional, Iterable, Any, Type

from ..interface import GSS, T, Acc, NewAcc
# This will be the name of the compiled C++ module
from leveled_gss_cpp import LeveledGssCpp

class Leveled_impl_cppGSS(GSS[T, Acc]):
    """
    A wrapper around the C++ LeveledGSS implementation.
    """
    _cpp_gss: LeveledGssCpp

    def __init__(self, cpp_gss_instance: LeveledGssCpp):
        self._cpp_gss = cpp_gss_instance

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> Leveled_impl_cppGSS[T, Acc]:
        cpp_gss = LeveledGssCpp.from_stacks(stacks)
        return cls(cpp_gss)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        stacks = self._cpp_gss.to_stacks()
        # Use ReferenceGSS to get canonical sorting for test comparison
        from .reference_impl import ReferenceGSS
        return ReferenceGSS.from_stacks(stacks).to_stacks()

    def push(self, value: T) -> Leveled_impl_cppGSS[T, Acc]:
        return Leveled_impl_cppGSS(self._cpp_gss.push(value))

    def pop(self) -> Leveled_impl_cppGSS[T, Acc]:
        return Leveled_impl_cppGSS(self._cpp_gss.pop())

    def popn(self, n: int) -> Leveled_impl_cppGSS[T, Acc]:
        return Leveled_impl_cppGSS(self._cpp_gss.popn(n))

    def is_empty(self) -> bool:
        return self._cpp_gss.is_empty()

    def isolate(self, value: Optional[T]) -> Leveled_impl_cppGSS[T, Acc]:
        return Leveled_impl_cppGSS(self._cpp_gss.isolate(value))

    def isolate_many(self, values: Iterable[Optional[T]]) -> Leveled_impl_cppGSS[T, Acc]:
        return Leveled_impl_cppGSS(self._cpp_gss.isolate_many(list(values)))

    def apply(self, func: Callable[[Acc], NewAcc], memo=None) -> GSS[T, NewAcc]:
        return Leveled_impl_cppGSS(self._cpp_gss.apply(func))

    def prune(self, predicate: Callable[[Acc], bool], memo=None) -> Leveled_impl_cppGSS[T, Acc]:
        return Leveled_impl_cppGSS(self._cpp_gss.prune(predicate))

    def apply_and_prune(self, mutator: Callable[[Acc], Optional[NewAcc]], memo=None) -> GSS[T, NewAcc]:
        return Leveled_impl_cppGSS(self._cpp_gss.apply_and_prune(mutator))

    def merge(self, other: Leveled_impl_cppGSS[T, Acc]) -> Leveled_impl_cppGSS[T, Acc]:
        return Leveled_impl_cppGSS(self._cpp_gss.merge(other._cpp_gss))

    @classmethod
    def merge_many(cls, gss_list: Iterable[Leveled_impl_cppGSS[T, Acc]]) -> Leveled_impl_cppGSS[T, Acc]:
        cpp_gss_list = [g._cpp_gss for g in gss_list]
        return cls(LeveledGssCpp.merge_many(cpp_gss_list))

    def peek(self) -> set[T]:
        return self._cpp_gss.peek()

    def reduce_acc(self) -> Optional[Acc]:
        return self._cpp_gss.reduce_acc()
