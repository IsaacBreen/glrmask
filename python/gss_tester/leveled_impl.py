from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Set, Tuple, Type, Union, Callable

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes
# ------------------------------

@dataclass(frozen=True, eq=True)
class InnerLeaf:
    pass


@dataclass(frozen=True, eq=True)
class InnerBranch(Generic[T]):
    # children: T -> depth -> LeveledGSSInner
    children: Dict[T, Dict[int, 'LeveledGSSInner[T]']]


@dataclass(frozen=True, eq=True)
class MiddleBranch(Generic[T, Acc]):
    # children: T -> depth -> (Acc, LeveledGSSInner)
    children: Dict[T, Dict[int, Tuple[Acc, 'LeveledGSSInner[T]']]]


@dataclass(frozen=True, eq=True)
class Branch(Generic[T, Acc]):
    # children: T -> depth -> LeveledGSS
    children: Dict[T, Dict[int, 'LeveledGSS[T, Acc]']]


@dataclass(frozen=True, eq=True)
class LeveledGSSInner(Generic[T]):
    inner: Union[InnerLeaf, InnerBranch[T]]

    @classmethod
    def from_stacks(cls: Type['LeveledGSSInner[T]'], stacks: List[List[T]]) -> 'LeveledGSSInner[T]':
        raise NotImplementedError

    def to_stacks(self) -> List[List[T]]:
        raise NotImplementedError


# ------------------------------
# Public LeveledGSS implementation
# ------------------------------

@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Union[MiddleBranch[T, Acc], Branch[T, Acc]]

    def __post_init__(self):
        self.validate_invariants()

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        raise NotImplementedError

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.push(value)
        return LeveledGSS.from_stacks(new_ref_impl.to_stacks())

    def pop(self) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.pop()
        return LeveledGSS.from_stacks(new_ref_impl.to_stacks())

    def is_empty(self) -> bool:
        return len(self.to_stacks()) == 0

    def isolate(self, value: T) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.isolate(value)
        return LeveledGSS.from_stacks(new_ref_impl.to_stacks())

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.apply(func)
        return LeveledGSS.from_stacks(new_ref_impl.to_stacks())

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.prune(predicate)
        return LeveledGSS.from_stacks(new_ref_impl.to_stacks())

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.merge(other)
        return LeveledGSS.from_stacks(new_ref_impl.to_stacks())

    def peek(self) -> Set[T]:
        return self.to_reference_impl().peek()

    def reduce_acc(self) -> Optional[Acc]:
        return self.to_reference_impl().reduce_acc()

    def validate_invariants(self) -> None:
        raise NotImplementedError


class InvariantViolation(Exception):
    pass
