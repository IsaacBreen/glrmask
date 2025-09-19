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
class WithAcc(Generic[T, Acc]):
    node: 'LeveledGSSInner[T]'
    acc: Acc


@dataclass(frozen=True, eq=True)
class Branch(Generic[T, Acc]):
    # children: T -> depth -> LeveledGSS
    children: Dict[object, Dict[int, 'LeveledGSS[T, Acc]']]


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
    inner: Union[WithAcc[T, Acc], Branch[T, Acc]]

    def __post_init__(self):
        _validate_invariants_node(self)

    # ---- GSS interface ----

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        raise NotImplementedError

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.push(value)
        return LeveledGSS.from_stacks(new_ref_impl._stacks)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.pop()
        return LeveledGSS.from_stacks(new_ref_impl._stacks)

    def is_empty(self) -> bool:
        return isinstance(self.inner, Branch) and not self.inner.children

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.isolate(value)
        return LeveledGSS.from_stacks(new_ref_impl._stacks)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.apply(func)
        return LeveledGSS.from_stacks(new_ref_impl._stacks)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.prune(predicate)
        return LeveledGSS.from_stacks(new_ref_impl._stacks)

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        ref_impl = self.to_reference_impl()
        new_ref_impl = ref_impl.merge(other)
        return LeveledGSS.from_stacks(new_ref_impl._stacks)

    def peek(self) -> Set[T]:
        return self.to_reference_impl().peek()

    def reduce_acc(self) -> Optional[Acc]:
        return self.to_reference_impl().reduce_acc()

    # Also expose a human-friendly validator
    def validate_invariants(self) -> None:
        _validate_invariants_node(self.inner)


# ------------------------------
# Invariant validation
# ------------------------------

class InvariantViolation(Exception):
    pass


def _validate_invariants_node(node: LeveledGSS[T, Acc]):
    raise NotImplementedError