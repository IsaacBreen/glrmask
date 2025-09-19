from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Set, Tuple, Type, Union, Callable

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes
# ------------------------------

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    # children: T -> depth -> LeveledGSS
    children: Dict[T, Dict[int, 'LeveledGSS[T, Acc]']]


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    node: Lower[T]
    acc: Acc


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    # children: T -> depth -> LeveledGSSInner
    children: Dict[T, Dict[int, 'Lower[T]']]
    is_leaf: bool

    @classmethod
    def from_stacks(cls: Type['Lower[T]'], stacks: List[List[T]]) -> 'Lower[T]':
        raise NotImplementedError

    def to_stacks(self) -> List[List[T]]:
        raise NotImplementedError

    def validate_invariants(self) -> None:
        raise NotImplementedError


class InvariantViolation(Exception):
    pass


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Union[Upper[T, Acc], Interface[T, Acc]]

    def __post_init__(self):
        self.validate_invariants()

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        raise NotImplementedError

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def pop(self) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def peek(self) -> Set[T]:
        raise NotImplementedError

    def reduce_acc(self) -> Optional[Acc]:
        raise NotImplementedError

    def validate_invariants(self) -> None:
        raise NotImplementedError
