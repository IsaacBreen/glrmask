from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Set, Tuple

from .interface import GSS, T, Acc
from .reference_impl import ReferenceGSS


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    _ref: ReferenceGSS[T, Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        return cls(ReferenceGSS(stacks))

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        return self._ref.to_stacks()

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        return LeveledGSS(self._ref.push(value))

    def pop(self) -> LeveledGSS[T, Acc]:
        return LeveledGSS(self._ref.pop())

    def is_empty(self) -> bool:
        return self._ref.is_empty()

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(self._ref.isolate(value))

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(self._ref.apply(func))

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(self._ref.prune(predicate))

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(self._ref.merge(other._ref))

    def peek(self) -> Set[T]:
        return self._ref.peek()

    def reduce_acc(self) -> Optional[Acc]:
        return self._ref.reduce_acc()
