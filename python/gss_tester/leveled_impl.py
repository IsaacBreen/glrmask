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
        # Invariant: lower must either have children or be a leaf (or both)
        if not self.children and not self.is_leaf:
            raise InvariantViolation("Lower node must have children or be a leaf.")

        # Recurse to children
        for children_at_depth in self.children.values():
            for child in children_at_depth.values():
                child.validate_invariants()


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
        if isinstance(self.inner, Upper):
            # Invariant: upper must have at least one child.
            all_children = [
                child
                for children_at_depth in self.inner.children.values()
                for child in children_at_depth.values()
            ]
            if not all_children:
                raise InvariantViolation("Upper node must have at least one child.")

            # Recurse validation to children.
            for child in all_children:
                child.validate_invariants()

            # Invariant: if all children are interfaces, they must not have equal accs.
            all_children_are_interfaces = all(isinstance(c.inner, Interface) for c in all_children)
            if all_children_are_interfaces:
                accs = [c.inner.acc for c in all_children]
                # Accumulators may not be hashable, so we can't use a set.
                # This is O(n^2) but likely fine for tests.
                if len(accs) > 1:
                    for i in range(len(accs)):
                        for j in range(i + 1, len(accs)):
                            if accs[i] == accs[j]:
                                raise InvariantViolation(
                                    "Upper with all-interface children has duplicate accumulators."
                                )

        elif isinstance(self.inner, Interface):
            # Delegate validation to the inner Lower node.
            self.inner.node.validate_invariants()
