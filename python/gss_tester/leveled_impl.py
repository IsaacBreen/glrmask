from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Type, Union

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes
# ------------------------------

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    inner: Upper[T, Acc] | Interface[T, Acc]


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    children: Dict[T, Dict[int, Upper[T, Acc]]]


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    node: Lower[T]
    acc: Acc


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    inner: LowerBranch[T] | Leaf


@dataclass(frozen=True, eq=True)
class LowerBranch(Generic[T]):
    children: Dict[T, Dict[int, Lower[T]]]


@dataclass(frozen=True, eq=True)
class Leaf:
    pass


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        raise NotImplementedError
    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        raise NotImplementedError
    def push(self, value: T) -> LeveledGSS[T, Acc]:
        raise NotImplementedError
    def pop(self) -> LeveledGSS[T, Acc]:
        raise NotImplementedError
    def is_empty(self) -> bool:
        raise NotImplementedError
    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        raise NotImplementedError
    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        raise NotImplementedError
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        raise NotImplementedError
    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        raise NotImplementedError
    def peek(self) -> Set[T]:
        raise NotImplementedError
    def reduce_acc(self) -> Optional[Acc]:
        raise NotImplementedError


def _validate_upper(node: Upper[T, Acc]):
    """Recursively validates invariants on Upper nodes."""
    if isinstance(node.inner, Upper):
        # Recurse on wrapper
        _validate_upper(node.inner)
    elif isinstance(node.inner, UpperBranch):
        branch = node.inner
        all_children = [
            child
            for children_by_val in branch.children.values()
            for child in children_by_val.values()
        ]

        # Invariant 1: If all children are interfaces, their accs must be unique.
        if all_children and all(isinstance(child.inner, Interface) for child in all_children):
            accs = [child.inner.acc for child in all_children]
            if len(set(accs)) != len(accs):
                raise AssertionError(
                    "Invariant violated: UpperBranch has children that are all Interfaces with duplicate accs."
                )

        # Recurse into children
        for child in all_children:
            _validate_upper(child)
    # Base case: node.inner is an Interface, do nothing further down this path.


def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Checks internal invariants of the LeveledGSS structure.
    Raises AssertionError if an invariant is violated.
    """
    # Check recursive invariants on the inner structure.
    _validate_upper(gss.inner)

    # Invariant 2: If inner is an interface and empty exists, their accs must differ.
    if isinstance(gss.inner, Interface) and gss.empty is not None:
        if gss.inner.acc == gss.empty:
            raise AssertionError(
                "Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator."
            )
