from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any
from collections import defaultdict

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes
# ------------------------------

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    inner: UpperBranch[T, Acc] | Interface[T, Acc]


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


# A shared, canonical leaf node
_LOWER_LEAF = Lower(Leaf())


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        def _build_recursively(
            stacks_dict: Dict[Tuple[T, ...], Acc]
        ) -> LeveledGSS[T, Acc]:
            """Recursively builds the LeveledGSS from a dictionary of stacks."""
            empty_acc = stacks_dict.pop((), None)

            # Group stacks by their first element
            groups: Dict[T, List[Tuple[Tuple[T, ...], Acc]]] = defaultdict(list)
            for stack, acc in stacks_dict.items():
                groups[stack[0]].append((stack[1:], acc))

            children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for val, sub_stacks in groups.items():
                sub_gss = _build_recursively(dict(sub_stacks))

                nodes = []
                # A sub-GSS with an `empty` acc means a stack terminated at `val`.
                if sub_gss.empty is not None:
                    nodes.append(Upper(Interface(_LOWER_LEAF, sub_gss.empty)))

                # A sub-GSS with children means some stacks continue deeper.
                if isinstance(sub_gss.inner.inner, UpperBranch) and sub_gss.inner.inner.children:
                    nodes.append(sub_gss.inner)

                if nodes:
                    children[val] = {i: node for i, node in enumerate(nodes)}

            return LeveledGSS(Upper(UpperBranch(children)), empty_acc)

        # 1. Merge stacks with identical values using the reference implementation.
        from .reference_impl import ReferenceGSS
        merged = {tuple(s): a for s, a in ReferenceGSS(stacks)._stacks}

        # 2. Build the structure from the merged stacks.
        return _build_recursively(merged)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        all_stacks: List[Tuple[List[T], Acc]] = []
        if self.empty is not None:
            all_stacks.append(([], self.empty))

        def traverse(node: Upper[T, Acc], prefix: List[T]):
            """Recursively traverses the tree structure to reconstruct stacks."""
            if isinstance(node.inner, Interface):
                # An Interface node signifies the end of a stack.
                # In this implementation, the Lower part is always a Leaf,
                # so the tail is empty.
                all_stacks.append((prefix, node.inner.acc))
            else:  # UpperBranch
                branch = node.inner
                for val, children_by_id in branch.children.items():
                    for child_node in children_by_id.values():
                        traverse(child_node, prefix + [val])

        traverse(self.inner, [])

        # Delegate canonicalization (sorting) to the reference implementation.
        from .reference_impl import ReferenceGSS
        return ReferenceGSS(all_stacks).to_stacks()

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().push(value).to_stacks())
    def pop(self) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().pop().to_stacks())
    def is_empty(self) -> bool:
        return self.to_reference_impl().is_empty()
    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().isolate(value).to_stacks())
    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().apply(func).to_stacks())
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().prune(predicate).to_stacks())
    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().merge(other.to_reference_impl()).to_stacks())
    def peek(self) -> Set[T]:
        return self.to_reference_impl().peek()
    def reduce_acc(self) -> Optional[Acc]:
        return self.to_reference_impl().reduce_acc()


def _validate_upper(node: Upper[T, Acc]):
    """Recursively validates invariants on Upper nodes."""
    if isinstance(node.inner, UpperBranch):
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
