from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any

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
        def _build_leveled_gss_recursively(
                stacks: Dict[Tuple[T, ...], Acc]
        ) -> LeveledGSS[T, Acc]:
            """
            Recursively builds a LeveledGSS from a dictionary of stacks.
            This approach correctly handles the prefix problem by treating terminations
            (empty stacks in the recursive context) and branches distinctly.
            """
            empty_acc = stacks.pop((), None)

            groups: Dict[T, List[Tuple[Tuple[T, ...], Acc]]] = {}
            for stack_tuple, acc in stacks.items():
                val = stack_tuple[0]
                if val not in groups:
                    groups[val] = []
                groups[val].append((stack_tuple[1:], acc))

            children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for val, sub_stacks_list in groups.items():
                sub_stacks_dict = dict(sub_stacks_list)
                sub_gss = _build_leveled_gss_recursively(sub_stacks_dict)

                # If the sub-GSS has an accumulator for the empty stack, it means
                # a stack in the parent context terminated at `val`. We represent this
                # with an Interface node.
                if sub_gss.empty is not None:
                    interface_node = Upper(Interface(_LOWER_LEAF, sub_gss.empty))
                    if val not in children:
                        children[val] = {}
                    children[val][len(children[val])] = interface_node

                # If the sub-GSS has branches, it means there are stacks longer than `val`.
                # We embed the inner structure of the sub-GSS as another child.
                if isinstance(sub_gss.inner.inner, UpperBranch) and sub_gss.inner.inner.children:
                    if val not in children:
                        children[val] = {}
                    children[val][len(children[val])] = sub_gss.inner

            return LeveledGSS(Upper(UpperBranch(children)), empty_acc)

        # 1. Merge stacks with identical values into a dictionary.
        merged: Dict[Tuple[T, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc

        # 2. Build the LeveledGSS structure recursively.
        return _build_leveled_gss_recursively(merged)

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

        # Sort for a canonical representation, as required by the interface.
        import json

        def _encode_for_sort(obj: Any) -> str:
            try:
                return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
            except Exception:
                return repr(obj)

        all_stacks.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        return all_stacks

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
