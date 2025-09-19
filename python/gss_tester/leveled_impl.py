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


# ------------------------------
# Invariant validation
# ------------------------------

class InvariantViolation(Exception):
    pass


def _validate_invariants_node(node: LeveledGSS[T, Acc]):
    def check(n: LeveledGSS[T, Acc]) -> None:
        # This function validates a LeveledGSS node and recurses on its children.
        inner = n.inner

        if isinstance(inner, WithAcc):
            # Check invariants on the inner LeveledGSSInner node.
            def check_inner(inner_node: LeveledGSSInner[T]) -> None:
                if isinstance(inner_node.inner, InnerBranch):
                    # Invariant: inner branch should always have at least 1 item.
                    if not inner_node.inner.children:
                        raise InvariantViolation("InnerBranch has no children.")
                    # Recurse on inner nodes
                    for depths in inner_node.inner.children.values():
                        for child in depths.values():
                            check_inner(child)

            check_inner(inner.node)

        elif isinstance(inner, Branch):
            if not inner.children:
                return  # This is a valid empty GSS.

            children_gss = [
                child_gss
                for depths in inner.children.values()
                for child_gss in depths.values()
            ]

            # Invariant for (outer) branch: should never have a child that is itself a(n outer) branch with zero items.
            for child in children_gss:
                if isinstance(child.inner, Branch) and not child.inner.children:
                    raise InvariantViolation("Branch has an empty Branch as a child.")

            # Invariant for (outer) branch: if one child is WithAcc, then at least one other child must either be not WithAcc or must have Acc unequal to the first child's.
            if len(children_gss) > 1:
                first_child_inner = children_gss[0].inner
                if isinstance(first_child_inner, WithAcc):
                    first_acc = first_child_inner.acc
                    all_children_are_withacc_with_same_acc = all(
                        isinstance(c.inner, WithAcc) and c.inner.acc == first_acc
                        for c in children_gss
                    )
                    if all_children_are_withacc_with_same_acc:
                        raise InvariantViolation(
                            "Branch with all WithAcc children having the same accumulator is not normalized."
                        )

            # Recurse
            for child in children_gss:
                check(child)

    check(node)


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

