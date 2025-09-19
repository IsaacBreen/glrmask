from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Dict, Generic, List, Optional, Set, Tuple, Type, Union, Callable

from .interface import GSS, T, Acc
from .reference_impl import ReferenceGSS


# ------------------------------
# Internal node classes
# ------------------------------

@dataclass
class InnerLeaf:
    pass



@dataclass
class InnerBranch(Generic[T]):
    # children: T -> depth -> LeveledGSSInner
    children: Dict[T, Dict[int, 'LeveledGSSInner[T]']]


@dataclass
class WithAcc(Generic[T, Acc]):
    node: 'LeveledGSSInner[T]'
    acc: Acc


@dataclass
class Branch(Generic[T, Acc]):
    # children: T_or_EPS -> depth -> LeveledGSS
    # Note: T_or_EPS is either a T value or the _EPS sentinel for "empty" stacks at this node.
    children: Dict[object, Dict[int, 'LeveledGSS[T, Acc]']]


@dataclass
class LeveledGSSInner(Generic[T]):
    inner: Union[InnerLeaf, InnerBranch[T]]


# ------------------------------
# Helpers to convert between ReferenceGSS and our leveled node representation
# ------------------------------

def _build_inner_from_sequences(seqs: List[List[T]]) -> Union[InnerLeaf, InnerBranch[T]]:
    raise NotImplementedError


def _build_leveled_from_pairs(pairs: List[Tuple[List[T], Acc]]) -> Union[WithAcc[T, Acc], Branch[T, Acc]]:
    raise NotImplementedError


def _normalize_suck_up(node: Union[WithAcc[T, Acc], Branch[T, Acc]]) -> Union[WithAcc[T, Acc], Branch[T, Acc]]:
    raise NotImplementedError

def _enumerate_pairs_from_node(node: Union[WithAcc[T, Acc], Branch[T, Acc]]) -> List[Tuple[List[T], Acc]]:
    raise NotImplementedError




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

class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Union[WithAcc[T, Acc], Branch[T, Acc]]

    def __init__(self, inner: Union[WithAcc[T, Acc], Branch[T, Acc]]):
        self.inner = inner

    # ---- GSS interface ----

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        node = _build_leveled_from_pairs(stacks)
        return cls(node)

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self.inner)
        new_pairs = [(vals + [value], acc) for vals, acc in pairs]
        return LeveledGSS.from_stacks(new_pairs)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self.inner)
        new_pairs = [(vals[:-1], acc) for vals, acc in pairs if vals]
        return LeveledGSS.from_stacks(new_pairs)

    def is_empty(self) -> bool:
        return isinstance(self.inner, Branch) and not self.inner.children

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self.inner)
        if value is None:
            filtered = [(v, a) for v, a in pairs if not v]
        else:
            filtered = [(v, a) for v, a in pairs if v and v[-1] == value]
        return LeveledGSS.from_stacks(filtered)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self.inner)
        applied = [(vals, func(acc)) for vals, acc in pairs]
        return LeveledGSS.from_stacks(applied)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self.inner)
        kept = [(v, a) for v, a in pairs if predicate(a)]
        return LeveledGSS.from_stacks(kept)

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        self_pairs = _enumerate_pairs_from_node(self.inner)
        other_ref = other.to_reference_impl()
        other_pairs = other_ref._stacks
        all_pairs = self_pairs + other_pairs
        return LeveledGSS.from_stacks(all_pairs)

    def peek(self) -> Set[T]:
        result: Set[T] = set()
        pairs = _enumerate_pairs_from_node(self.inner)
        for vals, _ in pairs:
            if vals:
                result.add(vals[-1])
        return result

    def reduce_acc(self) -> Optional[Acc]:
        pairs = _enumerate_pairs_from_node(self.inner)
        if not pairs:
            return None
        accs = [acc for _, acc in pairs]
        return reduce(lambda a, b: a.merge(b), accs)

    def to_reference_impl(self) -> 'ReferenceGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self.inner)
        return ReferenceGSS.from_stacks(pairs)

    # Also expose a human-friendly validator
    def validate_invariants(self) -> None:
        _validate_invariants_node(self.inner)

