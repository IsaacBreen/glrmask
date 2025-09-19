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
class Empty:
    pass


@dataclass
class LeveledGSSInner(Generic[T]):
    inner: Union[InnerLeaf, InnerBranch[T]]


# ------------------------------
# Helpers to convert between ReferenceGSS and our leveled node representation
# ------------------------------

def _build_inner_from_sequences(seqs: List[List[T]]) -> Union[InnerLeaf, InnerBranch[T]]:
    raise NotImplementedError


def _build_leveled_from_pairs(pairs: List[Tuple[List[T], Acc]]) -> Union[WithAcc[T, Acc], Branch[T, Acc], Empty]:
    raise NotImplementedError


def _normalize_suck_up(node: Union[WithAcc[T, Acc], Branch[T, Acc], Empty]) -> Union[WithAcc[T, Acc], Branch[T, Acc], Empty]:
    raise NotImplementedError

def _enumerate_pairs_from_node(node: Union[WithAcc[T, Acc], Branch[T, Acc], Empty]) -> List[Tuple[List[T], Acc]]:
    raise NotImplementedError




# ------------------------------
# Invariant validation
# ------------------------------

class InvariantViolation(Exception):
    pass


def _validate_invariants_node(node: Union[WithAcc[T, Acc], Branch[T, Acc], Empty]):
    def check(n: Union[WithAcc[T, Acc], Branch[T, Acc], Empty]) -> None:
        match n:
            case Empty() | WithAcc():
                return
            case Branch(children=children):
                kids = [ch._node for m in children.values() for ch in m.values()]
                for ch in kids:
                    check(ch)
                if kids and all(isinstance(ch, WithAcc) for ch in kids):
                    acc0 = kids[0].acc
                    if all(ch.acc == acc0 for ch in kids):
                        raise InvariantViolation("Suck-up opportunity not applied: Branch with uniform WithAcc children.")
    check(node)


# ------------------------------
# Public LeveledGSS implementation
# ------------------------------

class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):

    # Construction
    def __init__(self, node: Union[WithAcc[T, Acc], Branch[T, Acc], Empty]):
        self._node = node
        _validate_invariants_node(self._node)

    # ---- GSS interface ----

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        node = _build_leveled_from_pairs(stacks)
        return cls(node)

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self._node)
        new_pairs = [(vals + [value], acc) for vals, acc in pairs]
        return LeveledGSS.from_stacks(new_pairs)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self._node)
        new_pairs = [(vals[:-1], acc) for vals, acc in pairs if vals]
        return LeveledGSS.from_stacks(new_pairs)

    def is_empty(self) -> bool:
        return isinstance(self._node, Empty)

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self._node)
        if value is None:
            filtered = [(v, a) for v, a in pairs if not v]
        else:
            filtered = [(v, a) for v, a in pairs if v and v[-1] == value]
        return LeveledGSS.from_stacks(filtered)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self._node)
        applied = [(vals, func(acc)) for vals, acc in pairs]
        return LeveledGSS.from_stacks(applied)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self._node)
        kept = [(v, a) for v, a in pairs if predicate(a)]
        return LeveledGSS.from_stacks(kept)

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        self_pairs = _enumerate_pairs_from_node(self._node)
        other_ref = other.to_reference_impl()
        other_pairs = other_ref._stacks
        all_pairs = self_pairs + other_pairs
        return LeveledGSS.from_stacks(all_pairs)

    def peek(self) -> Set[T]:
        result: Set[T] = set()
        pairs = _enumerate_pairs_from_node(self._node)
        for vals, _ in pairs:
            if vals:
                result.add(vals[-1])
        return result

    def reduce_acc(self) -> Optional[Acc]:
        pairs = _enumerate_pairs_from_node(self._node)
        if not pairs:
            return None
        accs = [acc for _, acc in pairs]
        return reduce(lambda a, b: a.merge(b), accs)

    def to_reference_impl(self) -> 'ReferenceGSS[T, Acc]':
        pairs = _enumerate_pairs_from_node(self._node)
        return ReferenceGSS.from_stacks(pairs)

    # Also expose a human-friendly validator
    def validate_invariants(self) -> None:
        _validate_invariants_node(self._node)

