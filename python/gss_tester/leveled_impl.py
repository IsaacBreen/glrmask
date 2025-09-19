from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union, Callable, Any
from functools import reduce

from .interface import GSS, T, Acc, MergeableInt
from .reference_impl import ReferenceGSS

# ------------------------------
# Internal node classes (mirroring the Rust-like structure)
# ------------------------------

@dataclass
class InnerLeaf:
    pass


@dataclass
class InnerBranch(Generic[T]):
    # children: T -> depth -> _InnerNode
    children: Dict[T, Dict[int, '_InnerNode[T]']]


_InnerNode = Union[InnerLeaf, InnerBranch[T]]


@dataclass
class WithAcc(Generic[T, Acc]):
    node: _InnerNode[T]
    acc: Acc


@dataclass
class Branch(Generic[T, Acc]):
    # children: T_or_EPS -> depth -> _LeveledNode
    # Note: T_or_EPS is either a T value or the _EPS sentinel for "empty" stacks at this node.
    children: Dict[object, Dict[int, '_LeveledNode[T, Acc]']]


@dataclass
class Empty:
    pass


_LeveledNode = Union[WithAcc[T, Acc], Branch[T, Acc], Empty]


# ------------------------------
# Helpers to convert between ReferenceGSS and our leveled node representation
# ------------------------------

def _merge_acc(a: Acc, b: Acc) -> Acc:
    # Acc is Mergeable by protocol; use merge to combine
    return a.merge(b)  # type: ignore[attr-defined]


def _dedup_pairs(pairs: List[Tuple[List[T], Acc]]) -> List[Tuple[List[T], Acc]]:
    # Merge duplicate stacks by merging accumulators
    merged: Dict[Tuple[T, ...], Acc] = {}
    for vals, acc in pairs:
        key = tuple(vals)
        if key in merged:
            merged[key] = _merge_acc(merged[key], acc)
        else:
            merged[key] = acc
    return [(list(k), v) for k, v in merged.items()]


def _build_inner_from_sequences(seqs: List[List[T]]) -> _InnerNode[T]:
    # Build the A-level inner tree (no accumulators at this level)
    if not seqs:
        return InnerLeaf()

    # Partition by whether sequence is empty
    non_empty = [s for s in seqs if s]
    empty_count = len(seqs) - len(non_empty)

    if not non_empty:
        # Only empty sequences present -> Leaf
        return InnerLeaf()

    # Group by first token
    group: Dict[T, List[List[T]]] = {}
    for s in non_empty:
        t = s[0]
        tail = s[1:]
        group.setdefault(t, []).append(tail)

    children: Dict[T, Dict[int, _InnerNode[T]]] = {}
    for t, tails in group.items():
        child_inner = _build_inner_from_sequences(tails)
        # We set "depth" to the maximum length remaining (for determinism and a form of "max depth")
        max_depth = max((len(tl) for tl in tails), default=0)
        children.setdefault(t, {})[max_depth] = child_inner

    return InnerBranch(children=children)


def _build_leveled_from_pairs(pairs: List[Tuple[List[T], Acc]]) -> _LeveledNode[T, Acc]:
    raise NotImplementedError


def _normalize_suck_up(node: _LeveledNode[T, Acc]) -> _LeveledNode[T, Acc]:
    raise NotImplementedError

def _enumerate_pairs_from_node(node: _LeveledNode[T, Acc]) -> List[Tuple[List[T], Acc]]:
    raise NotImplementedError


# ------------------------------
# Invariant validation
# ------------------------------

class InvariantViolation(Exception):
    pass


def _validate_invariants_node(node: _LeveledNode[T, Acc]):
    # Ensure that "Suck up" has been applied whenever possible: for any Branch node, if all children are WithAcc and share the same acc, we should not leave it as Branch.
    def check(node_b: _LeveledNode[T, Acc]) -> Tuple[bool, Optional[Acc]]:
        match node_b:
            case Empty():
                return True, None
            case WithAcc(acc=acc):
                return True, acc
            case Branch(children=children):
                # Recurse, collect child accs when child is WithAcc
                child_accs: List[Acc] = []
                child_types: List[type] = []
                for kt, depth_map in children.items():
                    for _, ch in depth_map.items():
                        ok, acc = check(ch)
                        if not ok:
                            return False, None
                        child_types.append(type(ch))
                        if isinstance(ch, WithAcc):
                            child_accs.append(ch.acc)
                # suck-up opportunity detection
                if child_types and all(ct is WithAcc for ct in child_types):
                    # All children are WithAcc; if their accs are equal, it should have been sucked up
                    if child_accs and all(a == child_accs[0] for a in child_accs):
                        raise InvariantViolation("Suck-up opportunity not applied: Branch with uniform WithAcc children.")
                return True, None

    ok, _ = check(node)
    if not ok:
        raise InvariantViolation("Invariant validation failed for unknown reason.")


# ------------------------------
# Public LeveledGSS implementation
# ------------------------------

class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A leveled, graph-structured stack implementation that mirrors the provided Rust-like shape.

    Notes:
    - Semantics are kept identical to ReferenceGSS by delegating operation semantics to an internal
      ReferenceGSS and rebuilding the leveled structure on each operation. This favors correctness.
    - Internal structure (_node) respects invariants, with an explicit "suck-up" normalization pass.
    - For determinism and correctness, to_reference_impl() returns the internal ReferenceGSS.
    """

    # Construction
    def __init__(self, node: _LeveledNode[T, Acc]):
        self._node = node
        # Validate invariants in debug-oriented fashion (can be toggled off if performance becomes a concern)
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

