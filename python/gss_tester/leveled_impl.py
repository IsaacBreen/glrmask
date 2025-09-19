from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union, Callable, Any
from functools import reduce

from .interface import GSS, T, Acc, MergeableInt
from .reference_impl import ReferenceGSS

# Sentinel key to represent the "empty stack" child when a node needs to also contain an empty stack.
# This is purely internal; it never leaks out of to_reference_impl() or to_json.
_EPS = object()


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
    pairs = _dedup_pairs(pairs)

    if not pairs:
        return Empty()

    # Check if all stacks share the same accumulator; then we can store them under a single WithAcc node.
    accs = {acc for _, acc in pairs}
    if len(accs) == 1:
        vals_list = [vals for vals, _ in pairs]
        has_empty = any(not v for v in vals_list)
        has_non_empty = any(v for v in vals_list)

        if not (has_empty and has_non_empty):
            only_acc = next(iter(accs))
            inner = _build_inner_from_sequences(vals_list)
            return WithAcc(node=inner, acc=only_acc)
        # else: fall through to the general case which can handle mixed empty/non-empty

    # Otherwise, build an Internal node, partitioning by first symbol.
    # Empty stacks ([]) must still be representable: we attach them under the _EPS sentinel.
    children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}

    # Handle empty stacks (if any)
    empty_pairs = [(vals, acc) for vals, acc in pairs if not vals]
    if empty_pairs:
        # Recursively build a child node for these empty stacks.
        child = _build_leveled_from_pairs(empty_pairs)
        # Depth for empty is 0
        children.setdefault(_EPS, {})[0] = child

    # Non-empty stacks
    non_empty_pairs = [(vals, acc) for vals, acc in pairs if vals]
    by_first: Dict[T, List[Tuple[List[T], Acc]]] = {}
    for vals, acc in non_empty_pairs:
        by_first.setdefault(vals[0], []).append((vals[1:], acc))

    for t, tails_pairs in by_first.items():
        # Recursively build subtree for all stacks with first token == t
        child = _build_leveled_from_pairs(tails_pairs)
        # The "depth" we attach can be the maximum length (remaining) among those tails
        max_depth = max((len(v) for v, _ in tails_pairs), default=0)
        children.setdefault(t, {})[max_depth] = child

    node: _LeveledNode[T, Acc] = Branch(children=children)
    return _normalize_suck_up(node)


def _normalize_suck_up(node: _LeveledNode[T, Acc]) -> _LeveledNode[T, Acc]:
    # Bottom-up normalization: recursively transform children first.
    if isinstance(node, Empty):
        return node
    if isinstance(node, WithAcc):
        # Its inner is a pure A-level tree; no accs inside; nothing to do.
        return node
    if isinstance(node, Branch):
        # Normalize children first
        new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
        for key_t, depth_map in node.children.items():
            for depth, child in depth_map.items():
                norm = _normalize_suck_up(child)
                new_children.setdefault(key_t, {})[depth] = norm

        # Check suck-up condition: if all children are WithAcc and share the same acc
        # If there are no children, it's empty
        if not new_children:
            return Empty()

        # Flatten list of children
        child_list: List[Tuple[object, int, _LeveledNode[T, Acc]]] = []
        for kt, dm in new_children.items():
            for d, ch in dm.items():
                child_list.append((kt, d, ch))

        all_with_acc = all(isinstance(ch, WithAcc) for _, _, ch in child_list)
        if all_with_acc:
            accs: Set[Acc] = set(ch.acc for _, _, ch in child_list if isinstance(ch, WithAcc))
            if len(accs) == 1:
                # A suck-up would lose information if we need to represent both a terminal
                # state (from an _EPS child) and a branching state (from non-EPS children),
                # because the _InnerNode structure cannot do both.
                has_eps_child = any(kt is _EPS for kt, _, _ in child_list)
                has_non_eps_child = any(kt is not _EPS for kt, _, _ in child_list)
                if has_eps_child and has_non_eps_child:
                    return Branch(children=new_children)

                # We can "suck up": create a single WithAcc whose inner combines the A-level of all children.
                the_acc = next(iter(accs))
                # Build an A-level Internal whose children map keys to the inner nodes of the children.
                # The EPS child (empty) corresponds to retaining the empty sequence inside the A-level inner.
                # To incorporate EPS into A-level inner, we treat EPS -> Root child as representing an empty sequence.
                # We can merge all children's inner trees into one A-level node by creating an A-level Internal where
                # each edge t maps to the union of the children's inner nodes under that t. For simplicity, we do not
                # attempt to merge A-level siblings with the same t and different depths; we keep one entry per
                # (t, depth) pair.
                inner_children: Dict[T, Dict[int, _InnerNode[T]]] = {}
                has_empty = False
                # Collect A-level children from the WithAcc children
                for kt, d, ch in child_list:
                    chw = ch  # type: ignore[assignment]
                    assert isinstance(chw, WithAcc)
                    if kt is _EPS:
                        # This child represents an empty sequence among the group
                        # Representing empty within A-level is naturally done by allowing Root.
                        # We mark has_empty to ensure Root is included; but since a WithAcc's node can be Root
                        # and other children may also add edges, we simply union them.
                        # We union the child.inner into the overall A-level; if it's non-Root, include those edges;
                        # if it's Leaf, that means the empty sequence is present.
                        if isinstance(chw.node, InnerLeaf):
                            has_empty = True
                        else:
                            # This corresponds to "some non-empty A-level sequences that appear even though the parent
                            # edge was EPS". It can happen if the empty group collected non-empty A-level nodes via previous
                            # suck-ups; we include them.
                            for tt, dm in chw.node.children.items():  # type: ignore[union-attr]
                                for dd, inn in dm.items():
                                    inner_children.setdefault(tt, {})[dd] = inn
                            has_empty = True
                    else:
                        # Regular T key
                        if isinstance(chw.node, InnerLeaf):
                            # The child being Root under a non-EPS edge means: the sequence [kt] exists.
                            # So in A-level inner, we create an edge kt -> Root with some depth (we can use depth-1 non-negative)
                            # Keep the provided depth (already corresponds to remaining length); to be safe ensure >= 0
                            key_t: T = kt  # type: ignore[assignment]
                            depth_int = max(d - 1, 0)
                            inner_children.setdefault(key_t, {})[depth_int] = InnerLeaf()
                        else:
                            # Merge edges
                            key_t: T = kt  # type: ignore[assignment]
                            for dd, inn in ((max(d - 1, 0), chw.node),):  # one entry, but we may later elaborate
                                # We store the entire inner subtree under (key_t, depth-1)
                                inner_children.setdefault(key_t, {})[dd] = inn  # type: ignore[arg-type]

                # If there were no non-EPS children but has_empty is True and there are no other children,
                # then the inner is just Root.
                if not inner_children:
                    if has_empty:
                        inner: _InnerNode[T] = InnerLeaf()
                    else:
                        # Should not happen (no children => handled above), but keep safe.
                        inner = InnerLeaf()
                else:
                    inner = InnerBranch(children=inner_children)

                return WithAcc(node=inner, acc=the_acc)

        return Branch(children=new_children)

    # Should not reach
    return node


def _enumerate_pairs_from_node(node: _LeveledNode[T, Acc]) -> List[Tuple[List[T], Acc]]:
    # Enumerate (stack, acc) pairs represented by the leveled node
    result: List[Tuple[List[T], Acc]] = []

    def emit_from_inner(inner: _InnerNode[T], prefix: List[T], acc: Acc):
        if isinstance(inner, InnerLeaf):
            result.append((list(prefix), acc))
            return
        # InnerBranch
        for t, depth_map in inner.children.items():  # type: ignore[union-attr]
            for _, child in depth_map.items():
                emit_from_inner(child, prefix + [t], acc)

    def walk(node_b: _LeveledNode[T, Acc], prefix: List[T]):
        if isinstance(node_b, Empty):
            return
        if isinstance(node_b, WithAcc):
            emit_from_inner(node_b.node, prefix, node_b.acc)
            return
        # Branch
        for key_t, depth_map in node_b.children.items():
            for _, child in depth_map.items():
                if key_t is _EPS:
                    # Empty step: do not advance prefix
                    walk(child, prefix)
                else:
                    walk(child, prefix + [key_t])  # type: ignore[list-item]

    walk(node, [])
    return _dedup_pairs(result)


# ------------------------------
# Invariant validation
# ------------------------------

class InvariantViolation(Exception):
    pass


def _validate_invariants_node(node: _LeveledNode[T, Acc]):
    # Ensure that:
    # 1) Acc only exists at WithAcc nodes.
    # 2) _Inner nodes never contain any acc; only structure.
    # 3) "Suck up" has been applied whenever possible: for any Branch node,
    #    if all children are WithAcc and share the same acc, we should not leave it as Branch.
    # 4) If a WithAcc node has an InnerBranch child whose children are identical (structurally),
    #    that's fine; but _WithAcc's descendants should have no another acc (by construction).
    #
    # We traverse and check these constraints; for #3 we just detect a violation opportunity.

    def check_inner(inner: _InnerNode[T]):
        if isinstance(inner, InnerLeaf):
            return
        if isinstance(inner, InnerBranch):
            for t, depth_map in inner.children.items():
                # Keys must not be EPS at A-level
                if t is _EPS:
                    raise InvariantViolation("EPS sentinel leaked into A-level inner structure.")
                for _, child in depth_map.items():
                    check_inner(child)

    def check(node_b: _LeveledNode[T, Acc]) -> Tuple[bool, Optional[Acc]]:
        if isinstance(node_b, Empty):
            return True, None
        if isinstance(node_b, WithAcc):
            # Its inner must be pure structure
            check_inner(node_b.node)
            return True, node_b.acc
        if isinstance(node_b, Branch):
            # Recurse, collect child accs when child is WithAcc
            child_accs: List[Acc] = []
            child_types: List[type] = []
            for kt, depth_map in node_b.children.items():
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

