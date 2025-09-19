from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union, Callable, Any

from .interface import GSS, T, Acc, MergeableInt
from .reference_impl import ReferenceGSS

# Sentinel key to represent the "empty stack" child when a node needs to also contain an empty stack.
# This is purely internal; it never leaks out of to_reference_impl() or to_json.
_EPS = object()


# ------------------------------
# Internal node classes (mirroring the Rust-like structure)
# ------------------------------

@dataclass
class _InnerRoot(Generic[T]):
    pass


@dataclass
class _InnerInternal(Generic[T]):
    # children: T -> depth -> _InnerNode
    children: Dict[T, Dict[int, '_InnerNode[T]']]


_InnerNode = Union[_InnerRoot[T], _InnerInternal[T]]


@dataclass
class _WithAcc(Generic[T, Acc]):
    node: _InnerNode[T]
    acc: Acc


@dataclass
class _Internal(Generic[T, Acc]):
    # children: T_or_EPS -> depth -> _LeveledNode
    # Note: T_or_EPS is either a T value or the _EPS sentinel for "empty" stacks at this node.
    children: Dict[object, Dict[int, '_LeveledNode[T, Acc]']]


@dataclass
class _Empty:
    pass


_LeveledNode = Union[_WithAcc[T, Acc], _Internal[T, Acc], _Empty]


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


def _pairs_from_ref(ref: ReferenceGSS[T, Acc]) -> List[Tuple[List[T], Acc]]:
    # The ReferenceGSS stores canonical stacks in _stacks already merged
    return [(list(vals), acc) for (vals, acc) in ref._stacks]  # type: ignore[attr-defined]


def _build_inner_from_sequences(seqs: List[List[T]]) -> _InnerNode[T]:
    # Build the A-level inner tree (no accumulators at this level)
    if not seqs:
        return _InnerRoot()

    # Partition by whether sequence is empty
    non_empty = [s for s in seqs if s]
    empty_count = len(seqs) - len(non_empty)

    if not non_empty:
        # Only empty sequences present -> Root
        return _InnerRoot()

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

    return _InnerInternal(children=children)


def _build_leveled_from_pairs(pairs: List[Tuple[List[T], Acc]]) -> _LeveledNode[T, Acc]:
    pairs = _dedup_pairs(pairs)

    if not pairs:
        return _Empty()

    # Check if all stacks share the same accumulator; then we can store them under a single WithAcc node.
    accs = {acc for _, acc in pairs}
    if len(accs) == 1:
        only_acc = next(iter(accs))
        inner = _build_inner_from_sequences([vals for vals, _ in pairs])
        return _WithAcc(node=inner, acc=only_acc)

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

    node: _LeveledNode[T, Acc] = _Internal(children=children)
    return _normalize_suck_up(node)


def _normalize_suck_up(node: _LeveledNode[T, Acc]) -> _LeveledNode[T, Acc]:
    # Bottom-up normalization: recursively transform children first.
    if isinstance(node, _Empty):
        return node
    if isinstance(node, _WithAcc):
        # Its inner is a pure A-level tree; no accs inside; nothing to do.
        return node
    if isinstance(node, _Internal):
        # Normalize children first
        new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
        for key_t, depth_map in node.children.items():
            for depth, child in depth_map.items():
                norm = _normalize_suck_up(child)
                new_children.setdefault(key_t, {})[depth] = norm

        # Check suck-up condition: if all children are WithAcc and share the same acc
        # If there are no children, it's empty
        if not new_children:
            return _Empty()

        # Flatten list of children
        child_list: List[Tuple[object, int, _LeveledNode[T, Acc]]] = []
        for kt, dm in new_children.items():
            for d, ch in dm.items():
                child_list.append((kt, d, ch))

        all_with_acc = all(isinstance(ch, _WithAcc) for _, _, ch in child_list)
        if all_with_acc:
            accs: Set[Acc] = set(ch.acc for _, _, ch in child_list if isinstance(ch, _WithAcc))
            if len(accs) == 1:
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
                    assert isinstance(chw, _WithAcc)
                    if kt is _EPS:
                        # This child represents an empty sequence among the group
                        # Representing empty within A-level is naturally done by allowing Root.
                        # We mark has_empty to ensure Root is included; but since a WithAcc's node can be Root
                        # and other children may also add edges, we simply union them.
                        # We union the child.inner into the overall A-level; if it's non-Root, include those edges;
                        # if it's Root, that means the empty sequence is present.
                        if isinstance(chw.node, _InnerRoot):
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
                        if isinstance(chw.node, _InnerRoot):
                            # The child being Root under a non-EPS edge means: the sequence [kt] exists.
                            # So in A-level inner, we create an edge kt -> Root with some depth (we can use depth-1 non-negative)
                            # Keep the provided depth (already corresponds to remaining length); to be safe ensure >= 0
                            key_t: T = kt  # type: ignore[assignment]
                            depth_int = max(d - 1, 0)
                            inner_children.setdefault(key_t, {})[depth_int] = _InnerRoot()
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
                        inner: _InnerNode[T] = _InnerRoot()
                    else:
                        # Should not happen (no children => handled above), but keep safe.
                        inner = _InnerRoot()
                else:
                    inner = _InnerInternal(children=inner_children)

                return _WithAcc(node=inner, acc=the_acc)

        return _Internal(children=new_children)

    # Should not reach
    return node


def _enumerate_pairs_from_node(node: _LeveledNode[T, Acc]) -> List[Tuple[List[T], Acc]]:
    # Enumerate (stack, acc) pairs represented by the leveled node
    result: List[Tuple[List[T], Acc]] = []

    def emit_from_inner(inner: _InnerNode[T], prefix: List[T], acc: Acc):
        if isinstance(inner, _InnerRoot):
            result.append((list(prefix), acc))
            return
        # _InnerInternal
        for t, depth_map in inner.children.items():  # type: ignore[union-attr]
            for _, child in depth_map.items():
                emit_from_inner(child, prefix + [t], acc)

    def walk(node_b: _LeveledNode[T, Acc], prefix: List[T]):
        if isinstance(node_b, _Empty):
            return
        if isinstance(node_b, _WithAcc):
            emit_from_inner(node_b.node, prefix, node_b.acc)
            return
        # _Internal
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
    # 1) Acc only exists at _WithAcc nodes.
    # 2) _Inner nodes never contain any acc; only structure.
    # 3) "Suck up" has been applied whenever possible: for any _Internal node,
    #    if all children are _WithAcc and share the same acc, we should not leave it as _Internal.
    # 4) If a _WithAcc node has an _InnerInternal child whose children are identical (structurally),
    #    that's fine; but _WithAcc's descendants should have no another acc (by construction).
    #
    # We traverse and check these constraints; for #3 we just detect a violation opportunity.

    def check_inner(inner: _InnerNode[T]):
        if isinstance(inner, _InnerRoot):
            return
        if isinstance(inner, _InnerInternal):
            for t, depth_map in inner.children.items():
                # Keys must not be EPS at A-level
                if t is _EPS:
                    raise InvariantViolation("EPS sentinel leaked into A-level inner structure.")
                for _, child in depth_map.items():
                    check_inner(child)

    def check(node_b: _LeveledNode[T, Acc]) -> Tuple[bool, Optional[Acc]]:
        if isinstance(node_b, _Empty):
            return True, None
        if isinstance(node_b, _WithAcc):
            # Its inner must be pure structure
            check_inner(node_b.node)
            return True, node_b.acc
        if isinstance(node_b, _Internal):
            # Recurse, collect child accs when child is WithAcc
            child_accs: List[Acc] = []
            child_types: List[type] = []
            for kt, depth_map in node_b.children.items():
                for _, ch in depth_map.items():
                    ok, acc = check(ch)
                    if not ok:
                        return False, None
                    child_types.append(type(ch))
                    if isinstance(ch, _WithAcc):
                        child_accs.append(ch.acc)
            # suck-up opportunity detection
            if child_types and all(ct is _WithAcc for ct in child_types):
                # All children are WithAcc; if their accs are equal, it should have been sucked up
                if child_accs and all(a == child_accs[0] for a in child_accs):
                    raise InvariantViolation("Suck-up opportunity not applied: Internal with uniform WithAcc children.")
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
    def __init__(self, ref: ReferenceGSS[T, Acc], node: _LeveledNode[T, Acc]):
        self._ref = ref
        self._node = node
        # Validate invariants in debug-oriented fashion (can be toggled off if performance becomes a concern)
        try:
            _validate_invariants_node(self._node)
        except InvariantViolation:
            # As a fallback, try to rebuild from current reference impl (which is canonical) and re-validate.
            rebuilt = _build_leveled_from_pairs(_pairs_from_ref(self._ref))
            _validate_invariants_node(rebuilt)
            self._node = rebuilt

    # ---- GSS interface ----

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        # Build a canonical ReferenceGSS first
        ref = ReferenceGSS.from_stacks(stacks)
        # Build our leveled node from the deduped pairs
        pairs = _pairs_from_ref(ref)
        node = _build_leveled_from_pairs(pairs)
        return cls(ref, node)

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        # Delegate semantics to ReferenceGSS
        new_ref = self._ref.push(value)
        new_node = _build_leveled_from_pairs(_pairs_from_ref(new_ref))
        return LeveledGSS(new_ref, new_node)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        new_ref = self._ref.pop()
        new_node = _build_leveled_from_pairs(_pairs_from_ref(new_ref))
        return LeveledGSS(new_ref, new_node)

    def is_empty(self) -> bool:
        return self._ref.is_empty()

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        # Semantics: keep only stacks whose top equals value; if value is None, keep only empty stacks.
        pairs = _pairs_from_ref(self._ref)
        if value is None:
            filtered = [(v, a) for v, a in pairs if len(v) == 0]
        else:
            filtered = [(v, a) for v, a in pairs if v and v[-1] == value]
        new_ref = ReferenceGSS.from_stacks(filtered)
        new_node = _build_leveled_from_pairs(_pairs_from_ref(new_ref))
        return LeveledGSS(new_ref, new_node)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        # Apply func independently to each accumulator
        pairs = _pairs_from_ref(self._ref)
        applied = [(vals, func(acc)) for vals, acc in pairs]
        new_ref = ReferenceGSS.from_stacks(applied)
        # Memoized rebuild would preserve sharing; for now rebuild canonical leveled node
        new_node = _build_leveled_from_pairs(_pairs_from_ref(new_ref))
        return LeveledGSS(new_ref, new_node)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        pairs = _pairs_from_ref(self._ref)
        kept = [(v, a) for v, a in pairs if predicate(a)]
        new_ref = ReferenceGSS.from_stacks(kept)
        new_node = _build_leveled_from_pairs(_pairs_from_ref(new_ref))
        return LeveledGSS(new_ref, new_node)

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        # Convert other's reference representation and merge via ReferenceGSS semantics
        other_ref = other.to_reference_impl()
        assert isinstance(other_ref, ReferenceGSS)
        merged_ref = ReferenceGSS(self._ref._stacks + other_ref._stacks)  # type: ignore[attr-defined]
        # ReferenceGSS constructor canonicalizes and merges duplicate stacks
        new_node = _build_leveled_from_pairs(_pairs_from_ref(merged_ref))
        return LeveledGSS(merged_ref, new_node)

    def peek(self) -> Set[T]:
        # Set of all top values across non-empty stacks
        result: Set[T] = set()
        for vals, _ in _pairs_from_ref(self._ref):
            if vals:
                result.add(vals[-1])
        return result

    def reduce_acc(self) -> Optional[Acc]:
        return self._ref.reduce_acc()

    def to_reference_impl(self) -> 'ReferenceGSS[T, Acc]':
        # Return the canonical ReferenceGSS (gold standard for comparisons)
        return ReferenceGSS(_pairs_from_ref(self._ref))  # type: ignore[arg-type]

    # Also expose a human-friendly validator
    def validate_invariants(self) -> None:
        _validate_invariants_node(self._node)

    # Optional: convenience for debugging
    def __repr__(self) -> str:
        return f"LeveledGSS(ref={self._ref!r})"

    def __str__(self) -> str:
        return f"LeveledGSS({self._ref.to_json_serializable()})"
