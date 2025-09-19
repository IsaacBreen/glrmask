from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
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
# Efficient Traversal-Based Operations
# ------------------------------

def _node_max_depth(node: _LeveledNode[T, Acc]) -> int:
    if isinstance(node, _Empty):
        return 0
    if isinstance(node, _WithAcc):
        return _inner_max_depth(node.node)
    # _Internal
    max_d = 0
    for _, depth_map in node.children.items():
        for depth, child in depth_map.items():
            max_d = max(max_d, 1 + _node_max_depth(child))
    return max_d


def _inner_max_depth(inner: _InnerNode[T]) -> int:
    if isinstance(inner, _InnerRoot):
        return 0
    # _InnerInternal
    max_d = 0
    for _, depth_map in inner.children.items():
        for depth, child in depth_map.items():
            max_d = max(max_d, 1 + _inner_max_depth(child))
    return max_d


def _explode(node: _WithAcc[T, Acc]) -> _Internal[T, Acc]:
    """Converts a _WithAcc node into an equivalent _Internal node."""
    if isinstance(node.node, _InnerRoot):
        # Represents a single empty stack with an accumulator.
        # The child is _WithAcc because _EPS must lead to a node with an accumulator.
        return _Internal(children={_EPS: {0: _WithAcc(node=_InnerRoot(), acc=node.acc)}})

    # _InnerInternal
    new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
    for t, depth_map in node.node.children.items():
        for depth, child_inner in depth_map.items():
            new_children.setdefault(t, {})[depth] = _WithAcc(node=child_inner, acc=node.acc)
    return _Internal(children=new_children)


def _merge_inners(a: _InnerNode[T], b: _InnerNode[T]) -> _InnerNode[T]:
    if a is b: return a
    if isinstance(a, _InnerRoot): return b
    if isinstance(b, _InnerRoot): return a

    # Both are _InnerInternal
    new_children = {k: v.copy() for k, v in a.children.items()}
    for t, d_map_b in b.children.items():
        if t not in new_children:
            new_children[t] = d_map_b
        else:
            # Merge depth maps
            for d, child_b in d_map_b.items():
                if d not in new_children[t]:
                    new_children[t][d] = child_b
                else:
                    new_children[t][d] = _merge_inners(new_children[t][d], child_b)
    return _InnerInternal(children=new_children)


def _merge_nodes(a: _LeveledNode[T, Acc], b: _LeveledNode[T, Acc]) -> _LeveledNode[T, Acc]:
    if a is b: return a
    if isinstance(a, _Empty): return b
    if isinstance(b, _Empty): return a

    # If both are WithAcc with same acc, merge their inners.
    if isinstance(a, _WithAcc) and isinstance(b, _WithAcc) and a.acc == b.acc:
        return _WithAcc(node=_merge_inners(a.node, b.node), acc=a.acc)

    # Explode any WithAcc to Internal to proceed with merge.
    a_int = _explode(a) if isinstance(a, _WithAcc) else a
    b_int = _explode(b) if isinstance(b, _WithAcc) else b
    assert isinstance(a_int, _Internal)
    assert isinstance(b_int, _Internal)

    # Merge children of the two _Internal nodes.
    new_children = {k: v.copy() for k, v in a_int.children.items()}
    for t, d_map_b in b_int.children.items():
        if t not in new_children:
            new_children[t] = d_map_b
        else:
            for d, child_b in d_map_b.items():
                if d not in new_children[t]:
                    new_children[t][d] = child_b
                else:
                    new_children[t][d] = _merge_nodes(new_children[t][d], child_b)

    return _normalize_suck_up(_Internal(children=new_children))


def _merge_many_nodes(nodes: Iterable[_LeveledNode[T, Acc]]) -> _LeveledNode[T, Acc]:
    return reduce(_merge_nodes, nodes, _Empty())


def _pop_inner(inner: _InnerNode[T]) -> _InnerNode[T]:
    if isinstance(inner, _InnerRoot):
        return _InnerRoot() # Cannot pop empty, effectively remains empty.
    # Merge all children.
    return reduce(_merge_inners, (child for dm in inner.children.values() for child in dm.values()), _InnerRoot())


def _pop_node(node: _LeveledNode[T, Acc]) -> _LeveledNode[T, Acc]:
    if isinstance(node, _Empty):
        return _Empty()
    if isinstance(node, _WithAcc):
        return _WithAcc(node=_pop_inner(node.node), acc=node.acc)
    # _Internal: merge all non-EPS children
    children_to_merge = [child for t, dm in node.children.items() if t is not _EPS for child in dm.values()]
    return _merge_many_nodes(children_to_merge)


def _apply_node(node: _LeveledNode[T, Acc], func: Callable[[Acc], Acc], memo: Dict[int, _LeveledNode[T, Acc]]) -> _LeveledNode[T, Acc]:
    if id(node) in memo:
        return memo[id(node)]

    if isinstance(node, _Empty):
        result = _Empty()
    elif isinstance(node, _WithAcc):
        new_acc = func(node.acc)
        result = node if new_acc == node.acc else _WithAcc(node=node.node, acc=new_acc)
    else: # _Internal
        new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
        changed = False
        for t, d_map in node.children.items():
            for d, child in d_map.items():
                new_child = _apply_node(child, func, memo)
                if new_child is not child:
                    changed = True
                new_children.setdefault(t, {})[d] = new_child
        result = node if not changed else _normalize_suck_up(_Internal(children=new_children))

    memo[id(node)] = result
    return result


def _prune_node(node: _LeveledNode[T, Acc], predicate: Callable[[Acc], bool], memo: Dict[int, _LeveledNode[T, Acc]]) -> _LeveledNode[T, Acc]:
    if id(node) in memo:
        return memo[id(node)]

    if isinstance(node, _Empty):
        result = _Empty()
    elif isinstance(node, _WithAcc):
        result = node if predicate(node.acc) else _Empty()
    else: # _Internal
        new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
        changed = False
        for t, d_map in node.children.items():
            for d, child in d_map.items():
                new_child = _prune_node(child, predicate, memo)
                if new_child is not child:
                    changed = True
                if not isinstance(new_child, _Empty):
                    new_children.setdefault(t, {})[d] = new_child
        if not changed:
            result = node
        else:
            result = _Empty() if not new_children else _normalize_suck_up(_Internal(children=new_children))

    memo[id(node)] = result
    return result


def _peek_inner(inner: _InnerNode[T]) -> Set[T]:
    if isinstance(inner, _InnerRoot):
        return set()
    return set(inner.children.keys())


def _peek_node(node: _LeveledNode[T, Acc]) -> Set[T]:
    if isinstance(node, _Empty):
        return set()
    if isinstance(node, _WithAcc):
        return _peek_inner(node.node)
    # _Internal
    return {t for t in node.children.keys() if t is not _EPS}


def _reduce_acc_node(node: _LeveledNode[T, Acc]) -> Optional[Acc]:
    if isinstance(node, _Empty):
        return None
    if isinstance(node, _WithAcc):
        return node.acc
    # _Internal
    accs = [_reduce_acc_node(child) for dm in node.children.values() for child in dm.values()]
    valid_accs = [acc for acc in accs if acc is not None]
    if not valid_accs:
        return None
    return reduce(_merge_acc, valid_accs)


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
    def __init__(self, node: _LeveledNode[T, Acc]):
        self._node = node
        # Validate invariants in debug-oriented fashion (can be toggled off if performance becomes a concern)
        try:
            _validate_invariants_node(self._node)
        except InvariantViolation:
            # As a fallback, try to rebuild from pairs and re-validate.
            rebuilt = _build_leveled_from_pairs(_enumerate_pairs_from_node(self._node))
            _validate_invariants_node(rebuilt)
            self._node = rebuilt

    # ---- GSS interface ----

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        # Stacks are stored reversed to make push/pop efficient.
        reversed_stacks = [(s[::-1], acc) for s, acc in stacks]
        node = _build_leveled_from_pairs(reversed_stacks)
        return cls(node)

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        if self.is_empty():
            return self
        # With reversed stacks, push adds a new root.
        depth = _node_max_depth(self._node)
        new_node = _normalize_suck_up(_Internal(children={value: {depth: self._node}}))
        return LeveledGSS(new_node)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        return LeveledGSS(_pop_node(self._node))

    def is_empty(self) -> bool:
        return isinstance(self._node, _Empty)

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        # This is inefficient but correct. A fully structural implementation is complex.
        ref = self.to_reference_impl()
        new_ref = ref.isolate(value)
        return LeveledGSS.from_stacks(new_ref._stacks)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        return LeveledGSS(_apply_node(self._node, func, memo={}))

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        return LeveledGSS(_prune_node(self._node, predicate, memo={}))

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        if not isinstance(other, LeveledGSS):
            # Fallback for unknown GSS types
            other = LeveledGSS.from_stacks(other.to_reference_impl()._stacks)

        if self is other:
            return self
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        return LeveledGSS(_merge_nodes(self._node, other._node))

    def peek(self) -> Set[T]:
        return _peek_node(self._node)

    def reduce_acc(self) -> Optional[Acc]:
        return _reduce_acc_node(self._node)

    def to_reference_impl(self) -> 'ReferenceGSS[T, Acc]':
        # Enumerate reversed stacks and reverse them back.
        reversed_pairs = _enumerate_pairs_from_node(self._node)
        pairs = [(s[::-1], acc) for s, acc in reversed_pairs]
        return ReferenceGSS.from_stacks(pairs)

    # Also expose a human-friendly validator
    def validate_invariants(self) -> None:
        _validate_invariants_node(self._node)

    # Optional: convenience for debugging
    def __repr__(self) -> str:
        return f"LeveledGSS({self.to_json_serializable()!r})"

    def __str__(self) -> str:
        return f"LeveledGSS({self.to_json_serializable()})"
