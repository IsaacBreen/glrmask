from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


# ------------------------------
# Efficient structural transforms and analyses
# ------------------------------

def _append_token_inner(inner: _InnerNode[T], token: T, memo: Dict[int, _InnerNode[T]]) -> _InnerNode[T]:
    """
    Return a new Inner node representing all sequences from `inner`, with `token` appended to the end.
    Preserves sharing via memoization, and does not mutate the original node.
    """
    key = id(inner)
    if key in memo:
        return memo[key]

    if isinstance(inner, _InnerRoot):
        # Empty sequence becomes [token]
        res: _InnerNode[T] = _InnerInternal(children={token: {0: _InnerRoot()}})
        memo[key] = res
        return res

    # _InnerInternal
    new_children: Dict[T, Dict[int, _InnerNode[T]]] = {}
    for t, depth_map in inner.children.items():  # type: ignore[union-attr]
        for d, ch in depth_map.items():
            ch2 = _append_token_inner(ch, token, memo)
            # Depth increases by 1 since we've extended each sequence by one symbol
            new_depth = d + 1
            new_children.setdefault(t, {})[new_depth] = ch2
    res = _InnerInternal(children=new_children)
    memo[key] = res
    return res


def _append_token_node(node: _LeveledNode[T, Acc], token: T,
                       memo_node: Dict[int, _LeveledNode[T, Acc]],
                       memo_inner: Dict[int, _InnerNode[T]]) -> _LeveledNode[T, Acc]:
    """
    Append `token` to the end of every sequence represented by `node`.
    """
    k = id(node)
    if k in memo_node:
        return memo_node[k]

    if isinstance(node, _Empty):
        memo_node[k] = node
        return node
    if isinstance(node, _WithAcc):
        new_inner = _append_token_inner(node.node, token, memo_inner)
        res: _LeveledNode[T, Acc] = _WithAcc(node=new_inner, acc=node.acc)
        memo_node[k] = res
        return res
    # _Internal
    changed = False
    new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
    for key_t, depth_map in node.children.items():
        out_dm: Dict[int, _LeveledNode[T, Acc]] = {}
        for d, ch in depth_map.items():
            ch2 = _append_token_node(ch, token, memo_node, memo_inner)
            out_dm[d] = ch2
            if ch2 is not ch:
                changed = True
        new_children[key_t] = out_dm
    if not changed:
        memo_node[k] = node
        return node
    res_internal: _LeveledNode[T, Acc] = _Internal(children=new_children)
    res = _normalize_suck_up(res_internal)
    memo_node[k] = res
    return res


def _apply_node(node: _LeveledNode[T, Acc], func: Callable[[Acc], Acc],
                memo: Dict[int, _LeveledNode[T, Acc]]) -> _LeveledNode[T, Acc]:
    k = id(node)
    if k in memo:
        return memo[k]
    if isinstance(node, _Empty):
        memo[k] = node
        return node
    if isinstance(node, _WithAcc):
        new_acc = func(node.acc)
        # Reuse node if acc unchanged (by ==) to preserve sharing
        if new_acc == node.acc:
            memo[k] = node
            return node
        res: _LeveledNode[T, Acc] = _WithAcc(node=node.node, acc=new_acc)
        memo[k] = res
        return res
    # _Internal
    changed = False
    new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
    for key_t, dm in node.children.items():
        out_dm: Dict[int, _LeveledNode[T, Acc]] = {}
        for d, ch in dm.items():
            ch2 = _apply_node(ch, func, memo)
            out_dm[d] = ch2
            if ch2 is not ch:
                changed = True
        new_children[key_t] = out_dm
    if not changed:
        memo[k] = node
        return node
    res_internal: _LeveledNode[T, Acc] = _Internal(children=new_children)
    res = _normalize_suck_up(res_internal)
    memo[k] = res
    return res


def _prune_node(node: _LeveledNode[T, Acc], predicate: Callable[[Acc], bool],
                memo: Dict[int, _LeveledNode[T, Acc]]) -> _LeveledNode[T, Acc]:
    k = id(node)
    if k in memo:
        return memo[k]
    if isinstance(node, _Empty):
        memo[k] = node
        return node
    if isinstance(node, _WithAcc):
        if predicate(node.acc):
            memo[k] = node
            return node
        else:
            res: _LeveledNode[T, Acc] = _Empty()
            memo[k] = res
            return res
    # _Internal
    new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
    kept_any = False
    unchanged = True
    for key_t, dm in node.children.items():
        out_dm: Dict[int, _LeveledNode[T, Acc]] = {}
        for d, ch in dm.items():
            ch2 = _prune_node(ch, predicate, memo)
            if not isinstance(ch2, _Empty):
                out_dm[d] = ch2
                kept_any = True
            if ch2 is not ch:
                unchanged = False
        if out_dm:
            new_children[key_t] = out_dm
    if not kept_any:
        res2: _LeveledNode[T, Acc] = _Empty()
        memo[k] = res2
        return res2
    if unchanged:
        memo[k] = node
        return node
    res_internal: _LeveledNode[T, Acc] = _Internal(children=new_children)
    res = _normalize_suck_up(res_internal)
    memo[k] = res
    return res


def _inner_peek(inner: _InnerNode[T], memo: Dict[int, Tuple[Set[T], bool]]) -> Tuple[Set[T], bool]:
    """
    Return (tops, has_empty) for an inner node:
    - tops: set of last tokens across all sequences represented by the inner.
    - has_empty: True iff the empty sequence [] is present among the inner sequences.
    """
    k = id(inner)
    if k in memo:
        return memo[k]
    if isinstance(inner, _InnerRoot):
        res = (set(), True)
        memo[k] = res
        return res
    # _InnerInternal
    tops: Set[T] = set()
    has_empty = False
    for t, dm in inner.children.items():  # type: ignore[union-attr]
        for _, ch in dm.items():
            sub_tops, sub_empty = _inner_peek(ch, memo)
            tops |= sub_tops
            if sub_empty:
                tops.add(t)
        # Note: inner-level has no EPS; no need to handle it here.
    # No direct representation of "and empty" alongside edges at inner-level; Root handles that.
    res = (tops, has_empty)
    memo[k] = res
    return res


def _peek_node(node: _LeveledNode[T, Acc],
               memo_inner: Dict[int, Tuple[Set[T], bool]]) -> Tuple[Set[T], bool]:
    """
    Return (tops, has_empty) for a leveled node:
    - tops: set of last tokens across all sequences represented from this node.
    - has_empty: True iff the empty sequence [] is present from this node.
    """
    if isinstance(node, _Empty):
        return set(), False
    if isinstance(node, _WithAcc):
        return _inner_peek(node.node, memo_inner)
    # _Internal
    tops: Set[T] = set()
    has_empty = False
    for key_t, dm in node.children.items():
        for _, ch in dm.items():
            child_tops, child_empty = _peek_node(ch, memo_inner)
            tops |= child_tops
            if key_t is _EPS:
                # Empty step: propagate emptiness upward
                has_empty = has_empty or child_empty
            else:
                if child_empty:
                    tops.add(key_t)  # type: ignore[arg-type]
    return tops, has_empty


def _count_inner(inner: _InnerNode[T], memo: Dict[int, int]) -> int:
    """Count number of sequences represented by an inner node (Root counts as 1)."""
    k = id(inner)
    if k in memo:
        return memo[k]
    if isinstance(inner, _InnerRoot):
        memo[k] = 1
        return 1
    total = 0
    for _, dm in inner.children.items():  # type: ignore[union-attr]
        for _, ch in dm.items():
            total += _count_inner(ch, memo)
    memo[k] = total
    return total


def _repeat_merge(a: Acc, times: int) -> Optional[Acc]:
    """
    Merge `a` with itself `times` times, returning None if times == 0.
    Uses exponentiation by squaring for efficiency; assumes associativity of merge.
    """
    if times <= 0:
        return None
    result: Optional[Acc] = None
    base: Optional[Acc] = a
    n = times
    while n > 0:
        if n & 1:
            result = base if result is None else result.merge(base)  # type: ignore[union-attr]
        base = base.merge(base)  # type: ignore[union-attr]
        n >>= 1
    return result


def _reduce_node(node: _LeveledNode[T, Acc],
                 memo_inner_count: Dict[int, int],
                 memo_reduce: Dict[int, Optional[Acc]]) -> Optional[Acc]:
    k = id(node)
    if k in memo_reduce:
        return memo_reduce[k]
    if isinstance(node, _Empty):
        memo_reduce[k] = None
        return None
    if isinstance(node, _WithAcc):
        cnt = _count_inner(node.node, memo_inner_count)
        res = _repeat_merge(node.acc, cnt)
        memo_reduce[k] = res
        return res
    # _Internal
    acc_opt: Optional[Acc] = None
    for _, dm in node.children.items():
        for _, ch in dm.items():
            sub = _reduce_node(ch, memo_inner_count, memo_reduce)
            if sub is None:
                continue
            acc_opt = sub if acc_opt is None else acc_opt.merge(sub)
    memo_reduce[k] = acc_opt
    return acc_opt


# ------------------------------
# Invariant validation
# ------------------------------
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
    A leveled, graph-structured stack implementation optimized for sharing.

    Key properties:
    - Operations like apply, prune, push, peek, and reduce_acc are implemented via
      structure-preserving traversals with memoization to maximize node sharing.
    - merge short-circuits when the underlying node graph is identical (via `is`).
    - Invariants are validated; a normalization pass ("suck up") is applied where relevant.
    """

    # Construction
    def __init__(self, node: _LeveledNode[T, Acc]):
        self._node = node
        # Validate invariants (kept on to ensure structure health)
        try:
            _validate_invariants_node(self._node)
        except InvariantViolation:
            # As a fallback (paranoid), rebuild from canonical reference pairs.
            rebuilt = _build_leveled_from_pairs(_enumerate_pairs_from_node(self._node))
            _validate_invariants_node(rebuilt)
            self._node = rebuilt

    # ---- GSS interface ----

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        # Build via ReferenceGSS once to ensure canonical deduplication of identical stacks.
        ref = ReferenceGSS.from_stacks(stacks)
        node = _build_leveled_from_pairs(_pairs_from_ref(ref))
        return cls(node)

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        # Efficient traversal: append at inner-level and propagate
        memo_node: Dict[int, _LeveledNode[T, Acc]] = {}
        memo_inner: Dict[int, _InnerNode[T]] = {}
        transformed = _append_token_node(self._node, value, memo_node, memo_inner)
        # Ensure normalization where possible
        normalized = _normalize_suck_up(transformed)
        return LeveledGSS(normalized)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        # Correctness-first fallback: operate over explicit stacks then rebuild.
        pairs = _enumerate_pairs_from_node(self._node)
        new_pairs = [(vals[:-1], acc) for vals, acc in pairs if vals]
        new_node = _build_leveled_from_pairs(new_pairs)
        return LeveledGSS(new_node)

    def is_empty(self) -> bool:
        if isinstance(self._node, _Empty):
            return True
        # Cheap check: if node is WithAcc with inner Root? That still means at least one stack exists.
        # For Internal, presence of any child implies non-empty.
        # We'll do a quick structural check:
        if isinstance(self._node, _WithAcc):
            return False
        # _Internal
        return all(len(dm) == 0 for dm in self._node.children.values())

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        # Correctness-first fallback (filter by top value) due to complex last-token semantics with acc placement.
        pairs = _enumerate_pairs_from_node(self._node)
        if value is None:
            filtered = [(vals, acc) for vals, acc in pairs if not vals]
        else:
            filtered = [(vals, acc) for vals, acc in pairs if vals and vals[-1] == value]
        new_node = _build_leveled_from_pairs(filtered)
        return LeveledGSS(new_node)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        # Traversal-based, memoized
        memo: Dict[int, _LeveledNode[T, Acc]] = {}
        transformed = _apply_node(self._node, func, memo)
        normalized = _normalize_suck_up(transformed)
        return LeveledGSS(normalized)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        memo: Dict[int, _LeveledNode[T, Acc]] = {}
        transformed = _prune_node(self._node, predicate, memo)
        normalized = _normalize_suck_up(transformed)
        return LeveledGSS(normalized)

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        # Fast path: identical node graph -> return self
        if isinstance(other, LeveledGSS):
            if self._node is other._node:
                return self
            # Fall back to pair-based merge for correctness
            pairs_self = _enumerate_pairs_from_node(self._node)
            pairs_other = _enumerate_pairs_from_node(other._node)
            merged_ref = ReferenceGSS(pairs_self + pairs_other)  # canonicalizes and merges accs
            new_node = _build_leveled_from_pairs(_pairs_from_ref(merged_ref))
            return LeveledGSS(new_node)
        # Generic GSS: go through its reference impl
        other_ref = other.to_reference_impl()
        pairs_self = _enumerate_pairs_from_node(self._node)
        merged_ref = ReferenceGSS(pairs_self + _pairs_from_ref(other_ref))  # type: ignore[arg-type]
        new_node = _build_leveled_from_pairs(_pairs_from_ref(merged_ref))
        return LeveledGSS(new_node)

    def peek(self) -> Set[T]:
        # Traversal-based computation (no enumeration)
        tops, _ = _peek_node(self._node, {})
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        memo_inner_count: Dict[int, int] = {}
        memo_reduce: Dict[int, Optional[Acc]] = {}
        return _reduce_node(self._node, memo_inner_count, memo_reduce)

    def to_reference_impl(self) -> 'ReferenceGSS[T, Acc]':
        # Build canonical ReferenceGSS from the current leveled node
        return ReferenceGSS(_enumerate_pairs_from_node(self._node))  # type: ignore[arg-type]

    # Also expose a human-friendly validator
    def validate_invariants(self) -> None:
        _validate_invariants_node(self._node)

    # Optional: convenience for debugging
    def __repr__(self) -> str:
        return f"LeveledGSS(ref={self.to_reference_impl()!r})"

    def __str__(self) -> str:
        return f"LeveledGSS({self.to_reference_impl().to_json_serializable()})"

