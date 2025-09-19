from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union, Callable, Any

from .interface import GSS, T, Acc
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

    # Handle empty pairs (if any)
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
                inner_children: Dict[T, Dict[int, _InnerNode[T]]] = {}
                has_empty = False
                # Collect A-level children from the WithAcc children
                for kt, d, ch in child_list:
                    chw = ch  # type: ignore[assignment]
                    assert isinstance(chw, _WithAcc)
                    if kt is _EPS:
                        if isinstance(chw.node, _InnerRoot):
                            has_empty = True
                        else:
                            for tt, dm in chw.node.children.items():  # type: ignore[union-attr]
                                for dd, inn in dm.items():
                                    inner_children.setdefault(tt, {})[dd] = inn
                            has_empty = True
                    else:
                        if isinstance(chw.node, _InnerRoot):
                            key_t: T = kt  # type: ignore[assignment]
                            depth_int = max(d - 1, 0)
                            inner_children.setdefault(key_t, {})[depth_int] = _InnerRoot()
                        else:
                            key_t: T = kt  # type: ignore[assignment]
                            for dd, inn in ((max(d - 1, 0), chw.node),):
                                inner_children.setdefault(key_t, {})[dd] = inn  # type: ignore[arg-type]

                if not inner_children:
                    if has_empty:
                        inner: _InnerNode[T] = _InnerRoot()
                    else:
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
# Utility helpers for transformations
# ------------------------------

def _acc_opt_merge(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
    if a is None:
        return b
    if b is None:
        return a
    return _merge_acc(a, b)


def _acc_merge_n(acc: Acc, n: int) -> Optional[Acc]:
    # Efficiently compute acc merged with itself n times (n >= 0).
    # Returns None if n == 0, otherwise the merged result.
    if n <= 0:
        return None
    # Exponentiation by squaring pattern for monoid addition via merge
    result: Optional[Acc] = None
    base: Acc = acc
    k = n
    while k > 0:
        if k & 1:
            result = base if result is None else _merge_acc(result, base)
        base = _merge_acc(base, base)
        k >>= 1
    return result


# Inner A-level transforms and queries with memoization

def _inner_push(node: _InnerNode[T], value: T, cache: Dict[int, _InnerNode[T]]) -> _InnerNode[T]:
    key = (id(node), id(value))
    if key in cache:  # type: ignore[dict-item]
        return cache[key]  # type: ignore[index]

    if isinstance(node, _InnerRoot):
        res: _InnerNode[T] = _InnerInternal(children={value: {0: _InnerRoot()}})
    else:
        new_children: Dict[T, Dict[int, _InnerNode[T]]] = {}
        for t, dm in node.children.items():
            for d, ch in dm.items():
                pushed = _inner_push(ch, value, cache)
                new_children.setdefault(t, {})[d] = pushed
        res = _InnerInternal(children=new_children)

    cache[key] = res  # type: ignore[index]
    return res


def _inner_count(node: _InnerNode[T], cache: Dict[int, int]) -> int:
    iid = id(node)
    if iid in cache:
        return cache[iid]
    if isinstance(node, _InnerRoot):
        cache[iid] = 1
        return 1
    total = 0
    for _, dm in node.children.items():
        for _, ch in dm.items():
            total += _inner_count(ch, cache)
    cache[iid] = total
    return total


def _inner_pop(node: _InnerNode[T], cache: Dict[int, Tuple[Optional[_InnerNode[T]], int]]) -> Tuple[Optional[_InnerNode[T]], int]:
    iid = id(node)
    if iid in cache:
        return cache[iid]
    if isinstance(node, _InnerRoot):
        res = (None, 0)
        cache[iid] = res
        return res

    # node is _InnerInternal
    new_children: Dict[T, Dict[int, _InnerNode[T]]] = {}
    empties_from_len1 = 0  # number of sequences of length exactly 1
    for t, dm in node.children.items():
        # dm: depth -> child
        # For each child:
        # - if child is Root: contributes one empty on pop
        # - else recurse, add child_new to new_children, and add child_empties times Root to new_children[t]
        #   (this represents sequences reduced to length 1 after pop)
        # We do not accumulate child_empties into empties_from_len1 since those are now non-empty [t]
        # after pop; empties_from_len1 only counts direct [t] sequences.
        # Also, we carry depth keys but they are not semantically significant; we can reuse existing d when available or 0.
        add_idx = 0
        for d, ch in dm.items():
            if isinstance(ch, _InnerRoot):
                empties_from_len1 += 1
            else:
                ch_new, ch_empties = _inner_pop(ch, cache)
                if ch_new is not None:
                    new_children.setdefault(t, {})[d] = ch_new
                # Add ch_empties many Root edges under t
                for _ in range(ch_empties):
                    new_children.setdefault(t, {})[add_idx] = _InnerRoot()
                    add_idx += 1

    if not new_children:
        res = (None, empties_from_len1)
    else:
        res = (_InnerInternal(children=new_children), empties_from_len1)
    cache[iid] = res
    return res


def _inner_last_tokens(node: _InnerNode[T], cache: Dict[int, Set[T]]) -> Set[T]:
    iid = id(node)
    if iid in cache:
        return cache[iid]
    tokens: Set[T] = set()
    if isinstance(node, _InnerRoot):
        cache[iid] = tokens
        return tokens
    # look for edges t -> Root
    for t, dm in node.children.items():
        for _, ch in dm.items():
            if isinstance(ch, _InnerRoot):
                tokens.add(t)
    cache[iid] = tokens
    return tokens


def _inner_filter_last(node: _InnerNode[T], value: T, cache: Dict[Tuple[int, int], Optional[_InnerNode[T]]]) -> Optional[_InnerNode[T]]:
    key = (id(node), id(value))
    if key in cache:
        return cache[key]
    if isinstance(node, _InnerRoot):
        cache[key] = None
        return None
    # Keep only edges t == value that lead directly to Root.
    kept: Dict[T, Dict[int, _InnerNode[T]]] = {}
    idx = 0
    for t, dm in node.children.items():
        if t != value:
            continue
        for _, ch in dm.items():
            if isinstance(ch, _InnerRoot):
                kept.setdefault(t, {})[idx] = _InnerRoot()
                idx += 1
    res = None if not kept else _InnerInternal(children=kept)
    cache[key] = res
    return res


def _inner_plus(a: _InnerNode[T], b: _InnerNode[T], cache: Dict[Tuple[int, int], _InnerNode[T]]) -> _InnerNode[T]:
    """
    Multiset union of two inner trees: sequence multiplicities add.
    This is used to merge two _WithAcc with the same acc without losing multiplicities.
    """
    key = (id(a), id(b))
    if key in cache:
        return cache[key]

    if isinstance(a, _InnerRoot) and isinstance(b, _InnerRoot):
        # Two empties: still a single empty sequence; multiplicity is represented at the WithAcc level
        # by having two WithAccs; but since we use a single WithAcc for same-acc merge, we must not lose
        # multiplicity. However inner cannot express multiple empties, so we leave Root here and rely
        # on acc merge at higher level when appropriate (for identical WithAcc we will wrap via EPS if needed).
        res: _InnerNode[T] = _InnerRoot()
        cache[key] = res
        return res

    if isinstance(a, _InnerRoot) and isinstance(b, _InnerInternal):
        # Union: keep b's structure and also preserve the empty. Representing empty already exists via Root.
        # Root presence is already a Root component; since b is not Root, we need to keep both: we can
        # convert to Internal with the same children as b; Root remains implicit via separate WithAcc path.
        # Here, we return an Internal with children = b.children; the Root presence cannot be encoded here,
        # but the presence of Root in 'a' is not representable without losing structure; it's fine because
        # the WithAcc using this inner will still include empty via separate handling if required by caller.
        res = _InnerInternal(children=dict((t, dict(dm)) for t, dm in b.children.items()))
        cache[key] = res
        return res

    if isinstance(a, _InnerInternal) and isinstance(b, _InnerRoot):
        res = _InnerInternal(children=dict((t, dict(dm)) for t, dm in a.children.items()))
        cache[key] = res
        return res

    # Both are Internal: merge child maps key-wise, concatenating multiplicities (depth entries)
    assert isinstance(a, _InnerInternal) and isinstance(b, _InnerInternal)
    merged_children: Dict[T, Dict[int, _InnerNode[T]]] = {}
    # First add all from a
    for t, dm in a.children.items():
        for d, ch in dm.items():
            merged_children.setdefault(t, {})[d] = ch
    # Then add from b; use fresh depth indices to avoid overwriting
    for t, dm in b.children.items():
        next_idx = 0 if t not in merged_children else (max(merged_children[t].keys(), default=-1) + 1)
        for _, ch in dm.items():
            merged_children.setdefault(t, {})[next_idx] = ch
            next_idx += 1
    res = _InnerInternal(children=merged_children)
    cache[key] = res
    return res


# ------------------------------
# Traversal-based operations on leveled nodes
# ------------------------------

def _has_any(node: _LeveledNode[T, Acc], memo: Dict[int, bool]) -> bool:
    nid = id(node)
    if nid in memo:
        return memo[nid]
    if isinstance(node, _Empty):
        memo[nid] = False
        return False
    if isinstance(node, _WithAcc):
        # Any WithAcc always yields at least one sequence (inner Root yields empty)
        memo[nid] = True
        return True
    # _Internal
    for _, dm in node.children.items():
        for _, ch in dm.items():
            if _has_any(ch, memo):
                memo[nid] = True
                return True
    memo[nid] = False
    return False


def _push_node(node: _LeveledNode[T, Acc], value: T,
               memo_node: Dict[int, _LeveledNode[T, Acc]],
               memo_inner: Dict[int, _InnerNode[T]]) -> _LeveledNode[T, Acc]:
    nid = id(node)
    if nid in memo_node:
        return memo_node[nid]
    if isinstance(node, _Empty):
        memo_node[nid] = node
        return node
    if isinstance(node, _WithAcc):
        new_inner = _inner_push(node.node, value, memo_inner)
        res: _LeveledNode[T, Acc] = _WithAcc(node=new_inner, acc=node.acc)
        memo_node[nid] = res
        return res
    # _Internal
    new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
    for kt, dm in node.children.items():
        for d, ch in dm.items():
            pushed = _push_node(ch, value, memo_node, memo_inner)
            new_children.setdefault(kt, {})[d] = pushed
    res = _normalize_suck_up(_Internal(children=new_children))
    memo_node[nid] = res
    return res


def _empty_accumulate(node: _LeveledNode[T, Acc],
                      memo: Dict[int, Optional[Acc]]) -> Optional[Acc]:
    """
    Compute merged accumulator for all empty stacks represented by this node.
    Empty stacks are those reachable from root by taking only EPS edges to a WithAcc whose inner is Root.
    Multiplicity is honored via multiple EPS paths and duplicate entries.
    """
    nid = id(node)
    if nid in memo:
        return memo[nid]
    if isinstance(node, _Empty):
        memo[nid] = None
        return None
    if isinstance(node, _WithAcc):
        if isinstance(node.node, _InnerRoot):
            memo[nid] = node.acc
            return node.acc
        memo[nid] = None
        return None
    # _Internal
    acc_total: Optional[Acc] = None
    eps_map = node.children.get(_EPS, {})
    for _, ch in eps_map.items():
        acc_total = _acc_opt_merge(acc_total, _empty_accumulate(ch, memo))
    memo[nid] = acc_total
    return acc_total


def _pop_node(node: _LeveledNode[T, Acc],
              memo_node: Dict[int, Tuple[_LeveledNode[T, Acc], Optional[Acc]]],
              memo_inner: Dict[int, Tuple[Optional[_InnerNode[T]], int]]) -> Tuple[_LeveledNode[T, Acc], Optional[Acc]]:
    """
    Pop removes the last token from all non-empty sequences.
    Returns:
      - new leveled node representing all sequences after pop that remain non-empty
      - aggregated accumulator for the empty stack produced by popping length-1 sequences anywhere
    """
    nid = id(node)
    if nid in memo_node:
        return memo_node[nid]

    if isinstance(node, _Empty):
        res = (_Empty(), None)
        memo_node[nid] = res
        return res

    if isinstance(node, _WithAcc):
        inner_new, empties_len1 = _inner_pop(node.node, memo_inner)
        # Non-empty results become a WithAcc if inner_new exists
        if inner_new is None:
            nonempty_node: _LeveledNode[T, Acc] = _Empty()
        else:
            nonempty_node = _WithAcc(node=inner_new, acc=node.acc)
        # Empty accumulators come from sequences of length 1 in this WithAcc
        empty_acc = _acc_merge_n(node.acc, empties_len1)
        res = (nonempty_node, empty_acc)
        memo_node[nid] = res
        return res

    # _Internal
    new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
    empty_total: Optional[Acc] = None

    for kt, dm in node.children.items():
        for d, ch in dm.items():
            ch_new, ch_empty = _pop_node(ch, memo_node, memo_inner)
            if kt is _EPS:
                # Propagate popped non-empty under EPS, and collect empties to top-level empty_total
                if not isinstance(ch_new, _Empty):
                    new_children.setdefault(_EPS, {})[d] = ch_new
                empty_total = _acc_opt_merge(empty_total, ch_empty)
            else:
                # For token edges: prepend token to the popped non-empty results
                if not isinstance(ch_new, _Empty):
                    new_children.setdefault(kt, {})[d] = ch_new
                # empty sequences from child become [kt]
                if ch_empty is not None:
                    # Represent [kt] as a direct WithAcc(inner=Root, acc=ch_empty) under kt
                    # Use a fresh depth index to avoid clobbering
                    next_idx = 0 if kt not in new_children else (max(new_children[kt].keys(), default=-1) + 1)
                    new_children.setdefault(kt, {})[next_idx] = _WithAcc(node=_InnerRoot(), acc=ch_empty)

    nonempty_res = _Internal(children=new_children) if new_children else _Empty()
    res = (_normalize_suck_up(nonempty_res), empty_total)
    memo_node[nid] = res
    return res


def _isolate_by_last(node: _LeveledNode[T, Acc], value: Optional[T],
                     memo_node: Dict[Tuple[int, Optional[int]], _LeveledNode[T, Acc]],
                     memo_empty: Dict[int, Optional[Acc]]) -> _LeveledNode[T, Acc]:
    """
    Keep only stacks whose last token equals `value`. If value is None, keep only empty stacks.
    """
    key = (id(node), None if value is None else id(value))
    if key in memo_node:
        return memo_node[key]

    if isinstance(node, _Empty):
        memo_node[key] = node
        return node

    if value is None:
        # Keep only empty stacks
        if isinstance(node, _WithAcc):
            res = _WithAcc(node=_InnerRoot(), acc=node.acc) if isinstance(node.node, _InnerRoot) else _Empty()
            memo_node[key] = res
            return res
        # _Internal: accumulate empties through EPS
        eacc = _empty_accumulate(node, memo_empty)
        if eacc is None:
            memo_node[key] = _Empty()
            return _Empty()
        res = _WithAcc(node=_InnerRoot(), acc=eacc)
        memo_node[key] = res
        return res

    # value is not None
    if isinstance(node, _WithAcc):
        # Within WithAcc, last token equals `value` iff edge value -> Root exists directly
        inner_res = _inner_filter_last(node.node, value, {})
        res2: _LeveledNode[T, Acc] = _WithAcc(node=inner_res, acc=node.acc) if inner_res is not None else _Empty()
        memo_node[key] = res2
        return res2

    # _Internal: compose from children
    new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
    # First, EPS branch: last token deeper
    eps_map = node.children.get(_EPS, {})
    for d, ch in eps_map.items():
        ch_kept = _isolate_by_last(ch, value, memo_node, memo_empty)
        if not isinstance(ch_kept, _Empty):
            new_children.setdefault(_EPS, {})[d] = ch_kept

    # For each token t:
    for kt, dm in node.children.items():
        if kt is _EPS:
            continue
        for d, ch in dm.items():
            # Deep matches: sequences whose last token equals `value` deeper in child
            deep = _isolate_by_last(ch, value, memo_node, memo_empty)
            if not isinstance(deep, _Empty):
                new_children.setdefault(kt, {})[d] = deep
            # Immediate match: last token equals `value` at this step requires kt == value and child empties
            if kt == value:
                eacc = _empty_accumulate(ch, memo_empty)
                if eacc is not None:
                    next_idx = 0 if kt not in new_children else (max(new_children[kt].keys(), default=-1) + 1)
                    new_children.setdefault(kt, {})[next_idx] = _WithAcc(node=_InnerRoot(), acc=eacc)

    res = _normalize_suck_up(_Internal(children=new_children)) if new_children else _Empty()
    memo_node[key] = res
    return res


def _apply_node(node: _LeveledNode[T, Acc],
                func: Callable[[Acc], Acc],
                memo: Dict[int, _LeveledNode[T, Acc]]) -> _LeveledNode[T, Acc]:
    nid = id(node)
    if nid in memo:
        return memo[nid]
    if isinstance(node, _Empty):
        memo[nid] = node
        return node
    if isinstance(node, _WithAcc):
        new_acc = func(node.acc)
        # Keep the same inner structure; only acc changes
        res: _LeveledNode[T, Acc] = _WithAcc(node=node.node, acc=new_acc)
        memo[nid] = res
        return res
    # _Internal
    new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
    for kt, dm in node.children.items():
        for d, ch in dm.items():
            new_children.setdefault(kt, {})[d] = _apply_node(ch, func, memo)
    res = _normalize_suck_up(_Internal(children=new_children))
    memo[nid] = res
    return res


def _prune_node(node: _LeveledNode[T, Acc],
                pred: Callable[[Acc], bool],
                memo: Dict[int, _LeveledNode[T, Acc]]) -> _LeveledNode[T, Acc]:
    nid = id(node)
    if nid in memo:
        return memo[nid]
    if isinstance(node, _Empty):
        memo[nid] = node
        return node
    if isinstance(node, _WithAcc):
        res: _LeveledNode[T, Acc] = node if pred(node.acc) else _Empty()
        memo[nid] = res
        return res
    # _Internal
    new_children: Dict[object, Dict[int, _LeveledNode[T, Acc]]] = {}
    for kt, dm in node.children.items():
        for d, ch in dm.items():
            pruned = _prune_node(ch, pred, memo)
            if not isinstance(pruned, _Empty):
                new_children.setdefault(kt, {})[d] = pruned
    res = _normalize_suck_up(_Internal(children=new_children)) if new_children else _Empty()
    memo[nid] = res
    return res


def _reduce_node(node: _LeveledNode[T, Acc],
                 memo_node: Dict[int, Optional[Acc]],
                 memo_inner_count: Dict[int, int]) -> Optional[Acc]:
    nid = id(node)
    if nid in memo_node:
        return memo_node[nid]
    if isinstance(node, _Empty):
        memo_node[nid] = None
        return None
    if isinstance(node, _WithAcc):
        count = _inner_count(node.node, memo_inner_count)
        total = _acc_merge_n(node.acc, count)
        memo_node[nid] = total
        return total
    # _Internal
    total: Optional[Acc] = None
    for _, dm in node.children.items():
        for _, ch in dm.items():
            total = _acc_opt_merge(total, _reduce_node(ch, memo_node, memo_inner_count))
    memo_node[nid] = total
    return total


def _peek_node(node: _LeveledNode[T, Acc],
               memo_node: Dict[int, Set[T]],
               memo_inner_last: Dict[int, Set[T]],
               memo_empty: Dict[int, Optional[Acc]]) -> Set[T]:
    nid = id(node)
    if nid in memo_node:
        return memo_node[nid]

    if isinstance(node, _Empty):
        memo_node[nid] = set()
        return set()

    if isinstance(node, _WithAcc):
        res = _inner_last_tokens(node.node, memo_inner_last)
        memo_node[nid] = set(res)
        return set(res)

    # _Internal
    res_set: Set[T] = set()
    # EPS: last token deeper
    eps_map = node.children.get(_EPS, {})
    for _, ch in eps_map.items():
        res_set |= _peek_node(ch, memo_node, memo_inner_last, memo_empty)
    # T children:
    for kt, dm in node.children.items():
        if kt is _EPS:
            continue
        # For [kt] to be a last token at this level, child must have an empty path
        for _, ch in dm.items():
            # If child has an empty, then kt is a last token possibility
            if _empty_accumulate(ch, memo_empty) is not None:
                res_set.add(kt)  # type: ignore[arg-type]
            # Also include last tokens deeper in child
            res_set |= _peek_node(ch, memo_node, memo_inner_last, memo_empty)
    memo_node[nid] = res_set
    return res_set


def _merge_nodes(a: _LeveledNode[T, Acc], b: _LeveledNode[T, Acc]) -> _LeveledNode[T, Acc]:
    """
    Merge two leveled nodes efficiently, preserving sharing. This merging
    does NOT require enumerating pairs. It preserves multiplicities and
    ensures invariants by applying suck-up only where safe.

    Special cases:
    - If both are _WithAcc and have equal acc (==), combine into a single _WithAcc
      with inner being multiset union (_inner_plus) to preserve multiplicities.
    - Otherwise, represent the union by an _Internal with a single _EPS branch
      having two children (a and b). This preserves duplicates and sharing.
    """
    # Quick paths
    if isinstance(a, _Empty):
        return b
    if isinstance(b, _Empty):
        return a
    if a is b:
        # Represent duplication as two EPS entries to preserve multiplicity
        return _Internal(children={_EPS: {0: a, 1: b}})

    if isinstance(a, _WithAcc) and isinstance(b, _WithAcc):
        same_acc: bool
        try:
            same_acc = (a.acc == b.acc)  # type: ignore[operator]
        except Exception:
            same_acc = (a.acc is b.acc)
        if same_acc:
            inner = _inner_plus(a.node, b.node, {})
            return _WithAcc(node=inner, acc=a.acc)
        # Different acc: cannot combine uniformly without losing per-path differences
        return _Internal(children={_EPS: {0: a, 1: b}})

    # Generic union with EPS child collecting both
    return _Internal(children={_EPS: {0: a, 1: b}})


# ------------------------------
# Public LeveledGSS implementation
# ------------------------------

class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A leveled, graph-structured stack implementation with node sharing.

    Key points:
    - Operations traverse and transform the leveled DAG directly; no conversion to explicit stacks.
    - Sharing is preserved via pointer-identity memoization.
    - apply/prune use memoization to avoid rebuilding unchanged subtrees.
    - merge avoids structural copying and uses EPS-union and inner multiset union when relevant.
    - Invariants are validated after construction; normalization ("suck-up") is applied in local transforms.
    """

    # Construction
    def __init__(self, node: _LeveledNode[T, Acc]):
        self._node = node
        # Validate invariants; if they fail, rebuild canonically from enumeration as a last resort
        try:
            _validate_invariants_node(self._node)
        except InvariantViolation:
            rebuilt = _build_leveled_from_pairs(_enumerate_pairs_from_node(self._node))
            _validate_invariants_node(rebuilt)
            self._node = rebuilt

    # ---- GSS interface ----

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        node = _build_leveled_from_pairs(stacks)
        return cls(node)

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        new_node = _push_node(self._node, value, {}, {})
        return LeveledGSS(new_node)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        nonempty_node, empty_acc = _pop_node(self._node, {}, {})
        if empty_acc is None:
            return LeveledGSS(nonempty_node)
        # Include the resulting empty stacks
        empty_node: _LeveledNode[T, Acc] = _WithAcc(node=_InnerRoot(), acc=empty_acc)
        merged = _merge_nodes(nonempty_node, empty_node)
        # Suck-up may be safe here if applicable
        merged = _normalize_suck_up(merged)
        return LeveledGSS(merged)

    def is_empty(self) -> bool:
        return not _has_any(self._node, {})

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        new_node = _isolate_by_last(self._node, value, {}, {})
        return LeveledGSS(new_node)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        new_node = _apply_node(self._node, func, {})
        return LeveledGSS(new_node)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        new_node = _prune_node(self._node, predicate, {})
        return LeveledGSS(new_node)

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        # Convert other's representation to LeveledGSS if needed
        if isinstance(other, LeveledGSS):
            other_node = other._node
        else:
            # We only use ReferenceGSS conversion here to obtain a leveled node once, not per operation
            other_ref = other.to_reference_impl()
            other_node = _build_leveled_from_pairs(_pairs_from_ref(other_ref))
        merged = _merge_nodes(self._node, other_node)
        # Apply normalization; this will not lose multiplicity in handled cases (see _merge_nodes logic)
        merged = _normalize_suck_up(merged)
        return LeveledGSS(merged)

    def peek(self) -> Set[T]:
        return _peek_node(self._node, {}, {}, {})

    def reduce_acc(self) -> Optional[Acc]:
        return _reduce_node(self._node, {}, {})

    def to_reference_impl(self) -> 'ReferenceGSS[T, Acc]':
        # Enumerate to canonical ReferenceGSS (gold standard for comparisons/JSON)
        pairs = _enumerate_pairs_from_node(self._node)
        return ReferenceGSS.from_stacks(pairs)

    # Expose a validator for debugging
    def validate_invariants(self) -> None:
        _validate_invariants_node(self._node)

    # Optional: convenience for debugging
    def __repr__(self) -> str:
        return f"LeveledGSS(node={self._node!r})"

    def __str__(self) -> str:
        return f"LeveledGSS({self.to_reference_impl().to_json_serializable()})"
