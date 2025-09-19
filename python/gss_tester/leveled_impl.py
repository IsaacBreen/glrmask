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
    is_terminal: bool = False


_InnerNode = Union[InnerLeaf, InnerBranch[T]]


@dataclass
class WithAcc(Generic[T, Acc]):
    node: _InnerNode[T]
    acc: Acc


@dataclass
class Branch(Generic[T, Acc]):
    # children: T -> depth -> _LeveledNode
    children: Dict[T, Dict[int, '_LeveledNode[T, Acc]']]
    terminals: Optional['_LeveledNode[T, Acc]'] = None


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

    return InnerBranch(children=children, is_terminal=(empty_count > 0))


def _build_leveled_from_pairs(pairs: List[Tuple[List[T], Acc]]) -> _LeveledNode[T, Acc]:
    pairs = _dedup_pairs(pairs)

    if not pairs:
        return Empty()

    # Check if all stacks share the same accumulator; then we can store them under a single WithAcc node.
    accs = {acc for _, acc in pairs}
    if len(accs) == 1:
        vals_list = [vals for vals, _ in pairs]
        only_acc = next(iter(accs))
        inner = _build_inner_from_sequences(vals_list)
        return WithAcc(node=inner, acc=only_acc)
        # Note: mixed empty/non-empty stacks are also handled here because the inner builder can represent them.

    # Otherwise, build an Internal node, partitioning by first symbol.
    # Empty stacks ([]) must still be representable: we attach them under the _EPS sentinel.
    children: Dict[T, Dict[int, '_LeveledNode[T, Acc]']] = {}

    # Handle empty stacks (if any)
    empty_pairs = [(vals, acc) for vals, acc in pairs if not vals]
    terminals_node = _build_leveled_from_pairs(empty_pairs) if empty_pairs else None

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

    if not children:
        return terminals_node if terminals_node is not None else Empty()

    node: _LeveledNode[T, Acc] = Branch(children=children, terminals=terminals_node)
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
        new_children: Dict[T, Dict[int, _LeveledNode[T, Acc]]] = {}
        for key_t, depth_map in node.children.items():
            for depth, child in depth_map.items():
                norm = _normalize_suck_up(child)
                new_children.setdefault(key_t, {})[depth] = norm
        new_terminals = _normalize_suck_up(node.terminals) if node.terminals else None

        # Check suck-up condition: if all children are WithAcc and share the same acc
        # If there are no children, it's empty
        if not new_children:
            return new_terminals if new_terminals else Empty()

        # Flatten list of children
        child_nodes_from_children: List[_LeveledNode[T, Acc]] = []
        for kt, dm in new_children.items():
            child_nodes_from_children.extend(dm.values())

        all_child_nodes = child_nodes_from_children + ([new_terminals] if new_terminals else [])

        all_with_acc = all(isinstance(ch, WithAcc) for ch in all_child_nodes)
        if all_with_acc:
            accs: Set[Acc] = set(ch.acc for ch in all_child_nodes if isinstance(ch, WithAcc))
            if len(accs) == 1:
                # We can "suck up": create a single WithAcc whose inner combines the A-level of all children.
                the_acc = next(iter(accs))
                inner_children: Dict[T, Dict[int, _InnerNode[T]]] = {}
                has_empty = False

                # From children with T keys
                for kt, depth_map in new_children.items():
                    for d, ch in depth_map.items():
                        chw = ch # is WithAcc
                        assert isinstance(chw, WithAcc)
                        if isinstance(chw.node, InnerLeaf):
                            depth_int = max(d - 1, 0)
                            inner_children.setdefault(kt, {})[depth_int] = InnerLeaf()
                        else: # InnerBranch
                            inner_children.setdefault(kt, {})[max(d - 1, 0)] = chw.node

                # From terminals node
                if new_terminals:
                    term_w = new_terminals
                    assert isinstance(term_w, WithAcc)
                    if isinstance(term_w.node, InnerLeaf):
                        has_empty = True
                    else: # InnerBranch
                        # Merge its children
                        for t, dm in term_w.node.children.items():
                            inner_children.setdefault(t, {}).update(dm)
                        has_empty = term_w.node.is_terminal

                # If there were no non-EPS children but has_empty is True and there are no other children,
                # then the inner is just Root.
                if not inner_children:
                    if has_empty:
                        inner: _InnerNode[T] = InnerLeaf()
                    else:
                        inner = InnerLeaf()
                else:
                    inner = InnerBranch(children=inner_children, is_terminal=has_empty)

                return WithAcc(node=inner, acc=the_acc)

        return Branch(children=new_children, terminals=new_terminals)

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
        if inner.is_terminal:
            result.append((list(prefix), acc))
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
        if node_b.terminals:
            walk(node_b.terminals, prefix)
        for key_t, depth_map in node_b.children.items():
            for _, child in depth_map.items():
                walk(child, prefix + [key_t])

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
            all_child_nodes: List[_LeveledNode[T, Acc]] = []
            for kt, depth_map in node_b.children.items():
                all_child_nodes.extend(depth_map.values())
            if node_b.terminals:
                all_child_nodes.append(node_b.terminals)

            for ch in all_child_nodes:
                ok, acc = check(ch)
                if not ok:
                    return False, None

            # suck-up opportunity detection
            if all_child_nodes and all(isinstance(n, WithAcc) for n in all_child_nodes):
                child_accs = [
                    ch.acc
                    for ch in all_child_nodes
                    if isinstance(ch, WithAcc)
                ]
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

