from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Iterable, Callable, Any, Set, TypeVar, Generic
import json

from .interface import GSS, T, Acc


# Implementation overview
# ----------------------
# We represent the set of stacks as a trie over values of type T, but in reversed
# order (top-of-stack first). Every explicit stack path terminates with a sentinel
# edge label None (meaning "end of stack"). That means each stack [a, b, c] turns
# into the path [c, b, a, None].
#
# Accumulators exist at a single "level" within a subtree, enforced by a secondary
# layer of nodes (B-nodes) that overlay the structural trie (A-nodes):
#  - ANode: purely structural, maps edge label -> child ANode (labels include None).
#           It also stores nleaves = number of terminal edges (None) reachable under it.
#  - BNode: overlays an ANode, either:
#      * Wrap: holds a uniform multiset of accumulators for all stacks in the ANode
#               subtree (accs != None, children == None).
#      * Map:  distributes to children (accs == None, children is a dict label -> BNode).
#
# Invariants maintained:
#  - If a BNode is Wrap, none of its descendants hold accumulators; the accs are uniform
#    across the entire A-subtree.
#  - If a BNode is Map, the "accumulator level" is pushed exactly one level below (its children).
#  - If all children in a Map are Wrap and their acc lists are equal, they are "sucked up"
#    into the parent by turning the parent into a Wrap with that same acc list.
#  - We allow the acc multiset at a Wrap (tuple of Acc) to have multiple entries to represent
#    duplicates of identical stacks. This preserves reference semantics where duplicates
#    can exist until a pop/merge occurs (which merges duplicates for identical stacks).
#
# Operations:
#  - from_stacks builds the trie and overlay by inserting one stack at a time; then we normalize.
#  - push(value) is O(1) on structure: create a new A-root with a single child "value" -> old A-root,
#    and a new B-root as Map with the same single child. Normalize afterwards (which bubbles up
#    if possible).
#  - pop(merge_func) enumerates all stacks (path, acc), drops empty ones, pops top (path[1:]), and
#    merges accumulators for identical resulting stacks using merge_func; builds a new GSS via
#    from_stacks (with one acc per resulting stack).
#  - isolate(value) restricts to stacks whose top equals value (or empty if value is None).
#  - apply(func), prune(predicate) operate on accs at Wrap nodes; Maps recurse. Normalize at the end.
#  - peek() inspects top-of-stack values: for Wrap, look at underlying A-children labels (excluding None);
#    for Map, look at its children keys (excluding None).
#  - reduce_acc merges all accumulators across all stacks using merge_func, or returns None for no stacks.
#  - merge(gss_list, merge_func) flattens all stacks from each GSS and merges by resulting stack key
#    using merge_func, then builds a new GSS via from_stacks.
#
# Canonical equality and hashing are based on a canonical JSON-serializable representation:
#  a sorted list of [values (bottom-to-top), acc] pairs, exactly like the reference implementation.
#
# Notes:
#  - We use reversed stack representation internally for O(1) push and simpler pop logic.
#  - We avoid local config/cache advice as per guidance.
#  - We do not assume that T is hashable beyond what's needed to be a dict key (same as reference).
#  - We accept that reduce_acc order is only guaranteed if merge_func is order-independent (as requested).


# --------------------
# Internal structures
# --------------------

class _ANode(Generic[T]):
    __slots__ = ("children", "nleaves")

    def __init__(self, children: Dict[Optional[T], "_ANode[T]"]):
        # children: mapping Optional[T] -> _ANode (labels include None sentinel for end-of-stack)
        self.children: Dict[Optional[T], _ANode[T]] = children
        # Number of terminal (None) edges reachable under this node.
        # Since None edges are always terminal (we only create None at end-of-stack),
        # nleaves = (1 if None in children else 0) + sum(nleaves(child) for non-None children)
        cnt = 1 if (None in children) else 0
        for k, child in children.items():
            if k is not None:
                cnt += child.nleaves
        self.nleaves: int = cnt

    def with_single_child(self, label: Optional[T], child: "_ANode[T]") -> "_ANode[T]":
        return _ANode({label: child})

    @staticmethod
    def merge_nodes(nodes: List["_ANode[T]"]) -> "_ANode[T]":
        # Union/merge the A-trie shapes of several nodes.
        # Merge per-label recursively.
        merged_children: Dict[Optional[T], _ANode[T]] = {}
        by_label: Dict[Optional[T], List[_ANode[T]]] = {}
        for a in nodes:
            for label, ch in a.children.items():
                by_label.setdefault(label, []).append(ch)
        for label, lst in by_label.items():
            if len(lst) == 1:
                merged_children[label] = lst[0]
            else:
                merged_children[label] = _ANode.merge_nodes(lst)
        return _ANode(merged_children)


class _BNode(Generic[T, Acc]):
    __slots__ = ("a", "accs", "children")

    def __init__(self, a: _ANode[T], accs: Optional[Tuple[Acc, ...]], children: Optional[Dict[Optional[T], "_BNode[T, Acc]"]]):
        # Overlay for structural A-node `a`.
        # If accs is not None => Wrap: uniform multiset of accumulators across entire A-subtree.
        # If accs is None => Map: children mapping label->BNode over corresponding A-child.
        self.a: _ANode[T] = a
        self.accs: Optional[Tuple[Acc, ...]] = accs
        self.children: Optional[Dict[Optional[T], _BNode[T, Acc]]] = children

    @staticmethod
    def wrap(a: _ANode[T], accs: Iterable[Acc]) -> "_BNode[T, Acc]":
        # Normalize wrap: if accs is empty => return empty Map.
        accs_tuple = tuple(accs)
        if len(accs_tuple) == 0:
            return _BNode.map(a, {})
        return _BNode(a, accs_tuple, None)

    @staticmethod
    def map(a: _ANode[T], children: Dict[Optional[T], "_BNode[T, Acc]"]) -> "_BNode[T, Acc]":
        # Prune empty children and apply "suck-up" if all children are wraps with equal acc-lists.
        pruned: Dict[Optional[T], _BNode[T, Acc]] = {}
        for k, v in children.items():
            if not v.is_empty():
                pruned[k] = v
        if not pruned:
            return _BNode(a, None, {})  # empty map overlay (no stacks)

        # Check if all children are Wrap and have equal acc-lists
        first_accs: Optional[Tuple[Acc, ...]] = None
        all_wrap_equal = True
        for child in pruned.values():
            if child.accs is None:
                all_wrap_equal = False
                break
            if first_accs is None:
                first_accs = child.accs
            elif not _accs_equal(child.accs, first_accs):
                all_wrap_equal = False
                break

        if all_wrap_equal and first_accs is not None:
            # Suck up into parent
            return _BNode.wrap(a, first_accs)
        # Else keep as Map
        return _BNode(a, None, pruned)

    def is_wrap(self) -> bool:
        return self.accs is not None

    def is_map(self) -> bool:
        return self.accs is None

    def is_empty(self) -> bool:
        if self.accs is not None:
            return len(self.accs) == 0  # should be normalized away
        # Map case: empty if no children
        return not self.children

    def to_map(self) -> "_BNode[T, Acc]":
        # Expand Wrap into Map by replicating uniform accs to all children (including None if present).
        if self.is_map():
            return self
        assert self.accs is not None
        # For each A child, produce Wrap child with same accs
        child_nodes: Dict[Optional[T], _BNode[T, Acc]] = {}
        for label, a_child in self.a.children.items():
            child_nodes[label] = _BNode.wrap(a_child, self.accs)
        return _BNode.map(self.a, child_nodes)

    def normalized(self) -> "_BNode[T, Acc]":
        if self.is_wrap():
            return self if self.accs else _BNode.map(self.a, {})
        else:
            assert self.children is not None
            norm_children = {k: v.normalized() for k, v in self.children.items()}
            return _BNode.map(self.a, norm_children)

    def clone_with_children(self, new_children: Dict[Optional[T], "_BNode[T, Acc]"]) -> "_BNode[T, Acc]":
        return _BNode.map(self.a, new_children)


def _accs_equal(a: Tuple[Acc, ...], b: Tuple[Acc, ...]) -> bool:
    # Structural equality; order matters. This is fine because we never reorder internally.
    # Canonical JSON equality does not rely on acc equality here; it's for "suck up" optimization.
    return a == b


# --------------------------
# Main LeveledGSS class
# --------------------------

@dataclass(eq=False)
class LeveledGSS(GSS[T, Acc]):
    _a_root: _ANode[T]
    _b_root: _BNode[T, Acc]

    # ---------------
    # Constructors
    # ---------------

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> "LeveledGSS[T, Acc]":
        # Build A-trie (top-first + terminal None) and overlay by incremental insertion.
        if not stacks:
            # No stacks at all: empty structure.
            a_root = _ANode({})
            b_root = _BNode.map(a_root, {})
            return cls(a_root, b_root)

        # First, construct structural trie for all stacks (without accumulators).
        # We'll insert terminal None to the end of each reversed path.
        a_root = _build_anode([tuple(reversed(vals)) + (None,) for vals, _ in stacks])

        # Then insert each (path, acc) into an initially empty overlay.
        b_root = _BNode.map(a_root, {})
        for vals, acc in stacks:
            path = tuple(reversed(vals)) + (None,)
            b_root = _b_insert(b_root, path, acc)

        b_root = b_root.normalized()
        return cls(a_root, b_root)

    # ---------------
    # Core operations
    # ---------------

    def push(self, value: T) -> "LeveledGSS[T, Acc]":
        # New A-root with a single child "value" -> old A-root
        new_a_root = _ANode({value: self._a_root})
        # New B-root as a Map with that single edge pointing to old B-root
        new_b_root = _BNode.map(new_a_root, {value: self._b_root})
        new_b_root = new_b_root.normalized()
        return LeveledGSS(new_a_root, new_b_root)

    def pop(self, merge_func: Callable[[Acc, Acc], Acc]) -> "LeveledGSS[T, Acc]":
        # Enumerate all explicit stacks, drop empty ones, pop top symbol, and merge equal remainders.
        popped: Dict[Tuple[T, ...], Acc] = {}
        for path, acc in self._iter_stacks_accs():
            if len(path) == 0:
                # Was an empty stack; pop() removes it (reference drops empty stacks).
                continue
            remainder = path[1:]
            if remainder in popped:
                popped[remainder] = merge_func(popped[remainder], acc)
            else:
                popped[remainder] = acc

        # Build new GSS from the merged remainder stacks (one acc per stack).
        new_stacks: List[Tuple[List[T], Acc]] = [(list(reversed(rem)), acc) for rem, acc in popped.items()]
        return LeveledGSS.from_stacks(new_stacks)

    def is_empty(self) -> bool:
        # True iff exactly one active stack exists and that stack is empty ([]).
        # We count stacks by enumerating; early-exit to avoid full traversal when possible.
        count = 0
        empty_only = False
        for path, _acc in self._iter_stacks_accs():
            count += 1
            if count == 1:
                empty_only = (len(path) == 0)
            else:
                # More than one stack -> not empty by definition.
                return False
        return count == 1 and empty_only

    def isolate(self, value: Optional[T]) -> "LeveledGSS[T, Acc]":
        # Keep only stacks whose top equals `value` (or empty stacks if value is None).
        b = self._b_root
        # Ensure we are in Map form at the root for easy filtering.
        if b.is_wrap():
            b = b.to_map()

        # Select only the branch for `value`.
        new_children: Dict[Optional[T], _BNode[T, Acc]] = {}
        if b.children is not None:
            child = b.children.get(value)
            if child is not None:
                new_children[value] = child

        # A-root becomes the A-child at `value` if present; else empty.
        if value in self._a_root.children:
            new_a_root = self._a_root.children[value]  # type: ignore[index]
        else:
            new_a_root = _ANode({})

        new_b_root = _BNode.map(new_a_root, {value: new_children[value]} if value in new_children else {})
        new_b_root = new_b_root.normalized()
        return LeveledGSS(new_a_root, new_b_root)

    def apply(self, func: Callable[[Acc], Acc]) -> "LeveledGSS[T, Acc]":
        def _apply(b: _BNode[T, Acc]) -> _BNode[T, Acc]:
            if b.is_wrap():
                assert b.accs is not None
                new_accs = tuple(func(a) for a in b.accs)
                return _BNode.wrap(b.a, new_accs)
            else:
                assert b.children is not None
                new_children = {k: _apply(ch) for k, ch in b.children.items()}
                return _BNode.map(b.a, new_children)

        new_b_root = _apply(self._b_root).normalized()
        return LeveledGSS(self._a_root, new_b_root)

    def prune(self, predicate: Callable[[Acc], bool]) -> "LeveledGSS[T, Acc]":
        def _prune(b: _BNode[T, Acc]) -> _BNode[T, Acc]:
            if b.is_wrap():
                assert b.accs is not None
                kept = tuple(a for a in b.accs if predicate(a))
                return _BNode.wrap(b.a, kept)
            else:
                assert b.children is not None
                new_children = {k: _prune(ch) for k, ch in b.children.items()}
                return _BNode.map(b.a, new_children)

        new_b_root = _prune(self._b_root).normalized()
        return LeveledGSS(self._a_root, new_b_root)

    def peek(self) -> Set[T]:
        # Returns the set of all top-of-stack values across non-empty stacks.
        if self._b_root.is_wrap():
            # Look at structural A-children
            return {k for k in self._a_root.children.keys() if k is not None}
        else:
            assert self._b_root.children is not None
            return {k for k in self._b_root.children.keys() if k is not None}

    def reduce_acc(self, merge_func: Callable[[Acc, Acc], Acc]) -> Optional[Acc]:
        it = self._iter_stacks_accs()
        try:
            first_path, first_acc = next(it)
        except StopIteration:
            return None
        total = first_acc
        for _path, acc in it:
            total = merge_func(total, acc)
        return total

    @staticmethod
    def merge(gss_list: Iterable["LeveledGSS[T, Acc]"], merge_func: Callable[[Acc, Acc], Acc]) -> "LeveledGSS[T, Acc]":
        # Flatten all stacks and merge accumulators for identical stacks using merge_func.
        merged: Dict[Tuple[T, ...], Acc] = {}
        for g in gss_list:
            for path, acc in g._iter_stacks_accs():
                if path in merged:
                    merged[path] = merge_func(merged[path], acc)
                else:
                    merged[path] = acc
        stacks: List[Tuple[List[T], Acc]] = [(list(reversed(path)), acc) for path, acc in merged.items()]
        return LeveledGSS.from_stacks(stacks)

    # ---------------
    # JSON/equality/hash
    # ---------------

    def to_json_serializable(self) -> Any:
        # Canonical representation: list of [values_list (bottom->top), acc], sorted deterministically.
        pairs: List[Tuple[List[T], Acc]] = []
        for path, acc in self._iter_stacks_accs():
            # path is reversed (top-first). Convert back to bottom->top.
            pairs.append((list(reversed(path)), acc))

        def _encode_for_sort(obj: Any) -> str:
            try:
                return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
            except Exception:
                return repr(obj)

        pairs.sort(key=lambda p: (_encode_for_sort(p[0]), _encode_for_sort(p[1])))
        return [[vals, acc] for vals, acc in pairs]

    def __hash__(self):
        try:
            canonical = self.to_json_serializable()
            s = json.dumps(canonical, sort_keys=True, default=repr, separators=(",", ":"))
            return hash(s)
        except Exception:
            # Fallback: try to hash set of pairs; if fails, object hash.
            try:
                # This may fail if acc is unhashable
                flat = tuple((tuple(vals), acc) for vals, acc in self.to_json_serializable())
                return hash(flat)
            except Exception:
                return object.__hash__(self)

    # ---------------
    # Internals: enumeration and builders
    # ---------------

    def _iter_stacks_accs(self) -> Iterable[Tuple[Tuple[T, ...], Acc]]:
        # Enumerate (reversed_path_without_None, acc) for each explicit stack (including duplicates).
        # We avoid constructing explicit stacks eagerly when we can.
        if self._b_root.is_wrap():
            assert self._b_root.accs is not None
            # Uniform accs across all leaves of self._a_root.
            for path in _iter_paths(self._a_root, ()):
                for acc in self._b_root.accs:
                    yield (path, acc)
        else:
            assert self._b_root.children is not None
            yield from _iter_bnode_paths(self._b_root, ())


def _build_anode(paths: List[Tuple[Optional[T], ...]]) -> _ANode[T]:
    # Build a structural trie (A-nodes) from a list of paths (each ends with None).
    # We'll first create a nested dict-of-dicts representation and then convert recursively.
    nested: Dict[Optional[T], Any] = {}

    for path in paths:
        cur = nested
        for label in path:
            nxt = cur.get(label)
            if nxt is None:
                nxt = {}
                cur[label] = nxt
            cur = nxt

    def convert(d: Dict[Optional[T], Any]) -> _ANode[T]:
        children: Dict[Optional[T], _ANode[T]] = {}
        for label, sub in d.items():
            children[label] = convert(sub) if isinstance(sub, dict) else sub
        return _ANode(children)

    return convert(nested)


def _b_insert(b: _BNode[T, Acc], path: Tuple[Optional[T], ...], acc: Acc) -> _BNode[T, Acc]:
    # Insert a single (path, acc) into overlay, returning new normalized overlay.
    # path is a tuple of Optional[T], ending with None.
    if b.is_wrap():
        b = b.to_map()

    assert b.children is not None
    assert len(path) >= 1
    label = path[0]
    rest = path[1:]
    # The A structure must have this label child
    a_child = b.a.children.get(label)
    if a_child is None:
        # Path does not exist structurally; create corresponding A nodes on the fly.
        # This only happens if _build_anode didn't include the path; but from_stacks builds A first,
        # so this is typically unreachable. We still handle it defensively.
        a_child = _build_anode([path])

    child_b = b.children.get(label)
    if child_b is None:
        if len(rest) == 0:
            # We've reached the terminal (None). Create a Wrap with the acc.
            new_child = _BNode.wrap(a_child, [acc])
        else:
            # Need a Map deeper; create empty map on a_child, then insert rest.
            tmp = _BNode.map(a_child, {})
            new_child = _b_insert(tmp, rest, acc)
    else:
        if len(rest) == 0:
            # Reached terminal into an existing child overlay
            if child_b.is_wrap():
                assert child_b.accs is not None
                new_child = _BNode.wrap(child_b.a, list(child_b.accs) + [acc])
            else:
                # child is a Map; insert an empty (no further labels) shouldn't happen here because terminal None
                # is always the last label; if Map exists here, we insert into its child labeled by nothing,
                # but our structure ensures terminal (None) is last step and its child overlay should be a Wrap.
                # Fall back to recursive insertion to keep correctness.
                new_child = _b_insert(child_b, rest, acc)
        else:
            new_child = _b_insert(child_b, rest, acc)

    new_children = dict(b.children)
    new_children[label] = new_child.normalized()
    return _BNode.map(b.a, new_children)


def _iter_paths(a: _ANode[T], prefix: Tuple[T, ...]) -> Iterable[Tuple[T, ...]]:
    # Iterate all stack paths (reversed, top-first) under A node `a`, excluding the terminal None.
    for label, child in a.children.items():
        if label is None:
            # End-of-stack; yield current prefix.
            yield prefix
        else:
            yield from _iter_paths(child, prefix + (label,))


def _iter_bnode_paths(b: _BNode[T, Acc], prefix: Tuple[T, ...]) -> Iterable[Tuple[Tuple[T, ...], Acc]]:
    # Enumerate all stacks under B-node `b` (which must be a Map), with their accumulators.
    assert b.is_map()
    assert b.children is not None
    for label, child in b.children.items():
        if label is None:
            # terminal branch: child overlay at terminal sentinel
            if child.is_wrap():
                assert child.accs is not None
                for acc in child.accs:
                    yield (prefix, acc)
            else:
                # Invariant: terminal child should normally be Wrap; but handle defensively.
                yield from _iter_bnode_paths(child, prefix)
        else:
            if child.is_wrap():
                assert child.accs is not None
                for path in _iter_paths(child.a, prefix + (label,)):
                    for acc in child.accs:
                        yield (path, acc)
            else:
                yield from _iter_bnode_paths(child, prefix + (label,))


# -------------
# End of file
# -------------
