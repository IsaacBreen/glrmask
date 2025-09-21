from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Generator, TypeVar, Iterator

from ..interface import GSS, T, Acc

# ------------------------------------------------------------------------------
# Internal node classes and type alias (keep type/class structure unchanged)
# ------------------------------------------------------------------------------

# Keep the type alias exactly as authored (no changes to typing structure).
type Upper[T, Acc] = UpperBranch[T, Acc] | Interface[T, Acc]


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    """
    Upper tree node that carries:
      - children: next nodes keyed by top-of-stack value and depth bucket
      - empty: Optional[Acc] representing an end-of-stack at this point
    """
    children: Dict[T, Dict[int, Upper[T, Acc]]]
    empty: Optional[Acc]
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, '_max_depth', depth)

    def _all_children(self) -> Iterator[Upper[T, Acc]]:
        for children_at_depth in self.children.values():
            yield from children_at_depth.values()


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    """
    Boundary node that pins an accumulator value for all stacks beneath it.
    Its children are a 'lower' trie that encodes just stack shapes (no accumulators).
    """
    children: Dict[T, Dict[int, Lower[T]]]
    acc: Acc
    empty: Optional[Acc]
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, '_max_depth', depth)

    def _all_children(self) -> Iterator[Lower[T]]:
        for v_children in self.children.values():
            yield from v_children.values()


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    """
    Lower trie node: structure-only (no accumulators).
    'empty' marks whether a stack can end here.
    """
    children: Dict[T, Dict[int, Lower[T]]]
    empty: bool
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, '_max_depth', depth)

    def _all_children(self) -> Iterator[Lower[T]]:
        for v_children in self.children.values():
            yield from v_children.values()


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A compact, canonical GSS using a two-level representation:
      - Upper tree (UpperBranch/Interface) merges identical suffixes
      - Lower tree (beneath Interface) encodes shapes only (no Acc), while 'Interface'
        provides the Acc for all stacks in its Lower subtree.
    """
    inner: Upper[T, Acc]

    # ------------------------------
    # Validation (unchanged)
    # ------------------------------
    def __post_init__(self):
        if os.environ.get("GSS_TESTER_VALIDATE"):
            self._validate_max_depths()
            self._validate_no_promotions()
            self._validate_populated_nodes()

    def _validate_max_depths(self) -> None:
        self._validate_depths_node(self.inner)

    def _validate_depths_node(self, node: Upper[T, Acc]) -> None:
        if isinstance(node, Interface):
            def _validate_lower_recursively(n: Interface[T, Acc] | Lower[T]):
                for children_at_depth in n.children.values():  # type: ignore[attr-defined]
                    for depth, child in children_at_depth.items():
                        if depth != child._max_depth:
                            raise ValueError(
                                "LeveledGSS validation failed: incorrect max_depth for Lower child. "
                                f"Expected {depth}, got {child._max_depth}. Node: {n}"
                            )
                        _validate_lower_recursively(child)

            _validate_lower_recursively(node)
            return

        # UpperBranch
        for children_at_depth in node.children.values():
            for depth, child in children_at_depth.items():
                if depth != child._max_depth:
                    raise ValueError(
                        "LeveledGSS validation failed: incorrect max_depth for Upper child. "
                        f"Expected {depth}, got {child._max_depth}. Node: {node}"
                    )
                self._validate_depths_node(child)

    def _validate_no_promotions(self) -> None:
        if isinstance(self.inner, UpperBranch):
            self._validate_promotion_node(self.inner)

    def _validate_promotion_node(self, node: Upper[T, Acc]) -> None:
        if isinstance(node, Interface):
            return

        # UpperBranch: recurse, then check promotion possibility
        for children_at_depth in node.children.values():
            for child in children_at_depth.values():
                self._validate_promotion_node(child)

        all_children = list(node._all_children())
        if not all_children or not all(isinstance(child, Interface) for child in all_children):
            return

        accs: Set[Acc] = set()
        if node.empty is not None:
            accs.add(node.empty)
        for child in all_children:
            ic: Interface[T, Acc] = child  # type: ignore[assignment]
            accs.add(ic.acc)
            if ic.empty is not None:
                accs.add(ic.empty)

        if len(accs) == 1:
            raise ValueError(
                "LeveledGSS validation failed: an UpperBranch can be promoted to an Interface, "
                f"indicating a non-canonical structure. Node: {node}"
            )

    def _validate_populated_nodes(self) -> None:
        # Root may be empty.
        if isinstance(self.inner, UpperBranch) and not self.inner.children and self.inner.empty is None:
            return
        self._validate_node_is_populated(self.inner)

    def _validate_node_is_populated(self, node: Upper[T, Acc] | Lower[T]) -> None:
        if isinstance(node, UpperBranch):
            if not node.children and node.empty is None:
                raise ValueError(
                    "LeveledGSS validation failed: UpperBranch with no children and no empty accumulator "
                    f"found in a non-root position. Node: {node}"
                )
            for children_at_depth in node.children.values():
                for child in children_at_depth.values():
                    self._validate_node_is_populated(child)
        elif isinstance(node, Interface):
            if not node.children and node.empty is None:
                raise ValueError(
                    "LeveledGSS validation failed: Interface with no children and no empty accumulator found. "
                    f"Node: {node}"
                )
            for children_at_depth in node.children.values():
                for child in children_at_depth.values():
                    self._validate_node_is_populated(child)
        else:  # Lower
            if not node.children and not node.empty:
                raise ValueError(
                    "LeveledGSS validation failed: Lower node with no children and empty=False found. "
                    f"Node: {node}"
                )
            for children_at_depth in node.children.values():
                for child in children_at_depth.values():
                    self._validate_node_is_populated(child)

    # ------------------------------
    # Construction
    # ------------------------------
    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        """
        Build a canonical LeveledGSS from explicit stacks:
          1) Merge duplicate stacks (same values) by merging their accumulators.
          2) Compile into an Upper trie (top-of-stack first).
          3) Locally promote to Interface where possible.
        """
        # 1) Merge duplicates
        merged: Dict[Tuple[T, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            prev = merged.get(key)
            merged[key] = acc if prev is None else prev.merge(acc)

        root_empty: Optional[Acc] = merged.pop((), None) if () in merged else None
        # Reverse sequences to be top-of-stack first
        rev_items: List[Tuple[Tuple[T, ...], Acc]] = [(tuple(reversed(k)), acc) for k, acc in merged.items()]

        def build_from_rev(items: List[Tuple[Tuple[T, ...], Acc]], empty_acc: Optional[Acc]) -> Upper[T, Acc]:
            if not items:
                return UpperBranch(children={}, empty=empty_acc)

            # Group by first element; collect ends and tails
            ends: Dict[T, Acc] = {}
            tails: Dict[T, List[Tuple[Tuple[T, ...], Acc]]] = {}
            for seq, acc in items:
                v = seq[0]
                if len(seq) == 1:
                    prev = ends.get(v)
                    ends[v] = acc if prev is None else prev.merge(acc)
                else:
                    tails.setdefault(v, []).append((seq[1:], acc))

            built_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}

            def add_child(v: T, child: Upper[T, Acc]) -> None:
                bucket = built_children.setdefault(v, {})
                bucket[child._max_depth] = child

            # "End" stacks at this level become leaf UpperBranches with 'empty=acc'
            for v, acc in ends.items():
                add_child(v, UpperBranch(children={}, empty=acc))

            # Tails recurse
            for v, group in tails.items():
                add_child(v, build_from_rev(group, None))

            return try_promote(UpperBranch(children=built_children, empty=empty_acc))

        return LeveledGSS(build_from_rev(rev_items, root_empty))

    # ------------------------------
    # Public API
    # ------------------------------
    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """
        Convert to explicit stacks, canonicalize (merge duplicates), and sort deterministically.
        """
        def iter_lower(node: Lower[T], prefix_tos: List[T], acc: Acc) -> Iterator[Tuple[List[T], Acc]]:
            if node.empty:
                yield (list(reversed(prefix_tos)), acc)
            for v, kids in node.children.items():
                for child in kids.values():
                    yield from iter_lower(child, prefix_tos + [v], acc)

        def iter_upper(node: Upper[T, Acc], prefix_tos: List[T]) -> Iterator[Tuple[List[T], Acc]]:
            if isinstance(node, UpperBranch):
                if node.empty is not None:
                    yield (list(reversed(prefix_tos)), node.empty)
                for v, kids in node.children.items():
                    for child in kids.values():
                        yield from iter_upper(child, prefix_tos + [v])
            else:
                if node.empty is not None:
                    yield (list(reversed(prefix_tos)), node.empty)
                # A leaf Interface with no children and no explicit empty encodes a stack ending here with acc.
                if not node.children and node.empty is None:
                    yield (list(reversed(prefix_tos)), node.acc)
                for v, kids in node.children.items():
                    for child in kids.values():
                        yield from iter_lower(child, prefix_tos + [v], node.acc)

        return _canonicalize_and_sort(list(iter_upper(self.inner, [])))

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        """
        Push value onto the top of all stacks.
        Empty GSS remains empty.
        """
        if self.is_empty():
            return self

        if isinstance(self.inner, Interface):
            # After a push, everything previously under this Interface becomes a single Lower child under `value`.
            had_empty_stack = (self.inner.empty is not None) or (not self.inner.children)
            lower_node = Lower(children=self.inner.children, empty=had_empty_stack)
            new_children: Dict[T, Dict[int, Lower[T]]] = {value: {lower_node._max_depth: lower_node}}
            return LeveledGSS(Interface(children=new_children, acc=self.inner.acc, empty=None))

        # UpperBranch: wrap the entire structure under 'value'
        child = self.inner
        return LeveledGSS(UpperBranch(children={value: {child._max_depth: child}}, empty=None))

    def pop(self) -> LeveledGSS[T, Acc]:
        """
        Pop the top element from all non-empty stacks.
        Empty stacks are discarded.
        """
        ub = _as_upperbranch(self.inner)
        # Gather all children across top values
        children = [child for kids in ub.children.values() for child in kids.values()]
        if not children:
            return LeveledGSS(UpperBranch(children={}, empty=None))
        merged = _merge_many_upper(children)
        return LeveledGSS(try_promote(_as_upperbranch(merged)))

    def is_empty(self) -> bool:
        if isinstance(self.inner, UpperBranch):
            return not self.inner.children and self.inner.empty is None
        return False  # Interface always encodes at least one stack

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        """
        Keep only stacks whose top equals value. If value is None, keep only empty stacks.
        """
        if value is None:
            if isinstance(self.inner, UpperBranch):
                return LeveledGSS(UpperBranch(children={}, empty=self.inner.empty))
            empty_acc = _interface_empty_acc(self.inner)
            return LeveledGSS(UpperBranch(children={}, empty=empty_acc))

        if isinstance(self.inner, UpperBranch):
            filtered = {value: self.inner.children[value]} if value in self.inner.children else {}
            return LeveledGSS(try_promote(UpperBranch(children=filtered, empty=None)))
        else:
            if value not in self.inner.children:
                return LeveledGSS(UpperBranch(children={}, empty=None))
            filtered_children = {value: self.inner.children[value]}
            return LeveledGSS(Interface(children=filtered_children, acc=self.inner.acc, empty=None))

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        """
        Apply func to every accumulator carried by the structure.
        This affects:
          - UpperBranch.empty if present
          - Interface.acc and Interface.empty if present
        Lower nodes contain no accumulators.
        """
        memo: Dict[int, Upper[T, Acc]] = {}

        def transform(node: Upper[T, Acc]) -> Upper[T, Acc]:
            cached = memo.get(id(node))
            if cached is not None:
                return cached

            if isinstance(node, Interface):
                res: Upper[T, Acc] = Interface(
                    children=node.children,
                    acc=func(node.acc),
                    empty=(func(node.empty) if node.empty is not None else None),
                )
                memo[id(node)] = res
                return res

            # UpperBranch
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, kids in node.children.items():
                out_for_v: Dict[int, Upper[T, Acc]] = {}
                for child in kids.values():
                    tchild = transform(child)
                    out_for_v[tchild._max_depth] = tchild
                if out_for_v:
                    new_children[v] = out_for_v

            new_empty = func(node.empty) if node.empty is not None else None
            res_ub = UpperBranch(children=new_children, empty=new_empty)
            promoted = try_promote(res_ub)
            memo[id(node)] = promoted
            return promoted

        return LeveledGSS(transform(self.inner))

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        """
        Remove stacks whose accumulator fails predicate.
        For Interface:
          - If acc pruned and empty pruned => remove whole subtree
          - If acc pruned but empty kept => only empty-at-this-level survives (becomes UpperBranch with empty set)
          - If acc kept => keep subtree; possibly drop 'empty' if pruned
        For UpperBranch:
          - Recurse on children; drop child if it prunes to None
          - Drop 'empty' if predicate fails
        """
        memo: Dict[int, Optional[Upper[T, Acc]]] = {}

        def transform(node: Upper[T, Acc]) -> Optional[Upper[T, Acc]]:
            cached = memo.get(id(node))
            if cached is not None:
                return cached

            if isinstance(node, Interface):
                keep_acc = predicate(node.acc)
                keep_empty = node.empty is not None and predicate(node.empty)
                new_empty = node.empty if keep_empty else None

                if not keep_acc and not keep_empty:
                    memo[id(node)] = None
                    return None
                if not keep_acc and keep_empty:
                    res = UpperBranch(children={}, empty=new_empty)
                    memo[id(node)] = res
                    return res

                res = Interface(children=node.children, acc=node.acc, empty=new_empty)
                memo[id(node)] = res
                return res

            # UpperBranch
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, kids in node.children.items():
                out_for_v: Dict[int, Upper[T, Acc]] = {}
                for child in kids.values():
                    tchild = transform(child)
                    if tchild is not None:
                        out_for_v[tchild._max_depth] = tchild
                if out_for_v:
                    new_children[v] = out_for_v

            new_empty = node.empty if (node.empty is not None and predicate(node.empty)) else None
            if not new_children and new_empty is None:
                memo[id(node)] = None
                return None

            res_ub = UpperBranch(children=new_children, empty=new_empty)
            promoted = try_promote(res_ub)
            memo[id(node)] = promoted
            return promoted

        new_inner = transform(self.inner)
        if new_inner is None:
            return LeveledGSS(UpperBranch(children={}, empty=None))
        return LeveledGSS(new_inner)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(merge_upper(self.inner, other.inner))

    def peek(self) -> Set[T]:
        """
        Return the set of all top-of-stack values (ignores empty stacks).
        """
        return set(self.inner.children.keys())

    def reduce_acc(self) -> Optional[Acc]:
        """
        Merge the accumulators of all active stacks into a single optional value.
        """
        def emit_lower(node: Lower[T], acc: Acc) -> Iterator[Acc]:
            if node.empty:
                yield acc
            for children_at_depth in node.children.values():
                for child in children_at_depth.values():
                    yield from emit_lower(child, acc)

        def emit(node: Upper[T, Acc]) -> Iterator[Acc]:
            if isinstance(node, UpperBranch):
                if node.empty is not None:
                    yield node.empty
                for child in node._all_children():
                    yield from emit(child)
            else:
                if node.empty is not None:
                    yield node.empty
                if not node.children and node.empty is None:
                    yield node.acc
                for children_at_depth in node.children.values():
                    for child in children_at_depth.values():
                        yield from emit_lower(child, node.acc)

        it = iter(emit(self.inner))
        try:
            out = next(it)
        except StopIteration:
            return None
        for a in it:
            out = out.merge(a)
        return out


# ------------------------------------------------------------------------------
# Internal helpers (no new classes added)
# ------------------------------------------------------------------------------

def _as_upperbranch(node: Upper[T, Acc]) -> UpperBranch[T, Acc]:
    return node if isinstance(node, UpperBranch) else interface_to_upperbranch(node)


Node = TypeVar("Node")


def _merge_children_by_depth(
    c1: Dict[T, Dict[int, Node]],
    c2: Dict[T, Dict[int, Node]],
    merge_func: Callable[[Node, Node], Node],
) -> Dict[T, Dict[int, Node]]:
    """
    Merge children maps grouped first by value, then by depth. For each value and
    depth bucket, fold multiple nodes with merge_func.
    """
    merged: Dict[T, Dict[int, Node]] = {}

    def ingest(src: Dict[T, Dict[int, Node]]) -> None:
        for v, bucket in src.items():
            out_bucket = merged.setdefault(v, {})
            for n in bucket.values():
                d = getattr(n, "_max_depth")
                prev = out_bucket.get(d)
                out_bucket[d] = n if prev is None else merge_func(prev, n)

    ingest(c1)
    ingest(c2)
    return merged


def _merge_many_upper(nodes: List[Upper[T, Acc]]) -> Upper[T, Acc]:
    """
    Merge a non-empty list of Upper nodes into a single Upper node.
    """
    it = iter(nodes)
    acc = next(it)
    for n in it:
        acc = merge_upper(acc, n)
    return acc


def try_promote(node: UpperBranch[T, Acc]) -> Upper[T, Acc]:
    """
    If all children are Interface nodes and all accumulators (including node.empty when present)
    collapse to a single value, compress this UpperBranch into a single Interface whose children
    are a Lower forest.
    """
    all_children = list(node._all_children())
    if not all_children:
        return node
    if not all(isinstance(c, Interface) for c in all_children):
        return node

    accs: Set[Acc] = set()
    if node.empty is not None:
        accs.add(node.empty)
    for c in all_children:
        ic: Interface[T, Acc] = c  # type: ignore[assignment]
        accs.add(ic.acc)
        if ic.empty is not None:
            accs.add(ic.empty)

    if len(accs) <= 1:
        the_acc: Optional[Acc] = next(iter(accs)) if accs else None
        if the_acc is None:
            return UpperBranch(children={}, empty=None)

        l_children: Dict[T, Dict[int, Lower[T]]] = {}
        for v, kids in node.children.items():
            bucket: Dict[int, Lower[T]] = {}
            for child in kids.values():
                ci: Interface[T, Acc] = child  # type: ignore[assignment]
                lower = Lower(children=ci.children, empty=(ci.empty is not None))
                bucket[lower._max_depth] = lower
            if bucket:
                l_children[v] = bucket
        return Interface(children=l_children, acc=the_acc, empty=node.empty)

    return node


def interface_to_upperbranch(it: Interface[T, Acc]) -> UpperBranch[T, Acc]:
    """
    Convert an Interface into an UpperBranch. A Lower child becomes an Interface
    child (with the same acc) whose 'empty' reflects whether the lower node ended here.
    Also, if the Interface has no children and no explicit empty accumulator, it
    represents an end-of-stack with acc at this level; capture that in empty.
    """
    children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for v, kids in it.children.items():
        bucket: Dict[int, Upper[T, Acc]] = {}
        for lchild in kids.values():
            child_it = Interface(children=lchild.children, acc=it.acc, empty=(it.acc if lchild.empty else None))
            bucket[child_it._max_depth] = child_it
        if bucket:
            children[v] = bucket

    new_empty = it.empty
    if not it.children and new_empty is None:
        new_empty = it.acc
    return UpperBranch(children=children, empty=new_empty)


def _merge_opt(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
    if a is None:
        return b
    if b is None:
        return a
    return a.merge(b)


def merge_upper(u1: Upper[T, Acc], u2: Upper[T, Acc]) -> Upper[T, Acc]:
    if u1 is u2:
        return u1
    if isinstance(u1, Interface) and isinstance(u2, Interface):
        return merge_interfaces(u1, u2)
    if isinstance(u1, UpperBranch) and isinstance(u2, UpperBranch):
        return merge_upperbranches(u1, u2)

    # Mixed types: convert Interfaces to UpperBranches and merge
    ub1 = u1 if isinstance(u1, UpperBranch) else interface_to_upperbranch(u1)
    ub2 = u2 if isinstance(u2, UpperBranch) else interface_to_upperbranch(u2)
    return merge_upperbranches(ub1, ub2)  # type: ignore[arg-type]


def merge_upperbranches(a: UpperBranch[T, Acc], b: UpperBranch[T, Acc]) -> Upper[T, Acc]:
    if a is b:
        return a
    new_empty = _merge_opt(a.empty, b.empty)
    merged_children = _merge_children_by_depth(a.children, b.children, merge_upper)
    return try_promote(UpperBranch(children=merged_children, empty=new_empty))


def merge_interfaces(a: Interface[T, Acc], b: Interface[T, Acc]) -> Upper[T, Acc]:
    # If both interfaces use the same accumulator, keep Interface form
    if a.acc == b.acc:
        new_empty = _merge_opt(a.empty, b.empty)
        merged_children = _merge_children_by_depth(a.children, b.children, merge_lower)
        return Interface(children=merged_children, acc=a.acc, empty=new_empty)

    # Different accumulators => lift to UpperBranch form and merge
    return merge_upperbranches(interface_to_upperbranch(a), interface_to_upperbranch(b))


def merge_lower(l1: Lower[T], l2: Lower[T]) -> Lower[T]:
    # Merge emptiness (logical OR) and children structurally
    new_empty = l1.empty or l2.empty
    merged_children = _merge_children_by_depth(l1.children, l2.children, merge_lower)
    return Lower(children=merged_children, empty=new_empty)


def _interface_empty_acc(it: Interface[T, Acc]) -> Optional[Acc]:
    """
    Determine the accumulator for empty stacks at an Interface boundary.
    If explicit empty exists, return it; if no children exist and no explicit empty,
    the leaf represents an empty stack with 'acc'.
    """
    if it.empty is not None:
        return it.empty
    if not it.children:
        return it.acc
    return None


# ------------------------------------------------------------------------------
# Canonicalization helpers for stacks (keep behavior aligned with ReferenceGSS)
# ------------------------------------------------------------------------------

def _encode_for_sort(obj: Any) -> str:
    """
    Deterministic encoding used to sort stacks. Mirrors ReferenceGSS._get_canonical_sorted_stacks.
    """
    import json
    try:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    except Exception:
        return repr(obj)


def _canonicalize_and_sort(items: List[Tuple[List[T], Acc]]) -> List[Tuple[List[T], Acc]]:
    """
    Merge duplicate stacks (by values) by merging their accumulators, then
    sort deterministically using the same ordering as ReferenceGSS.
    """
    merged: Dict[Tuple[T, ...], Acc] = {}
    for vals, acc in items:
        key = tuple(vals)
        prev = merged.get(key)
        merged[key] = acc if prev is None else prev.merge(acc)

    out = [(list(k), v) for k, v in merged.items()]
    out.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
    return out


__all__ = ["LeveledGSS"]
