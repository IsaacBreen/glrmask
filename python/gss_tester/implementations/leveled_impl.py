from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Generator, TypeVar, Iterator

from ..interface import GSS, T, Acc
from .reference_impl import ReferenceGSS

# ------------------------------------------------------------------------------
# Internal node classes and type alias
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

    def _all_children(self) -> Generator[Upper[T, Acc], None, None]:
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
        Build a canonical LeveledGSS from explicit stacks.
        We first canonicalize by merging duplicate stacks (without sorting cost),
        then compile into our trie representation and perform local promotion.
        """
        # Canonicalize duplicates while keeping order-free structure; no sorting overhead.
        canonical_stacks = ReferenceGSS(stacks)._stacks

        # Build a simple trie in Python dicts:
        # Each node: { value: {"end": Optional[Acc], "sub": <subtrie> } }
        trie: Dict[T, Dict[str, Any]] = {}
        root_empty: Optional[Acc] = None

        for vals, acc in canonical_stacks:
            if not vals:
                root_empty = acc
                continue
            node = trie
            # Traverse from top-of-stack downwards
            rev = list(reversed(vals))
            for i, v in enumerate(rev):
                entry = node.setdefault(v, {"end": None, "sub": {}})
                if i == len(rev) - 1:
                    entry["end"] = acc  # leaf (end-of-stack)
                else:
                    node = entry["sub"]

        def build_upper(subtrie: Dict[T, Dict[str, Any]], empty_acc: Optional[Acc]) -> Upper[T, Acc]:
            # Compile children first
            built_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, e in subtrie.items():
                nodes_for_v: List[Upper[T, Acc]] = []

                end_acc: Optional[Acc] = e.get("end")
                sub = e.get("sub", {})

                if end_acc is not None:
                    nodes_for_v.append(UpperBranch(children={}, empty=end_acc))
                if sub:
                    nodes_for_v.append(build_upper(sub, None))

                if nodes_for_v:
                    built_children[v] = {n._max_depth: n for n in nodes_for_v}

            # Assemble as UpperBranch then try to promote if possible
            ub = UpperBranch(children=built_children, empty=empty_acc)
            return try_promote(ub)

        return LeveledGSS(build_upper(trie, root_empty))

    # ------------------------------
    # Public API
    # ------------------------------
    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """
        Convert back to explicit stacks, merging duplicates and sorting for stability.
        """
        res: List[Tuple[List[T], Acc]] = []

        def emit_stack(pref: List[T], acc: Acc) -> None:
            res.append((list(reversed(pref)), acc))

        def dfs_lower(node: Lower[T], pref: List[T], acc: Acc) -> None:
            if node.empty:
                emit_stack(pref, acc)
            for v, kids in node.children.items():
                for child in kids.values():
                    dfs_lower(child, pref + [v], acc)

        def dfs_upper(node: Upper[T, Acc], pref: List[T]) -> None:
            if isinstance(node, UpperBranch):
                if node.empty is not None:
                    emit_stack(pref, node.empty)
                for v, kids in node.children.items():
                    for child in kids.values():
                        dfs_upper(child, pref + [v])
            else:
                if node.empty is not None:
                    emit_stack(pref, node.empty)
                if not node.children and node.empty is None:
                    emit_stack(pref, node.acc)
                else:
                    for v, kids in node.children.items():
                        for child in kids.values():
                            dfs_lower(child, pref + [v], node.acc)

        dfs_upper(self.inner, [])
        # Canonicalize (merge duplicates) + sort deterministically
        return ReferenceGSS(res).to_stacks()

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        """
        Push value onto the top of all stacks.
        Empty GSS remains empty.
        """
        if self.is_empty():
            return self

        if isinstance(self.inner, Interface):
            # Pushing onto an Interface: wrap its lower subtree under the new top value.
            # The new Lower node is "empty" if and only if there was an empty stack previously.
            had_empty_stack = (self.inner.empty is not None) or (not self.inner.children)
            lower_node = Lower(children=self.inner.children, empty=had_empty_stack)
            new_children: Dict[T, Dict[int, Lower[T]]] = {value: {lower_node._max_depth: lower_node}}
            return LeveledGSS(Interface(children=new_children, acc=self.inner.acc, empty=None))

        # UpperBranch: attach the previous root under the new value.
        child = self.inner
        return LeveledGSS(UpperBranch(children={value: {child._max_depth: child}}, empty=None))

    def pop(self) -> LeveledGSS[T, Acc]:
        """
        Pop the top element from all non-empty stacks.
        Empty stacks are discarded.
        """
        ub = self.inner if isinstance(self.inner, UpperBranch) else interface_to_upperbranch(self.inner)

        # Combine all children (across all top values) since we pop the top-of-stack.
        merged: Upper[T, Acc] = UpperBranch(children={}, empty=None)
        for child in ub._all_children():
            merged = merge_upper(merged, child)

        merged = try_promote(merged if isinstance(merged, UpperBranch) else interface_to_upperbranch(merged))
        return LeveledGSS(merged)

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
            # Interface: preserve empty stacks. If it's a leaf Interface with implicit empty,
            # we must capture that as an empty accumulator.
            empty_acc: Optional[Acc]
            if self.inner.empty is not None:
                empty_acc = self.inner.empty
            elif not self.inner.children:
                empty_acc = self.inner.acc
            else:
                empty_acc = None
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
                new_acc = func(node.acc)
                new_empty = func(node.empty) if node.empty is not None else None
                res: Upper[T, Acc] = Interface(children=node.children, acc=new_acc, empty=new_empty)
                memo[id(node)] = res
                return res

            # UpperBranch
            new_empty = func(node.empty) if node.empty is not None else None
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, kids in node.children.items():
                out_for_v: Dict[int, Upper[T, Acc]] = {}
                for child in kids.values():
                    tchild = transform(child)
                    out_for_v[tchild._max_depth] = tchild
                if out_for_v:
                    new_children[v] = out_for_v

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
            new_empty = node.empty if (node.empty is not None and predicate(node.empty)) else None
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, kids in node.children.items():
                out_for_v: Dict[int, Upper[T, Acc]] = {}
                for child in kids.values():
                    tchild = transform(child)
                    if tchild is not None:
                        out_for_v[tchild._max_depth] = tchild
                if out_for_v:
                    new_children[v] = out_for_v

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
        Reduce/merge all accumulators across all active stacks.
        """
        def emit_lower(node: Lower[T], acc: Acc) -> Generator[Acc, None, None]:
            if node.empty:
                yield acc
            for children_at_depth in node.children.values():
                for child in children_at_depth.values():
                    yield from emit_lower(child, acc)

        def emit(node: Upper[T, Acc]) -> Generator[Acc, None, None]:
            if isinstance(node, UpperBranch):
                if node.empty is not None:
                    yield node.empty
                for children_at_depth in node.children.values():
                    for child in children_at_depth.values():
                        yield from emit(child)
            else:
                if node.empty is not None:
                    yield node.empty
                if not node.children and node.empty is None:
                    yield node.acc
                for children_at_depth in node.children.values():
                    for child in children_at_depth.values():
                        yield from emit_lower(child, node.acc)

        g = emit(self.inner)
        try:
            out = next(g)
        except StopIteration:
            return None
        for a in g:
            out = out.merge(a)
        return out


# ------------------------------------------------------------------------------
# Internal helpers (no new classes added)
# ------------------------------------------------------------------------------

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

    for v in set(c1.keys()) | set(c2.keys()):
        map1 = c1.get(v, {})
        map2 = c2.get(v, {})
        if not map1 and not map2:
            continue

        out_for_v: Dict[int, Node] = {}

        def _accumulate(node: Node) -> None:
            d = getattr(node, "_max_depth")
            prev = out_for_v.get(d)
            out_for_v[d] = node if prev is None else merge_func(prev, node)

        for n in map1.values():
            _accumulate(n)
        for n in map2.values():
            _accumulate(n)

        if out_for_v:
            merged[v] = out_for_v

    return merged


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


def lower_to_upper(l: Lower[T], acc: Acc) -> Upper[T, Acc]:
    """
    Convert a Lower subtree into an Upper subtree with a uniform accumulator 'acc'.
    """
    children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for v, kids in l.children.items():
        bucket: Dict[int, Upper[T, Acc]] = {}
        for lchild in kids.values():
            up_child = lower_to_upper(lchild, acc)
            bucket[up_child._max_depth] = up_child
        if bucket:
            children[v] = bucket
    ub = UpperBranch(children=children, empty=(acc if l.empty else None))
    return try_promote(ub)

__all__ = ["LeveledGSS"]
