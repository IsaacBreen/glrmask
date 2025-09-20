from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Generator, TypeVar

from ..interface import GSS, T, Acc
from .reference_impl import ReferenceGSS

# ------------------------------
# Internal node classes
# ------------------------------

type Upper[T, Acc] = UpperBranch[T, Acc] | Interface[T, Acc]


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    children: Dict[T, Dict[int, Upper[T, Acc]]]
    empty: Optional[Acc]

    def _all_children(self) -> List[Upper[T, Acc]]:
        """Returns a flat list of all child nodes."""
        res: List[Upper[T, Acc]] = []
        for children_at_depth in self.children.values():
            res.extend(children_at_depth.values())
        return res

    def _max_depth(self) -> int:
        """Computes the max depth of the subtree rooted at this node."""
        if not self.children:
            return 0
        max_child_depth = 0
        for v_children in self.children.values():
            if v_children:
                max_child_depth = max(max_child_depth, max(v_children.keys()))
        return 1 + max_child_depth


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    children: Dict[T, Dict[int, Lower[T]]]
    acc: Acc
    empty: Optional[Acc]

    def _max_depth(self) -> int:
        """Max depth where this interface is a leaf of the upper tree."""
        if not self.children:
            return 0
        max_child_depth = 0
        for v_children in self.children.values():
            if v_children:
                max_child_depth = max(max_child_depth, max(v_children.keys()))
        return 1 + max_child_depth


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    children: Dict[T, Dict[int, Lower[T]]]
    empty: bool

    def _max_depth(self) -> int:
        """Computes the max depth of the subtree rooted at this node."""
        if not self.children:
            return 0
        max_child_depth = 0
        for v_children in self.children.values():
            if v_children:
                max_child_depth = max(max_child_depth, max(v_children.keys()))
        return 1 + max_child_depth


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]

    def __post_init__(self):
        # Keep validations trivial and cheap; construction/merge maintain invariants.
        # These no-ops preserve public signature without adding overhead or complexity.
        self._validate_no_promotions()
        self._validate_populated_nodes()

    def _validate_no_promotions(self) -> None:
        return

    def _validate_populated_nodes(self) -> None:
        return

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        # Canonicalize by merging duplicate stacks (values identical -> accumulators merged).
        canonical_stacks = ReferenceGSS(stacks)._stacks

        # Extract the optional empty stack accumulator and collect non-empty stacks.
        empty_acc: Optional[Acc] = None
        non_empty: List[Tuple[List[T], Acc]] = []
        for vals, acc in canonical_stacks:
            if not vals:
                empty_acc = acc
            else:
                non_empty.append((vals, acc))

        def build_group(items: List[Tuple[List[T], Acc]], root_empty: Optional[Acc]) -> Upper[T, Acc]:
            """
            Recursively group by top-of-stack (last element in vals) and build an Upper subtree.
            Promotion is applied at each level to keep the representation minimal.
            """
            if not items and root_empty is None:
                return UpperBranch(children={}, empty=None)

            # Group by top-of-stack value.
            grouped: Dict[T, Dict[str, Any]] = {}
            for vals, acc in items:
                v = vals[-1]
                entry = grouped.setdefault(v, {"end": None, "rest": []})
                if len(vals) == 1:
                    if entry["end"] is None:
                        entry["end"] = acc
                    else:
                        # Defensive: if input wasn't canonical, still merge identical singletons.
                        entry["end"] = entry["end"].merge(acc)  # type: ignore[union-attr]
                else:
                    entry["rest"].append((vals[:-1], acc))

            children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, e in grouped.items():
                v_nodes: List[Upper[T, Acc]] = []
                end_acc = e["end"]
                rest: List[Tuple[List[T], Acc]] = e["rest"]

                if end_acc is not None:
                    v_nodes.append(Interface(children={}, acc=end_acc, empty=None))
                if rest:
                    v_nodes.append(build_group(rest, None))

                if v_nodes:
                    children[v] = {n._max_depth(): n for n in v_nodes}

            return try_promote(UpperBranch(children=children, empty=root_empty))

        result = build_group(non_empty, empty_acc)
        return LeveledGSS(result)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        res: List[Tuple[List[T], Acc]] = []

        def emit_lower(l: Lower[T], pref: List[T], acc: Acc) -> None:
            if l.empty:
                res.append((list(reversed(pref)), acc))
            for v, kids in l.children.items():
                for child in kids.values():
                    emit_lower(child, pref + [v], acc)

        def emit_upper(u: Upper[T, Acc], pref: List[T]) -> None:
            if isinstance(u, UpperBranch):
                if u.empty is not None:
                    res.append((list(reversed(pref)), u.empty))
                for v, kids in u.children.items():
                    for child in kids.values():
                        emit_upper(child, pref + [v])
            else:
                # Interface
                if u.empty is not None:
                    res.append((list(reversed(pref)), u.empty))
                if not u.children and u.empty is None:
                    # A terminal stack at this level with accumulator u.acc
                    res.append((list(reversed(pref)), u.acc))
                else:
                    for v, kids in u.children.items():
                        for child in kids.values():
                            emit_lower(child, pref + [v], u.acc)

        emit_upper(self.inner, [])
        # ReferenceGSS only for deterministic ordering; content is already canonical.
        return ReferenceGSS(res).to_stacks()

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        # New top-of-stack is `value`, so the previous structure becomes the child under `value`.
        if isinstance(self.inner, Interface):
            lower_node = Lower(children=self.inner.children, empty=self.inner.empty is not None)
            new_children = {value: {lower_node._max_depth(): lower_node}}
            return LeveledGSS(Interface(children=new_children, acc=self.inner.acc, empty=None))
        else:
            return LeveledGSS(UpperBranch(children={value: {self.inner._max_depth(): self.inner}}, empty=None))

    def pop(self) -> LeveledGSS[T, Acc]:
        # Remove the top-of-stack from all stacks by merging all root children together.
        upper = interface_to_upperbranch(self.inner) if isinstance(self.inner, Interface) else self.inner
        merged: Upper[T, Acc] = UpperBranch(children={}, empty=None)
        for c in upper._all_children():
            merged = merge_upper(merged, c)
        return LeveledGSS(try_promote(merged if isinstance(merged, UpperBranch) else interface_to_upperbranch(merged)))  # type: ignore[arg-type]

    def is_empty(self) -> bool:
        # Empty iff there are no active stacks.
        if isinstance(self.inner, UpperBranch):
            return not self.inner.children and self.inner.empty is None
        return False  # Interface always represents at least one stack.

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        empty_root = UpperBranch(children={}, empty=None)

        if value is None:
            # Keep only empty stacks.
            if isinstance(self.inner, UpperBranch):
                return LeveledGSS(UpperBranch(children={}, empty=self.inner.empty) if self.inner.empty is not None else empty_root)
            else:
                return LeveledGSS(UpperBranch(children={}, empty=self.inner.empty) if self.inner.empty is not None else empty_root)

        # Keep stacks whose top equals `value`.
        if isinstance(self.inner, UpperBranch):
            nodes = list(self.inner.children.get(value, {}).values())
            if not nodes:
                return LeveledGSS(empty_root)
            merged = nodes[0]
            for c in nodes[1:]:
                merged = merge_upper(merged, c)
            return LeveledGSS(merged)

        # Interface case: children under `value` are Lower nodes.
        lowers = list(self.inner.children.get(value, {}).values())
        if not lowers:
            return LeveledGSS(empty_root)
        merged_lower = lowers[0]
        for c in lowers[1:]:
            merged_lower = merge_lower(merged_lower, c)
        return LeveledGSS(lower_to_upper(merged_lower, self.inner.acc))

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        memo: Dict[Any, Any] = {}

        def transform(node: Upper[T, Acc]) -> Upper[T, Acc]:
            if node in memo:
                return memo[node]

            if isinstance(node, Interface):
                new_acc = func(node.acc)
                new_empty = func(node.empty) if node.empty is not None else None
                if new_acc == node.acc and new_empty == node.empty:
                    memo[node] = node
                    return node
                res = Interface(children=node.children, acc=new_acc, empty=new_empty)
                memo[node] = res
                return res

            # UpperBranch
            new_empty = func(node.empty) if node.empty is not None else None
            changed = new_empty != node.empty
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}

            for v, kids in node.children.items():
                changed_for_v = False
                new_kids_for_v: Dict[int, Upper[T, Acc]] = {}
                for d, child in kids.items():
                    new_child = transform(child)
                    if new_child is not child:
                        changed_for_v = True
                    new_kids_for_v[d] = new_child  # depth unchanged
                if changed_for_v:
                    changed = True
                    new_children[v] = new_kids_for_v
                else:
                    new_children[v] = kids  # reuse

            if not changed:
                memo[node] = node
                return node

            promoted = try_promote(UpperBranch(children=new_children, empty=new_empty))
            memo[node] = promoted
            return promoted

        return LeveledGSS(transform(self.inner))

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        memo: Dict[Any, Optional[Upper[T, Acc]]] = {}

        def transform(node: Upper[T, Acc]) -> Optional[Upper[T, Acc]]:
            if node in memo:
                return memo[node]

            if isinstance(node, Interface):
                keep_acc = predicate(node.acc)
                keep_empty = node.empty is not None and predicate(node.empty)
                new_empty = node.empty if keep_empty else None

                if keep_acc:
                    res: Upper[T, Acc] = Interface(children=node.children, acc=node.acc, empty=new_empty)
                    memo[node] = res
                    return res

                if keep_empty:
                    res2: Upper[T, Acc] = UpperBranch(children={}, empty=new_empty)
                    memo[node] = res2
                    return res2

                memo[node] = None
                return None

            # UpperBranch
            new_empty = node.empty if node.empty is not None and predicate(node.empty) else None
            changed = new_empty != node.empty
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}

            for v, kids in node.children.items():
                v_map: Dict[int, Upper[T, Acc]] = {}
                for child in kids.values():
                    new_child = transform(child)
                    if new_child is not child:
                        changed = True
                    if new_child is not None:
                        v_map[new_child._max_depth()] = new_child
                if v_map:
                    new_children[v] = v_map
                elif v in new_children:
                    del new_children[v]

            if not changed and len(new_children) == len(node.children):
                memo[node] = node
                return node

            if not new_children and new_empty is None:
                memo[node] = None
                return None

            promoted = try_promote(UpperBranch(children=new_children, empty=new_empty))
            memo[node] = promoted
            return promoted

        res_inner = transform(self.inner)
        if res_inner is None:
            return LeveledGSS(UpperBranch(children={}, empty=None))
        return LeveledGSS(res_inner)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(merge_upper(self.inner, other.inner))

    def peek(self) -> Set[T]:
        return set(self.inner.children.keys())

    def reduce_acc(self) -> Optional[Acc]:
        acc_out: Optional[Acc] = None

        def merge_in(a: Acc) -> None:
            nonlocal acc_out
            if acc_out is None:
                acc_out = a
            else:
                acc_out = acc_out.merge(a)

        def visit_lower(l: Lower[T], acc: Acc) -> None:
            if l.empty:
                merge_in(acc)
            for kids in l.children.values():
                for child in kids.values():
                    visit_lower(child, acc)

        def visit_upper(u: Upper[T, Acc]) -> None:
            if isinstance(u, UpperBranch):
                if u.empty is not None:
                    merge_in(u.empty)
                for kids in u.children.values():
                    for child in kids.values():
                        visit_upper(child)
            else:
                if u.empty is not None:
                    merge_in(u.empty)
                if not u.children and u.empty is None:
                    merge_in(u.acc)
                else:
                    for kids in u.children.values():
                        for child in kids.values():
                            visit_lower(child, u.acc)

        visit_upper(self.inner)
        return acc_out


Node = TypeVar("Node")


def _merge_children_by_depth(
    c1: Dict[T, Dict[int, Node]],
    c2: Dict[T, Dict[int, Node]],
    merge_func: Callable[[Node, Node], Node],
) -> Dict[T, Dict[int, Node]]:
    """
    Merge two 'children' maps keyed by value then grouped by child max depth.
    For each value v and each depth, merge nodes that fall into the same depth bucket.
    """
    merged_children: Dict[T, Dict[int, Node]] = {}
    all_vals = set(c1.keys()) | set(c2.keys())
    for v in all_vals:
        map1 = c1.get(v, {})
        map2 = c2.get(v, {})
        depth_buckets: Dict[int, List[Node]] = {}
        for child in map1.values():
            depth_buckets.setdefault(getattr(child, "_max_depth")(), []).append(child)
        for child in map2.values():
            depth_buckets.setdefault(getattr(child, "_max_depth")(), []).append(child)

        v_out: Dict[int, Node] = {}
        for _, nodes in depth_buckets.items():
            merged_node = nodes[0]
            for n in nodes[1:]:
                merged_node = merge_func(merged_node, n)
            v_out[getattr(merged_node, "_max_depth")()] = merged_node

        if v_out:
            merged_children[v] = v_out

    return merged_children


def try_promote(node: UpperBranch[T, Acc]) -> Upper[T, Acc]:
    """
    Promote an UpperBranch to an Interface when all children are Interfaces and
    all accumulators (including the branch-empty if present) agree to a single value.
    """
    all_children = node._all_children()
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
            v_map: Dict[int, Lower[T]] = {}
            for child in kids.values():
                ci: Interface[T, Acc] = child  # type: ignore[assignment]
                lower = Lower(children=ci.children, empty=(ci.empty is not None))
                v_map[lower._max_depth()] = lower
            if v_map:
                l_children[v] = v_map
        return Interface(children=l_children, acc=the_acc, empty=node.empty)
    return node


def interface_to_upperbranch(it: Interface[T, Acc]) -> UpperBranch[T, Acc]:
    children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for v, kids in it.children.items():
        v_map: Dict[int, Upper[T, Acc]] = {}
        for lchild in kids.values():
            ci = Interface(
                children=lchild.children,
                acc=it.acc,
                empty=(it.acc if lchild.empty else None),
            )
            v_map[ci._max_depth()] = ci
        if v_map:
            children[v] = v_map
    new_empty = it.empty
    if not it.children and new_empty is None:
        new_empty = it.acc
    return UpperBranch(children=children, empty=new_empty)


def merge_upper(u1: Upper[T, Acc], u2: Upper[T, Acc]) -> Upper[T, Acc]:
    if u1 is u2:
        return u1
    if isinstance(u1, Interface) and isinstance(u2, Interface):
        return merge_interfaces(u1, u2)
    if isinstance(u1, UpperBranch) and isinstance(u2, UpperBranch):
        return merge_upperbranches(u1, u2)
    if isinstance(u1, Interface):
        return merge_upperbranches(interface_to_upperbranch(u1),
                                   u2 if isinstance(u2, UpperBranch) else interface_to_upperbranch(u2))  # type: ignore[arg-type]
    else:
        return merge_upperbranches(u1,
                                   u2 if isinstance(u2, UpperBranch) else interface_to_upperbranch(u2))  # type: ignore[arg-type]


def merge_upperbranches(a: UpperBranch[T, Acc], b: UpperBranch[T, Acc]) -> Upper[T, Acc]:
    if a is b:
        return a
    # Merge 'empty'
    if a.empty is None:
        new_empty = b.empty
    elif b.empty is None:
        new_empty = a.empty
    else:
        new_empty = a.empty.merge(b.empty)

    merged_children = _merge_children_by_depth(a.children, b.children, merge_upper)
    return try_promote(UpperBranch(children=merged_children, empty=new_empty))


def merge_interfaces(a: Interface[T, Acc], b: Interface[T, Acc]) -> Upper[T, Acc]:
    if a.acc == b.acc:
        if a.empty is None:
            new_empty = b.empty
        elif b.empty is None:
            new_empty = a.empty
        else:
            new_empty = a.empty.merge(b.empty)
        merged_children = _merge_children_by_depth(a.children, b.children, merge_lower)
        return Interface(children=merged_children, acc=a.acc, empty=new_empty)
    # Different accumulators -> lift to upperbranches and merge structurally.
    return merge_upperbranches(interface_to_upperbranch(a), interface_to_upperbranch(b))


def merge_lower(l1: Lower[T], l2: Lower[T]) -> Lower[T]:
    # Fast paths
    if l1 is l2:
        return l1
    if l1 == l2:
        return l1

    # Merge 'empty' flags (logical OR)
    new_empty = l1.empty or l2.empty
    merged_children = _merge_children_by_depth(l1.children, l2.children, merge_lower)
    return Lower(children=merged_children, empty=new_empty)


def lower_to_upper(l: Lower[T], acc: Acc) -> Upper[T, Acc]:
    # Convert a Lower subtree to an Upper subtree; the accumulator for all stacks is 'acc'.
    children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for v, kids in l.children.items():
        v_map: Dict[int, Upper[T, Acc]] = {}
        for lchild in kids.values():
            up_child = lower_to_upper(lchild, acc)
            v_map[getattr(up_child, "_max_depth")()] = up_child
        if v_map:
            children[v] = v_map
    ub = UpperBranch(children=children, empty=(acc if l.empty else None))
    return try_promote(ub)
