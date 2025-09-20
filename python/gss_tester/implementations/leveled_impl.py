from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Generator

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

    def _all_children(self) -> Generator[Upper[T, Acc], None, None]:
        """Returns an iterator over all child nodes."""
        for children_at_depth in self.children.values():
            yield from children_at_depth.values()

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
        """
        Computes the max depth of the subtree. For an Interface, this is based
        on the Lower children. An Interface is a leaf in the Upper tree.
        """
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
        self._validate_no_promotions()

    def _validate_no_promotions(self) -> None:
        """
        Recursively validates that no UpperBranch nodes can be promoted to an Interface.
        An UpperBranch can be promoted if all its children are Interfaces and they all
        (including the UpperBranch's own empty slot) represent the same single accumulator value.
        """
        if isinstance(self.inner, UpperBranch):
            self._validate_node(self.inner)

    def _validate_node(self, node: Upper[T, Acc]) -> None:
        """Recursive helper for validation."""
        if isinstance(node, Interface):
            # An Interface node has Lower children. We need to validate their depths recursively.
            def _validate_lower_recursively(n: Interface[T, Acc] | Lower[T]):
                for children_at_depth in n.children.values():
                    for depth, child in children_at_depth.items():
                        if depth != child._max_depth():
                            raise ValueError(
                                "LeveledGSS validation failed: incorrect max_depth for Lower child. "
                                f"Expected {depth}, got {child._max_depth()}. Node: {n}"
                            )
                        _validate_lower_recursively(child)

            _validate_lower_recursively(node)
            return  # Leaf of the upper tree

        # It must be an UpperBranch
        # First, recurse on children and check their depths
        for children_at_depth in node.children.values():
            for depth, child in children_at_depth.items():
                if depth != child._max_depth():
                    raise ValueError(
                        "LeveledGSS validation failed: incorrect max_depth for Upper child. "
                        f"Expected {depth}, got {child._max_depth()}. Node: {node}"
                    )
                self._validate_node(child)

        # Now, check for promotion condition on the current node
        all_children = list(node._all_children())

        if not all_children or not all(isinstance(child, Interface) for child in all_children):
            return  # Cannot promote

        # All children are Interfaces. Gather all accumulators.
        accs: Set[Acc] = set()
        if node.empty is not None:
            accs.add(node.empty)

        for child in all_children:
            interface_child: Interface[T, Acc] = child
            accs.add(interface_child.acc)
            if interface_child.empty is not None:
                accs.add(interface_child.empty)

        if len(accs) == 1:
            raise ValueError(
                "LeveledGSS validation failed: an UpperBranch can be promoted to an Interface, "
                f"indicating a non-canonical structure. Node: {node}"
            )

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        # Use ReferenceGSS to canonicalize stacks by merging accumulators.
        # We access _stacks directly to avoid the sorting done by to_stacks().
        canonical_stacks = ReferenceGSS(stacks)._stacks

        empty_acc: Optional[Acc] = None
        # A simple trie: { val: { "end": Optional[Acc], "sub": <subtrie> } }
        trie: Dict[T, Dict[str, Any]] = {}

        for vals, acc in canonical_stacks:
            if not vals:
                empty_acc = acc
                continue
            node = trie
            for i, v in enumerate(reversed(vals)):
                entry = node.setdefault(v, {"end": None, "sub": {}})
                if i == len(vals) - 1:
                    # Since input is canonical, there's no need to merge.
                    entry["end"] = acc
                else:
                    node = entry["sub"]

        def build(d: Dict[T, Dict[str, Any]], root_empty: Optional[Acc] = None) -> Upper[T, Acc]:
            # Build children recursively
            children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            all_child_nodes: List[Upper[T, Acc]] = []
            for v, e in d.items():
                nodes_for_v: List[Upper[T, Acc]] = []
                end_acc = e.get("end")
                sub = e.get("sub", {})
                if end_acc is not None:
                    nodes_for_v.append(Interface(children={}, acc=end_acc, empty=None))
                if sub:
                    nodes_for_v.append(build(sub))
                if nodes_for_v:
                    children[v] = {n._max_depth(): n for n in nodes_for_v}
                    all_child_nodes.extend(nodes_for_v)

            # Check for promotion
            if all(isinstance(child, Interface) for child in all_child_nodes):
                accs: Set[Acc] = set()
                for c in all_child_nodes:
                    # This must be an Interface, based on the check above.
                    accs.add(c.acc)
                    if c.empty is not None:
                        accs.add(c.empty)

                if root_empty is not None:
                    accs.add(root_empty)

                if len(accs) <= 1:
                    the_acc = accs.pop() if accs else None
                    if the_acc is None:
                        # This is a truly empty GSS.
                        return UpperBranch(children={}, empty=None)

                    def build_lower(sub_d: Dict[T, Dict[str, Any]]) -> Lower[T]:
                        l_children: Dict[T, Dict[int, Lower[T]]] = {}
                        for v_l, e_l in sub_d.items():
                            sub_l = e_l.get("sub", {})
                            has_end = e_l.get("end") is not None
                            sub_lower = build_lower(sub_l) if sub_l else Lower(children={}, empty=False)
                            node_for_v = Lower(children=sub_lower.children, empty=has_end)
                            l_children[v_l] = {node_for_v._max_depth(): node_for_v}
                        return Lower(children=l_children, empty=False)

                    lower_tree = build_lower(d)
                    return Interface(
                        children=lower_tree.children,
                        acc=the_acc,
                        empty=root_empty
                    )

            return UpperBranch(children=children, empty=root_empty)

        return LeveledGSS(build(trie, empty_acc))

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        res: List[Tuple[List[T], Acc]] = []

        def dfs_lower(l: Lower[T], pref: List[T], acc: Acc) -> None:
            if l.empty:
                res.append((list(reversed(pref)), acc))
            for v, kids in l.children.items():
                for child in kids.values():
                    dfs_lower(child, pref + [v], acc)

        def dfs_upper(u: Upper[T, Acc], pref: List[T]) -> None:
            if isinstance(u, UpperBranch):
                if u.empty is not None:
                    res.append((list(reversed(pref)), u.empty))
                for v, kids in u.children.items():
                    for child in kids.values():
                        dfs_upper(child, pref + [v])
            elif isinstance(u, Interface):
                # The interface's `empty` slot is for a stack ending at `pref`.
                if u.empty is not None:
                    res.append((list(reversed(pref)), u.empty))

                if not u.children:
                    # If there are no lower children, this interface represents the end of a stack
                    # with accumulator u.acc, but only if a stack for `pref` wasn't already added via `u.empty`.
                    if u.empty is None:
                        res.append((list(reversed(pref)), u.acc))
                else:
                    # The interface's `children` are for stacks extending `pref`.
                    # All these stacks share accumulator `u.acc`.
                    for v, kids in u.children.items():
                        for child in kids.values():
                            dfs_lower(child, pref + [v], u.acc)

        dfs_upper(self.inner, [])

        # The internal representation is canonical. We use ReferenceGSS to sort
        # the stacks into a canonical list representation.
        return ReferenceGSS(res).to_stacks()

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        if isinstance(self.inner, Interface):
            lower_node = Lower(children=self.inner.children, empty=self.inner.empty is not None)
            new_children = {value: {lower_node._max_depth(): lower_node}}
            return LeveledGSS(Interface(children=new_children, acc=self.inner.acc, empty=None))
        else:
            return LeveledGSS(UpperBranch(children={value: {self.inner._max_depth(): self.inner}}, empty=None))
    def pop(self) -> LeveledGSS[T, Acc]:
        if isinstance(self.inner, Interface):
            upper_branch = interface_to_upperbranch(self.inner)
        else:
            upper_branch = self.inner
        all_children = list(upper_branch._all_children())
        merged = all_children[0]
        for c in all_children[1:]:
            merged = merge_upper(merged, c)
        return LeveledGSS(merged)

    def is_empty(self) -> bool:
        # An empty GSS is represented by an UpperBranch with no children and no empty accumulator.
        if isinstance(self.inner, UpperBranch):
            return not self.inner.children and self.inner.empty is None
        # An Interface always represents at least one stack, as it has an accumulator.
        return False

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        empty_gss_inner = UpperBranch(children={}, empty=None)

        if value is None:
            # Keep only empty stacks.
            empty_acc: Optional[Acc] = None
            if isinstance(self.inner, UpperBranch):
                empty_acc = self.inner.empty
            elif isinstance(self.inner, Interface):
                empty_acc = self.inner.empty

            if empty_acc is not None:
                return LeveledGSS(UpperBranch(children={}, empty=empty_acc))
            return LeveledGSS(empty_gss_inner)

        # Keep stacks with `value` at the top.
        if isinstance(self.inner, UpperBranch):
            children_to_merge = self.inner.children.get(value, {}).values()
            if not children_to_merge:
                return LeveledGSS(empty_gss_inner)

            it = iter(children_to_merge)
            merged = next(it)
            for c in it:
                merged = merge_upper(merged, c)
            return LeveledGSS(merged)

        # self.inner is an Interface
        children_to_merge = self.inner.children.get(value, {}).values()
        if not children_to_merge:
            return LeveledGSS(empty_gss_inner)

        it = iter(children_to_merge)
        merged_lower = next(it)
        for c in it:
            merged_lower = merge_lower(merged_lower, c)

        # Convert the merged Lower subtree back to an Upper tree with the Interface's accumulator.
        upper_node = lower_to_upper(merged_lower, self.inner.acc)
        return LeveledGSS(upper_node)

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

            # It's an UpperBranch
            new_empty = func(node.empty) if node.empty is not None else None

            changed = new_empty != node.empty
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}

            for v, kids in node.children.items():
                new_kids_for_v: Dict[int, Upper[T, Acc]] = {}
                any_child_changed_for_v = False
                for d, child in kids.items():
                    new_child = transform(child)
                    if new_child is not child:
                        any_child_changed_for_v = True
                    # Depth does not change in apply, so we can reuse `d`.
                    new_kids_for_v[d] = new_child

                if any_child_changed_for_v:
                    changed = True
                    new_children[v] = new_kids_for_v
                else:
                    new_children[v] = kids  # Reuse child dict

            if not changed:
                memo[node] = node
                return node

            res = UpperBranch(children=new_children, empty=new_empty)
            promoted = try_promote(res)
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

                if keep_acc and new_empty == node.empty:
                    memo[node] = node
                    return node

                if not keep_acc and not keep_empty:
                    memo[node] = None
                    return None

                if not keep_acc and keep_empty:
                    res = UpperBranch(children={}, empty=new_empty)
                    memo[node] = res
                    return res

                # keep_acc is True, but empty might have been pruned.
                res = Interface(children=node.children, acc=node.acc, empty=new_empty)
                memo[node] = res
                return res

            # It's an UpperBranch
            new_empty = node.empty if node.empty is not None and predicate(node.empty) else None
            changed = new_empty != node.empty

            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, kids in node.children.items():
                new_kids_for_v: Dict[int, Upper[T, Acc]] = {}
                child_map_changed = False
                for d, child in kids.items():
                    new_child = transform(child)
                    if new_child is not child:
                        child_map_changed = True
                    if new_child is not None:
                        new_kids_for_v[new_child._max_depth()] = new_child

                if len(new_kids_for_v) != len(kids):
                    child_map_changed = True

                if child_map_changed:
                    changed = True
                    if new_kids_for_v:
                        new_children[v] = new_kids_for_v
                else:
                    new_children[v] = kids  # Reuse

            if not changed:
                memo[node] = node
                return node

            if not new_children and new_empty is None:
                memo[node] = None
                return None

            res = UpperBranch(children=new_children, empty=new_empty)
            promoted = try_promote(res)
            memo[node] = promoted
            return promoted

        res_inner = transform(self.inner)
        if res_inner is None:
            return LeveledGSS(UpperBranch(children={}, empty=None))
        return LeveledGSS(res_inner)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(merge_upper(self.inner, other.inner))
    def peek(self) -> Set[T]:
        if isinstance(self.inner, Interface):
            return set(self.inner.children.keys())
        else:
            return set(self.inner.children.keys())

    def reduce_acc(self) -> Optional[Acc]:
        def generate_accs_lower(node: Lower[T], acc: Acc) -> Generator[Acc, None, None]:
            if node.empty:
                yield acc
            for children_at_depth in node.children.values():
                for child in children_at_depth.values():
                    yield from generate_accs_lower(child, acc)

        def generate_accs(node: Upper[T, Acc]) -> Generator[Acc, None, None]:
            if isinstance(node, UpperBranch):
                if node.empty is not None:
                    yield node.empty
                for children_at_depth in node.children.values():
                    for child in children_at_depth.values():
                        yield from generate_accs(child)
            elif isinstance(node, Interface):
                if node.empty is not None:
                    yield node.empty

                # The case where the interface itself is a stack end.
                if not node.children and node.empty is None:
                    yield node.acc

                for children_at_depth in node.children.values():
                    for child in children_at_depth.values():
                        yield from generate_accs_lower(child, node.acc)

        gen = generate_accs(self.inner)
        try:
            reduced_acc = next(gen)
        except StopIteration:
            return None

        for acc in gen:
            reduced_acc = reduced_acc.merge(acc)
        return reduced_acc


def try_promote(node: UpperBranch[T, Acc]) -> Upper[T, Acc]:
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

    # Merge children grouped by value and depth
    merged_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    all_vals = set(a.children.keys()) | set(b.children.keys())
    for v in all_vals:
        amap = a.children.get(v, {})
        bmap = b.children.get(v, {})
        depth_buckets: Dict[int, List[Upper[T, Acc]]] = {}
        for child in amap.values():
            depth_buckets.setdefault(child._max_depth(), []).append(child)
        for child in bmap.values():
            depth_buckets.setdefault(child._max_depth(), []).append(child)
        v_out: Dict[int, Upper[T, Acc]] = {}
        for d, nodes in depth_buckets.items():
            merged_node = nodes[0]
            for n in nodes[1:]:
                merged_node = merge_upper(merged_node, n)
            v_out[merged_node._max_depth()] = merged_node
        if v_out:
            merged_children[v] = v_out

    return try_promote(UpperBranch(children=merged_children, empty=new_empty))

def merge_interfaces(a: Interface[T, Acc], b: Interface[T, Acc]) -> Upper[T, Acc]:
    if a.acc == b.acc:
        if a.empty is None:
            new_empty = b.empty
        elif b.empty is None:
            new_empty = a.empty
        else:
            new_empty = a.empty.merge(b.empty)
        merged_children: Dict[T, Dict[int, Lower[T]]] = {}
        all_vals = set(a.children.keys()) | set(b.children.keys())
        for v in all_vals:
            amap = a.children.get(v, {})
            bmap = b.children.get(v, {})
            depth_buckets: Dict[int, List[Lower[T]]] = {}
            for child in amap.values():
                depth_buckets.setdefault(child._max_depth(), []).append(child)
            for child in bmap.values():
                depth_buckets.setdefault(child._max_depth(), []).append(child)
            v_out: Dict[int, Lower[T]] = {}
            for d, nodes in depth_buckets.items():
                merged_node = nodes[0]
                for n in nodes[1:]:
                    merged_node = merge_lower(merged_node, n)
                v_out[merged_node._max_depth()] = merged_node
            if v_out:
                merged_children[v] = v_out
        return Interface(children=merged_children, acc=a.acc, empty=new_empty)
    return merge_upperbranches(interface_to_upperbranch(a), interface_to_upperbranch(b))

def merge_lower(l1: Lower[T], l2: Lower[T]) -> Lower[T]:
    # Fast paths
    if l1 is l2:
        return l1
    if l1 == l2:
        return l1

    # Merge 'empty' flags (logical OR)
    new_empty = l1.empty or l2.empty

    # Merge children grouped by value and by child max depth
    merged_children: Dict[T, Dict[int, Lower[T]]] = {}
    all_vals = set(l1.children.keys()) | set(l2.children.keys())
    for v in all_vals:
        l1map = l1.children.get(v, {})
        l2map = l2.children.get(v, {})

        depth_buckets: Dict[int, List[Lower[T]]] = {}
        for child in l1map.values():
            depth_buckets.setdefault(child._max_depth(), []).append(child)
        for child in l2map.values():
            depth_buckets.setdefault(child._max_depth(), []).append(child)

        v_out: Dict[int, Lower[T]] = {}
        for _, nodes in depth_buckets.items():
            merged_node = nodes[0]
            for n in nodes[1:]:
                merged_node = merge_lower(merged_node, n)
            # Key by the resulting node's max depth to keep depth-index invariant
            v_out[merged_node._max_depth()] = merged_node

        if v_out:
            merged_children[v] = v_out

    return Lower(children=merged_children, empty=new_empty)


def lower_to_upper(l: Lower[T], acc: Acc) -> Upper[T, Acc]:
    # Convert a Lower subtree to an Upper subtree; the accumulator for all stacks is 'acc'.
    children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for v, kids in l.children.items():
        v_map: Dict[int, Upper[T, Acc]] = {}
        for lchild in kids.values():
            up_child = lower_to_upper(lchild, acc)
            v_map[up_child._max_depth()] = up_child
        if v_map:
            children[v] = v_map
    ub = UpperBranch(children=children, empty=(acc if l.empty else None))
    return try_promote(ub)
