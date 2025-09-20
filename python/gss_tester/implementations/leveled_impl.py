from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Generator, TypeVar

from ..interface import GSS, T, Acc
from .reference_impl import ReferenceGSS

# ------------------------------
# Internal node classes and type aliases
# ------------------------------

type Upper[T, Acc] = UpperBranch[T, Acc] | Interface[T, Acc]
Node = TypeVar("Node")


def _max_depth_from_children(children: Dict[Any, Dict[int, Any]]) -> int:
    """
    Computes the max depth for a node given its children mapping:
    children: value -> { depth_of_child: child_node }
    """
    if not children:
        return 0
    max_child_depth = 0
    for depth_map in children.values():
        if depth_map:
            max_child_depth = max(max_child_depth, max(depth_map.keys()))
    return 1 + max_child_depth


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    """
    Upper tree node that branches by value and holds either:
    - children: value -> depth -> (UpperBranch | Interface)
    - empty: Optional accumulator for a stack that ends at this prefix
    """
    children: Dict[T, Dict[int, Upper[T, Acc]]]
    empty: Optional[Acc]

    def _all_children(self) -> Generator[Upper[T, Acc], None, None]:
        for depth_map in self.children.values():
            yield from depth_map.values()

    def _max_depth(self) -> int:
        return _max_depth_from_children(self.children)


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    """
    Interface node separating the upper and lower trees.
    It stores a single accumulator that applies to all stacks below,
    and optionally an 'empty' accumulator if a stack ends at this prefix.
    """
    children: Dict[T, Dict[int, 'Lower[T]']]
    acc: Acc
    empty: Optional[Acc]

    def _max_depth(self) -> int:
        # Depth for an Interface is based on the depths of Lower children.
        return _max_depth_from_children(self.children)


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    """
    Lower tree node that stores only shape (no accumulator values).
    'empty' indicates whether a stack ends at this prefix (True/False).
    """
    children: Dict[T, Dict[int, 'Lower[T]']]
    empty: bool

    def _max_depth(self) -> int:
        return _max_depth_from_children(self.children)


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A leveled (upper/lower) implementation of Graph-Structured Stacks.
    The representation ensures canonical form through:
    - Depth-indexed children maps (the second key equals the child's max depth).
    - Promotion: an UpperBranch may be replaced by an Interface when all children
      are Interfaces with a single accumulator (including consideration of empties).
    - Validation hooks in __post_init__.
    """
    inner: Upper[T, Acc]

    def __post_init__(self):
        self._validate_max_depths()
        self._validate_no_promotions()
        self._validate_populated_nodes()

    # ------------------------------
    # Validation helpers
    # ------------------------------

    def _validate_max_depths(self) -> None:
        """Recursively validates that the `_max_depth` of each node is correct."""
        self._validate_depths_node(self.inner)

    def _validate_depths_node(self, node: Upper[T, Acc]) -> None:
        """Recursive helper for validating max_depth on Upper nodes."""
        if isinstance(node, Interface):
            # Validate Lower depths recursively under the interface.
            def _validate_lower_recursively(n: Interface[T, Acc] | Lower[T]) -> None:
                for depth_map in n.children.values():
                    for depth, child in depth_map.items():
                        if depth != child._max_depth():
                            raise ValueError(
                                "LeveledGSS validation failed: incorrect max_depth for Lower child. "
                                f"Expected {depth}, got {child._max_depth()}. Node: {n}"
                            )
                        _validate_lower_recursively(child)

            _validate_lower_recursively(node)
            return

        # UpperBranch: validate children and recurse
        for depth_map in node.children.values():
            for depth, child in depth_map.items():
                if depth != child._max_depth():
                    raise ValueError(
                        "LeveledGSS validation failed: incorrect max_depth for Upper child. "
                        f"Expected {depth}, got {child._max_depth()}. Node: {node}"
                    )
                self._validate_depths_node(child)

    def _validate_no_promotions(self) -> None:
        """
        Recursively validates that no UpperBranch nodes can be promoted to an Interface.
        An UpperBranch can be promoted if all its children are Interfaces and they all
        (including the UpperBranch's own empty slot) represent the same single accumulator value.
        """
        if isinstance(self.inner, UpperBranch):
            self._validate_promotion_node(self.inner)

    def _validate_promotion_node(self, node: Upper[T, Acc]) -> None:
        """Recursive helper for promotion validation."""
        if isinstance(node, Interface):
            return

        # Recurse first
        for child in node._all_children():
            self._validate_promotion_node(child)

        # Check promotion condition
        all_children = list(node._all_children())
        if not all_children or not all(isinstance(c, Interface) for c in all_children):
            return

        accs: Set[Acc] = set()
        if node.empty is not None:
            accs.add(node.empty)
        for c in all_children:
            ic: Interface[T, Acc] = c  # type: ignore[assignment]
            accs.add(ic.acc)
            if ic.empty is not None:
                accs.add(ic.empty)

        if len(accs) == 1:
            raise ValueError(
                "LeveledGSS validation failed: an UpperBranch can be promoted to an Interface, "
                f"indicating a non-canonical structure. Node: {node}"
            )

    def _validate_populated_nodes(self) -> None:
        """
        Validates that every node represents at least one stack, with the
        exception of the root UpperBranch for an empty GSS.
        """
        # The root can be an empty UpperBranch (representing an empty GSS).
        if isinstance(self.inner, UpperBranch) and not self.inner.children and self.inner.empty is None:
            return
        self._validate_node_is_populated(self.inner)

    def _validate_node_is_populated(self, node: Upper[T, Acc] | Lower[T]) -> None:
        """Recursive helper for validation."""
        if isinstance(node, UpperBranch):
            if not node.children and node.empty is None:
                raise ValueError(
                    "LeveledGSS validation failed: UpperBranch with no children and no empty accumulator "
                    f"found in a non-root position. Node: {node}"
                )
            for depth_map in node.children.values():
                for child in depth_map.values():
                    self._validate_node_is_populated(child)

        elif isinstance(node, Interface):
            if not node.children and node.empty is None:
                raise ValueError(
                    "LeveledGSS validation failed: Interface with no children and no empty accumulator found. "
                    f"Node: {node}"
                )
            for depth_map in node.children.values():
                for child in depth_map.values():
                    self._validate_node_is_populated(child)

        else:  # Lower
            if not node.children and not node.empty:
                raise ValueError(
                    "LeveledGSS validation failed: Lower node with no children and empty=False found. "
                    f"Node: {node}"
                )
            for depth_map in node.children.values():
                for child in depth_map.values():
                    self._validate_node_is_populated(child)

    # ------------------------------
    # Construction from explicit stacks
    # ------------------------------

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        """
        Build a leveled representation from explicit stacks. Uses ReferenceGSS to
        canonicalize (merge duplicate stacks) before building.
        """
        # Use ReferenceGSS to canonicalize stacks by merging accumulators.
        canonical_stacks = ReferenceGSS(stacks)._stacks  # Access internal to avoid sorting

        # Build a simple trie keyed by reversed stack (top-to-bottom in traversal here).
        # Structure: { value: { "end": Optional[Acc], "sub": <subtrie> } }
        root_empty: Optional[Acc] = None
        trie: Dict[T, Dict[str, Any]] = {}

        for values, acc in canonical_stacks:
            if not values:
                root_empty = acc
                continue
            node = trie
            for i, v in enumerate(reversed(values)):
                entry = node.setdefault(v, {"end": None, "sub": {}})
                if i == len(values) - 1:
                    # End of the path for this stack
                    entry["end"] = acc
                else:
                    node = entry["sub"]

        def build_lower_from_trie(subtrie: Dict[T, Dict[str, Any]]) -> Lower[T]:
            """
            Convert a trie dictionary (below some interface) into a Lower subtree.
            For each entry:
              - 'end' indicates the stack ends after current value.
              - 'sub' continues the Lower subtree.
            """
            l_children: Dict[T, Dict[int, Lower[T]]] = {}
            for v, e in subtrie.items():
                sub = e.get("sub", {})
                has_end = e.get("end") is not None
                # Build lower for children (if any). Even if no sub, maintain the shape.
                sub_lower = build_lower_from_trie(sub) if sub else Lower(children={}, empty=False)
                # Construct current node combining sub_lower's children with the 'end' flag at this level.
                node_for_v = Lower(children=sub_lower.children, empty=has_end)
                l_children[v] = {node_for_v._max_depth(): node_for_v}
            return Lower(children=l_children, empty=False)

        def build_upper_from_trie(d: Dict[T, Dict[str, Any]], empty_at_root: Optional[Acc]) -> Upper[T, Acc]:
            """
            Convert the trie rooted at this point into an Upper subtree.
            Initially produces UpperBranch nodes, then tries promotion to Interface
            when possible.
            """
            # Construct children for this level
            children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            child_nodes: List[Upper[T, Acc]] = []

            for v, e in d.items():
                nodes_for_v: List[Upper[T, Acc]] = []
                end_acc = e.get("end")
                sub = e.get("sub", {})

                if end_acc is not None:
                    nodes_for_v.append(Interface(children={}, acc=end_acc, empty=None))
                if sub:
                    nodes_for_v.append(build_upper_from_trie(sub, None))

                if nodes_for_v:
                    depth_map: Dict[int, Upper[T, Acc]] = {n._max_depth(): n for n in nodes_for_v}
                    children[v] = depth_map
                    child_nodes.extend(nodes_for_v)

            # If all children are Interfaces and all accumulators unify (including empties),
            # build a single Interface with a Lower subtree.
            if child_nodes and all(isinstance(c, Interface) for c in child_nodes):
                accs: Set[Acc] = set()
                if empty_at_root is not None:
                    accs.add(empty_at_root)
                for c in child_nodes:
                    ic: Interface[T, Acc] = c  # type: ignore[assignment]
                    accs.add(ic.acc)
                    if ic.empty is not None:
                        accs.add(ic.empty)

                if len(accs) <= 1:
                    the_acc = next(iter(accs)) if accs else None
                    if the_acc is None:
                        # Truly empty GSS
                        return UpperBranch(children={}, empty=None)

                    lower_tree = build_lower_from_trie(d)
                    return Interface(children=lower_tree.children, acc=the_acc, empty=empty_at_root)

            # Otherwise, remain an UpperBranch
            return UpperBranch(children=children, empty=empty_at_root)

        return LeveledGSS(build_upper_from_trie(trie, root_empty))

    # ------------------------------
    # Interface implementation
    # ------------------------------

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """Flatten the structure to explicit stacks, then canonicalize ordering via ReferenceGSS."""
        res: List[Tuple[List[T], Acc]] = []

        def dfs_lower(l: Lower[T], prefix: List[T], acc: Acc) -> None:
            if l.empty:
                res.append((list(reversed(prefix)), acc))
            for v, depth_map in l.children.items():
                for child in depth_map.values():
                    dfs_lower(child, prefix + [v], acc)

        def dfs_upper(u: Upper[T, Acc], prefix: List[T]) -> None:
            if isinstance(u, UpperBranch):
                if u.empty is not None:
                    res.append((list(reversed(prefix)), u.empty))
                for v, depth_map in u.children.items():
                    for child in depth_map.values():
                        dfs_upper(child, prefix + [v])

            else:  # Interface
                if u.empty is not None:
                    res.append((list(reversed(prefix)), u.empty))

                if not u.children:
                    # The interface itself is a stack end if no lower children and no explicit empty.
                    if u.empty is None:
                        res.append((list(reversed(prefix)), u.acc))
                else:
                    for v, depth_map in u.children.items():
                        for child in depth_map.values():
                            dfs_lower(child, prefix + [v], u.acc)

        dfs_upper(self.inner, [])
        return ReferenceGSS(res).to_stacks()

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        """
        Pushes a value onto all active stack heads.
        For an Interface, convert its 'empty' condition at the child level into the Lower.empty flag.
        """
        if isinstance(self.inner, Interface):
            lower_node = Lower(children=self.inner.children, empty=self.inner.empty is not None)
            new_children = {value: {lower_node._max_depth(): lower_node}}
            return LeveledGSS(Interface(children=new_children, acc=self.inner.acc, empty=None))

        # UpperBranch
        return LeveledGSS(UpperBranch(children={value: {self.inner._max_depth(): self.inner}}, empty=None))

    def pop(self) -> LeveledGSS[T, Acc]:
        """
        For all active stacks, create new stacks by removing the top value.
        Empty stacks do not contribute to pop.
        """
        root: UpperBranch[T, Acc]
        if isinstance(self.inner, Interface):
            root = interface_to_upperbranch(self.inner)
        else:
            root = self.inner

        merged: Upper[T, Acc] = UpperBranch(children={}, empty=None)
        for child in root._all_children():
            merged = merge_upper(merged, child)
        merged = try_promote(merged if isinstance(merged, UpperBranch) else interface_to_upperbranch(merged))  # type: ignore[arg-type]
        return LeveledGSS(merged)

    def is_empty(self) -> bool:
        # An empty GSS is an UpperBranch with no children and no empty accumulator.
        return isinstance(self.inner, UpperBranch) and not self.inner.children and self.inner.empty is None

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        """
        Keeps only the stacks that have `value` at the top.
        If `value` is None, keeps only the empty stacks.
        """
        empty_inner = UpperBranch(children={}, empty=None)

        if value is None:
            # Keep only empty stacks.
            empty_acc: Optional[Acc]
            if isinstance(self.inner, UpperBranch):
                empty_acc = self.inner.empty
            else:
                empty_acc = self.inner.empty

            if empty_acc is not None:
                return LeveledGSS(UpperBranch(children={}, empty=empty_acc))
            return LeveledGSS(empty_inner)

        # Filter stacks with `value` at the top.
        if isinstance(self.inner, UpperBranch):
            children_to_merge = list(self.inner.children.get(value, {}).values())
            if not children_to_merge:
                return LeveledGSS(empty_inner)

            merged: Upper[T, Acc] = children_to_merge[0]
            for c in children_to_merge[1:]:
                merged = merge_upper(merged, c)
            return LeveledGSS(merged)

        # Inner is Interface
        children_to_merge = list(self.inner.children.get(value, {}).values())
        if not children_to_merge:
            return LeveledGSS(empty_inner)

        merged_lower = children_to_merge[0]
        for c in children_to_merge[1:]:
            merged_lower = merge_lower(merged_lower, c)

        # Convert merged Lower subtree back to an Upper subtree with the Interface's accumulator.
        upper_node = lower_to_upper(merged_lower, self.inner.acc)
        return LeveledGSS(upper_node)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        """
        Applies a function to each accumulator, preserving structure when unchanged.
        """
        memo: Dict[int, Any] = {}

        def transform(node: Upper[T, Acc]) -> Upper[T, Acc]:
            cached = memo.get(id(node))
            if cached is not None:
                return cached

            if isinstance(node, Interface):
                new_acc = func(node.acc)
                new_empty = func(node.empty) if node.empty is not None else None

                if new_acc == node.acc and new_empty == node.empty:
                    memo[id(node)] = node
                    return node

                res = Interface(children=node.children, acc=new_acc, empty=new_empty)
                memo[id(node)] = res
                return res

            # UpperBranch
            new_empty = func(node.empty) if node.empty is not None else None
            changed = new_empty != node.empty

            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, depth_map in node.children.items():
                # Depth is unchanged in apply(), so we can reuse keys.
                new_map: Dict[int, Upper[T, Acc]] = {}
                child_changed_for_v = False
                for d, child in depth_map.items():
                    new_child = transform(child)
                    if new_child is not child:
                        child_changed_for_v = True
                    new_map[d] = new_child

                if child_changed_for_v:
                    changed = True
                    new_children[v] = new_map
                else:
                    new_children[v] = depth_map  # Reuse

            if not changed:
                memo[id(node)] = node
                return node

            res = UpperBranch(children=new_children, empty=new_empty)
            promoted = try_promote(res)
            memo[id(node)] = promoted
            return promoted

        return LeveledGSS(transform(self.inner))

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        """
        Removes stacks from the GSS based on a predicate on their accumulator.
        If `predicate(acc)` returns False, the stack is removed.
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

                if keep_acc and new_empty == node.empty:
                    memo[id(node)] = node
                    return node

                if not keep_acc and not keep_empty:
                    memo[id(node)] = None
                    return None

                if not keep_acc and keep_empty:
                    res = UpperBranch(children={}, empty=new_empty)
                    memo[id(node)] = res
                    return res

                # keep_acc is True; keep children and possibly pruned empty.
                res = Interface(children=node.children, acc=node.acc, empty=new_empty)
                memo[id(node)] = res
                return res

            # UpperBranch
            new_empty = node.empty if node.empty is not None and predicate(node.empty) else None
            changed = new_empty != node.empty

            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, depth_map in node.children.items():
                new_map: Dict[int, Upper[T, Acc]] = {}
                child_map_changed = False

                for _, child in depth_map.items():
                    new_child = transform(child)
                    if new_child is not child:
                        child_map_changed = True
                    if new_child is not None:
                        new_map[new_child._max_depth()] = new_child

                if len(new_map) != len(depth_map):
                    child_map_changed = True

                if child_map_changed:
                    changed = True
                    if new_map:
                        new_children[v] = new_map
                else:
                    new_children[v] = depth_map  # Reuse

            if not changed:
                memo[id(node)] = node
                return node

            if not new_children and new_empty is None:
                memo[id(node)] = None
                return None

            res = UpperBranch(children=new_children, empty=new_empty)
            promoted = try_promote(res)
            memo[id(node)] = promoted
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
        """
        Merges the accumulators of all active stacks into a single optional value.
        Returns None if there are no active stacks.
        """

        def generate_accs_lower(node: Lower[T], acc: Acc) -> Generator[Acc, None, None]:
            if node.empty:
                yield acc
            for depth_map in node.children.values():
                for child in depth_map.values():
                    yield from generate_accs_lower(child, acc)

        def generate_accs(node: Upper[T, Acc]) -> Generator[Acc, None, None]:
            if isinstance(node, UpperBranch):
                if node.empty is not None:
                    yield node.empty
                for depth_map in node.children.values():
                    for child in depth_map.values():
                        yield from generate_accs(child)
            else:  # Interface
                if node.empty is not None:
                    yield node.empty
                if not node.children and node.empty is None:
                    yield node.acc
                for depth_map in node.children.values():
                    for child in depth_map.values():
                        yield from generate_accs_lower(child, node.acc)

        gen = generate_accs(self.inner)
        try:
            reduced = next(gen)
        except StopIteration:
            return None

        for a in gen:
            reduced = reduced.merge(a)
        return reduced


# ------------------------------
# Helper functions
# ------------------------------

def _merge_children_by_depth(
    c1: Dict[T, Dict[int, Node]],
    c2: Dict[T, Dict[int, Node]],
    merge_func: Callable[[Node, Node], Node],
) -> Dict[T, Dict[int, Node]]:
    """
    Merge two children maps:
      value -> { depth_of_child: child_node }
    Children having the same value and depth are merged using merge_func, and
    children with the same value but different depths are grouped and merged
    bucket-wise so that one child remains per resultant depth.
    """
    merged: Dict[T, Dict[int, Node]] = {}
    all_vals = set(c1.keys()) | set(c2.keys())

    for v in all_vals:
        map1 = c1.get(v, {})
        map2 = c2.get(v, {})

        # Bucket children by their own max depth
        buckets: Dict[int, List[Node]] = {}
        for child in map1.values():
            buckets.setdefault(child._max_depth(), []).append(child)
        for child in map2.values():
            buckets.setdefault(child._max_depth(), []).append(child)

        out_map: Dict[int, Node] = {}
        for _, nodes in buckets.items():
            merged_node = nodes[0]
            for n in nodes[1:]:
                merged_node = merge_func(merged_node, n)
            out_map[merged_node._max_depth()] = merged_node

        if out_map:
            merged[v] = out_map

    return merged


def try_promote(node: UpperBranch[T, Acc]) -> Upper[T, Acc]:
    """
    If the UpperBranch has only Interface children and all accumulators (including
    node.empty and Interface empties) are identical or absent, promote to a single Interface.
    """
    children = list(node._all_children())
    if not children or not all(isinstance(c, Interface) for c in children):
        return node

    accs: Set[Acc] = set()
    if node.empty is not None:
        accs.add(node.empty)
    for c in children:
        ic: Interface[T, Acc] = c  # type: ignore[assignment]
        accs.add(ic.acc)
        if ic.empty is not None:
            accs.add(ic.empty)

    if len(accs) <= 1:
        the_acc = next(iter(accs)) if accs else None
        if the_acc is None:
            return UpperBranch(children={}, empty=None)

        l_children: Dict[T, Dict[int, Lower[T]]] = {}
        for v, depth_map in node.children.items():
            v_map: Dict[int, Lower[T]] = {}
            for child in depth_map.values():
                ic: Interface[T, Acc] = child  # type: ignore[assignment]
                lower = Lower(children=ic.children, empty=(ic.empty is not None))
                v_map[lower._max_depth()] = lower
            if v_map:
                l_children[v] = v_map

        return Interface(children=l_children, acc=the_acc, empty=node.empty)

    return node


def interface_to_upperbranch(it: Interface[T, Acc]) -> UpperBranch[T, Acc]:
    """
    Convert an Interface node to an UpperBranch by lifting its Lower children back to Interfaces.
    """
    children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for v, depth_map in it.children.items():
        v_map: Dict[int, Upper[T, Acc]] = {}
        for lchild in depth_map.values():
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
        # Interface with no lower children and no empty means the interface itself ends a stack.
        new_empty = it.acc
    return UpperBranch(children=children, empty=new_empty)


def merge_upper(u1: Upper[T, Acc], u2: Upper[T, Acc]) -> Upper[T, Acc]:
    """
    Merge two Upper nodes, preserving canonical form and applying promotion when possible.
    """
    if u1 is u2:
        return u1
    if isinstance(u1, Interface) and isinstance(u2, Interface):
        return merge_interfaces(u1, u2)
    if isinstance(u1, UpperBranch) and isinstance(u2, UpperBranch):
        return merge_upperbranches(u1, u2)

    # Mixed types: normalize to UpperBranch for merging
    ub1 = interface_to_upperbranch(u1) if isinstance(u1, Interface) else u1
    ub2 = interface_to_upperbranch(u2) if isinstance(u2, Interface) else u2
    return merge_upperbranches(ub1, ub2)


def merge_upperbranches(a: UpperBranch[T, Acc], b: UpperBranch[T, Acc]) -> Upper[T, Acc]:
    """
    Merge two UpperBranch nodes, merging empties and children by depth and value.
    """
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
    """
    Merge two Interfaces. If they share the same accumulator, stay as an Interface.
    Otherwise, convert both to UpperBranches and merge at that level.
    """
    if a.acc == b.acc:
        if a.empty is None:
            new_empty = b.empty
        elif b.empty is None:
            new_empty = a.empty
        else:
            new_empty = a.empty.merge(b.empty)

        merged_children = _merge_children_by_depth(a.children, b.children, merge_lower)
        return Interface(children=merged_children, acc=a.acc, empty=new_empty)

    # Different accumulators: normalize to UpperBranch and merge at upper level
    return merge_upperbranches(interface_to_upperbranch(a), interface_to_upperbranch(b))


def merge_lower(l1: Lower[T], l2: Lower[T]) -> Lower[T]:
    """
    Merge two Lower nodes, OR-ing 'empty' flags and merging children by depth and value.
    """
    # Fast paths
    if l1 is l2:
        return l1
    if l1 == l2:
        return l1

    new_empty = l1.empty or l2.empty
    merged_children = _merge_children_by_depth(l1.children, l2.children, merge_lower)
    return Lower(children=merged_children, empty=new_empty)


def lower_to_upper(l: Lower[T], acc: Acc) -> Upper[T, Acc]:
    """
    Convert a Lower subtree to an Upper subtree; the same accumulator 'acc' applies to all stacks below.
    """
    children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for v, depth_map in l.children.items():
        v_map: Dict[int, Upper[T, Acc]] = {}
        for lchild in depth_map.values():
            up_child = lower_to_upper(lchild, acc)
            v_map[up_child._max_depth()] = up_child
        if v_map:
            children[v] = v_map

    ub = UpperBranch(children=children, empty=(acc if l.empty else None))
    return try_promote(ub)
