from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Generator, TypeVar, Iterator, Iterable
from collections import Counter, defaultdict

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
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, '_max_depth', depth)

    def _all_children(self) -> Generator[Upper[T, Acc], None, None]:
        """Returns an iterator over all child nodes."""
        for children_at_depth in self.children.values():
            yield from children_at_depth.values()


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
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
    children: Dict[T, Dict[int, Lower[T]]]
    empty: bool
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, '_max_depth', depth)

    def _all_children(self) -> Iterator[Lower[T]]:
        for v_children in self.children.values():
            yield from v_children.values()


@dataclass(frozen=True)
class LeveledGSSStats(Generic[T, Acc]):
    # Stack-level statistics
    total_stacks: int
    empty_stacks: int
    non_empty_stacks: int
    min_stack_length: Optional[int]
    max_stack_length: Optional[int]
    avg_stack_length: Optional[float]
    median_stack_length: Optional[float]
    length_histogram: Dict[int, int]
    top_values_distribution: Dict[T, int]
    top_values: Set[T]

    # Structure-level statistics (unique nodes/edges in the DAG)
    num_upperbranch_nodes: int
    num_interface_nodes: int
    num_lower_nodes: int
    total_unique_nodes: int

    upper_edges: int                      # UpperBranch -> (Upper | Interface)
    interface_to_lower_edges: int         # Interface -> Lower
    lower_edges: int                      # Lower -> Lower
    total_edges: int

    max_upper_depth: int                  # inner._max_depth at root (Upper tree)
    max_lower_depth: int                  # max _max_depth across Lower nodes

    # Value/accumulator coverage
    distinct_values_count: int
    distinct_values: Set[T]
    unique_accumulators_count: int
    unique_accumulators: Set[Acc]
    total_accumulator_instances: int
    accumulator_sharing_ratio: float

    # "Empty" flags / terminal interfaces
    num_upper_with_empty: int
    num_interfaces_with_empty: int
    num_lower_terminal_nodes: int
    num_interface_implicit_terminals: int  # Interface nodes that represent a terminal stack via acc (no children, empty=None)

    # Multi-depth slot metrics
    num_multi_depth_slots_upper: int       # Count of (UpperBranch node, value) pairs with >1 children at different depths
    num_multi_depth_slots_lower: int       # Same for the lower layer (Interface and Lower children)
    max_multiplicity_per_value_upper: int
    max_multiplicity_per_value_lower: int

    # Sharing/graph metrics
    average_in_degree: float               # average incoming edges across nodes with at least one incoming edge
    max_in_degree: int
    structural_sharing_factor: float       # edges / max(1, nodes - 1) — >1 implies sharing

    # Potential canonicalization insights (non-fatal)
    promotable_upper_nodes: int            # UpperBranch nodes that could be promoted to Interface

    def _fmt_subset(self, s: Set[Any], max_items: int = 10) -> str:
        if not s:
            return "{}"
        items = list(s)
        shown = ", ".join(repr(x) for x in items[:max_items])
        suffix = "" if len(items) <= max_items else ", ..."
        return "{" + shown + suffix + "}"

    def _fmt_hist(self, h: Dict[int, int], max_items: int = 12) -> str:
        if not h:
            return "{}"
        keys = sorted(h.keys())
        shown_pairs = []
        for k in keys[:max_items]:
            shown_pairs.append(f"{k}:{h[k]}")
        suffix = "" if len(keys) <= max_items else ", ..."
        return "{" + ", ".join(shown_pairs) + suffix + "}"

    def __str__(self) -> str:
        lines: List[str] = []
        lines.append("LeveledGSSStats")
        lines.append(f"- stacks: total={self.total_stacks}, empty={self.empty_stacks}, non_empty={self.non_empty_stacks}")
        lines.append(f"- lengths: min={self.min_stack_length}, max={self.max_stack_length}, avg={self.avg_stack_length}, median={self.median_stack_length}")
        lines.append(f"- length_histogram: {self._fmt_hist(self.length_histogram)}")
        lines.append(f"- top_values: {len(self.top_values)} distinct -> {self._fmt_subset(self.top_values)}")
        lines.append(f"- top_values_distribution (counts by top-of-stack value): size={len(self.top_values_distribution)}")
        # Try to display a small preview of top_values_distribution
        if self.top_values_distribution:
            sample_items = list(self.top_values_distribution.items())[:10]
            lines.append("  " + ", ".join(f"{repr(k)}:{v}" for k, v in sample_items) + (" ..." if len(self.top_values_distribution) > 10 else ""))

        lines.append("- structure:")
        lines.append(f"  nodes: UpperBranch={self.num_upperbranch_nodes}, Interface={self.num_interface_nodes}, Lower={self.num_lower_nodes}, total={self.total_unique_nodes}")
        lines.append(f"  edges: upper={self.upper_edges}, interface_to_lower={self.interface_to_lower_edges}, lower={self.lower_edges}, total={self.total_edges}")
        lines.append(f"  depths: max_upper_depth={self.max_upper_depth}, max_lower_depth={self.max_lower_depth}")

        lines.append("- values/accumulators:")
        lines.append(f"  distinct_values_count={self.distinct_values_count}, sample={self._fmt_subset(self.distinct_values)}")
        lines.append(f"  unique_accumulators_count={self.unique_accumulators_count} (physically stored)")
        lines.append(f"  total_accumulator_instances={self.total_accumulator_instances} (storage slots used)")
        lines.append(f"  total_stacks={self.total_stacks} (logical paths)")
        lines.append(f"  accumulator_sharing_ratio={self.accumulator_sharing_ratio:.4f} (unique_accs/total_stacks)")

        lines.append("- empties/terminals:")
        lines.append(f"  upper_with_empty={self.num_upper_with_empty} (nodes representing a true empty stack)")
        lines.append(f"  interfaces_with_empty={self.num_interfaces_with_empty} (nodes representing a true empty stack)")
        lines.append(f"  lower_terminal_nodes={self.num_lower_terminal_nodes} (nodes where a stack can end)")
        lines.append(f"  interface_implicit_terminals={self.num_interface_implicit_terminals} (interfaces with no children)")

        lines.append("- multi-depth slots:")
        lines.append(f"  num_multi_depth_slots_upper={self.num_multi_depth_slots_upper}, max_multiplicity_per_value_upper={self.max_multiplicity_per_value_upper}")
        lines.append(f"  num_multi_depth_slots_lower={self.num_multi_depth_slots_lower}, max_multiplicity_per_value_lower={self.max_multiplicity_per_value_lower}")

        lines.append("- sharing/graph:")
        lines.append(f"  average_in_degree={self.average_in_degree}, max_in_degree={self.max_in_degree}, structural_sharing_factor={self.structural_sharing_factor}")

        lines.append("- canonicalization opportunities (non-fatal):")
        lines.append(f"  promotable_upper_nodes={self.promotable_upper_nodes}")
        return "\n".join(lines)


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]

    def __post_init__(self):
        if os.environ.get("GSS_TESTER_VALIDATE"):
            self._validate()

    def _validate(self):
        self._validate_max_depths()
        self._validate_no_promotions()
        self._validate_populated_nodes()

    def _validate_max_depths(self) -> None:
        """Recursively validates that the `_max_depth` of each node is correct."""
        self._validate_depths_node(self.inner)

    def _validate_depths_node(self, node: Upper[T, Acc]) -> None:
        """Recursive helper for validating max_depth on Upper nodes."""
        if isinstance(node, Interface):
            # An Interface node has Lower children. We need to validate their depths recursively.
            def _validate_lower_recursively(n: Interface[T, Acc] | Lower[T]):
                for children_at_depth in n.children.values():
                    for depth, child in children_at_depth.items():
                        if depth != child._max_depth:
                            raise ValueError(
                                "LeveledGSS validation failed: incorrect max_depth for Lower child. "
                                f"Expected {depth}, got {child._max_depth}. Node: {n}"
                            )
                        _validate_lower_recursively(child)

            _validate_lower_recursively(node)
            return  # Leaf of the upper tree

        # It must be an UpperBranch
        # Recurse on children and check their depths
        for children_at_depth in node.children.values():
            for depth, child in children_at_depth.items():
                if depth != child._max_depth:
                    raise ValueError(
                        "LeveledGSS validation failed: incorrect max_depth for Upper child. "
                        f"Expected {depth}, got {child._max_depth}. Node: {node}"
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
            return  # Leaf of the upper tree

        # It must be an UpperBranch
        # First, recurse on children
        for children_at_depth in node.children.values():
            for child in children_at_depth.values():
                self._validate_promotion_node(child)

        # Now, check for promotion condition on the current node
        # All children are Interfaces. Gather all accumulators.
        accs: Set[Acc] = set()
        if node.empty is not None:
            accs.add(node.empty)

        for child in node._all_children():
            interface_child: Interface[T, Acc] = child  # type: ignore[assignment]
            accs.add(interface_child.acc)
            if interface_child.empty is not None:
                accs.add(interface_child.empty)

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
        # The root can be an empty UpperBranch, which represents an empty GSS.
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
        elif isinstance(node, Lower):
            if not node.children and not node.empty:
                raise ValueError(
                    "LeveledGSS validation failed: Lower node with no children and empty=False found. "
                    f"Node: {node}"
                )
            for children_at_depth in node.children.values():
                for child in children_at_depth.values():
                    self._validate_node_is_populated(child)

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
                    # Represent a leaf/end-of-stack using an UpperBranch with the accumulator in 'empty', then promote
                    nodes_for_v.append(try_promote(UpperBranch(children={}, empty=end_acc)))
                if sub:
                    nodes_for_v.append(build(sub))
                if nodes_for_v:
                    children[v] = {n._max_depth: n for n in nodes_for_v}
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
                            l_children[v_l] = {node_for_v._max_depth: node_for_v}
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
        if self.is_empty():
            return self
        if isinstance(self.inner, Interface):
            lower_node = Lower(children=self.inner.children, empty=self.inner.empty is not None)
            new_children = {value: {lower_node._max_depth: lower_node}}
            return LeveledGSS(Interface(children=new_children, acc=self.inner.acc, empty=None))
        else:
            return LeveledGSS(UpperBranch(children={value: {self.inner._max_depth: self.inner}}, empty=None))
    def pop(self) -> LeveledGSS[T, Acc]:
        upper_branch = self.inner if isinstance(self.inner, UpperBranch) else interface_to_upperbranch(self.inner)
        all_children = list(upper_branch._all_children())
        merged = reduce(merge_upper, all_children[1:], all_children[0]) if all_children else UpperBranch(children={}, empty=None)
        merged = try_promote(merged)
        return LeveledGSS(merged)
    def popn(self, n: int) -> LeveledGSS[T, Acc]:
        # Fast path: popn(0) is a no-op. Preserve identity to match the default
        # implementation's behavior (crucial for deterministic fuzzing).
        if n <= 0:
            return self
        all_children: Dict[int, Upper[T, Acc]] = {id(self.inner): self.inner}
        for _ in range(n):
            def to_upperbranch(upper: Upper[T, Acc]) -> UpperBranch[T, Acc]:
                return upper if isinstance(upper, UpperBranch) else interface_to_upperbranch(upper)
            all_children = {id(child): child for parent in all_children.values() for child in to_upperbranch(parent)._all_children()}
        all_children: List[Upper[T, Acc]] = list(all_children.values())
        merged = reduce(merge_upper, all_children[1:], all_children[0]) if all_children else UpperBranch(children={}, empty=None)
        merged = try_promote(merged)
        return LeveledGSS(merged)


    def is_empty(self) -> bool:
        # An empty GSS is represented by an UpperBranch with no children and no empty accumulator.
        if isinstance(self.inner, UpperBranch):
            return not self.inner.children and self.inner.empty is None
        # An Interface always represents at least one stack, as it has an accumulator.
        return False

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        if value is None:
            # Keep only empty stacks.
            if isinstance(self.inner, UpperBranch):
                empty_acc = self.inner.empty
            else:
                empty_acc = self.inner.empty
            new_root = UpperBranch(children={}, empty=empty_acc)
            # Promote to canonical form if applicable (avoids validator errors).
            return LeveledGSS(try_promote(new_root))

        # Keep stacks with `value` at the top.
        if isinstance(self.inner, UpperBranch):
            filtered_children = {value: self.inner.children[value]} if value in self.inner.children else {}
            return LeveledGSS(try_promote(UpperBranch(children=filtered_children, empty=None)))
        else:
            if value not in self.inner.children:
                return LeveledGSS(UpperBranch(children={}, empty=None))
            else:
                filtered_children = {value: self.inner.children[value]} if value in self.inner.children else {}
                return LeveledGSS(Interface(children=filtered_children, acc=self.inner.acc, empty=None))

    def isolate_many(self, values: Iterable[Optional[T]]) -> LeveledGSS[T, Acc]:
        values_set = set(values)

        new_empty: Optional[Acc] = None
        if None in values_set and isinstance(self.inner, (UpperBranch, Interface)):
            new_empty = self.inner.empty

        if isinstance(self.inner, UpperBranch):
            filtered_children = {
                v: kids for v, kids in self.inner.children.items() if v in values_set
            }
            new_inner = try_promote(UpperBranch(children=filtered_children, empty=new_empty))
            return LeveledGSS(new_inner)
        else:  # The root is an Interface
            filtered_children = {
                v: kids for v, kids in self.inner.children.items() if v in values_set
            }

            if filtered_children:
                # Children remain, so we build an Interface.
                new_inner = Interface(children=filtered_children, acc=self.inner.acc, empty=new_empty)
                return LeveledGSS(new_inner)
            else:
                # No children remain. The result only contains the empty stack (if requested).
                # This is represented by an UpperBranch.
                new_inner = try_promote(UpperBranch(children={}, empty=new_empty))
                return LeveledGSS(new_inner)

    def apply(self, func: Callable[[Acc], Acc], memo: Optional[Dict[int, Any]] = None) -> LeveledGSS[T, Acc]:
        if memo is None:
            memo = {}

        def transform(node: Upper[T, Acc]) -> Upper[T, Acc]:
            if id(node) in memo:
                return memo[id(node)]

            if isinstance(node, Interface):
                new_acc = func(node.acc)
                new_empty = func(node.empty) if node.empty is not None else None

                if new_acc == node.acc and new_empty == node.empty:
                    memo[id(node)] = node
                    return node

                res = Interface(children=node.children, acc=new_acc, empty=new_empty)
                memo[id(node)] = res
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
                memo[id(node)] = node
                return node

            res = UpperBranch(children=new_children, empty=new_empty)
            promoted = try_promote(res)
            memo[id(node)] = promoted
            return promoted

        return LeveledGSS(transform(self.inner))

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        memo: Dict[int, Optional[Upper[T, Acc]]] = {}

        def transform(node: Upper[T, Acc]) -> Optional[Upper[T, Acc]]:
            if id(node) in memo:
                return memo[id(node)]

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
                    promoted = try_promote(res)
                    memo[id(node)] = promoted
                    return promoted

                # keep_acc is True, but empty might have been pruned.
                res = Interface(children=node.children, acc=node.acc, empty=new_empty)
                memo[id(node)] = res
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
                        new_kids_for_v[new_child._max_depth] = new_child

                if len(new_kids_for_v) != len(kids):
                    child_map_changed = True

                if child_map_changed:
                    changed = True
                    if new_kids_for_v:
                        new_children[v] = new_kids_for_v
                else:
                    new_children[v] = kids  # Reuse

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

    def apply_and_prune(self, mutator: Callable[[Acc], Optional[Acc]]) -> LeveledGSS[T, Acc]:
        """
        Fast single-pass implementation of apply_and_prune for LeveledGSS.
        - mutator(acc) -> Optional[Acc]
            * return None to prune stacks carrying `acc`
            * return Acc (possibly unchanged) to keep/update stacks
        This fuses the behavior of `apply` and `prune` and minimizes reconstruction.
        """
        acc_cache: Dict[int, Optional[Acc]] = {}

        def mutate_acc(a: Acc) -> Optional[Acc]:
            k = id(a)
            if k in acc_cache:
                return acc_cache[k]
            r = mutator(a)
            acc_cache[k] = r
            return r

        memo: Dict[int, Optional[Upper[T, Acc]]] = {}

        def transform(node: Upper[T, Acc]) -> Optional[Upper[T, Acc]]:
            nid = id(node)
            if nid in memo:
                return memo[nid]

            if isinstance(node, Interface):
                # Mutate/prune the primary accumulator
                new_acc_opt = mutate_acc(node.acc)
                # Mutate/prune the explicit empty, if present
                new_empty_opt = mutate_acc(node.empty) if node.empty is not None else None

                keep_acc = new_acc_opt is not None
                keep_empty = new_empty_opt is not None

                if not keep_acc and not keep_empty:
                    memo[nid] = None
                    return None

                if not keep_acc and keep_empty:
                    # Acc is pruned, but the interface's explicit empty survives as a terminal stack.
                    # Promote the leaf to maintain canonical form (Interface with no children).
                    res = UpperBranch(children={}, empty=new_empty_opt)  # type: ignore[arg-type]
                    promoted = try_promote(res)
                    memo[nid] = promoted
                    return promoted

                # keep_acc is True
                new_acc = new_acc_opt  # type: ignore[assignment]
                # Detect if anything changed; children are reused verbatim.
                changed = (new_acc != node.acc) or (
                    (node.empty is not None and new_empty_opt != node.empty)
                )
                if not changed:
                    memo[nid] = node
                    return node

                res = Interface(children=node.children, acc=new_acc, empty=new_empty_opt)
                memo[nid] = res
                return res

            # UpperBranch
            if node.empty is not None:
                new_empty_opt = mutate_acc(node.empty)
                empty_changed = new_empty_opt != node.empty
            else:
                new_empty_opt = None
                empty_changed = False

            changed = empty_changed
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}

            for v, kids in node.children.items():
                new_kids_for_v: Dict[int, Upper[T, Acc]] = {}
                child_map_changed = False
                for d, child in kids.items():
                    new_child = transform(child)
                    if new_child is not child:
                        child_map_changed = True
                    if new_child is not None:
                        new_kids_for_v[new_child._max_depth] = new_child

                if len(new_kids_for_v) != len(kids):
                    child_map_changed = True

                if child_map_changed:
                    changed = True
                    if new_kids_for_v:
                        new_children[v] = new_kids_for_v
                else:
                    new_children[v] = kids  # Reuse

            if not changed:
                memo[nid] = node
                return node

            if not new_children and new_empty_opt is None:
                memo[nid] = None
                return None

            res = UpperBranch(children=new_children, empty=new_empty_opt)
            promoted = try_promote(res)
            memo[nid] = promoted
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

    def stats(self) -> LeveledGSSStats[T, Acc]:
        """
        Compute a comprehensive set of statistics for this LeveledGSS without flattening to stacks.
        Where possible, dynamic programming is used to avoid enumerating all stacks.
        """
        # --------------------
        # Helpers: histograms of path lengths (suffix lengths from a node to any terminal stack)
        # --------------------
        lower_hist_cache: Dict[int, Dict[int, int]] = {}
        upper_hist_cache: Dict[int, Dict[int, int]] = {}

        def _merge_hist_inplace(dst: Dict[int, int], src: Dict[int, int], offset: int = 0) -> None:
            if offset == 0:
                for k, v in src.items():
                    dst[k] = dst.get(k, 0) + v
            else:
                for k, v in src.items():
                    kk = k + offset
                    dst[kk] = dst.get(kk, 0) + v

        def _hist_lower(node: Lower[T]) -> Dict[int, int]:
            key = id(node)
            cached = lower_hist_cache.get(key)
            if cached is not None:
                return cached
            hist: Dict[int, int] = {}
            if node.empty:
                hist[0] = hist.get(0, 0) + 1
            for kids in node.children.values():
                for child in kids.values():
                    ch = _hist_lower(child)
                    _merge_hist_inplace(hist, ch, offset=1)
            lower_hist_cache[key] = hist
            return hist

        def _hist_upper(node: Upper[T, Acc]) -> Dict[int, int]:
            key = id(node)
            cached = upper_hist_cache.get(key)
            if cached is not None:
                return cached
            hist: Dict[int, int] = {}
            if isinstance(node, UpperBranch):
                if node.empty is not None:
                    hist[0] = hist.get(0, 0) + 1
                for kids in node.children.values():
                    for child in kids.values():
                        ch = _hist_upper(child)
                        _merge_hist_inplace(hist, ch, offset=1)
            else:
                # Interface
                if not node.children and node.empty is None:
                    # Terminal via acc
                    hist[0] = hist.get(0, 0) + 1
                if node.empty is not None:
                    # Terminal via explicit empty
                    hist[0] = hist.get(0, 0) + 1
                for kids in node.children.values():
                    for child in kids.values():
                        ch = _hist_lower(child)
                        _merge_hist_inplace(hist, ch, offset=1)
            upper_hist_cache[key] = hist
            return hist

        def _hist_total(h: Dict[int, int]) -> int:
            return sum(h.values())

        # Root histogram (suffix lengths from root)
        root_hist = _hist_upper(self.inner)
        total_stacks = _hist_total(root_hist)
        empty_stacks = root_hist.get(0, 0)
        non_empty_stacks = total_stacks - empty_stacks
        if total_stacks > 0:
            min_len = min(root_hist.keys())
            max_len = max(root_hist.keys())
            avg_len = sum(k * v for k, v in root_hist.items()) / total_stacks
            # median from histogram
            def median_from_hist(h: Dict[int, int], n: int) -> float:
                keys = sorted(h.keys())
                if n % 2 == 1:
                    k_idx = (n + 1) // 2
                    cum = 0
                    for L in keys:
                        cum += h[L]
                        if cum >= k_idx:
                            return float(L)
                    return float(keys[-1])
                else:
                    k1 = n // 2
                    k2 = k1 + 1
                    cum = 0
                    L1 = keys[0]
                    L2 = keys[0]
                    hit1 = False
                    for L in keys:
                        cum += h[L]
                        if not hit1 and cum >= k1:
                            L1 = L
                            hit1 = True
                        if cum >= k2:
                            L2 = L
                            break
                    return (L1 + L2) / 2.0
            median_len: Optional[float] = median_from_hist(root_hist, total_stacks)
        else:
            min_len = None
            max_len = None
            avg_len = None
            median_len = None

        # --------------------
        # Top-of-stack distribution
        # --------------------
        top_values_distribution: Dict[T, int] = {}
        if isinstance(self.inner, UpperBranch):
            for v, kids in self.inner.children.items():
                cnt = 0
                for child in kids.values():
                    cnt += _hist_total(_hist_upper(child))
                if cnt:
                    top_values_distribution[v] = cnt
        else:
            # Interface at root
            for v, kids in self.inner.children.items():
                cnt = 0
                for child in kids.values():
                    cnt += _hist_total(_hist_lower(child))
                if cnt:
                    top_values_distribution[v] = cnt
        top_values: Set[T] = set(top_values_distribution.keys())

        # --------------------
        # Structural scan (unique nodes/edges, depths, values, accumulators, multi-depth slots, sharing)
        # --------------------
        visited_upperbranch: Set[int] = set()
        visited_interface: Set[int] = set()
        visited_lower: Set[int] = set()

        num_upperbranch_nodes = 0
        num_interface_nodes = 0
        num_lower_nodes = 0

        upper_edges = 0
        interface_to_lower_edges = 0
        lower_edges = 0

        distinct_values: Set[T] = set()
        unique_accumulators: Set[Acc] = set()
        total_accumulator_instances = 0

        num_upper_with_empty = 0
        num_interfaces_with_empty = 0
        num_lower_terminal_nodes = 0
        num_interface_implicit_terminals = 0

        num_multi_depth_slots_upper = 0
        num_multi_depth_slots_lower = 0
        max_multiplicity_per_value_upper = 1
        max_multiplicity_per_value_lower = 1

        max_lower_depth = 0

        incoming_edges: Dict[int, int] = {}

        def bump_incoming(child_obj: Any) -> None:
            cid = id(child_obj)
            incoming_edges[cid] = incoming_edges.get(cid, 0) + 1

        # Promotable UpperBranch nodes
        promotable_upper_nodes = 0

        def is_promotable(node: UpperBranch[T, Acc]) -> bool:
            all_children = list(node._all_children())
            if not all_children:
                return False
            if not all(isinstance(c, Interface) for c in all_children):
                return False
            accs: Set[Acc] = set()
            if node.empty is not None:
                accs.add(node.empty)
            for c in all_children:
                ic: Interface[T, Acc] = c  # type: ignore[assignment]
                accs.add(ic.acc)
                if ic.empty is not None:
                    accs.add(ic.empty)
            return len(accs) == 1

        # Traverse graph collecting the structural stats
        # Use queues to ensure we visit each UNIQUE node once for node-level data,
        # but we still count edges (which are properties of parents) when visiting each unique parent.
        upper_queue: List[Upper[T, Acc]] = [self.inner]
        lower_queue: List[Lower[T]] = []

        while upper_queue:
            node = upper_queue.pop()
            if isinstance(node, UpperBranch):
                nid = id(node)
                if nid not in visited_upperbranch:
                    visited_upperbranch.add(nid)
                    num_upperbranch_nodes += 1
                    if node.empty is not None:
                        num_upper_with_empty += 1
                        unique_accumulators.add(node.empty)
                        total_accumulator_instances += 1
                    # edges and values
                    for v, kids in node.children.items():
                        distinct_values.add(v)
                        # multi-depth slot counting for upper layer
                        if len(kids) > 1:
                            num_multi_depth_slots_upper += 1
                            if len(kids) > max_multiplicity_per_value_upper:
                                max_multiplicity_per_value_upper = len(kids)
                        for child in kids.values():
                            upper_edges += 1
                            bump_incoming(child)
                            upper_queue.append(child)
                    # promotable?
                    if is_promotable(node):
                        promotable_upper_nodes += 1
            else:
                # Interface
                nid = id(node)
                if nid not in visited_interface:
                    visited_interface.add(nid)
                    num_interface_nodes += 1
                    unique_accumulators.add(node.acc)
                    total_accumulator_instances += 1
                    if node.empty is not None:
                        num_interfaces_with_empty += 1
                        unique_accumulators.add(node.empty)
                        total_accumulator_instances += 1
                    if not node.children and node.empty is None:
                        num_interface_implicit_terminals += 1
                    # edges to lower and values
                    for v, kids in node.children.items():
                        distinct_values.add(v)
                        # multi-depth slot counting for lower layer at interface boundary
                        if len(kids) > 1:
                            num_multi_depth_slots_lower += 1
                            if len(kids) > max_multiplicity_per_value_lower:
                                max_multiplicity_per_value_lower = len(kids)
                        for child in kids.values():
                            interface_to_lower_edges += 1
                            bump_incoming(child)
                            lower_queue.append(child)

        while lower_queue:
            node = lower_queue.pop()
            nid = id(node)
            if nid in visited_lower:
                continue
            visited_lower.add(nid)
            num_lower_nodes += 1
            if node.empty:
                num_lower_terminal_nodes += 1
            if node._max_depth > max_lower_depth:
                max_lower_depth = node._max_depth
            # edges and values
            for v, kids in node.children.items():
                distinct_values.add(v)
                # multi-depth at lower layer
                if len(kids) > 1:
                    num_multi_depth_slots_lower += 1
                    if len(kids) > max_multiplicity_per_value_lower:
                        max_multiplicity_per_value_lower = len(kids)
                for child in kids.values():
                    lower_edges += 1
                    bump_incoming(child)
                    lower_queue.append(child)

        total_unique_nodes = num_upperbranch_nodes + num_interface_nodes + num_lower_nodes
        total_edges = upper_edges + interface_to_lower_edges + lower_edges
        max_upper_depth = self.inner._max_depth
        distinct_values_count = len(distinct_values)
        unique_accumulators_count = len(unique_accumulators)

        # Accumulator counts
        if total_stacks > 0:
            accumulator_sharing_ratio = unique_accumulators_count / total_stacks
        else:
            accumulator_sharing_ratio = 1.0  # No stacks, so no redundancy.

        # Sharing metrics
        if incoming_edges:
            max_in_degree = max(incoming_edges.values())
            # average over nodes that actually have an incoming edge
            average_in_degree = sum(incoming_edges.values()) / len(incoming_edges)
        else:
            max_in_degree = 0
            average_in_degree = 0.0
        structural_sharing_factor = total_edges / float(max(1, total_unique_nodes - 1))

        return LeveledGSSStats(
            total_stacks=total_stacks,
            empty_stacks=empty_stacks,
            non_empty_stacks=non_empty_stacks,
            min_stack_length=min_len,
            max_stack_length=max_len,
            avg_stack_length=avg_len,
            median_stack_length=median_len,
            length_histogram=dict(sorted(root_hist.items())),
            top_values_distribution=top_values_distribution,
            top_values=top_values,
            num_upperbranch_nodes=num_upperbranch_nodes,
            num_interface_nodes=num_interface_nodes,
            num_lower_nodes=num_lower_nodes,
            total_unique_nodes=total_unique_nodes,
            upper_edges=upper_edges,
            interface_to_lower_edges=interface_to_lower_edges,
            lower_edges=lower_edges,
            total_edges=total_edges,
            max_upper_depth=max_upper_depth,
            max_lower_depth=max_lower_depth,
            distinct_values_count=distinct_values_count,
            distinct_values=distinct_values,
            unique_accumulators_count=unique_accumulators_count,
            unique_accumulators=unique_accumulators,
            total_accumulator_instances=total_accumulator_instances,
            accumulator_sharing_ratio=accumulator_sharing_ratio,
            num_upper_with_empty=num_upper_with_empty,
            num_interfaces_with_empty=num_interfaces_with_empty,
            num_lower_terminal_nodes=num_lower_terminal_nodes,
            num_interface_implicit_terminals=num_interface_implicit_terminals,
            num_multi_depth_slots_upper=num_multi_depth_slots_upper,
            num_multi_depth_slots_lower=num_multi_depth_slots_lower,
            max_multiplicity_per_value_upper=max_multiplicity_per_value_upper,
            max_multiplicity_per_value_lower=max_multiplicity_per_value_lower,
            average_in_degree=average_in_degree,
            max_in_degree=max_in_degree,
            structural_sharing_factor=structural_sharing_factor,
            promotable_upper_nodes=promotable_upper_nodes,
        )


Node = TypeVar("Node")

def _merge_optional_acc(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
    if a is None:
        return b
    if b is None:
        return a
    return a.merge(b)


def _merge_children_by_depth(
    c1: Dict[T, Dict[int, Node]],
    c2: Dict[T, Dict[int, Node]],
    merge_func: Callable[[Node, Node], Node],
) -> Dict[T, Dict[int, Node]]:
    merged_children: Dict[T, Dict[int, Node]] = {}
    all_vals = c1.keys() | c2.keys()
    for v in all_vals:
        nodes_by_depth: Dict[int, list[Node]] = defaultdict(list)
        children_c1 = c1.get(v, {}).items()
        children_c2 = c2.get(v, {}).items()
        for depth, child in chain(children_c1, children_c2):
            nodes_by_depth[depth].append(child)
        if not nodes_by_depth:
            continue
        v_out = {
            (merged := reduce(merge_func, nodes))._max_depth: merged
            for nodes in nodes_by_depth.values()
        }
        merged_children[v] = v_out
    return merged_children

def try_promote(node: UpperBranch[T, Acc]) -> Upper[T, Acc]:
    all_children = list(node._all_children())
    if not all_children:
        # Leaf UpperBranch: if it represents an explicit empty stack (empty is not None),
        # it can be represented canonically as an Interface with no children.
        if node.empty is not None:
            return Interface(children={}, acc=node.empty, empty=node.empty)
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
                v_map[lower._max_depth] = lower
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
            v_map[ci._max_depth] = ci
        if v_map:
            children[v] = v_map
    new_empty = it.empty
    if not it.children and new_empty is None:
        new_empty = it.acc
    return UpperBranch(children=children, empty=new_empty)

def merge_upper(u1: Upper[T, Acc], u2: Upper[T, Acc]) -> Upper[T, Acc]:
    if u1 is u2:
        return u1
    # If both are the same type, use the appropriate merge function
    if isinstance(u1, Interface) and isinstance(u2, Interface):
        return merge_interfaces(u1, u2)
    if isinstance(u1, UpperBranch) and isinstance(u2, UpperBranch):
        return merge_upperbranches(u1, u2)
    # Mixed types: convert Interface(s) to UpperBranch and merge
    ub1 = u1 if isinstance(u1, UpperBranch) else interface_to_upperbranch(u1)
    ub2 = u2 if isinstance(u2, UpperBranch) else interface_to_upperbranch(u2)
    return merge_upperbranches(ub1, ub2)  # type: ignore[arg-type]

def merge_upperbranches(a: UpperBranch[T, Acc], b: UpperBranch[T, Acc]) -> Upper[T, Acc]:
    if a is b:
        return a
    new_empty = _merge_optional_acc(a.empty, b.empty)
    merged_children = _merge_children_by_depth(a.children, b.children, merge_upper)
    return try_promote(UpperBranch(children=merged_children, empty=new_empty))

def merge_interfaces(a: Interface[T, Acc], b: Interface[T, Acc]) -> Upper[T, Acc]:
    if a.acc == b.acc:
        new_empty = _merge_optional_acc(a.empty, b.empty)
        merged_children = _merge_children_by_depth(a.children, b.children, merge_lower)
        return Interface(children=merged_children, acc=a.acc, empty=new_empty)
    if a.children is b.children:
        new_acc = a.acc.merge(b.acc)
        new_empty = _merge_optional_acc(a.empty, b.empty)
        return Interface(children=a.children, acc=new_acc, empty=new_empty)
    return merge_upperbranches(interface_to_upperbranch(a), interface_to_upperbranch(b))

def merge_lower(l1: Lower[T], l2: Lower[T]) -> Lower[T]:
    # Fast paths
    if l1 is l2:
        return l1
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
            v_map[up_child._max_depth] = up_child
        if v_map:
            children[v] = v_map
    ub = UpperBranch(children=children, empty=(acc if l.empty else None))
    return try_promote(ub)
