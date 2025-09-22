from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from ..interface import GSS, T, Acc, NewAcc
from .reference_impl import ReferenceGSS

# ------------------------------
# Internal Lower node (acc-agnostic)
# ------------------------------


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    """
    A simple, accumulator-agnostic trie that stores sets of stacks.
    Orientation matches LeveledGSS Lower:
      - Each edge key is a top-of-stack value.
      - The path order is from top-of-stack towards the bottom.
      - 'empty=True' marks that the current prefix corresponds to a complete stack.

    children:
      Dict[value, Dict[depth, Lower]]
        For each edge label 'value', we may keep multiple children bucketed by their max-depth.
    empty:
      True if the current node denotes a complete stack at this position.
    _max_depth:
      The maximum depth of any subtree under this node. 0 for leaf (no children).
    """
    children: Dict[T, Dict[int, "Lower[T]"]]
    empty: bool
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = (
            max(child._max_depth for child in _all_children_lower(self.children)) + 1
            if self.children
            else 0
        )
        object.__setattr__(self, "_max_depth", depth)


# ------------------------------
# Utility helpers (Lower ops)
# ------------------------------

NodeT = TypeVar("NodeT")


def _all_children_lower(
    children: Dict[T, Dict[int, Lower[T]]]
) -> Iterable[Lower[T]]:
    for v_children in children.values():
        for child in v_children.values():
            yield child


def _merge_children_by_depth(
    c1: Dict[T, Dict[int, NodeT]],
    c2: Dict[T, Dict[int, NodeT]],
    merge_func: Callable[[NodeT, NodeT], NodeT],
    get_depth: Callable[[NodeT], int],
) -> Dict[T, Dict[int, NodeT]]:
    """
    Merges two dictionaries of children, grouping by value, then by node depth.
    Nodes of the same value and same depth are merged using merge_func.
    """
    merged_children: Dict[T, Dict[int, NodeT]] = {}
    all_vals = set(c1.keys()) | set(c2.keys())
    for v in all_vals:
        map1 = c1.get(v, {})
        map2 = c2.get(v, {})
        depth_buckets: Dict[int, List[NodeT]] = {}
        for child in map1.values():
            depth_buckets.setdefault(get_depth(child), []).append(child)
        for child in map2.values():
            depth_buckets.setdefault(get_depth(child), []).append(child)

        v_out: Dict[int, NodeT] = {}
        for _, nodes in depth_buckets.items():
            merged_node = nodes[0]
            for n in nodes[1:]:
                merged_node = merge_func(merged_node, n)
            v_out[get_depth(merged_node)] = merged_node

        if v_out:
            merged_children[v] = v_out

    return merged_children


def merge_lower(l1: Lower[T], l2: Lower[T]) -> Lower[T]:
    # Fast paths
    if l1 is l2:
        return l1
    if l1 == l2:
        return l1

    new_empty = l1.empty or l2.empty
    merged_children = _merge_children_by_depth(
        l1.children, l2.children, merge_lower, lambda n: n._max_depth  # type: ignore[arg-type]
    )
    return Lower(children=merged_children, empty=new_empty)


def push_lower(root: Lower[T], value: T) -> Lower[T]:
    """
    Push a value onto all stacks represented by this Lower root.
    Effectively wraps the root under a new root whose only edge is `value`.
    """
    return Lower(children={value: {root._max_depth: root}}, empty=False)


def pop_lower(root: Lower[T]) -> Lower[T]:
    """
    Pop the top value from all stacks represented by this Lower root.
    - Empty stacks (root.empty == True) produce no output (they are discarded).
    - The popped result is the union of all child subtrees.
    """
    # Merge all children across all values/depths.
    it = iter(_all_children_lower(root.children))
    try:
        first = next(it)
    except StopIteration:
        # No children -> no stacks remain after pop
        return Lower(children={}, empty=False)
    acc = first
    for child in it:
        acc = merge_lower(acc, child)
    return acc


def isolate_lower(root: Lower[T], value: Optional[T]) -> Lower[T]:
    """
    Keep only stacks at this root that have `value` at the top.
    If value is None, keep only the empty stack (if present).
    """
    if value is None:
        # Keep only the empty stack marker at this node.
        return Lower(children={}, empty=root.empty)
    # Keep only the branch for `value`, if any.
    kept = root.children.get(value, {})
    return Lower(children={value: kept} if kept else {}, empty=False)


def lower_is_empty(root: Lower[T]) -> bool:
    """
    Returns True iff the Lower represents no stacks at all.
    """
    return (not root.empty) and (not root.children)


def dfs_collect_stacks(
    root: Lower[T], acc: Acc, out: List[Tuple[List[T], Acc]], pref: Optional[List[T]] = None
) -> None:
    """
    Collects (stack, acc) pairs from a Lower trie into `out`.
    The stack is returned in natural order [bottom...top].
    """
    if pref is None:
        pref = []
    if root.empty:
        out.append((list(reversed(pref)), acc))
    for v, kids in root.children.items():
        for child in kids.values():
            dfs_collect_stacks(child, acc, out, pref + [v])


def build_lower_from_stacks(stacks: List[List[T]]) -> Lower[T]:
    """
    Build a Lower trie from a list of stack value lists (for a single accumulator).
    """
    # Trie node structure: {"end": bool, "sub": {value: node}}
    root: Dict[str, Any] = {"end": False, "sub": {}}

    for vals in stacks:
        if not vals:
            root["end"] = True
            continue
        node = root
        rev = list(reversed(vals))
        for i, v in enumerate(rev):
            sub = node["sub"].setdefault(v, {"end": False, "sub": {}})
            if i == len(rev) - 1:
                sub["end"] = True
            else:
                node = sub

    def to_lower(n: Dict[str, Any]) -> Lower[T]:
        l_children: Dict[T, Dict[int, Lower[T]]] = {}
        for v, child in n["sub"].items():
            lower_child = to_lower(child)
            l_children[v] = {lower_child._max_depth: lower_child}
        return Lower(children=l_children, empty=bool(n["end"]))

    return to_lower(root)


# ------------------------------
# Public SimpleGSS implementation
# ------------------------------


@dataclass(frozen=True, eq=True)
class SimpleGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A simple Graph-Structured Stack that stores a mapping from accumulator -> Lower trie.
    The trie ('Lower') is completely accumulator-agnostic and represents sets of stacks.
    All cross-accumulator canonicalization (merging duplicate stacks with merged accumulators)
    is performed by converting to a ReferenceGSS when needed.

    Invariant we maintain after "canonicalizing" operations:
      - The sets of stacks across different accumulator keys are disjoint.
    """
    acc_trees: Dict[Acc, Lower[T]]

    # ---------------
    # Constructors
    # ---------------

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> "SimpleGSS[T, Acc]":
        """
        Build a SimpleGSS from explicit stacks. We first canonicalize with ReferenceGSS
        to merge identical stacks' accumulators, then group by accumulator and build
        per-acc Lower tries.
        """
        # Canonicalize stacks by merging duplicates
        canonical = ReferenceGSS(stacks)._stacks  # merged but not necessarily sorted
        return cls._from_merged_stacks(canonical)

    @classmethod
    def _from_merged_stacks(cls, merged_stacks: List[Tuple[List[T], Acc]]) -> "SimpleGSS[T, Acc]":
        """
        Build from stacks that are already merged w.r.t identical stack contents.
        """
        by_acc: Dict[Acc, List[List[T]]] = {}
        for vals, acc in merged_stacks:
            by_acc.setdefault(acc, []).append(vals)

        acc_trees: Dict[Acc, Lower[T]] = {}
        for acc, stacks_for_acc in by_acc.items():
            root = build_lower_from_stacks(stacks_for_acc)
            if not lower_is_empty(root):
                acc_trees[acc] = root

        return SimpleGSS(acc_trees=acc_trees)

    # ---------------
    # Conversions
    # ---------------

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """
        Returns a canonical, sorted list of stacks by delegating to ReferenceGSS for sorting
        and any final merging (should be no-ops in canonical states).
        """
        pairs: List[Tuple[List[T], Acc]] = []
        for acc, root in self.acc_trees.items():
            dfs_collect_stacks(root, acc, pairs)
        return ReferenceGSS(pairs).to_stacks()

    # ---------------
    # Core operations
    # ---------------

    def push(self, value: T) -> "SimpleGSS[T, Acc]":
        """
        Push value onto all active stacks. If this GSS is empty, returns itself.
        Pushing maintains disjointness across accumulators; no canonicalization needed.
        """
        if self.is_empty():
            return self
        new_map: Dict[Acc, Lower[T]] = {}
        for acc, root in self.acc_trees.items():
            new_map[acc] = push_lower(root, value)
        return SimpleGSS(acc_trees=new_map)

    def pop(self) -> "SimpleGSS[T, Acc]":
        """
        Pop from all non-empty stacks. Empty stacks are discarded.
        Popping can create duplicates across accumulators, so we canonicalize afterwards.
        """
        if self.is_empty():
            return self
        temp_map: Dict[Acc, Lower[T]] = {}
        for acc, root in self.acc_trees.items():
            new_root = pop_lower(root)
            if not lower_is_empty(new_root):
                temp_map[acc] = new_root

        # Canonicalize across accumulators by rebuilding via ReferenceGSS
        pairs: List[Tuple[List[T], Acc]] = []
        for acc, root in temp_map.items():
            dfs_collect_stacks(root, acc, pairs)
        canonical = ReferenceGSS(pairs)._stacks
        return SimpleGSS._from_merged_stacks(canonical)

    def is_empty(self) -> bool:
        return not self.acc_trees

    def isolate(self, value: Optional[T]) -> "SimpleGSS[T, Acc]":
        """
        Keep only stacks whose top equals `value`. If value is None, keep only empty stacks.
        """
        if self.is_empty():
            return self

        if value is None:
            # Keep only empty stacks; this may create cross-acc duplicates, so canonicalize via ReferenceGSS.
            pairs: List[Tuple[List[T], Acc]] = []
            for acc, root in self.acc_trees.items():
                if root.empty:
                    pairs.append(([], acc))
            canonical = ReferenceGSS(pairs)._stacks
            return SimpleGSS._from_merged_stacks(canonical)

        # value is not None: filter each acc's root to only that top value.
        new_map: Dict[Acc, Lower[T]] = {}
        for acc, root in self.acc_trees.items():
            filtered_root = isolate_lower(root, value)
            if not lower_is_empty(filtered_root):
                new_map[acc] = filtered_root
        return SimpleGSS(acc_trees=new_map)

    def apply(self, func: Callable[[Acc], NewAcc]) -> "SimpleGSS[T, NewAcc]":
        """
        Apply a function to each accumulator. If two accumulators map to the same new value,
        merge their tries.
        """
        if self.is_empty():
            return SimpleGSS(acc_trees={})

        new_map: Dict[NewAcc, Lower[T]] = {}
        for acc, root in self.acc_trees.items():
            new_acc = func(acc)
            if new_acc in new_map:
                new_map[new_acc] = merge_lower(new_map[new_acc], root)
            else:
                new_map[new_acc] = root
        # No cross-acc duplicates can appear here since stack sets were disjoint already.
        return SimpleGSS(acc_trees=new_map) # type: ignore[arg-type]

    def prune(self, predicate: Callable[[Acc], bool]) -> "SimpleGSS[T, Acc]":
        """
        Remove all stacks whose accumulator does not satisfy predicate. Since predicate
        depends only on the accumulator, we remove whole subtries per-acc.
        """
        if self.is_empty():
            return self
        new_map: Dict[Acc, Lower[T]] = {
            acc: root for acc, root in self.acc_trees.items() if predicate(acc)
        }
        return SimpleGSS(acc_trees=new_map)

    def merge(self, other: "SimpleGSS[T, Acc]") -> "SimpleGSS[T, Acc]":
        """
        Merge two GSSs. We first union per-acc tries (merging on matching acc),
        then canonicalize across accumulators to merge any duplicate stacks with
        merged accumulators.
        """
        if self is other:
            return self
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        temp_map: Dict[Acc, Lower[T]] = dict(self.acc_trees)
        for acc, root in other.acc_trees.items():
            if acc in temp_map:
                temp_map[acc] = merge_lower(temp_map[acc], root)
            else:
                temp_map[acc] = root

        # Canonicalize across accumulators (in case both sides had identical stacks under different accs)
        pairs: List[Tuple[List[T], Acc]] = []
        for acc, root in temp_map.items():
            dfs_collect_stacks(root, acc, pairs)
        canonical = ReferenceGSS(pairs)._stacks
        return SimpleGSS._from_merged_stacks(canonical)

    # ---------------
    # Observers
    # ---------------

    def peek(self) -> Set[T]:
        """
        Return the set of all values at the top of any stack.
        """
        tops: Set[T] = set()
        for root in self.acc_trees.values():
            tops.update(root.children.keys())
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        """
        Merge the accumulators across all active stacks into a single Acc.
        Returns None if there are no active stacks.
        """
        if self.is_empty():
            return None
        it = iter(self.acc_trees.keys())
        try:
            acc = next(it)
        except StopIteration:
            return None
        for a in it:
            acc = acc.merge(a)  # type: ignore[assignment]
        return acc
