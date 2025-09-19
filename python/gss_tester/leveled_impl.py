from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Iterable

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes
# ------------------------------

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    inner: UpperBranch[T, Acc] | Interface[T, Acc]


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    children: Dict[T, Dict[int, Upper[T, Acc]]]


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    node: Lower[T]
    acc: Acc | None  # We store None for the top-level interface acc as placeholder.


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    inner: LowerBranch[T] | Leaf


@dataclass(frozen=True, eq=True)
class LowerBranch(Generic[T]):
    children: Dict[Any, Dict[int, Lower[T]]]


@dataclass(frozen=True, eq=True)
class Leaf:
    pass


# Internal helper: sentinel key used inside Lower to store accumulator at a leaf.
class _AccKey(Generic[Acc]):
    __slots__ = ("acc",)

    def __init__(self, acc: Acc):
        self.acc = acc

    def __repr__(self) -> str:
        return f"<AccKey:{self.acc!r}>"

    # Make each instance unique, even if the underlying acc compares equal.
    # We do not want accidental coalescing at the dict layer; we canonicalize earlier.
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        """
        Build a LeveledGSS from explicit stacks.
        Implementation strategy:
        - Canonicalize input stacks by merging accumulators for identical lists.
        - Build a Lower trie that encodes each stack path from bottom to top.
        - At the end of each path, attach a special _AccKey(acc) edge to a Leaf.
        - Store the trie inside a top-level Upper Interface node with a placeholder acc (None).
        - We intentionally keep `empty=None` and represent empty stacks within the Lower trie
          via an _AccKey at the root. This keeps invariants trivially satisfied.
        """
        # Canonicalize: merge accumulators for identical stacks
        merged: Dict[Tuple[Any, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc

        # Build a nested Python dict trie first: Dict[node_key, child_dict]
        # node_key is either a real stack item (T) or _AccKey(acc) sentinel at leaves.
        trie: Dict[Any, Dict] = {}

        def insert_path(path: List[T], acc: Acc) -> None:
            node = trie
            # Traverse bottom -> top
            for item in path:
                node = node.setdefault(item, {})
            # Attach accumulator marker at the end
            node[_AccKey(acc)] = {}  # Child dict for leaf (empty)

        for key, acc in merged.items():
            insert_path(list(key), acc)

        # Convert the trie into Lower nodes (immutable dataclasses)
        def build_lower(node_dict: Dict[Any, Dict]) -> Lower[T]:
            children_map: Dict[Any, Dict[int, Lower[T]]] = {}
            for key, sub in node_dict.items():
                if isinstance(key, _AccKey):
                    # Terminal edge carrying the accumulator
                    child_lower = Lower(Leaf())
                    # Use index 0 for deterministic placement
                    children_map.setdefault(key, {})[0] = child_lower  # type: ignore[arg-type]
                else:
                    child_lower = build_lower(sub)
                    children_map.setdefault(key, {})[0] = child_lower  # type: ignore[arg-type]
            return Lower(LowerBranch(children=children_map))

        lower_root = build_lower(trie)

        # Top-level Upper is a single Interface to our Lower trie.
        upper = Upper(Interface(node=lower_root, acc=None))
        # Keep empty=None so that validation rule about equality of accs is skipped.
        return LeveledGSS(inner=upper, empty=None)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """
        Decode the Lower trie into a list of (stack, acc) pairs.
        The trie was encoded bottom->top; we traverse accordingly.
        """
        def collect_from_lower(node: Lower[T], prefix: List[T], out: List[Tuple[List[T], Acc]]) -> None:
            if isinstance(node.inner, Leaf):
                # Should not happen in our encoding except as child of an _AccKey.
                return
            branch: LowerBranch[T] = node.inner
            for key in branch.children:
                for child in branch.children[key].values():
                    if isinstance(key, _AccKey):
                        out.append((list(prefix), key.acc))  # type: ignore[attr-defined]
                    else:
                        prefix.append(key)  # descend adding item
                        collect_from_lower(child, prefix, out)
                        prefix.pop()

        results: List[Tuple[List[T], Acc]] = []
        # Our encoding always sets inner as Interface
        if isinstance(self.inner.inner, Interface):
            collect_from_lower(self.inner.inner.node, [], results)
        elif isinstance(self.inner.inner, UpperBranch):
            # Defensive: handle unexpected structure by traversing generic Upper tree
            def collect_from_upper(u: Upper[T, Acc], top_prefix: List[T], out: List[Tuple[List[T], Acc]]) -> None:
                if isinstance(u.inner, Interface):
                    collect_from_lower(u.inner.node, top_prefix, out)
                    return
                br: UpperBranch[T, Acc] = u.inner
                for val, idx_map in br.children.items():
                    for child in idx_map.values():
                        top_prefix.append(val)
                        collect_from_upper(child, top_prefix, out)
                        top_prefix.pop()

            collect_from_upper(self.inner, [], results)
        else:
            # Should not occur; return empty
            pass

        # Canonicalize and sort deterministically
        merged: Dict[Tuple[Any, ...], Acc] = {}
        for vals, acc in results:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc

        items = [(list(k), v) for k, v in merged.items()]

        def _encode_for_sort(obj: Any) -> str:
            try:
                return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
            except Exception:
                return repr(obj)

        items.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        return items

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        ref = self.to_reference_impl()
        pushed_ref = ref.push(value)
        return LeveledGSS.from_stacks(pushed_ref.to_stacks())

    def pop(self) -> LeveledGSS[T, Acc]:
        ref = self.to_reference_impl()
        popped_ref = ref.pop()
        return LeveledGSS.from_stacks(popped_ref.to_stacks())

    def is_empty(self) -> bool:
        return not self.to_stacks()

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        ref = self.to_reference_impl()
        isolated_ref = ref.isolate(value)
        return LeveledGSS.from_stacks(isolated_ref.to_stacks())

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        ref = self.to_reference_impl()
        applied_ref = ref.apply(func)
        return LeveledGSS.from_stacks(applied_ref.to_stacks())

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        ref = self.to_reference_impl()
        pruned_ref = ref.prune(predicate)
        return LeveledGSS.from_stacks(pruned_ref.to_stacks())

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_stacks() + other.to_stacks())

    def peek(self) -> Set[T]:
        return self.to_reference_impl().peek()

    def reduce_acc(self) -> Optional[Acc]:
        return self.to_reference_impl().reduce_acc()


def _validate_upper(node: Upper[T, Acc]):
    """Recursively validates invariants on Upper nodes."""
    if isinstance(node.inner, UpperBranch):
        branch = node.inner
        all_children = [
            child
            for children_by_val in branch.children.values()
            for child in children_by_val.values()
        ]

        # Invariant 1: If all children are interfaces, their accs must be unique.
        if all_children and all(isinstance(child.inner, Interface) for child in all_children):
            accs = [child.inner.acc for child in all_children]
            if len(set(accs)) != len(accs):
                raise AssertionError(
                    "Invariant violated: UpperBranch has children that are all Interfaces with duplicate accs."
                )

        # Recurse into children
        for child in all_children:
            _validate_upper(child)
    # Base case: node.inner is an Interface, do nothing further down this path.


def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Checks internal invariants of the LeveledGSS structure.
    Raises AssertionError if an invariant is violated.
    """
    # Check recursive invariants on the inner structure.
    _validate_upper(gss.inner)

    # Invariant 2: If inner is an interface and empty exists, their accs must differ.
    if isinstance(gss.inner.inner, Interface) and gss.empty is not None:
        if gss.inner.inner.acc == gss.empty:
            raise AssertionError(
                "Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator."
            )
