from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any
from collections import defaultdict

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
    acc: Acc


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    inner: LowerBranch[T] | Leaf


@dataclass(frozen=True, eq=True)
class LowerBranch(Generic[T]):
    children: Dict[T, Dict[int, Lower[T]]]


@dataclass(frozen=True, eq=True)
class Leaf:
    pass


# A shared, canonical leaf node
_LOWER_LEAF = Lower(Leaf())


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        # Canonicalize first using the reference implementation
        from .reference_impl import ReferenceGSS
        merged = ReferenceGSS(stacks).to_stacks()

        empty_acc: Optional[Acc] = None
        # A simple trie: { val: { "i": [acc, ...], "b": <subtrie> } }
        trie: Dict[T, Dict[str, Any]] = {}

        for vals, acc in merged:
            if not vals:
                empty_acc = acc
                continue
            node = trie
            for i, v in enumerate(vals):
                entry = node.setdefault(v, {"i": [], "b": {}})
                if i == len(vals) - 1:
                    entry["i"].append(acc)
                else:
                    node = entry["b"]

        def build(d: Dict[T, Dict[str, Any]]) -> Upper[T, Acc]:
            children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, e in d.items():
                nodes: List[Upper[T, Acc]] = [Upper(Interface(_LOWER_LEAF, a)) for a in e["i"]]
                # Always add a branch node (empty or not) to keep structure uniform and avoid edge invariants.
                branch_child = build(e["b"]) if e["b"] else Upper(UpperBranch({}))
                nodes.append(branch_child)
                children[v] = {i: n for i, n in enumerate(nodes)}
            return Upper(UpperBranch(children))

        return LeveledGSS(build(trie), empty_acc)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        res: List[Tuple[List[T], Acc]] = []
        if self.empty is not None:
            res.append(([], self.empty))

        def dfs(u: Upper[T, Acc], pref: List[T]) -> None:
            if isinstance(u.inner, Interface):
                res.append((pref, u.inner.acc))
                return
            for v, kids in u.inner.children.items():
                for child in kids.values():
                    dfs(child, pref + [v])

        dfs(self.inner, [])
        from .reference_impl import ReferenceGSS
        return ReferenceGSS(res).to_stacks()

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().push(value).to_stacks())
    def pop(self) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().pop().to_stacks())
    def is_empty(self) -> bool:
        return self.to_reference_impl().is_empty()
    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().isolate(value).to_stacks())
    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().apply(func).to_stacks())
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().prune(predicate).to_stacks())
    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().merge(other.to_reference_impl()).to_stacks())
    def peek(self) -> Set[T]:
        return self.to_reference_impl().peek()
    def reduce_acc(self) -> Optional[Acc]:
        return self.to_reference_impl().reduce_acc()


def _get_upper_children(branch: UpperBranch[T, Acc]) -> List[Upper[T, Acc]]:
    """Helper to get all children from an UpperBranch."""
    return [child for children_by_val in branch.children.values() for child in children_by_val.values()]


def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Checks internal invariants of the LeveledGSS structure.
    Raises AssertionError if an invariant is violated.
    """
    def _validate_upper(node: Upper[T, Acc]):
        """Recursively validates invariants on Upper nodes."""
        if not isinstance(node.inner, UpperBranch):
            return  # Base case: node is an Interface.
        all_children = _get_upper_children(node.inner)
        # Invariant 1: If all children are interfaces, there must be more than one unique acc.
        if all_children and all(isinstance(child.inner, Interface) for child in all_children):
            if len({child.inner.acc for child in all_children}) > 1:
                raise AssertionError("Invariant violated: UpperBranch has children that are all Interfaces with duplicate accs.")
        # Recurse into children
        for child in all_children:
            _validate_upper(child)
    _validate_upper(gss.inner)
    # Invariant 2: If inner is an interface and empty exists, their accs must differ.
    if isinstance(gss.inner, Interface) and gss.empty is not None and gss.inner.acc == gss.empty:
        raise AssertionError("Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator.")
