from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any

from ..interface import GSS, T, Acc


# ------------------------------
# Internal node classes
# ------------------------------

type Upper[T, Acc] = UpperBranch[T, Acc] | Interface[T, Acc]


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    children: Dict[T, Dict[int, Upper[T, Acc]]]
    empty: Optional[Acc]

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

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        # Canonicalize first using the reference implementation
        from .reference_impl import ReferenceGSS
        merged = ReferenceGSS.from_stacks(stacks).to_stacks()

        empty_acc: Optional[Acc] = None
        # A simple trie: { val: { "end": Optional[Acc], "sub": <subtrie> } }
        trie: Dict[T, Dict[str, Any]] = {}

        for vals, acc in merged:
            if not vals:
                empty_acc = acc
                continue
            node = trie
            for i, v in enumerate(vals):
                entry = node.setdefault(v, {"end": None, "sub": {}})
                if i == len(vals) - 1:
                    if entry["end"] is None:
                        entry["end"] = acc
                    else:
                        entry["end"] = entry["end"].merge(acc)  # type: ignore[attr-defined]
                else:
                    node = entry["sub"]

        def build(d: Dict[T, Dict[str, Any]], root_empty: Optional[Acc] = None) -> UpperBranch[T, Acc]:
            children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, e in d.items():
                nodes: List[Upper[T, Acc]] = []
                end_acc = e.get("end")
                sub = e.get("sub", {})
                if end_acc is not None:
                    nodes.append(Interface(children={}, acc=end_acc, empty=None))
                if sub:
                    nodes.append(build(sub))
                if nodes:
                    children[v] = {n._max_depth(): n for n in nodes}
            return UpperBranch(children=children, empty=root_empty)

        return LeveledGSS(build(trie, empty_acc))

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        res: List[Tuple[List[T], Acc]] = []
        if isinstance(self.inner, Interface):
            res.append(([], self.inner.acc))
            if self.inner.empty is not None:
                res.append(([], self.inner.empty))
        elif isinstance(self.inner, UpperBranch):
            if self.inner.empty is not None:
                res.append(([], self.inner.empty))

            def dfs(u: UpperBranch[T, Acc], pref: List[T]) -> None:
                for v, kids in u.children.items():
                    for child in kids.values():
                        if isinstance(child, Interface):
                            res.append((pref + [v], child.acc))
                        else:
                            dfs(child, pref + [v])

            dfs(self.inner, [])
        from .reference_impl import ReferenceGSS
        return ReferenceGSS.from_stacks(res).to_stacks()

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        return LeveledGSS(UpperBranch(children={value: {self.inner._max_depth(): self.inner}}, empty=None))
    def pop(self) -> LeveledGSS[T, Acc]:
        if isinstance(self.inner, Interface):
            return LeveledGSS(UpperBranch(children={}, empty=None))
        all_children: List[Upper] = [child for _, max_depth_to_children in self.inner.children.items() for child in max_depth_to_children.values()]
        new_inner: Upper[T, Acc] = _merge_uppers(all_children)
        if isinstance(new_inner, Interface):
            new_inner = _convert_interface_to_upper_brach(new_inner)
        return LeveledGSS(new_inner)
    def is_empty(self) -> bool:
        return len(self.to_stacks()) == 0

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().isolate(value).to_stacks())

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().apply(func).to_stacks())
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().prune(predicate).to_stacks())
    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().merge(other.to_reference_impl()).to_stacks())
    def peek(self) -> Set[T]:
        tops: Set[T] = set()
        for vals, _ in self.to_stacks():
            if vals:
                tops.add(vals[-1])
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        from functools import reduce
        items = self.to_stacks()
        if not items:
            return None
        accs = [acc for _, acc in items]
        return reduce(lambda a, b: a.merge(b), accs)
