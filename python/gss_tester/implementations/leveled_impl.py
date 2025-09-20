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


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    children: Dict[T, Dict[int, Lower[T]]]
    acc: Acc
    empty: Optional[Acc]


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    children: Dict[T, Dict[int, Lower[T]]]
    empty: bool


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

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

        def build(d: Dict[T, Dict[str, Any]]) -> Upper[T, Acc]:
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
                    children[v] = {i: n for i, n in enumerate(nodes)}
            return UpperBranch(children=children, empty=None)

        return LeveledGSS(build(trie), empty_acc)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        res: List[Tuple[List[T], Acc]] = []
        if self.empty is not None:
            res.append(([], self.empty))

        def dfs(u: Upper[T, Acc], pref: List[T]) -> None:
            if isinstance(u, Interface):
                res.append((pref, u.acc))
                return
            # u is UpperBranch
            for v, kids in u.children.items():
                for child in kids.values():
                    if isinstance(child, Interface):
                        res.append((pref + [v], child.acc))
                    else:
                        dfs(child, pref + [v])

        if isinstance(self.inner, Interface):
            # root is an interface: treat as stack with empty prefix
            res.append(([], self.inner.acc))
        else:
            dfs(self.inner, [])
        from .reference_impl import ReferenceGSS
        return ReferenceGSS.from_stacks(res).to_stacks()

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().push(value).to_stacks())
    def pop(self) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().pop().to_stacks())
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
