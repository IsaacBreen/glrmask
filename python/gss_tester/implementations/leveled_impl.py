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
    inner: UpperBranch[T, Acc]

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
                    children[v] = {i: n for i, n in enumerate(nodes)}
            return UpperBranch(children=children, empty=root_empty)

        return LeveledGSS(build(trie, empty_acc))

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        res: List[Tuple[List[T], Acc]] = []
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
        # Recursively push by transforming each leaf Interface into a deeper
        # UpperBranch with the new `value` as the next symbol.
        def push_upper(u: UpperBranch[T, Acc]) -> UpperBranch[T, Acc]:
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, kids in u.children.items():
                alts: List[Upper[T, Acc]] = []
                for child in kids.values():
                    if isinstance(child, Interface):
                        # Previously ended at ... + [v]; after push, it ends at ... + [v, value]
                        pushed_branch = UpperBranch(
                            children={value: {0: Interface(children={}, acc=child.acc, empty=None)}},
                            empty=None,
                        )
                        alts.append(pushed_branch)
                    else:
                        alts.append(push_upper(child))
                if alts:
                    new_children[v] = {i: alt for i, alt in enumerate(alts)}
            return UpperBranch(children=new_children, empty=None)

        base = push_upper(self.inner)
        # Handle pushing onto the empty stack at the root (if present).
        root_children: Dict[T, Dict[int, Upper[T, Acc]]] = dict(base.children)
        if self.inner.empty is not None:
            existing_alts: List[Upper[T, Acc]] = list(root_children.get(value, {}).values())
            existing_alts.append(Interface(children={}, acc=self.inner.empty, empty=None))
            root_children[value] = {i: alt for i, alt in enumerate(existing_alts)}
        return LeveledGSS(UpperBranch(children=root_children, empty=None))

    def pop(self) -> LeveledGSS[T, Acc]:
        # Recursively pop by:
        # - Converting Interface children at a node into that node's empty (popping removes the last symbol).
        # - Recursing into UpperBranch children; their popped empties become Interfaces under the symbol,
        #   and their children carry on as deeper alternatives.
        def pop_upper(u: UpperBranch[T, Acc]) -> UpperBranch[T, Acc]:
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            accum_here: Optional[Acc] = None

            for v, kids in u.children.items():
                alts: List[Upper[T, Acc]] = []
                for child in kids.values():
                    if isinstance(child, Interface):
                        if accum_here is None:
                            accum_here = child.acc
                        else:
                            accum_here = accum_here.merge(child.acc)  # type: ignore[attr-defined]
                    else:
                        popped = pop_upper(child)
                        # If popping the subtree yields an empty at that level, surface it
                        # as an Interface under the current symbol `v`.
                        if popped.empty is not None:
                            alts.append(Interface(children={}, acc=popped.empty, empty=None))
                        # Keep deeper structure if any.
                        if popped.children:
                            alts.append(UpperBranch(children=popped.children, empty=None))
                if alts:
                    new_children[v] = {i: alt for i, alt in enumerate(alts)}

            # We intentionally ignore u.empty here (popping from empty is invalid).
            return UpperBranch(children=new_children, empty=accum_here)

        return LeveledGSS(pop_upper(self.inner))

    def is_empty(self) -> bool:
        return len(self.to_stacks()) == 0

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().isolate(value).to_stacks())

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().apply(func).to_stacks())
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().prune(predicate).to_stacks())
    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        # Structural merge without flattening to full stacks.
        # We keep at most one Interface and one UpperBranch alternative per symbol.
        from typing import Tuple

        def merge_opt_acc(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
            if a is None:
                return b
            if b is None:
                return a
            return a.merge(b)  # type: ignore[attr-defined]

        memo: Dict[Tuple[int, int], UpperBranch[T, Acc]] = {}

        def merge_upper(u1: UpperBranch[T, Acc], u2: UpperBranch[T, Acc]) -> UpperBranch[T, Acc]:
            if u1 is u2:
                return u1
            key = (id(u1), id(u2))
            if key in memo:
                return memo[key]

            merged_empty = merge_opt_acc(u1.empty, u2.empty)
            keys: Set[T] = set(u1.children.keys()) | set(u2.children.keys())
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}

            for v in keys:
                alts: List[Upper[T, Acc]] = []
                kids1 = u1.children.get(v)
                kids2 = u2.children.get(v)

                # Merge all Interface children by combining their acc (and empty if present).
                iface_acc: Optional[Acc] = None
                iface_empty: Optional[Acc] = None

                def absorb_interface(ifc: Interface[T, Acc]) -> None:
                    nonlocal iface_acc, iface_empty
                    if iface_acc is None:
                        iface_acc = ifc.acc
                    else:
                        iface_acc = iface_acc.merge(ifc.acc)  # type: ignore[attr-defined]
                    if ifc.empty is not None:
                        if iface_empty is None:
                            iface_empty = ifc.empty
                        else:
                            iface_empty = iface_empty.merge(ifc.empty)  # type: ignore[attr-defined]

                if kids1 is not None:
                    for c in kids1.values():
                        if isinstance(c, Interface):
                            absorb_interface(c)
                if kids2 is not None:
                    for c in kids2.values():
                        if isinstance(c, Interface):
                            absorb_interface(c)
                if iface_acc is not None:
                    alts.append(Interface(children={}, acc=iface_acc, empty=iface_empty))

                # Merge all UpperBranch children by folding them with merge_upper.
                sub_list: List[UpperBranch[T, Acc]] = []
                if kids1 is not None:
                    for c in kids1.values():
                        if isinstance(c, UpperBranch):
                            sub_list.append(c)
                if kids2 is not None:
                    for c in kids2.values():
                        if isinstance(c, UpperBranch):
                            sub_list.append(c)

                if len(sub_list) == 1:
                    alts.append(sub_list[0])
                elif len(sub_list) >= 2:
                    merged_sub = sub_list[0]
                    for s in sub_list[1:]:
                        merged_sub = merge_upper(merged_sub, s)
                    alts.append(merged_sub)

                if alts:
                    new_children[v] = {i: alt for i, alt in enumerate(alts)}

            res = UpperBranch(children=new_children, empty=merged_empty)
            memo[key] = res
            return res

        return LeveledGSS(merge_upper(self.inner, other.inner))
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
