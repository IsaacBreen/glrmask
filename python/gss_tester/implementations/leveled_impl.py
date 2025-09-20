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
        # True if no stacks: no root empty and no Interface anywhere.
        if self.inner.empty is not None:
            return False

        def has_interface(u: UpperBranch[T, Acc]) -> bool:
            for kids in u.children.values():
                for child in kids.values():
                    if isinstance(child, Interface):
                        return True
                    else:
                        if has_interface(child):
                            return True
            return False

        return not has_interface(self.inner)

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        # Keep only stacks whose top equals `value`; if value is None, keep only the empty stack.
        if value is None:
            return LeveledGSS(UpperBranch(children={}, empty=self.inner.empty))

        def iso_upper(u: UpperBranch[T, Acc]) -> UpperBranch[T, Acc]:
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, kids in u.children.items():
                iface_acc: Optional[Acc] = None
                branches: List[Upper[T, Acc]] = []
                for child in kids.values():
                    if isinstance(child, Interface):
                        if v == value:
                            iface_acc = child.acc if iface_acc is None else iface_acc.merge(child.acc)  # type: ignore[attr-defined]
                    else:
                        sub = iso_upper(child)
                        if sub.children or sub.empty is not None:
                            branches.append(sub)
                alts: List[Upper[T, Acc]] = []
                if iface_acc is not None:
                    alts.append(Interface(children={}, acc=iface_acc, empty=None))
                alts.extend(branches)
                if alts:
                    new_children[v] = {i: alt for i, alt in enumerate(alts)}
            return UpperBranch(children=new_children, empty=None)

        return LeveledGSS(iso_upper(self.inner))

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        # Apply func to each stack accumulator structurally.
        def map_upper(u: UpperBranch[T, Acc]) -> UpperBranch[T, Acc]:
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, kids in u.children.items():
                iface_acc: Optional[Acc] = None
                branch_alts: List[Upper[T, Acc]] = []
                for child in kids.values():
                    if isinstance(child, Interface):
                        new_acc = func(child.acc)
                        iface_acc = new_acc if iface_acc is None else iface_acc.merge(new_acc)  # type: ignore[attr-defined]
                    else:
                        branch_alts.append(map_upper(child))
                alts: List[Upper[T, Acc]] = []
                if iface_acc is not None:
                    alts.append(Interface(children={}, acc=iface_acc, empty=None))
                alts.extend(branch_alts)
                if alts:
                    new_children[v] = {i: alt for i, alt in enumerate(alts)}
            # Preserve non-root empties unchanged (they are not stacks); pop recomputes from Interfaces.
            return UpperBranch(children=new_children, empty=u.empty)

        mapped = map_upper(self.inner)
        new_empty = self.inner.empty
        if new_empty is not None:
            new_empty = func(new_empty)
        return LeveledGSS(UpperBranch(children=mapped.children, empty=new_empty))
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        # Remove stacks whose accumulator does not satisfy predicate.
        def prune_upper(u: UpperBranch[T, Acc]) -> UpperBranch[T, Acc]:
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, kids in u.children.items():
                iface_acc: Optional[Acc] = None
                branches: List[Upper[T, Acc]] = []
                for child in kids.values():
                    if isinstance(child, Interface):
                        if predicate(child.acc):
                            iface_acc = child.acc if iface_acc is None else iface_acc.merge(child.acc)  # type: ignore[attr-defined]
                    else:
                        sub = prune_upper(child)
                        if sub.children:
                            branches.append(sub)
                alts: List[Upper[T, Acc]] = []
                if iface_acc is not None:
                    alts.append(Interface(children={}, acc=iface_acc, empty=None))
                alts.extend(branches)
                if alts:
                    new_children[v] = {i: alt for i, alt in enumerate(alts)}
            # Preserve non-root empties unchanged (they are not stacks).
            return UpperBranch(children=new_children, empty=u.empty)
        pruned = prune_upper(self.inner)
        new_empty = self.inner.empty
        if new_empty is not None and not predicate(new_empty):
            new_empty = None
        return LeveledGSS(UpperBranch(children=pruned.children, empty=new_empty))
    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        # Fast path: identical objects
        if self is other or self.inner is other.inner:
            return self

        def _merge_opt_acc(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
            if a is None:
                return b
            if b is None:
                return a
            return a.merge(b)  # type: ignore[attr-defined]

        def merge_branch(u1: UpperBranch[T, Acc], u2: UpperBranch[T, Acc]) -> UpperBranch[T, Acc]:
            # Identity shortcut
            if u1 is u2:
                return u1

            # Merge root/non-root empties (used by pop semantics)
            new_empty: Optional[Acc] = _merge_opt_acc(u1.empty, u2.empty)

            # Deterministic key order: keys from u1 followed by new keys from u2
            keys: List[T] = list(u1.children.keys())
            for k in u2.children.keys():
                if k not in u1.children:
                    keys.append(k)

            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v in keys:
                kids1 = u1.children.get(v, {})
                kids2 = u2.children.get(v, {})

                # Accumulate interface accs and collect branch alts
                iface_acc: Optional[Acc] = None
                branch_nodes: List[UpperBranch[T, Acc]] = []

                for child in kids1.values():
                    if isinstance(child, Interface):
                        iface_acc = child.acc if iface_acc is None else iface_acc.merge(child.acc)  # type: ignore[attr-defined]
                    else:
                        branch_nodes.append(child)
                for child in kids2.values():
                    if isinstance(child, Interface):
                        iface_acc = child.acc if iface_acc is None else iface_acc.merge(child.acc)  # type: ignore[attr-defined]
                    else:
                        branch_nodes.append(child)

                # Deduplicate branch nodes by identity to avoid obvious duplicates
                seen_ids = set()
                uniq_branches: List[UpperBranch[T, Acc]] = []
                for bn in branch_nodes:
                    bid = id(bn)
                    if bid not in seen_ids:
                        seen_ids.add(bid)
                        uniq_branches.append(bn)

                alts: List[Upper[T, Acc]] = []
                # Merge all branch alts into one if present
                if uniq_branches:
                    merged_branch = uniq_branches[0]
                    for bn in uniq_branches[1:]:
                        merged_branch = merge_branch(merged_branch, bn)
                    # Drop useless empty branches (no children and no empty)
                    if merged_branch.children or merged_branch.empty is not None:
                        alts.append(merged_branch)

                # Single Interface per symbol with merged accumulator
                if iface_acc is not None:
                    alts.insert(0, Interface(children={}, acc=iface_acc, empty=None))

                if alts:
                    new_children[v] = {i: alt for i, alt in enumerate(alts)}

            return UpperBranch(children=new_children, empty=new_empty)

        merged_inner = merge_branch(self.inner, other.inner)
        return LeveledGSS(merged_inner)
    def peek(self) -> Set[T]:
        # Collect labels `v` for which an Interface exists under `v` anywhere.
        tops: Set[T] = set()

        def dfs(u: UpperBranch[T, Acc]) -> None:
            for v, kids in u.children.items():
                any_iface = False
                for child in kids.values():
                    if isinstance(child, Interface):
                        any_iface = True
                    else:
                        dfs(child)
                if any_iface:
                    tops.add(v)

        dfs(self.inner)
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        # Merge accumulators from all stacks structurally.
        acc_res: Optional[Acc] = self.inner.empty

        def dfs(u: UpperBranch[T, Acc]) -> None:
            nonlocal acc_res
            for kids in u.children.values():
                for child in kids.values():
                    if isinstance(child, Interface):
                        if acc_res is None:
                            acc_res = child.acc
                        else:
                            acc_res = acc_res.merge(child.acc)  # type: ignore[attr-defined]
                    else:
                        dfs(child)

        dfs(self.inner)
        return acc_res
