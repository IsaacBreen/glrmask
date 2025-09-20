from __future__ import annotations
from dataclasses import dataclass, field
from functools import reduce
import json
from typing import List, Tuple, Callable, Set, Any, Type, Optional, Dict

from ..interface import GSS, T, Acc

# Using tuples for nodes to make them hashable and immutable.
# A node is defined by its value and a pointer to its parent.
# The parent pointer is an index into the previous level's node list.
_Node = Tuple[T, Optional[int]]

# A level is a tuple of unique nodes. The tuple provides a canonical ordering and indexing.
_Level = Tuple[_Node, ...]


@dataclass(frozen=True, eq=False)
class LeveledGSS(GSS[T, Acc]):
    """
    A GSS implementation that organizes the graph into levels by stack depth.
    This allows for structural sharing of stack prefixes.

    - `_levels`: A tuple of levels. `_levels[i]` contains all unique nodes at depth `i+1`.
    - `_heads`: A dictionary mapping a head node identifier to its accumulator.
      The key `(level_idx, node_idx)` points to `_levels[level_idx][node_idx]`.
    - `_empty_acc`: An optional accumulator for the single empty stack.
    """
    _levels: Tuple[_Level, ...] = field(default_factory=tuple)
    _heads: Dict[Tuple[int, int], Acc] = field(default_factory=dict)
    _empty_acc: Optional[Acc] = None

    @classmethod
    def from_stacks(cls: Type['LeveledGSS'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        levels_builder: List[Dict[_Node, int]] = []
        heads_builder: Dict[Tuple[int, int], Acc] = {}
        empty_acc: Optional[Acc] = None

        sorted_stacks = sorted(stacks, key=lambda s: len(s[0]))

        for vals, acc in sorted_stacks:
            if not vals:
                if empty_acc is None:
                    empty_acc = acc
                else:
                    empty_acc = empty_acc.merge(acc)
                continue

            parent_idx: Optional[int] = None
            for i, val in enumerate(vals):
                if len(levels_builder) <= i:
                    levels_builder.append({})
                
                node = (val, parent_idx)
                
                if node not in levels_builder[i]:
                    new_idx = len(levels_builder[i])
                    levels_builder[i][node] = new_idx
                
                parent_idx = levels_builder[i][node]

            level_idx = len(vals) - 1
            node_idx = parent_idx
            head_key = (level_idx, node_idx)
            if head_key in heads_builder:
                heads_builder[head_key] = heads_builder[head_key].merge(acc)
            else:
                heads_builder[head_key] = acc
        
        final_levels = tuple(
            tuple(node for node, _ in sorted(level_map.items(), key=lambda item: item[1]))
            for level_map in levels_builder
        )

        return cls(_levels=final_levels, _heads=heads_builder, _empty_acc=empty_acc)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        stacks = []
        if self._empty_acc is not None:
            stacks.append(([], self._empty_acc))

        for (level_idx, node_idx), acc in self._heads.items():
            stack = []
            curr_level_idx = level_idx
            curr_node_idx = node_idx
            while curr_level_idx >= 0:
                node = self._levels[curr_level_idx][curr_node_idx]
                value, parent_idx = node
                stack.append(value)
                if parent_idx is None:
                    break
                curr_level_idx -= 1
                curr_node_idx = parent_idx
            stacks.append((stack[::-1], acc))
        
        def _encode_for_sort(obj: Any) -> str:
            try:
                return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
            except Exception:
                return repr(obj)

        stacks.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        return stacks

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        new_stacks = [
            (vals + [value], acc) for vals, acc in self.to_stacks()
        ]
        return self.from_stacks(new_stacks)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        new_stacks = [
            (vals[:-1], acc) for vals, acc in self.to_stacks() if vals
        ]
        return self.from_stacks(new_stacks)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        return self.from_stacks([(vals, func(acc)) for vals, acc in self.to_stacks()])

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        return self.from_stacks([(vals, acc) for vals, acc in self.to_stacks() if predicate(acc)])

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        return self.from_stacks(self.to_stacks() + other.to_stacks())

    def is_empty(self) -> bool:
        return not self._heads and self._empty_acc is None

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        if value is None:
            return self.__class__(_empty_acc=self._empty_acc)

        new_heads = {
            head: acc for head, acc in self._heads.items()
            if self._levels[head[0]][head[1]][0] == value
        }
        return self.__class__(_levels=self._levels, _heads=new_heads)

    def peek(self) -> Set[T]:
        return {self._levels[level_idx][node_idx][0] for level_idx, node_idx in self._heads}

    def reduce_acc(self) -> Optional[Acc]:
        all_accs = list(self._heads.values())
        if self._empty_acc is not None:
            all_accs.append(self._empty_acc)

        if not all_accs:
            return None
        
        return reduce(lambda a, b: a.merge(b), all_accs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GSS):
            return NotImplemented
        return self.to_stacks() == other.to_stacks()
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any
from collections import defaultdict

from ..interface import GSS, T, Acc


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
        # The GSS is empty if there's no accumulator for the empty stack, and
        # the inner trie has no children. from_stacks ensures inner is an UpperBranch.
        return self.empty is None and not self.inner.inner.children

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        if value is None:
            return LeveledGSS(Upper(UpperBranch({})), self.empty)

        def filter_node(u: Upper[T, Acc]) -> Optional[Upper[T, Acc]]:
            if isinstance(u.inner, Interface):
                return None

            # It's an UpperBranch
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, children_map in u.inner.children.items():
                new_v_children: Dict[int, Upper[T, Acc]] = {}

                # If v is the value we're looking for, keep its interface children.
                if v == value:
                    for i, child in children_map.items():
                        if isinstance(child.inner, Interface):
                            new_v_children[i] = child

                # For all branch children, recurse.
                for i, child in children_map.items():
                    if isinstance(child.inner, UpperBranch):
                        filtered_child = filter_node(child)
                        if filtered_child:
                            new_v_children[i] = filtered_child

                if new_v_children:
                    new_children[v] = new_v_children

            if not new_children:
                return None
            return Upper(UpperBranch(new_children))

        new_inner = filter_node(self.inner)
        if not new_inner:
            new_inner = Upper(UpperBranch({}))

        return LeveledGSS(new_inner, None)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().apply(func).to_stacks())
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().prune(predicate).to_stacks())
    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().merge(other.to_reference_impl()).to_stacks())
    def peek(self) -> Set[T]:
        tops: Set[T] = set()

        def dfs(u: Upper[T, Acc]):
            if isinstance(u.inner, UpperBranch):
                for v, children_map in u.inner.children.items():
                    if any(isinstance(child.inner, Interface) for child in children_map.values()):
                        tops.add(v)

                    for child in children_map.values():
                        dfs(child)

        dfs(self.inner)
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        from functools import reduce
        accs: List[Acc] = []
        if self.empty is not None:
            accs.append(self.empty)

        def collect_accs(u: Upper[T, Acc]):
            if isinstance(u.inner, Interface):
                accs.append(u.inner.acc)
            elif isinstance(u.inner, UpperBranch):
                for children_map in u.inner.children.values():
                    for child in children_map.values():
                        collect_accs(child)

        collect_accs(self.inner)

        if not accs:
            return None

        return reduce(lambda a, b: a.merge(b), accs)

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
