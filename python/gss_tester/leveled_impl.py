from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import json
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple
 
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

# ------------------------------
# Helper functions
# ------------------------------

_LOWER_LEAF = Lower(Leaf())
_LOWER_CACHE: Dict[Tuple, Lower] = {}
_LOWER_RECONSTRUCT_CACHE: Dict[Lower, List] = {}

def _get_lower(stack: Tuple[T, ...]) -> Lower[T]:
    if not stack:
        return _LOWER_LEAF
    if stack in _LOWER_CACHE:
        return _LOWER_CACHE[stack]
    
    val = stack[-1]
    prefix = stack[:-1]
    parent = _get_lower(prefix)
    
    node = Lower(LowerBranch({val: {hash(parent): parent}}))
    _LOWER_CACHE[stack] = node
    return node

def _reconstruct_from_lower(node: Lower[T]) -> List[T]:
    if node in _LOWER_RECONSTRUCT_CACHE:
        return _LOWER_RECONSTRUCT_CACHE[node]
    
    if isinstance(node.inner, Leaf):
        return []
    
    branch: LowerBranch[T] = node.inner
    val, children_by_id = next(iter(branch.children.items()))
    child = next(iter(children_by_id.values()))
    
    res = _reconstruct_from_lower(child) + [val]
    _LOWER_RECONSTRUCT_CACHE[node] = res
    return res

def _deconstruct_lower(node: Lower[T]) -> Optional[Tuple[T, Lower[T]]]:
    if isinstance(node.inner, Leaf):
        return None
    branch: LowerBranch[T] = node.inner
    val, children_by_id = next(iter(branch.children.items()))
    child = next(iter(children_by_id.values()))
    return val, child

_UPPER_EMPTY = Upper(UpperBranch({}))

def _is_empty_upper(node: Upper[T, Acc]) -> bool:
    return node == _UPPER_EMPTY

def _build_upper(stacks: Dict[Tuple[T, ...], Acc]) -> Upper[T, Acc]:
    if not stacks:
        return _UPPER_EMPTY

    if len(stacks) == 1:
        stack, acc = next(iter(stacks.items()))
        return Upper(Interface(_get_lower(stack), acc))

    children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    groups: Dict[T, Dict[Tuple[T, ...], Acc]] = {}

    for stack, acc in stacks.items():
        top = stack[-1]
        groups.setdefault(top, {})[stack] = acc

    for top, group_stacks in groups.items():
        popped_stacks = {s[:-1]: a for s, a in group_stacks.items()}
        child_node = _build_upper(popped_stacks)
        if not _is_empty_upper(child_node):
            children.setdefault(top, {})[hash(child_node)] = child_node
    
    return Upper(UpperBranch(children))

def _collect_stacks(node: Upper[T, Acc], suffix: List[T], results: List[Tuple[List[T], Acc]]):
    if isinstance(node.inner, Interface):
        prefix = _reconstruct_from_lower(node.inner.node)
        results.append((prefix + suffix, node.inner.acc))
    elif isinstance(node.inner, UpperBranch):
        for val, children_by_id in node.inner.children.items():
            for child in children_by_id.values():
                _collect_stacks(child, [val] + suffix, results)

def _encode_for_sort(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    except Exception:
        return repr(obj)

def _collect_accs(node: Upper[T, Acc]) -> List[Acc]:
    if _is_empty_upper(node):
        return []
    if isinstance(node.inner, Interface):
        return [node.inner.acc]
    
    accs = []
    if isinstance(node.inner, UpperBranch):
        for children_by_id in node.inner.children.values():
            for child in children_by_id.values():
                accs.extend(_collect_accs(child))
    return accs

# ------------------------------
# GSS Implementation
# ------------------------------

@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        merged: Dict[Tuple[T, ...], Acc] = {}
        for stack, acc in stacks:
            key = tuple(stack)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc
        
        empty_acc = merged.pop((), None)
        inner_node = _build_upper(merged)
        return cls(inner_node, empty_acc)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        results = []
        if self.empty is not None:
            results.append(([], self.empty))
        
        _collect_stacks(self.inner, [], results)
        results.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        return results

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        if self.is_empty():
            return self

        pushed_inner = _UPPER_EMPTY
        if not _is_empty_upper(self.inner):
            pushed_inner = Upper(UpperBranch({value: {hash(self.inner): self.inner}}))
        
        new_gss = LeveledGSS(pushed_inner, None)

        if self.empty is not None:
            gss_from_empty = LeveledGSS.from_stacks([([value], self.empty)])
            new_gss = new_gss.merge(gss_from_empty)
            
        return new_gss

    def pop(self) -> LeveledGSS[T, Acc]:
        new_inner, accs_for_empty = _pop_upper(self.inner)
        
        new_empty = self.empty
        if accs_for_empty:
            merged_new_empty = reduce(lambda a, b: a.merge(b), accs_for_empty)
            if new_empty is None:
                new_empty = merged_new_empty
            else:
                new_empty = new_empty.merge(merged_new_empty)
        
        return LeveledGSS(new_inner, new_empty)

    def is_empty(self) -> bool:
        return self.empty is None and _is_empty_upper(self.inner)

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        if value is None:
            return LeveledGSS(_UPPER_EMPTY, self.empty)
        
        if _is_empty_upper(self.inner):
            return LeveledGSS(_UPPER_EMPTY, None)

        if isinstance(self.inner.inner, Interface):
            deconstruction = _deconstruct_lower(self.inner.inner.node)
            if deconstruction and deconstruction[0] == value:
                return self
            return LeveledGSS(_UPPER_EMPTY, None)

        branch: UpperBranch[T, Acc] = self.inner.inner
        children_at_val = branch.children.get(value)
        if not children_at_val:
            return LeveledGSS(_UPPER_EMPTY, None)
        
        new_inner = Upper(UpperBranch({value: children_at_val}))
        return LeveledGSS(new_inner, None)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        new_empty = func(self.empty) if self.empty is not None else None
        new_inner = _apply_upper(self.inner, func)
        return LeveledGSS(new_inner, new_empty)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        new_empty = self.empty if self.empty is not None and predicate(self.empty) else None
        new_inner = _prune_upper(self.inner, predicate)
        return LeveledGSS(new_inner, new_empty)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        new_inner = _merge_uppers(self.inner, other.inner)
        
        new_empty = self.empty
        if other.empty is not None:
            if new_empty is None:
                new_empty = other.empty
            else:
                new_empty = new_empty.merge(other.empty)
        
        return LeveledGSS(new_inner, new_empty)

    def peek(self) -> Set[T]:
        if isinstance(self.inner.inner, UpperBranch):
            return set(self.inner.inner.children.keys())
        if isinstance(self.inner.inner, Interface):
            deconstruction = _deconstruct_lower(self.inner.inner.node)
            if deconstruction:
                return {deconstruction[0]}
        return set()

    def reduce_acc(self) -> Optional[Acc]:
        accs = _collect_accs(self.inner)
        if self.empty is not None:
            accs.append(self.empty)
        
        if not accs:
            return None
        
        return reduce(lambda a, b: a.merge(b), accs)


def _merge_many_uppers(nodes: List[Upper[T, Acc]]) -> Upper[T, Acc]:
    if not nodes:
        return _UPPER_EMPTY
    return reduce(_merge_uppers, nodes)

def _merge_uppers(u1: Upper[T, Acc], u2: Upper[T, Acc]) -> Upper[T, Acc]:
    if _is_empty_upper(u1): return u2
    if _is_empty_upper(u2): return u1

    if isinstance(u1.inner, Interface) and isinstance(u2.inner, Interface):
        if u1.inner.node == u2.inner.node:
            new_acc = u1.inner.acc.merge(u2.inner.acc)
            return Upper(Interface(u1.inner.node, new_acc))
        else:
            s1 = {tuple(_reconstruct_from_lower(u1.inner.node)): u1.inner.acc}
            s2 = {tuple(_reconstruct_from_lower(u2.inner.node)): u2.inner.acc}
            s1.update(s2)
            return _build_upper(s1)

    if isinstance(u1.inner, UpperBranch) and isinstance(u2.inner, Interface):
        u1, u2 = u2, u1

    if isinstance(u1.inner, Interface) and isinstance(u2.inner, UpperBranch):
        deconstruction = _deconstruct_lower(u1.inner.node)
        if deconstruction is None: return u2

        top, popped_lower = deconstruction
        node_to_insert = Upper(Interface(popped_lower, u1.inner.acc))
        
        branch_children = dict(u2.inner.children)
        children_at_top = dict(branch_children.get(top, {}))
        
        h = hash(node_to_insert)
        if h in children_at_top:
            merged_child = _merge_uppers(children_at_top[h], node_to_insert)
            del children_at_top[h]
            children_at_top[hash(merged_child)] = merged_child
        else:
            children_at_top[h] = node_to_insert
            
        branch_children[top] = children_at_top
        return Upper(UpperBranch(branch_children))

    if isinstance(u1.inner, UpperBranch) and isinstance(u2.inner, UpperBranch):
        b1, b2 = u1.inner, u2.inner
        new_children = dict(b1.children)
        for val, children2_by_id in b2.children.items():
            if val not in new_children:
                new_children[val] = children2_by_id
            else:
                children1_by_id = dict(new_children[val])
                for h2, child2 in children2_by_id.items():
                    if h2 in children1_by_id:
                        child1 = children1_by_id[h2]
                        merged = _merge_uppers(child1, child2)
                        del children1_by_id[h2]
                        children1_by_id[hash(merged)] = merged
                    else:
                        children1_by_id[h2] = child2
                new_children[val] = children1_by_id
        return Upper(UpperBranch(new_children))
    
    raise TypeError(f"Unsupported merge combination: {type(u1.inner)}, {type(u2.inner)}")

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

def _pop_upper(node: Upper[T, Acc]) -> Tuple[Upper[T, Acc], List[Acc]]:
    if _is_empty_upper(node):
        return _UPPER_EMPTY, []

    if isinstance(node.inner, Interface):
        deconstruction = _deconstruct_lower(node.inner.node)
        if deconstruction is None:
            return _UPPER_EMPTY, [node.inner.acc]
        else:
            _, popped_lower = deconstruction
            return Upper(Interface(popped_lower, node.inner.acc)), []

    branch: UpperBranch[T, Acc] = node.inner
    nodes_to_merge = []
    for children_by_id in branch.children.values():
        nodes_to_merge.extend(children_by_id.values())
    
    if not nodes_to_merge:
        return _UPPER_EMPTY, []

    return _merge_many_uppers(nodes_to_merge), []

def _apply_upper(node: Upper[T, Acc], func: Callable[[Acc], Acc]) -> Upper[T, Acc]:
    if _is_empty_upper(node):
        return _UPPER_EMPTY
    
    if isinstance(node.inner, Interface):
        return Upper(Interface(node.inner.node, func(node.inner.acc)))

    branch: UpperBranch[T, Acc] = node.inner
    new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for val, children_by_id in branch.children.items():
        new_children_by_id = {}
        for child in children_by_id.values():
            new_child = _apply_upper(child, func)
            new_children_by_id[hash(new_child)] = new_child
        new_children[val] = new_children_by_id
    return Upper(UpperBranch(new_children))

def _prune_upper(node: Upper[T, Acc], predicate: Callable[[Acc], bool]) -> Upper[T, Acc]:
    if _is_empty_upper(node):
        return _UPPER_EMPTY
    if isinstance(node.inner, Interface):
        return node if predicate(node.inner.acc) else _UPPER_EMPTY

    branch: UpperBranch[T, Acc] = node.inner
    new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for val, children_by_id in branch.children.items():
        new_children_by_id = {}
        for child in children_by_id.values():
            new_child = _prune_upper(child, predicate)
            if not _is_empty_upper(new_child):
                new_children_by_id[hash(new_child)] = new_child
        if new_children_by_id:
            new_children[val] = new_children_by_id
    return _build_upper({}) if not new_children else Upper(UpperBranch(new_children))


def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Checks internal invariants of the LeveledGSS structure.
    Raises AssertionError if an invariant is violated.
    """
    # Check recursive invariants on the inner structure.
    _validate_upper(gss.inner)

    # Invariant 2: If inner is an interface and empty exists, their accs must differ.
    if isinstance(gss.inner, Interface) and gss.empty is not None:
        if gss.inner.acc == gss.empty:
            raise AssertionError(
                "Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator."
            )
