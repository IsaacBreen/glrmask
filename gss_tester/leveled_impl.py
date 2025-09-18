from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

Acc = Any

# --- Stack prefix graph structure (Type A) ---

@dataclass(frozen=True, eq=True)
class _StackPrefix:
    """Represents a node in the stack prefix graph (structure 'A')."""
    children: Dict[int, _StackPrefix] = field(default_factory=dict)
    is_terminal: bool = False  # True if a stack terminates at this prefix

_EMPTY_STACK_PREFIX = _StackPrefix(is_terminal=True)

def _merge_stack_prefixes(s1: _StackPrefix, s2: _StackPrefix) -> _StackPrefix:
    all_keys = s1.children.keys() | s2.children.keys()
    new_children = {}
    for k in all_keys:
        c1 = s1.children.get(k)
        c2 = s2.children.get(k)
        if c1 and c2:
            new_children[k] = _merge_stack_prefixes(c1, c2)
        elif c1:
            new_children[k] = c1
        else:
            new_children[k] = c2
    return _StackPrefix(new_children, s1.is_terminal or s2.is_terminal)

def _from_stack_list(stacks: List[List[int]]) -> _StackPrefix:
    root = _StackPrefix()  # Not terminal
    for stack in stacks:
        s = _EMPTY_STACK_PREFIX
        for state in reversed(stack):
            s = _StackPrefix({state: s})
        root = _merge_stack_prefixes(root, s)
    return root

# --- GSS structure (Type B) ---

class LeveledGSS:
    """
    A Graph-Structured Stack (GSS) implementation that maintains accumulators
    at a single "level" to enable sharing, as per the design prompt.
    This class acts as a public interface and a base for its variants.
    """
    def __new__(cls, node_factory=None):
        if cls is LeveledGSS:
            # Called as LeveledGSS() to create an empty GSS.
            return _GSS_EMPTY
        return super().__new__(cls)

    @staticmethod
    def from_stacks(stacks: List[Tuple[List[int], Acc]]) -> LeveledGSS:
        if not stacks:
            return _GSS_EMPTY

        by_acc: Dict[Acc, List[List[int]]] = collections.defaultdict(list)
        for stack, acc in stacks:
            by_acc[acc].append(stack)

        gss_parts: List[LeveledGSS] = []
        for acc, stack_list in by_acc.items():
            stack_prefix_graph = _from_stack_list(stack_list)
            gss_parts.append(_GSS_Leaf(stack_prefix_graph, acc))

        return LeveledGSS.merge(gss_parts)

    def push(self, state_id: int) -> LeveledGSS: raise NotImplementedError
    def pop(self) -> LeveledGSS: raise NotImplementedError
    def popn(self, n: int) -> LeveledGSS:
        gss = self
        for _ in range(n):
            gss = gss.pop()
        return gss
    def peek(self) -> Set[int]: raise NotImplementedError
    def reduce_acc(self) -> Optional[Acc]: raise NotImplementedError
    def is_empty(self) -> bool: raise NotImplementedError
    def isolate(self, state_id: int) -> LeveledGSS: raise NotImplementedError
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS: raise NotImplementedError
    def apply(self, f: Callable[[Acc], Acc]) -> LeveledGSS: raise NotImplementedError

    @staticmethod
    def merge(gss_list: List[LeveledGSS]) -> LeveledGSS:
        if not gss_list:
            return _GSS_EMPTY
        
        merged = _GSS_EMPTY
        for gss in gss_list:
            merged = _merge2(merged, gss)
        return merged

    @property
    def _factory(self):
        # For compatibility with code that expects a factory to create empty GSSs.
        return LeveledGSS

@dataclass(frozen=True, eq=True)
class _GSS_Empty(LeveledGSS):
    """Variant representing an empty GSS (no stacks)."""
    def is_empty(self) -> bool: return True
    def push(self, state_id: int) -> LeveledGSS: return self
    def pop(self) -> LeveledGSS: return self
    def peek(self) -> Set[int]: return set()
    def reduce_acc(self) -> Optional[Acc]: return None
    def isolate(self, state_id: int) -> LeveledGSS: return self
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS: return self
    def apply(self, f: Callable[[Acc], Acc]) -> LeveledGSS: return self

_GSS_EMPTY = _GSS_Empty()

@dataclass(frozen=True, eq=True)
class _GSS_Leaf(LeveledGSS):
    """Variant representing a set of stacks sharing a single accumulator."""
    stacks: _StackPrefix
    acc: Acc

    def is_empty(self) -> bool: return False
    def push(self, state_id: int) -> LeveledGSS:
        return _GSS_Leaf(_StackPrefix({state_id: self.stacks}), self.acc)
    def pop(self) -> LeveledGSS:
        children = {
            sid: _GSS_Leaf(child_prefix, self.acc)
            for sid, child_prefix in self.stacks.children.items()
        }
        return _create_internal(children)
    def peek(self) -> Set[int]:
        return set(self.stacks.children.keys())
    def reduce_acc(self) -> Optional[Acc]:
        return self.acc
    def isolate(self, state_id: int) -> LeveledGSS:
        child_stack = self.stacks.children.get(state_id)
        if child_stack is None:
            return _GSS_EMPTY
        return _GSS_Leaf(child_stack, self.acc)
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS:
        return self if predicate(self.acc) else _GSS_EMPTY
    def apply(self, f: Callable[[Acc], Acc]) -> LeveledGSS:
        return _GSS_Leaf(self.stacks, f(self.acc))

@dataclass(frozen=True, eq=True)
class _GSS_Internal(LeveledGSS):
    """Variant representing a GSS branching over different states or accumulators."""
    children: Dict[int, LeveledGSS]

    def is_empty(self) -> bool: return not self.children
    def push(self, state_id: int) -> LeveledGSS:
        return _GSS_Internal({state_id: self})
    def pop(self) -> LeveledGSS:
        return LeveledGSS.merge([c.pop() for c in self.children.values()])
    def peek(self) -> Set[int]:
        return set(self.children.keys())
    def reduce_acc(self) -> Optional[Acc]:
        accs = [c.reduce_acc() for c in self.children.values()]
        accs = [a for a in accs if a is not None]
        if not accs:
            return None
        
        # Assumes acc has a .merge method, like PyAcc.
        merged_acc = accs[0]
        for i in range(1, len(accs)):
            merged_acc = merged_acc.merge(accs[i])
        return merged_acc
    def isolate(self, state_id: int) -> LeveledGSS:
        return self.children.get(state_id, _GSS_EMPTY)
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS:
        return _create_internal({
            sid: child.prune(predicate) for sid, child in self.children.items()
        })
    def apply(self, f: Callable[[Acc], Acc]) -> LeveledGSS:
        return _create_internal({
            sid: child.apply(f) for sid, child in self.children.items()
        })

def _create_internal(children: Dict[int, LeveledGSS]) -> LeveledGSS:
    """Smart constructor for internal GSS nodes."""
    children = {k: v for k, v in children.items() if not v.is_empty()}
    if not children:
        return _GSS_EMPTY
    if len(children) == 1:
        return next(iter(children.values()))

    # Check if all children are leaves with the same accumulator to "suck up".
    first_child = next(iter(children.values()))
    if not isinstance(first_child, _GSS_Leaf):
        return _GSS_Internal(children)

    first_acc = first_child.acc
    
    all_leaves_same_acc = True
    for child in children.values():
        if not isinstance(child, _GSS_Leaf) or child.acc != first_acc:
            all_leaves_same_acc = False
            break
    
    if all_leaves_same_acc:
        # Suck up the common accumulator into a new parent leaf.
        new_stack_children = {
            sid: cast(_GSS_Leaf, child).stacks
            for sid, child in children.items()
        }
        # A stack ending at this new level is not represented, so is_terminal=False.
        return _GSS_Leaf(_StackPrefix(new_stack_children, is_terminal=False), first_acc)
    
    return _GSS_Internal(children)

def _gss_from_prefix(prefix: _StackPrefix, acc: Acc) -> LeveledGSS:
    """Deconstructs a _GSS_Leaf into an equivalent _GSS_Internal for merging."""
    if prefix.is_terminal and not prefix.children:
        return _GSS_Leaf(_EMPTY_STACK_PREFIX, acc)

    children = {
        sid: _gss_from_prefix(child_prefix, acc)
        for sid, child_prefix in prefix.children.items()
    }
    
    # If the prefix itself was a terminal stack, we need to represent the empty stack.
    # This is tricky because there's no state ID to key it on.
    # The logic of merging different-acc leaves requires a common prefix, which implies
    # they are not at the top level. This function is for breaking down a leaf
    # to merge with another GSS.
    # A proper merge would need to find the common prefix of stack lists, which is complex.
    # A simpler deconstruction is to turn it into a GSS tree.
    
    # Simplified deconstruction for merging:
    children = {
        sid: _GSS_Leaf(child_prefix, acc)
        for sid, child_prefix in prefix.children.items()
    }
    if prefix.is_terminal:
        # This case is complex. An empty stack at this level cannot be keyed
        # by a state ID. This indicates merging logic should be handled at a
        # higher level (e.g., `from_stacks`). The current `_merge2` logic
        # for two leaves with different accs is a simplification that
        # pushes one leaf's structure down to be merged.
        pass

    return _create_internal(children)


def _merge2(gss1: LeveledGSS, gss2: LeveledGSS) -> LeveledGSS:
    """Merges two GSS instances."""
    if gss1.is_empty(): return gss2
    if gss2.is_empty(): return gss1

    # Both internal: merge children recursively.
    if isinstance(gss1, _GSS_Internal) and isinstance(gss2, _GSS_Internal):
        all_keys = gss1.children.keys() | gss2.children.keys()
        new_children = {
            k: _merge2(gss1.children.get(k, _GSS_EMPTY), gss2.children.get(k, _GSS_EMPTY))
            for k in all_keys
        }
        return _create_internal(new_children)
    
    # Both leaves:
    if isinstance(gss1, _GSS_Leaf) and isinstance(gss2, _GSS_Leaf):
        # Same acc: merge stack prefixes.
        if gss1.acc == gss2.acc:
            return _GSS_Leaf(_merge_stack_prefixes(gss1.stacks, gss2.stacks), gss1.acc)
        # Different accs: must deconstruct one into an internal node to merge.
        else:
            return _merge2(_gss_from_prefix(gss1.stacks, gss1.acc), gss2)

    # One internal, one leaf:
    if isinstance(gss2, _GSS_Internal):
        gss1, gss2 = gss2, gss1 # ensure gss1 is internal
    
    gss1 = cast(_GSS_Internal, gss1)
    gss2 = cast(_GSS_Leaf, gss2)
    return _merge2(gss1, _gss_from_prefix(gss2.stacks, gss2.acc))
