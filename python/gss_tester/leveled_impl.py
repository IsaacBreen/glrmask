from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from collections import defaultdict

from .interface import GSS, T, Acc, Mergeable
from .reference_impl import ReferenceGSS


# Sentinel used to represent "epsilon" transitions (i.e., empty stack) inside the
# leveled edge maps. This allows us to represent an empty stack alongside non-empty
# stacks even when the parent node is an Internal node.
# This is consistent with the interface method isolate(None) (None means "empty top").
EPSILON: None = None

@dataclass(frozen=True, eq=True)
class Unit(Mergeable):
    def merge(self, other: 'Unit') -> 'Unit':
        return self

UNIT = Unit()


# -----------------------------
# Inner GSS (LeveledGSSInner)
# -----------------------------

@dataclass(frozen=True, eq=True)
class BranchInner(Generic[T]):
    children: Dict[Optional[T], Dict[int, 'LeveledGSSInner[T]']]

@dataclass(frozen=True, eq=True)
class Leaf:
    """Represents a GSS containing just one empty stack: {[]}, a leaf in the trie. """
    pass

LeveledGSSInnerVariant = Union[BranchInner[T], Leaf]

@dataclass(frozen=True, eq=True)
class LeveledGSSInner(Generic[T]):
    """
    An inner Leveled GSS that only represents a set of stacks, without accumulators.
    The accumulator type is fixed to `Unit`.
    This is used for the inner nodes in the main LeveledGSS.
    """
    variant: LeveledGSSInnerVariant

    # --- Constructors ---
    @classmethod
    def leaf(cls) -> 'LeveledGSSInner[T]':
        return cls(Leaf())

    @classmethod
    def internal(cls, children: Dict[Optional[T], Dict[int, 'LeveledGSSInner[T]']]) -> Optional['LeveledGSSInner[T]']:
        cleaned = {t: depths for t, depths in children.items() if depths}
        
        if not cleaned:
            return None

        if set(cleaned.keys()) == {EPSILON} and set(cleaned[EPSILON].keys()) == {0}:
            if isinstance(cleaned[EPSILON][0].variant, Leaf):
                return cls.leaf()
        
        return cls(BranchInner(cleaned))

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Unit]]) -> Optional['LeveledGSSInner[T]']:
        seqs = {tuple(s[0]) for s in stacks}
        return _build_inner_from_seqs(cls, seqs)

    def pop(self) -> Optional['LeveledGSSInner[T]']:
        from functools import reduce
        match self.variant:
            case Leaf():
                return None
            case BranchInner(children):
                popped_children = [
                    child
                    for t, by_depth in children.items()
                    if t is not EPSILON
                    for child in by_depth.values()
                ]
                if not popped_children:
                    return None
                return reduce(lambda acc, next: acc.merge(next), popped_children)

    def isolate(self, value: Optional[T]) -> Optional['LeveledGSSInner[T]']:
        match self.variant:
            case Leaf():
                return self if value is None else None
            case BranchInner(children):
                key = EPSILON if value is None else value
                by_depth = children.get(key, {})
                if not by_depth:
                    return None

                if key is EPSILON:
                    return self.leaf()

                return LeveledGSSInner.internal({key: by_depth})

    def merge(self, other: 'LeveledGSSInner[T]') -> 'LeveledGSSInner[T]':
        return _merge_inner_recursive(self, other, {})

    def peek(self) -> Set[T]:
        match self.variant:
            case Leaf():
                return set()
            case BranchInner(children):
                return {cast(T, t) for t in children.keys() if t is not EPSILON}

    # --- LeveledA specific helpers ---
    def has_epsilon(self) -> bool:
        match self.variant:
            case Leaf():
                return True
            case BranchInner(children):
                return EPSILON in children

    def max_depth(self) -> int:
        match self.variant:
            case Leaf():
                return 0
            case BranchInner(children):
                max_d = 0
                for t, by_depth in children.items():
                    if t is EPSILON: continue
                    max_d = max(max_d, *by_depth.keys())
                return max_d

    def enumerate_stacks(self) -> Iterator[Tuple[T, ...]]:
        match self.variant:
            case Leaf():
                yield tuple()
            case BranchInner(children):
                for t, by_depth in children.items():
                    for _, child in by_depth.items():
                        for tail in child.enumerate_stacks():
                            if t is EPSILON:
                                yield tail
                            else:
                                yield tail + (cast(T, t),)
    
    def validate_invariants(self) -> None:
        errors: List[str] = []
        _validate_inner(self, errors)
        if errors:
            raise ValueError("LeveledGSSInner invariant violations:\n" + "\n".join(f"- {e}" for e in errors))

def _build_inner_from_seqs(cls: Type[LeveledGSSInner[T]], seqs: Iterable[Tuple[T, ...]]) -> Optional[LeveledGSSInner[T]]:
    buckets: DefaultDict[Optional[T], DefaultDict[int, List[Tuple[T, ...]]]] = defaultdict(lambda: defaultdict(list))
    has_empty = False
    has_any = False
    for s in seqs:
        has_any = True
        if not s:
            has_empty = True
        else:
            t, d, prefix = s[-1], len(s), s[:-1]
            buckets[t][d].append(prefix)
    
    if not has_any:
        return None

    children: Dict[Optional[T], Dict[int, LeveledGSSInner[T]]] = {}
    if has_empty:
        children[EPSILON] = {0: cls.leaf()}

    for t, by_depth in buckets.items():
        children[t] = {}
        for d, group in by_depth.items():
            child = _build_inner_from_seqs(cls, group)
            if child is not None:
                children[t][d] = child
    
    return cls.internal(children)

def _validate_inner(a: Optional[LeveledGSSInner[T]], errors: List[str]) -> None:
    if a is None:
        return
    match a.variant:
        case Leaf():
            return
        case BranchInner(children):
            for t, by_depth in children.items():
                for d, ch in by_depth.items():
                    if t is EPSILON:
                        if d != 0: errors.append("Inner-layer EPSILON edge with non-zero depth")
                        if not isinstance(ch.variant, Leaf): errors.append("Inner-layer EPSILON edge must point to Leaf")
                    else:
                        if d < 1: errors.append("Inner-layer non-epsilon edge has depth < 1")
                        child_max = ch.max_depth()
                        if d != child_max + 1:
                            errors.append(f"Inner-layer edge depth mismatch: expected {child_max + 1}, found {d}")
                    _validate_inner(ch, errors)

def _merge_inner_recursive(
    n1: LeveledGSSInner[T],
    n2: LeveledGSSInner[T],
    memo: Dict[Tuple[int, int], LeveledGSSInner[T]]
) -> LeveledGSSInner[T]:
    if n1 == n2:
        return n1
    key = (id(n1), id(n2)) if id(n1) < id(n2) else (id(n2), id(n1))
    if key in memo:
        return memo[key]

    res: LeveledGSSInner[T]
    match (n1.variant, n2.variant):
        case (Leaf(), Leaf()):
            res = n1
        case (Leaf(), BranchInner(c2)):
            if EPSILON in c2:
                res = n2
            else:
                new_children = {t: d.copy() for t, d in c2.items()}
                new_children[EPSILON] = {0: LeveledGSSInner.leaf()}
                merged_node = LeveledGSSInner.internal(new_children)
                assert merged_node is not None, "Adding a child should not result in an empty node"
                res = merged_node
        case (BranchInner(_), Leaf()):
            res = _merge_inner_recursive(n2, n1, memo)
        case (BranchInner(c1), BranchInner(c2)):
            new_children: Dict[Optional[T], Dict[int, LeveledGSSInner[T]]] = defaultdict(dict)
            all_keys = c1.keys() | c2.keys()

            for t in all_keys:
                by_depth1 = c1.get(t, {})
                by_depth2 = c2.get(t, {})
                all_depths = by_depth1.keys() | by_depth2.keys()

                for d in all_depths:
                    child1 = by_depth1.get(d)
                    child2 = by_depth2.get(d)

                    if child1 and child2:
                        merged_child = _merge_inner_recursive(child1, child2, memo)
                        new_children[t][d] = merged_child
                    elif child1:
                        new_children[t][d] = child1
                    elif child2:
                        new_children[t][d] = child2
            
            merged_node = LeveledGSSInner.internal(dict(new_children))
            assert merged_node is not None, "Merging non-empty nodes should result in a non-empty node"
            res = merged_node

    memo[key] = res
    return res


# -----------------------------
# B-layer (main) node variants
# -----------------------------

@dataclass(frozen=True, eq=True)
class Constant(Generic[T, Acc]):
    node: LeveledGSSInner[T]
    acc: Acc

@dataclass(frozen=True, eq=True)
class Branch(Generic[T, Acc]):
    children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']]

@dataclass(frozen=True, eq=True)
class Empty:
    pass

LeveledVariant = Union[Constant[T, Acc], Branch[T, Acc], Empty]

# -----------------------------
# Main LeveledGSS
# -----------------------------

@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    variant: LeveledVariant

    @classmethod
    def empty(cls) -> 'LeveledGSS[T, Acc]':
        return cls(Empty())

    @classmethod
    def with_acc(cls, node: Optional[LeveledGSSInner[T]], acc: Acc) -> 'LeveledGSS[T, Acc]':
        if node is None:
            return cls.empty()
        return cls(Constant(node=node, acc=acc))

    @classmethod
    def internal(cls, children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']]) -> 'LeveledGSS[T, Acc]':
        cleaned: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = {}
        for t, depths in children.items():
            c2 = {d: ch for d, ch in depths.items() if not ch.is_empty()}
            if c2:
                cleaned[t] = c2
        if not cleaned:
            return cls.empty()
        return cls(Branch(children=cleaned))

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        merged: Dict[Tuple[T, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc
        return _build_from_stack_map(cls, merged)

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        if self.is_empty():
            return self
        max_depth = _max_depth(self)
        new_depth = max_depth + 1
        node = self.internal({value: {new_depth: self}})
        return _canonicalize(node)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        match self.variant:
            case Empty():
                return self
            case Constant(node=a, acc=acc):
                popped_node = a.pop()
                return LeveledGSS.with_acc(popped_node, acc)
            case Branch(children=children):
                result = LeveledGSS.empty()
                for t, by_depth in children.items():
                    if t is EPSILON:
                        continue
                    for child in by_depth.values():
                        result = result.merge(child)
                return result

    def is_empty(self) -> bool:
        return isinstance(self.variant, Empty)

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        match self.variant:
            case Empty():
                return self
            case Branch(children=children):
                key = EPSILON if value is None else value
                by_depth = children.get(key, {})
                if not by_depth:
                    return LeveledGSS.empty()
                return LeveledGSS.internal({key: by_depth})
            case Constant(node=a, acc=acc):
                isolated_node = a.isolate(value)
                return LeveledGSS.with_acc(isolated_node, acc)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        memo_b: Dict[int, 'LeveledGSS[T, Acc]'] = {}

        def apply_b(node: 'LeveledGSS[T, Acc]') -> 'LeveledGSS[T, Acc]':
            key = id(node)
            if key in memo_b: return memo_b[key]

            res: 'LeveledGSS[T, Acc]'
            match node.variant:
                case Empty():
                    res = node
                case Constant(node=inner_node, acc=acc):
                    new_acc = func(acc)
                    res = node if new_acc == acc else LeveledGSS.with_acc(inner_node, new_acc)
                case Branch(children=children):
                    changed = False
                    new_children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = {}
                    for t, by_depth in children.items():
                        inner_map: Dict[int, 'LeveledGSS[T, Acc]'] = {}
                        for d, child in by_depth.items():
                            new_child = apply_b(child)
                            inner_map[d] = new_child
                            if new_child is not child: changed = True
                        new_children[t] = inner_map
                    res = _canonicalize(LeveledGSS.internal(new_children)) if changed else node

            memo_b[key] = res
            return res

        return apply_b(self)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        def prune_b(node: 'LeveledGSS[T, Acc]') -> 'LeveledGSS[T, Acc]':
            match node.variant:
                case Empty():
                    return node
                case Constant(acc=acc):
                    return node if predicate(acc) else LeveledGSS.empty()
                case Branch(children=children):
                    new_children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = {}
                    for t, by_depth in children.items():
                        new_by_depth: Dict[int, 'LeveledGSS[T, Acc]'] = {}
                        for d, ch in by_depth.items():
                            pr = prune_b(ch)
                            if not pr.is_empty(): new_by_depth[d] = pr
                        if new_by_depth: new_children[t] = new_by_depth
                    return _canonicalize(LeveledGSS.internal(new_children))
        return prune_b(self)

    def merge(self, other: 'GSS[T, Acc]') -> 'LeveledGSS[T, Acc]':
        other_leveled: 'LeveledGSS[T, Acc]'
        if isinstance(other, LeveledGSS):
            other_leveled = other
        else:
            ref = other.to_reference_impl()
            other_leveled = LeveledGSS.from_stacks(ref._stacks)

        return _merge_leveled(self, other_leveled, {})

    def peek(self) -> Set[T]:
        match self.variant:
            case Empty():
                return set()
            case Branch(children=children):
                return {cast(T, t) for t in children.keys() if t is not EPSILON}
            case Constant(node=node):
                return node.peek()

    def reduce_acc(self) -> Optional[Acc]:
        match self.variant:
            case Empty():
                return None
            case Constant(acc=acc):
                return acc
            case Branch(children=children):
                acc_opt: Optional[Acc] = None
                for _, by_depth in children.items():
                    for _, ch in by_depth.items():
                        sub = ch.reduce_acc()
                        if sub is not None:
                            acc_opt = sub if acc_opt is None else acc_opt.merge(sub)
                return acc_opt

    def to_reference_impl(self) -> 'ReferenceGSS[T, Acc]':
        stacks: List[Tuple[List[T], Acc]] = list(_enumerate(self))
        return ReferenceGSS.from_stacks(stacks)

    def validate_invariants(self) -> None:
        errors: List[str] = []
        _validate(self, errors)
        if errors:
            raise ValueError("LeveledGSS invariant violations:\n" + "\n".join(f"- {e}" for e in errors))

def _distribute_acc(
    node: LeveledGSSInner[T],
    acc: Acc,
    memo: Dict[int, 'LeveledGSS[T, Acc]']
) -> 'LeveledGSS[T, Acc]':
    key = id(node)
    if key in memo:
        return memo[key]

    res: 'LeveledGSS[T, Acc]'
    match node.variant:
        case Leaf():
            res = LeveledGSS.with_acc(LeveledGSSInner.leaf(), acc)
        case BranchInner(children):
            new_children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = defaultdict(dict)
            for t, by_depth in children.items():
                for d, child_a in by_depth.items():
                    new_children[t][d] = _distribute_acc(child_a, acc, memo)
            res = LeveledGSS.internal(dict(new_children))
    
    memo[key] = res
    return res


def _merge_leveled(
    b1: 'LeveledGSS[T, Acc]',
    b2: 'LeveledGSS[T, Acc]',
    memo: Dict[Tuple[int, int], 'LeveledGSS[T, Acc]']
) -> 'LeveledGSS[T, Acc]':
    if b1.is_empty():
        return b2
    if b2.is_empty():
        return b1
    if b1 == b2:
        return b1

    key = (id(b1), id(b2)) if id(b1) < id(b2) else (id(b2), id(b1))
    if key in memo:
        return memo[key]

    res: 'LeveledGSS[T, Acc]'
    match (b1.variant, b2.variant):
        case (Constant(a1, acc1), Constant(a2, acc2)):
            if a1 == a2:
                res = LeveledGSS.with_acc(a1, acc1.merge(acc2))
            elif acc1 == acc2:
                merged_a = a1.merge(a2)
                res = LeveledGSS.with_acc(merged_a, acc1)
            else:
                b1_dist = _distribute_acc(a1, acc1, {})
                b2_dist = _distribute_acc(a2, acc2, {})
                res = _merge_leveled(b1_dist, b2_dist, memo)
        
        case (Constant(a1, acc1), Branch(c2)):
            if isinstance(a1.variant, Leaf):
                # This is the GSS representing {[]: acc1}. Merge it into the Branch.
                empty_stack_node = b1
                
                # Get the part of b2 representing empty stacks.
                b2_empty_children_by_depth = c2.get(EPSILON, {})
                b2_empty_child = b2_empty_children_by_depth.get(0, LeveledGSS.empty())
                
                # Merge our empty stack node with b2's empty stack node.
                # This recursive call is safe as it will hit (Constant, Constant) or (Constant, Empty).
                merged_empty_child = _merge_leveled(empty_stack_node, b2_empty_child, memo)
                
                # Construct the new children map for the result, handling the EPSILON child separately.
                new_children = {t: d for t, d in c2.items() if t is not EPSILON}
                if not merged_empty_child.is_empty():
                    new_children[EPSILON] = {0: merged_empty_child}

                res = _canonicalize(LeveledGSS.internal(new_children))
            else:
                # Original logic for non-leaf constants: distribute the acc and merge the resulting branches.
                b1_dist = _distribute_acc(a1, acc1, {})
                res = _merge_leveled(b1_dist, b2, memo)

        case (Branch(_), Constant(a2, acc2)):
            res = _merge_leveled(b2, b1, memo)

        case (Branch(c1), Branch(c2)):
            new_children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = defaultdict(dict)
            all_keys = c1.keys() | c2.keys()

            for t in all_keys:
                by_depth1 = c1.get(t, {})
                by_depth2 = c2.get(t, {})
                all_depths = by_depth1.keys() | by_depth2.keys()

                for d in all_depths:
                    child1 = by_depth1.get(d)
                    child2 = by_depth2.get(d)

                    if child1 and child2:
                        merged_child = _merge_leveled(child1, child2, memo)
                        if not merged_child.is_empty():
                            new_children[t][d] = merged_child
                    elif child1:
                        new_children[t][d] = child1
                    elif child2:
                        new_children[t][d] = child2
            
            res = _canonicalize(LeveledGSS.internal(dict(new_children)))

    memo[key] = res
    return res

def _build_from_stack_map(cls: Type[LeveledGSS[T, Acc]], stack_map: Dict[Tuple[T, ...], Acc]) -> LeveledGSS[T, Acc]:
    if not stack_map:
        return cls.empty()

    accs = list(stack_map.values())
    if all(acc == accs[0] for acc in accs):
        acc = accs[0]
        stacks = [(list(k), UNIT) for k in stack_map.keys()]
        inner = LeveledGSSInner.from_stacks(stacks) # type: ignore
        return _canonicalize(cls.with_acc(inner, acc))

    buckets: DefaultDict[Optional[T], DefaultDict[int, Dict[Tuple[T, ...], Acc]]] = defaultdict(lambda: defaultdict(dict))
    for seq, acc in stack_map.items():
        n = len(seq)
        if n == 0:
            buckets[EPSILON][0][tuple()] = acc
        else:
            t, prefix = seq[-1], seq[:-1]
            buckets[t][n][prefix] = acc

    children: Dict[Optional[T], Dict[int, LeveledGSS[T, Acc]]] = {}
    for t, by_depth in buckets.items():
        children[t] = {}
        for depth, submap in by_depth.items():
            child = _build_from_stack_map(cls, submap)
            if not child.is_empty():
                children[t][depth] = child

    return _canonicalize(cls.internal(children))

def _canonicalize(node: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
    match node.variant:
        case Empty() | Constant():
            return node
        case Branch(children):
            acc_val: Optional[Acc] = None
            all_with_acc = True
            child_nodes: List[Tuple[Optional[T], int, LeveledGSSInner[T]]] = []

            for t, by_depth in children.items():
                for d, ch in by_depth.items():
                    if not all_with_acc: break
                    match ch.variant:
                        case Constant(node=a_node, acc=child_acc):
                            if acc_val is None: acc_val = child_acc
                            elif child_acc != acc_val: all_with_acc = False
                            child_nodes.append((t, d, a_node))
                        case _:
                            all_with_acc = False
                if not all_with_acc: break

            if all_with_acc and acc_val is not None:
                a_children: Dict[Optional[T], Dict[int, LeveledGSSInner[T]]] = defaultdict(dict)
                for t, d, a_node in child_nodes:
                    a_children[t][d] = a_node
                a_node = LeveledGSSInner.internal(dict(a_children))
                return LeveledGSS.with_acc(a_node, acc_val)

            return node

def _enumerate(g: LeveledGSS[T, Acc]) -> Iterator[Tuple[List[T], Acc]]:
    match g.variant:
        case Empty():
            return
        case Constant(node=node, acc=acc):
            for seq in node.enumerate_stacks():
                yield (list(seq), acc)
        case Branch(children=children):
            for t, by_depth in children.items():
                for _, child in by_depth.items():
                    for tail, acc in _enumerate(child):
                        if t is not EPSILON:
                            tail.append(cast(T, t))
                        yield (tail, acc)

def _to_stack_map(g: LeveledGSS[T, Acc]) -> Dict[Tuple[T, ...], Acc]:
    d: Dict[Tuple[T, ...], Acc] = {}
    for seq_list, acc in _enumerate(g):
        d[tuple(seq_list)] = acc
    return d

def _max_depth(g: LeveledGSS[T, Acc]) -> int:
    match g.variant:
        case Empty():
            return 0
        case Constant(node=node):
            return node.max_depth()
        case Branch(children=children):
            max_d = 0
            for _, by_depth in children.items():
                max_d = max(max_d, *by_depth.keys())
            return max_d

def _validate(g: LeveledGSS[T, Acc], errors: List[str]) -> None:
    match g.variant:
        case Empty():
            return
        case Constant(node=node):
            try:
                node.validate_invariants()
            except ValueError as e:
                errors.append(f"Contained LeveledGSSInner has invariant violations: {e}")
            return
        case Branch(children=children):
            acc_set: List[Acc] = []
            every_child_with_acc = True
            for t, by_depth in children.items():
                for d, ch in by_depth.items():
                    if ch.is_empty():
                        errors.append("Internal node contains an Empty child; should be pruned")
                    
                    if t is EPSILON:
                        if d != 0: errors.append("EPSILON edge at B-layer with non-zero depth")
                    else:
                        if d < 1: errors.append("Non-epsilon B-layer edge has depth < 1")
                        child_max = _max_depth(ch)
                        if d != child_max + 1:
                            errors.append(f"B-layer edge depth mismatch: expected {child_max + 1}, found {d}")

                    match ch.variant:
                        case Constant(acc=acc): acc_set.append(acc)
                        case _: every_child_with_acc = False
                    _validate(ch, errors)

            if every_child_with_acc and acc_set and all(a == acc_set[0] for a in acc_set):
                errors.append("Internal node has all Constant children with equal accs; should be sucked up")
