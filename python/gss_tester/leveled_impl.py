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
        if self is other:
            return self

        match self.variant, other.variant:
            case Leaf(), Leaf():
                return self

            case Leaf(), BranchInner(other_children):
                if EPSILON in other_children:
                    return other  # other already contains the empty stack
                new_children = {t: d.copy() for t, d in other_children.items()}
                new_children[EPSILON] = {0: LeveledGSSInner.leaf()}
                return LeveledGSSInner(BranchInner(new_children))

            case BranchInner(self_children), Leaf():
                if EPSILON in self_children:
                    return self
                new_children = {t: d.copy() for t, d in self_children.items()}
                new_children[EPSILON] = {0: LeveledGSSInner.leaf()}
                return LeveledGSSInner(BranchInner(new_children))

            case BranchInner(self_children), BranchInner(other_children):
                new_children: Dict[Optional[T], Dict[int, 'LeveledGSSInner[T]']] = {}
                all_keys = self_children.keys() | other_children.keys()

                for t in all_keys:
                    self_by_depth = self_children.get(t, {})
                    other_by_depth = other_children.get(t, {})
                    
                    new_by_depth: Dict[int, 'LeveledGSSInner[T]'] = {}
                    all_depths = self_by_depth.keys() | other_by_depth.keys()

                    for d in all_depths:
                        self_child = self_by_depth.get(d)
                        other_child = other_by_depth.get(d)

                        if self_child and other_child:
                            new_by_depth[d] = self_child.merge(other_child)
                        elif self_child:
                            new_by_depth[d] = self_child
                        else:  # other_child
                            new_by_depth[d] = other_child
                    
                    if new_by_depth:
                        new_children[t] = new_by_depth
                
                result = LeveledGSSInner.internal(new_children)
                if result is None:
                    raise ValueError("Internal error: merge of LeveledGSSInner produced None")
                return result

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
                    for _, child in by_depth.items():
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
        if not isinstance(other, LeveledGSS):
            other_ref = other.to_reference_impl()
            if other_ref.is_empty():
                return self
            other_leveled = LeveledGSS.from_stacks(other_ref._stacks)
        else:
            other_leveled = other

        return _merge_leveled(self, other_leveled, memo={})

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

def _add_constant_to_children(
    children: MutableMapping[Optional[T], Dict[int, LeveledGSS[T, Acc]]],
    g_const: LeveledGSS[T, Acc],
    memo: Dict[Tuple[int, int], LeveledGSS[T, Acc]],
) -> None:
    v_const = g_const.variant
    assert isinstance(v_const, Constant)
    a, acc = v_const.node, v_const.acc

    match a.variant:
        case Leaf():
            child1 = LeveledGSS.with_acc(LeveledGSSInner.leaf(), acc)
            child2 = children.get(EPSILON, {}).get(0, LeveledGSS.empty())
            merged = _merge_leveled(child1, child2, memo)
            if EPSILON not in children:
                children[EPSILON] = {}
            children[EPSILON][0] = merged
        case BranchInner(ac):
            for t, by_depth in ac.items():
                if t not in children:
                    children[t] = {}
                for d, a_child in by_depth.items():
                    child1 = LeveledGSS.with_acc(a_child, acc)
                    child2 = children.get(t, {}).get(d, LeveledGSS.empty())
                    merged = _merge_leveled(child1, child2, memo)
                    children[t][d] = merged


def _merge_leveled(
    g1: LeveledGSS[T, Acc], g2: LeveledGSS[T, Acc], memo: Dict[Tuple[int, int], LeveledGSS[T, Acc]]
) -> LeveledGSS[T, Acc]:
    if g1 is g2: return g1
    if g1.is_empty(): return g2
    if g2.is_empty(): return g1

    key = (id(g1), id(g2)) if id(g1) < id(g2) else (id(g2), id(g1))
    if key in memo:
        return memo[key]

    res: LeveledGSS[T, Acc]
    v1, v2 = g1.variant, g2.variant

    if isinstance(v1, Branch) and isinstance(v2, Constant):
        g1, g2, v1, v2 = g2, g1, v2, v1

    match v1, v2:
        case Constant(a1, acc1), Constant(a2, acc2):
            if acc1 == acc2:
                res = LeveledGSS.with_acc(a1.merge(a2), acc1)
            else:
                new_children: Dict[Optional[T], Dict[int, LeveledGSS[T, Acc]]] = {}
                _add_constant_to_children(new_children, g1, memo)
                _add_constant_to_children(new_children, g2, memo)
                res = _canonicalize(LeveledGSS.internal(new_children))
        
        case Constant(), Branch():
            assert isinstance(v2, Branch)
            new_children = {t: d.copy() for t, d in v2.children.items()}
            _add_constant_to_children(new_children, g1, memo)
            res = _canonicalize(LeveledGSS.internal(new_children))

        case Branch(c1), Branch(c2):
            new_children = {}
            all_keys = c1.keys() | c2.keys()
            for t in all_keys:
                c1_by_depth = c1.get(t, {})
                c2_by_depth = c2.get(t, {})
                new_by_depth = {}
                all_depths = c1_by_depth.keys() | c2_by_depth.keys()
                for d in all_depths:
                    child1 = c1_by_depth.get(d, LeveledGSS.empty())
                    child2 = c2_by_depth.get(d, LeveledGSS.empty())
                    new_by_depth[d] = _merge_leveled(child1, child2, memo)
                if new_by_depth:
                    new_children[t] = new_by_depth
            res = _canonicalize(LeveledGSS.internal(new_children))
        
        case _, _:
            raise TypeError(f"Unhandled merge case: {type(v1)} and {type(v2)}")

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
