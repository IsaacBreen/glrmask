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
class InternalInner(Generic[T]):
    children: Dict[Optional[T], Dict[int, 'LeveledGSSInner[T]']]

@dataclass(frozen=True, eq=True)
class RootInner:
    """Represents a GSS containing just one empty stack: {[]}. """
    pass

LeveledGSSInnerVariant = Union[InternalInner[T], RootInner]

@dataclass(frozen=True, eq=True)
class LeveledGSSInner(GSS[T, Unit], Generic[T]):
    """
    An inner Leveled GSS that only represents a set of stacks, without accumulators.
    The accumulator type is fixed to `Unit`.
    This is used for the inner nodes in the main LeveledGSS.
    """
    variant: Optional[LeveledGSSInnerVariant]

    # --- Constructors ---
    @classmethod
    def empty(cls) -> 'LeveledGSSInner[T]':
        return cls(None)

    @classmethod
    def root(cls) -> 'LeveledGSSInner[T]':
        return cls(RootInner())

    @classmethod
    def internal(cls, children: Dict[Optional[T], Dict[int, 'LeveledGSSInner[T]']]) -> 'LeveledGSSInner[T]':
        cleaned: Dict[Optional[T], Dict[int, 'LeveledGSSInner[T]']] = {}
        for t, depths in children.items():
            c2 = {d: ch for d, ch in depths.items() if not ch.is_empty()}
            if c2:
                cleaned[t] = c2
        
        if not cleaned:
            return cls.empty()

        if set(cleaned.keys()) == {EPSILON} and set(cleaned[EPSILON].keys()) == {0}:
            if isinstance(cleaned[EPSILON][0].variant, RootInner):
                return cls.root()
        
        return cls(InternalInner(cleaned))

    # --- GSS Interface ---
    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Unit]]) -> 'LeveledGSSInner[T]':
        seqs = {tuple(s[0]) for s in stacks}
        return _build_inner_from_seqs(cls, seqs)

    def push(self, value: T) -> 'LeveledGSSInner[T]':
        if self.is_empty():
            return self
        max_d = self.max_depth()
        new_depth = max_d + 1
        return self.internal({value: {new_depth: self}})

    def pop(self) -> 'LeveledGSSInner[T]':
        match self.variant:
            case None:
                return self
            case RootInner():
                return self.empty()
            case InternalInner(children):
                result = self.empty()
                for t, by_depth in children.items():
                    if t is EPSILON:
                        continue
                    for _, child in by_depth.items():
                        result = result.merge(child)
                return result

    def is_empty(self) -> bool:
        return self.variant is None

    def isolate(self, value: Optional[T]) -> 'LeveledGSSInner[T]':
        match self.variant:
            case None:
                return self
            case RootInner():
                return self if value is None else self.empty()
            case InternalInner(children):
                key = EPSILON if value is None else value
                by_depth = children.get(key, {})
                if not by_depth:
                    return self.empty()
                
                if key is EPSILON:
                    return self.root()

                return self.internal({key: by_depth})

    def apply(self, func: Callable[[Unit], Unit]) -> 'LeveledGSSInner[T]':
        return self

    def prune(self, predicate: Callable[[Unit], bool]) -> 'LeveledGSSInner[T]':
        return self if predicate(UNIT) else self.empty()

    def merge(self, other: 'GSS[T, Unit]') -> 'LeveledGSSInner[T]':
        if other is self or other.is_empty():
            return self
        if self.is_empty():
            return cast(LeveledGSSInner[T], other)
        
        s1 = {s for s in self.enumerate_stacks()}
        s2 = {s for s in cast(LeveledGSSInner[T], other).enumerate_stacks()}
        return _build_inner_from_seqs(self.__class__, s1.union(s2))

    def peek(self) -> Set[T]:
        match self.variant:
            case None | RootInner():
                return set()
            case InternalInner(children):
                return {cast(T, t) for t in children.keys() if t is not EPSILON}

    def reduce_acc(self) -> Optional[Unit]:
        return UNIT if not self.is_empty() else None

    def to_reference_impl(self) -> 'ReferenceGSS[T, Unit]':
        stacks = [(list(vals), UNIT) for vals in self.enumerate_stacks()]
        return ReferenceGSS.from_stacks(stacks)

    # --- LeveledA specific helpers ---
    def has_epsilon(self) -> bool:
        match self.variant:
            case RootInner():
                return True
            case InternalInner(children):
                return EPSILON in children
            case _:
                return False

    def max_depth(self) -> int:
        match self.variant:
            case None | RootInner():
                return 0
            case InternalInner(children):
                max_d = 0
                for t, by_depth in children.items():
                    if t is EPSILON: continue
                    max_d = max(max_d, *by_depth.keys())
                return max_d

    def enumerate_stacks(self) -> Iterator[Tuple[T, ...]]:
        match self.variant:
            case None:
                return
            case RootInner():
                yield tuple()
            case InternalInner(children):
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

def _build_inner_from_seqs(cls: Type[LeveledGSSInner[T]], seqs: Iterable[Tuple[T, ...]]) -> LeveledGSSInner[T]:
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
        return cls.empty()

    children: Dict[Optional[T], Dict[int, LeveledGSSInner[T]]] = {}
    if has_empty:
        children[EPSILON] = {0: cls.root()}

    for t, by_depth in buckets.items():
        children[t] = {}
        for d, group in by_depth.items():
            child = _build_inner_from_seqs(cls, group)
            if not child.is_empty():
                children[t][d] = child
    
    return cls.internal(children)

def _validate_inner(a: LeveledGSSInner[T], errors: List[str]) -> None:
    match a.variant:
        case None | RootInner():
            return
        case InternalInner(children):
            for t, by_depth in children.items():
                for d, ch in by_depth.items():
                    if ch.is_empty():
                        errors.append("Inner-layer internal node contains an Empty child")
                    if t is EPSILON:
                        if d != 0: errors.append("Inner-layer EPSILON edge with non-zero depth")
                        if not isinstance(ch.variant, RootInner): errors.append("Inner-layer EPSILON edge must point to Root")
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
class WithAcc(Generic[T, Acc]):
    node: LeveledGSSInner[T]
    acc: Acc

@dataclass(frozen=True, eq=True)
class Internal(Generic[T, Acc]):
    children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']]

@dataclass(frozen=True, eq=True)
class Empty:
    pass

LeveledVariant = Union[WithAcc[T, Acc], Internal[T, Acc], Empty]

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
    def with_acc(cls, node: LeveledGSSInner[T], acc: Acc) -> 'LeveledGSS[T, Acc]':
        if node.is_empty():
            return cls.empty()
        return cls(WithAcc(node=node, acc=acc))

    @classmethod
    def internal(cls, children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']]) -> 'LeveledGSS[T, Acc]':
        cleaned: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = {}
        for t, depths in children.items():
            c2 = {d: ch for d, ch in depths.items() if not ch.is_empty()}
            if c2:
                cleaned[t] = c2
        if not cleaned:
            return cls.empty()
        return cls(Internal(children=cleaned))

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
            case WithAcc(node=a, acc=acc):
                return LeveledGSS.with_acc(a.pop(), acc)
            case Internal(children=children):
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
            case Internal(children=children):
                key = EPSILON if value is None else value
                by_depth = children.get(key, {})
                if not by_depth:
                    return LeveledGSS.empty()
                return LeveledGSS.internal({key: by_depth})
            case WithAcc(node=a, acc=acc):
                return LeveledGSS.with_acc(a.isolate(value), acc)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        memo_b: Dict[int, 'LeveledGSS[T, Acc]'] = {}

        def apply_b(node: 'LeveledGSS[T, Acc]') -> 'LeveledGSS[T, Acc]':
            key = id(node)
            if key in memo_b: return memo_b[key]
            
            res: 'LeveledGSS[T, Acc]'
            match node.variant:
                case Empty():
                    res = node
                case WithAcc(node=inner_node, acc=acc):
                    new_acc = func(acc)
                    res = node if new_acc == acc else LeveledGSS.with_acc(inner_node, new_acc)
                case Internal(children=children):
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
                case WithAcc(acc=acc):
                    return node if predicate(acc) else LeveledGSS.empty()
                case Internal(children=children):
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
        if other is self: return self
        if self.is_empty(): return cast(LeveledGSS[T, Acc], other)
        if other.is_empty(): return self

        d1 = _to_stack_map(self)
        d2 = _to_stack_map(cast(LeveledGSS[T, Acc], other))
        for k, acc2 in d2.items():
            d1[k] = d1[k].merge(acc2) if k in d1 else acc2
        return _build_from_stack_map(self.__class__, d1)

    def peek(self) -> Set[T]:
        match self.variant:
            case Empty():
                return set()
            case Internal(children=children):
                return {cast(T, t) for t in children.keys() if t is not EPSILON}
            case WithAcc(node=node):
                return node.peek()

    def reduce_acc(self) -> Optional[Acc]:
        match self.variant:
            case Empty():
                return None
            case WithAcc(acc=acc):
                return acc
            case Internal(children=children):
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

def _build_from_stack_map(cls: Type[LeveledGSS[T, Acc]], stack_map: Dict[Tuple[T, ...], Acc]) -> LeveledGSS[T, Acc]:
    if not stack_map:
        return cls.empty()

    accs = list(stack_map.values())
    if all(acc == accs[0] for acc in accs):
        acc = accs[0]
        stacks = [(list(k), UNIT) for k in stack_map.keys()]
        inner = LeveledGSSInner.from_stacks(stacks)
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
        case Empty() | WithAcc():
            return node
        case Internal(children):
            acc_val: Optional[Acc] = None
            all_with_acc = True
            child_nodes: List[Tuple[Optional[T], int, LeveledGSSInner[T]]] = []

            for t, by_depth in children.items():
                for d, ch in by_depth.items():
                    if not all_with_acc: break
                    match ch.variant:
                        case WithAcc(node=a_node, acc=child_acc):
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
        case WithAcc(node=node, acc=acc):
            for seq in node.enumerate_stacks():
                yield (list(seq), acc)
        case Internal(children=children):
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
        case WithAcc(node=node):
            return node.max_depth()
        case Internal(children=children):
            max_d = 0
            for _, by_depth in children.items():
                max_d = max(max_d, *by_depth.keys())
            return max_d

def _validate(g: LeveledGSS[T, Acc], errors: List[str]) -> None:
    match g.variant:
        case Empty():
            return
        case WithAcc(node=node):
            try:
                node.validate_invariants()
            except ValueError as e:
                errors.append(f"Contained LeveledGSSInner has invariant violations: {e}")
            return
        case Internal(children=children):
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
                        case WithAcc(acc=acc): acc_set.append(acc)
                        case _: every_child_with_acc = False
                    _validate(ch, errors)

            if every_child_with_acc and acc_set and all(a == acc_set[0] for a in acc_set):
                errors.append("Internal node has all WithAcc children with equal accs; should be sucked up")
