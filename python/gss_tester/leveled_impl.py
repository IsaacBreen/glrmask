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


# -----------------------------
# Inner-node (A-layer) types
# -----------------------------

@dataclass(frozen=True, eq=True)
class Root:
    """A-level node representing the presence of an empty tail only."""
    pass


@dataclass(frozen=True, eq=True)
class InternalInner(Generic[T]):
    """
    A-level node with a map of edges:
    - key is T (or EPSILON for an empty step)
    - value is a dict keyed by 'max depth' (usize in Rust) -> LeveledGSSInner[T]
    """
    children: Dict[Optional[T], Dict[int, 'LeveledGSSInner[T]']]


LeveledGSSInner = Union[Root, InternalInner[T]]


# -----------------------------
# B-layer (main) node variants
# -----------------------------

@dataclass(frozen=True, eq=True)
class WithAcc(Generic[T, Acc]):
    node: LeveledGSSInner
    acc: Acc


@dataclass(frozen=True, eq=True)
class Internal(Generic[T, Acc]):
    # children: T (or EPSILON) -> max depth -> LeveledGSS
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
    """
    Leveled graph-structured stack. This mirrors the Rust shape:

    enum LeveledGSS<T, Acc> {
        WithAcc { node: LeveledGSSInner<T>, acc: Acc },
        Internal(HashMap<T, HashMap<usize, Rc<LeveledGSS<T, Acc>>>>),
        Empty,
    }

    and the inner A-layer:

    enum LeveledGSSInner<T> {
        Root,
        InternalInner(HashMap<T, HashMap<usize, Rc<LeveledGSSInner<T>>>>),
    }
    """
    variant: LeveledVariant

    # -----------------------------
    # Constructors
    # -----------------------------

    @classmethod
    def empty(cls) -> 'LeveledGSS[T, Acc]':
        return cls(Empty())

    @classmethod
    def with_acc(cls, node: LeveledGSSInner, acc: Acc) -> 'LeveledGSS[T, Acc]':
        return cls(WithAcc(node=node, acc=acc))

    @classmethod
    def internal(cls, children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']]) -> 'LeveledGSS[T, Acc]':
        # Cleanup empty entries to keep canonical-ish form
        cleaned: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = {}
        for t, depths in children.items():
            c2 = {d: ch for d, ch in depths.items() if not ch.is_empty()}
            if c2:
                cleaned[t] = c2
        if not cleaned:
            return cls.empty()
        return cls(Internal(children=cleaned))

    # -----------------------------
    # GSS interface implementation
    # -----------------------------

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        """
        Build a LeveledGSS from explicit stacks.
        We first merge duplicate stacks (identical value lists) by merging their Accs.
        Then we build a leveled DAG honoring "suck-up" invariants and the T->max_depth edge scheme.
        Depth is counted as the total length of the stack at the point of the edge (top is the last element).
        """
        # Merge identical stacks by merging their accumulators
        merged: Dict[Tuple[T, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc

        return _build_b_from_stack_map(cls, merged)

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        """
        Pushes `value` onto all active stacks. If there are no stacks, stays empty.
        We create a new Internal root edge keyed by (value, max_depth_after_push),
        with this GSS as the child (which enumerates tails).
        """
        if self.is_empty():
            return self

        max_depth = _max_depth_b(self)
        new_depth = max_depth + 1
        # Single edge from new root to previous graph
        node = self.internal({value: {new_depth: self}})
        # Suck-up if all children are WithAcc and equal accs
        return _canonicalize_b(node)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        """
        Pops top value from all non-empty stacks. Empty stacks are discarded.
        Semantics: For B Internal, drop one level by unioning all children.
        For B WithAcc, transform A-children edges into B-children.
        """
        v = self.variant
        if isinstance(v, Empty):
            return self

        if isinstance(v, WithAcc):
            # Replace A-edges with B-edges carrying the same acc in WithAcc children
            acc = v.acc
            a = v.node
            # Build children for B Internal by reading A edges
            children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = {}
            for (t, d, a_child) in _iter_a_edges(a):
                if t is EPSILON:
                    # Popping empty stacks: discard (do not produce any stack)
                    continue
                # Child after popping: WithAcc whose A-root is a_child
                # Depth on the B edge remains the same d from A (total length at this point)
                by_depth = children.setdefault(t, {})
                by_depth[d] = LeveledGSS.with_acc(a_child, acc)
            return _canonicalize_b(LeveledGSS.internal(children))

        # Internal
        # Pop removes the top: just return the union of children at one level down.
        result = LeveledGSS.empty()
        iv = cast(Internal[T, Acc], v)
        for t, by_depth in iv.children.items():
            for _, child in by_depth.items():
                result = result.merge(child)
        return result

    def is_empty(self) -> bool:
        v = self.variant
        if isinstance(v, Empty):
            return True
        if isinstance(v, Internal):
            # internal with no children is technically empty; we canonicalize away, but be safe
            return not v.children
        # WithAcc always represents at least one stack
        return False

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        """
        Keeps only the stacks that have `value` at the top.
        value is None => keep only empty stacks.
        """
        v = self.variant
        if isinstance(v, Empty):
            return self

        if isinstance(v, Internal):
            # Keep only edges for the requested top value (or EPSILON for empty)
            key = EPSILON if value is None else value
            by_depth = v.children.get(key, {})
            # Union all children under this key
            result = LeveledGSS.empty()
            for _, child in by_depth.items():
                result = result.merge(child)
            return result

        # WithAcc: keep only paths in A whose top element equals `value`
        w = cast(WithAcc[T, Acc], v)
        acc = w.acc
        a = w.node

        if value is None:
            # Keep only empty tails
            # If there is at least one epsilon edge (None, 0) or if A is Root, we keep a single empty stack.
            # Represent it as WithAcc(Root, acc).
            if _a_has_epsilon(a) or isinstance(a, Root):
                return LeveledGSS.with_acc(Root(), acc)
            return LeveledGSS.empty()

        # value is not None
        # Build new Internal node with edges only for 'value'
        children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = {}
        for (t, d, a_child) in _iter_a_edges(a):
            if t == value:
                children.setdefault(value, {})[d] = LeveledGSS.with_acc(a_child, acc)
        return _canonicalize_b(LeveledGSS.internal(children))

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        """
        Applies a function to each accumulator. We use memoization to preserve sharing.
        If the transformed acc equals the original (==), we reuse the same node structure.
        """
        memo_b: Dict[int, 'LeveledGSS[T, Acc]'] = {}
        memo_a: Dict[int, LeveledGSSInner] = {}

        def apply_b(node: 'LeveledGSS[T, Acc]') -> 'LeveledGSS[T, Acc]':
            key = id(node)
            if key in memo_b:
                return memo_b[key]
            v = node.variant
            if isinstance(v, Empty):
                memo_b[key] = node
                return node
            if isinstance(v, WithAcc):
                new_acc = func(v.acc)
                # If no change, reuse
                if new_acc == v.acc:
                    memo_b[key] = node
                    return node
                # Node structure unchanged; we can reuse A-node instance to preserve sharing
                res = LeveledGSS.with_acc(_apply_a(v.node, memo_a), new_acc)
                memo_b[key] = res
                return res
            # Internal: map children
            changed = False
            new_children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = {}
            for t, by_depth in v.children.items():
                inner_map: Dict[int, 'LeveledGSS[T, Acc]'] = {}
                for d, child in by_depth.items():
                    new_child = apply_b(child)
                    inner_map[d] = new_child
                    if new_child is not child:
                        changed = True
                new_children[t] = inner_map
            if not changed:
                memo_b[key] = node
                return node
            res = _canonicalize_b(LeveledGSS.internal(new_children))
            memo_b[key] = res
            return res

        def _apply_a(inner: LeveledGSSInner, memo: Dict[int, LeveledGSSInner]) -> LeveledGSSInner:
            k = id(inner)
            if k in memo:
                return memo[k]
            if isinstance(inner, Root):
                memo[k] = inner
                return inner
            # InternalInner, children unchanged here
            changed = False
            new_children: Dict[Optional[T], Dict[int, LeveledGSSInner]] = {}
            for t, by_depth in inner.children.items():
                inner_map: Dict[int, LeveledGSSInner] = {}
                for d, ch in by_depth.items():
                    new_ch = _apply_a(ch, memo)
                    inner_map[d] = new_ch
                    if new_ch is not ch:
                        changed = True
                new_children[t] = inner_map
            if not changed:
                memo[k] = inner
                return inner
            norm = _normalize_a(InternalInner(new_children))
            memo[k] = norm
            return norm

        return apply_b(self)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        """
        Remove stacks whose acc does not satisfy predicate.
        Implementation uses structural recursion and canonicalization.
        """
        def prune_b(node: 'LeveledGSS[T, Acc]') -> 'LeveledGSS[T, Acc]':
            v = node.variant
            if isinstance(v, Empty):
                return node
            if isinstance(v, WithAcc):
                if predicate(v.acc):
                    return node
                return LeveledGSS.empty()
            # Internal: prune children
            new_children: Dict[Optional[T], Dict[int, 'LeveledGSS[T, Acc]']] = {}
            for t, by_depth in v.children.items():
                new_by_depth: Dict[int, 'LeveledGSS[T, Acc]'] = {}
                for d, ch in by_depth.items():
                    pr = prune_b(ch)
                    if not pr.is_empty():
                        new_by_depth[d] = pr
                if new_by_depth:
                    new_children[t] = new_by_depth
            return _canonicalize_b(LeveledGSS.internal(new_children))

        return prune_b(self)

    def merge(self, other: 'LeveledGSS[T, Acc]') -> 'LeveledGSS[T, Acc]':
        """
        Merge two LeveledGSS instances by combining accumulators for identical stacks.
        We use a straightforward approach: expand both to dict-of-stacks, merge accs, rebuild.
        """
        if other is self:
            return self
        # Fast paths
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        d1 = _to_stack_map(self)
        d2 = _to_stack_map(other)
        # Combine: merge accs for identical keys
        for k, acc2 in d2.items():
            if k in d1:
                d1[k] = d1[k].merge(acc2)
            else:
                d1[k] = acc2
        return _build_b_from_stack_map(self.__class__, d1)

    def peek(self) -> Set[T]:
        """
        Set of top-of-stack values across non-empty stacks.
        """
        v = self.variant
        if isinstance(v, Empty):
            return set()
        if isinstance(v, Internal):
            values: Set[T] = set()
            for t, _ in v.children.items():
                if t is not EPSILON:
                    values.add(cast(T, t))
            return values
        # WithAcc: peek from A-edges at top
        values: Set[T] = set()
        for (t, _d, _child) in _iter_a_edges(v.node):
            if t is not EPSILON:
                values.add(cast(T, t))
        return values

    def reduce_acc(self) -> Optional[Acc]:
        """
        Merge all accumulators of active stacks. Returns None if there are no stacks.
        """
        v = self.variant
        if isinstance(v, Empty):
            return None
        if isinstance(v, WithAcc):
            return v.acc
        # Internal: merge children's reductions
        acc_opt: Optional[Acc] = None
        iv = cast(Internal[T, Acc], v)
        for _, by_depth in iv.children.items():
            for _, ch in by_depth.items():
                sub = ch.reduce_acc()
                if sub is None:
                    continue
                if acc_opt is None:
                    acc_opt = sub
                else:
                    acc_opt = acc_opt.merge(sub)
        return acc_opt

    def to_reference_impl(self) -> 'ReferenceGSS[T, Acc]':
        """
        Expand to the canonical ReferenceGSS by enumerating stacks.
        """
        stacks: List[Tuple[List[T], Acc]] = []
        for vals, acc in _enumerate_b(self):
            stacks.append((list(vals), acc))
        return ReferenceGSS.from_stacks(stacks)

    # -----------------------------
    # Invariants validation
    # -----------------------------

    def validate_invariants(self) -> None:
        """
        Validate:
        - Acc only appears in WithAcc variant (never in A-layer, never in Internal or Empty).
        - Suck-up: If an Internal node has all children WithAcc and their accs are equal, the node should be WithAcc.
        - No Empty children inside Internal's children map.
        - Depth labels are consistent:
            * For B-layer Internal edges (t, d) -> child: d == 1 + max_depth(child)
              (with EPSILON edges allowed only when d == 0 and child is WithAcc with Root A-node).
            * For A-layer InternalInner edges (t, d) -> child: d == (len of sequence under child) + (1 if t != EPSILON else 0)
              Practically: d must equal 0 for EPSILON; for t != EPSILON, d >= 1 and equals 1 + max_depth(child A-node).
        """
        errors: List[str] = []
        _validate_b(self, errors)
        if errors:
            # Raise a single error with all collected messages
            raise ValueError("LeveledGSS invariant violations:\n" + "\n".join(f"- {e}" for e in errors))


# -----------------------------
# Builders and Helpers
# -----------------------------

def _build_b_from_stack_map(cls: Type[LeveledGSS[T, Acc]], stack_map: Dict[Tuple[T, ...], Acc]) -> LeveledGSS[T, Acc]:
    """
    Build B-layer node from a map: stack (tuple of T) -> Acc.
    Top-of-stack is the last element of the tuple.
    """
    if not stack_map:
        return cls.empty()

    # If all accs are equal, we can attach the acc at this level (WithAcc) and
    # build A-layer for all sequences at once, then normalize.
    accs = list(stack_map.values())
    all_equal = all(acc == accs[0] for acc in accs)
    if all_equal:
        acc = accs[0]
        # Build A-layer representing all stacks in stack_map
        inner = _build_a_from_seqs([k for k in stack_map.keys()])
        return _canonicalize_b(cls.with_acc(inner, acc))

    # Otherwise, we must build an Internal node with children grouped by (top T, depth)
    buckets: DefaultDict[Optional[T], DefaultDict[int, Dict[Tuple[T, ...], Acc]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for seq, acc in stack_map.items():
        n = len(seq)
        if n == 0:
            # empty stack: use EPSILON and depth 0
            buckets[EPSILON][0][tuple()] = acc
        else:
            t = seq[-1]
            prefix = seq[:-1]
            buckets[t][n][prefix] = acc

    children: Dict[Optional[T], Dict[int, LeveledGSS[T, Acc]]] = {}
    for t, by_depth in buckets.items():
        children[t] = {}
        for depth, submap in by_depth.items():
            # Build child from prefix map
            child = _build_b_from_stack_map(cls, submap)
            if not child.is_empty():
                children[t][depth] = child

    return _canonicalize_b(cls.internal(children))


def _build_a_from_seqs(seqs: Iterable[Tuple[T, ...]]) -> LeveledGSSInner:
    """
    Build A-layer node from a set of sequences (tuples of T). Top-of-stack is last element.
    We use EPSILON (None) with depth 0 to record the presence of empty sequence alongside others.
    """
    # Partition by (top, depth)
    buckets: DefaultDict[Optional[T], DefaultDict[int, List[Tuple[T, ...]]]] = defaultdict(lambda: defaultdict(list))
    count = 0
    has_empty = False
    for s in seqs:
        count += 1
        if len(s) == 0:
            has_empty = True
        else:
            t = s[-1]
            d = len(s)
            prefix = s[:-1]
            buckets[t][d].append(prefix)

    if count == 0:
        # No sequences: degenerate, return Root to be safe
        return Root()

    children: Dict[Optional[T], Dict[int, LeveledGSSInner]] = {}

    if has_empty:
        # Epsilon edge marks presence of empty sequence
        children.setdefault(EPSILON, {})[0] = Root()

    for t, by_depth in buckets.items():
        for d, group in by_depth.items():
            child = _build_a_from_seqs(group)
            children.setdefault(t, {})[d] = child

    return _normalize_a(InternalInner(children))


def _normalize_a(node: LeveledGSSInner) -> LeveledGSSInner:
    """
    Normalize A-layer nodes:
    - If only epsilon(0) -> Root is present, collapse to Root.
    - Remove empty maps defensively.
    """
    if isinstance(node, Root):
        return node
    # Remove empty nested maps
    cleaned: Dict[Optional[T], Dict[int, LeveledGSSInner]] = {}
    for t, by_depth in node.children.items():
        new_by_depth: Dict[int, LeveledGSSInner] = {}
        for d, ch in by_depth.items():
            if isinstance(ch, InternalInner) and not ch.children:
                # A-layer internal with no children => collapse to Root
                new_by_depth[d] = Root()
            else:
                new_by_depth[d] = ch
        if new_by_depth:
            cleaned[t] = new_by_depth

    # Collapse to Root if it's exactly epsilon -> 0 -> Root
    if set(cleaned.keys()) == {EPSILON} and set(cleaned[EPSILON].keys()) == {0}:
        if isinstance(cleaned[EPSILON][0], Root):
            return Root()

    return InternalInner(cleaned)


def _canonicalize_b(node: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
    """
    Canonicalize a B-layer node:
    - Remove empty children (already done in internal constructor).
    - If node is Internal and all its children are WithAcc with equal acc, suck-up acc to parent:
      transform to WithAcc with an A-layer whose children mirror the B-layer's edges.
    """
    v = node.variant
    if isinstance(v, Empty):
        return node
    if isinstance(v, WithAcc):
        # Optionally normalize A-layer
        return LeveledGSS.with_acc(_normalize_a(v.node), v.acc)
    # Internal
    children = v.children
    # Check for suck-up
    acc_val: Optional[Acc] = None
    all_with_acc = True
    for _t, by_depth in children.items():
        for _d, ch in by_depth.items():
            chv = ch.variant
            if not isinstance(chv, WithAcc):
                all_with_acc = False
                break
            if acc_val is None:
                acc_val = chv.acc
            else:
                if chv.acc != acc_val:
                    all_with_acc = False
                    break
        if not all_with_acc:
            break

    if all_with_acc and acc_val is not None:
        # Build A-layer for parent from all children edges: (t, d) -> child A-node
        a_children: Dict[Optional[T], Dict[int, LeveledGSSInner]] = {}
        for t, by_depth in children.items():
            for d, ch in by_depth.items():
                chv = cast(WithAcc[T, Acc], ch.variant)
                a_children.setdefault(t, {})[d] = chv.node
        a_node = _normalize_a(InternalInner(a_children))
        return LeveledGSS.with_acc(a_node, acc_val)

    # Else keep as is
    return node


# -----------------------------
# Enumerators
# -----------------------------

def _iter_a_edges(a: LeveledGSSInner) -> Iterator[Tuple[Optional[T], int, LeveledGSSInner]]:
    """
    Iterate A-layer edges as triples (t, depth, child).
    Root yields a single epsilon edge to Root with depth 0 to represent only-empty set,
    but we avoid yielding for Root here; Root by itself means "only empty tail".
    We will treat Root specially in callers where needed.
    """
    if isinstance(a, Root):
        # Representing only empty; callers handle this explicitly
        return
        yield  # type: ignore[misc]
    for t, by_depth in a.children.items():
        for d, child in by_depth.items():
            yield (t, d, child)


def _enumerate_a(a: LeveledGSSInner) -> Iterator[Tuple[T, ...]]:
    """
    Enumerate all tails represented by an A-layer node.
    Top-of-stack is last element; edges append at the end.
    """
    if isinstance(a, Root):
        # Only empty tail
        yield tuple()
        return
    for t, by_depth in a.children.items():
        for _d, child in by_depth.items():
            if t is EPSILON:
                # epsilon: propagate child's tails (should be Root)
                for tail in _enumerate_a(child):
                    yield tail
                continue
            for tail in _enumerate_a(child):
                yield tail + (cast(T, t),)


def _enumerate_b(g: LeveledGSS[T, Acc]) -> Iterator[Tuple[Tuple[T, ...], Acc]]:
    """
    Enumerate all stacks (values, acc) represented by a B-layer node.
    """
    v = g.variant
    if isinstance(v, Empty):
        return
        yield  # type: ignore[misc]
    if isinstance(v, WithAcc):
        for seq in _enumerate_a(v.node):
            yield (seq, v.acc)
        return
    for t, by_depth in v.children.items():
        for _d, child in by_depth.items():
            for tail, acc in _enumerate_b(child):
                if t is EPSILON:
                    # represent empty top: no element added
                    yield (tail, acc)
                else:
                    yield (tail + (cast(T, t),), acc)


def _to_stack_map(g: LeveledGSS[T, Acc]) -> Dict[Tuple[T, ...], Acc]:
    d: Dict[Tuple[T, ...], Acc] = {}
    for seq, acc in _enumerate_b(g):
        d[seq] = acc
    return d


# -----------------------------
# Depth utilities
# -----------------------------

def _max_depth_b(g: LeveledGSS[T, Acc]) -> int:
    """
    Compute max stack length represented by the B-layer node.
    We use depth labels when possible to avoid full enumeration.
    """
    v = g.variant
    if isinstance(v, Empty):
        return 0
    if isinstance(v, WithAcc):
        return _max_depth_a(v.node)
    # Internal: max of all registered depths on edges
    max_d = 0
    for _t, by_depth in v.children.items():
        for d in by_depth.keys():
            if d > max_d:
                max_d = d
    return max_d


def _max_depth_a(a: LeveledGSSInner) -> int:
    """
    Max length among tails represented by the A-layer node.
    """
    if isinstance(a, Root):
        return 0
    max_d = 0
    for t, by_depth in a.children.items():
        for d, child in by_depth.items():
            if t is EPSILON:
                # epsilon edges must be depth 0; ignore for max
                continue
            if d > max_d:
                max_d = d
    return max_d


def _a_has_epsilon(a: LeveledGSSInner) -> bool:
    if isinstance(a, Root):
        return True
    return EPSILON in a.children and 0 in a.children[EPSILON]


# -----------------------------
# Invariant validation helpers
# -----------------------------

def _validate_b(g: LeveledGSS[T, Acc], errors: List[str]) -> None:
    v = g.variant
    if isinstance(v, Empty):
        return
    if isinstance(v, WithAcc):
        _validate_a(v.node, errors)
        return
    # Internal
    # 1) Ensure no Empty children; also collect WithAcc accs to check suck-up
    acc_set: List[Acc] = []
    every_child_with_acc = True
    for t, by_depth in v.children.items():
        for d, ch in by_depth.items():
            if ch.is_empty():
                errors.append("Internal node contains an Empty child; should be pruned")
            chv = ch.variant
            # 2) Validate B-depth consistency:
            # For EPSILON, only depth 0 should appear and child must represent empty stacks only
            if t is EPSILON:
                if d != 0:
                    errors.append("EPSILON edge at B-layer with non-zero depth")
            # For non-epsilon, depth should be >= 1 and ideally reflect child max depth + 1
            # We check consistency but allow equal-or-greater as 'max depth' by design.
            if t is not EPSILON:
                if d < 1:
                    errors.append("Non-epsilon B-layer edge has depth < 1")
                # child_max = _max_depth_b(ch)
                # Allow equality (exact) or being a max; we enforce exact equality here to keep it strict.
                child_max = _max_depth_b(ch)
                if d != child_max + 1:
                    # We accept only exact canonical depth: current = child_max + 1
                    errors.append(f"B-layer edge depth mismatch: expected {child_max + 1}, found {d}")

            if isinstance(chv, WithAcc):
                acc_set.append(chv.acc)
            else:
                every_child_with_acc = False
            _validate_b(ch, errors)

    # 3) Suck-up check
    if every_child_with_acc and acc_set:
        all_equal = all(a == acc_set[0] for a in acc_set)
        if all_equal:
            errors.append("Internal node has all WithAcc children with equal accs; should be sucked up into parent")

def _validate_a(a: LeveledGSSInner, errors: List[str]) -> None:
    if isinstance(a, Root):
        return
    # For A-layer internal edges, we enforce:
    # - EPSILON edges must have depth=0 and lead to Root
    # - Non-epsilon edges must have depth >=1 and equal child_max + 1
    for t, by_depth in a.children.items():
        for d, ch in by_depth.items():
            if t is EPSILON:
                if d != 0:
                    errors.append("EPSILON edge at A-layer with non-zero depth")
                if not isinstance(ch, Root):
                    errors.append("EPSILON edge at A-layer should point to Root")
                continue
            if d < 1:
                errors.append("Non-epsilon A-layer edge has depth < 1")
            child_max = _max_depth_a(ch)
            if d != child_max + 1:
                errors.append(f"A-layer edge depth mismatch: expected {child_max + 1}, found {d}")
            _validate_a(ch, errors)


# -----------------------------
# END
# -----------------------------
