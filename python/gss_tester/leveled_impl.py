from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, TypeVar, Callable, Any
import json

from .interface import GSS, T, Acc


# ------------------------------------------------------------------------------
# Internal node structures
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Shape(Generic[T]):
    """
    Acc-free shape trie for stacks (top-first orientation).
    - terms: number of stacks that end exactly at this node (multiplicity count).
    - children: map from top-of-stack symbol to deeper shapes.
    """
    terms: int
    children: Dict[T, 'Shape[T]']

    def is_empty(self) -> bool:
        return self.terms == 0 and not self.children


class BNode(Generic[T, Acc]):
    """Union type base class for BranchNode and UniformNode."""
    __slots__ = ()

    def is_empty(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class BranchNode(BNode[T, Acc]):
    """
    Branch node: No uniform accumulator at this level.
    - terms: Acc values for stacks that end exactly at this node (duplicates allowed).
    - children: map from top-of-stack symbol to BNode.
    This node may contain children that are either BranchNode or UniformNode.
    """
    terms: Tuple[Acc, ...]
    children: Dict[T, BNode[T, Acc]]

    def is_empty(self) -> bool:
        return len(self.terms) == 0 and not self.children


@dataclass(frozen=True, slots=True)
class UniformNode(BNode[T, Acc]):
    """
    Uniform node: All stacks in this subtree share the same accumulator (acc).
    - acc: shared accumulator for the entire subtree.
    - shape: an acc-free shape trie that encodes where stacks end (via counts) and how they continue.
    Invariants:
      - Under a UniformNode, there are no child Acc values stored; only counts exist within Shape.
      - If a UniformNode has a parent UniformNode, the parent must absorb it (we never create nested Uniforms).
    """
    acc: Acc
    shape: Shape[T]

    def is_empty(self) -> bool:
        return self.shape.is_empty()


# ------------------------------------------------------------------------------
# Helpers for node construction and canonicalization
# ------------------------------------------------------------------------------

def _shape_normalize(s: Shape[T]) -> Shape[T]:
    """Drop empty children for a clean shape."""
    if not s.children:
        return s
    pruned: Dict[T, Shape[T]] = {}
    for k, ch in s.children.items():
        if not ch.is_empty():
            pruned[k] = ch
    if len(pruned) is not len(s.children):
        return Shape(s.terms, pruned)
    return s


def _mk_uniform(acc: Acc, shape: Shape[T]) -> BNode[T, Acc]:
    """Create a Uniform node if shape not empty; otherwise return empty Branch."""
    shape = _shape_normalize(shape)
    if shape.is_empty():
        return BranchNode((), {})
    return UniformNode(acc, shape)


def _unsuck(u: UniformNode[T, Acc]) -> BranchNode[T, Acc]:
    """
    Expand a Uniform node into a Branch node by distributing the uniform acc to:
      - 'terms' as repeated acc values,
      - each child shape becomes a Uniform child with the same acc.
    """
    terms = (u.acc,) * u.shape.terms
    children: Dict[T, BNode[T, Acc]] = {k: UniformNode(u.acc, ch) for k, ch in u.shape.children.items()}
    return BranchNode(terms, children)


def _canonicalize_branch(terms: Iterable[Acc], children: Dict[T, BNode[T, Acc]]) -> BNode[T, Acc]:
    """
    Canonicalize a Branch node by:
      - Dropping empty children,
      - Sucking up to a Uniform node if:
          * every child is Uniform with the same acc, and
          * every acc in `terms` equals that acc.
      - Otherwise, return a clean Branch.
    """
    terms_tuple: Tuple[Acc, ...] = tuple(terms)
    if children:
        # Drop empty children first
        pruned_children: Dict[T, BNode[T, Acc]] = {}
        for k, ch in children.items():
            if isinstance(ch, UniformNode):
                if not ch.shape.is_empty():
                    pruned_children[k] = ch
            else:
                # Branch child
                if not ch.is_empty():
                    pruned_children[k] = ch

        if pruned_children:
            # Check for suck-up possibility
            first_child = next(iter(pruned_children.values()))
            if isinstance(first_child, UniformNode):
                acc0 = first_child.acc
                all_children_uniform = True
                all_equal_acc = True
                for ch in pruned_children.values():
                    if not isinstance(ch, UniformNode):
                        all_children_uniform = False
                        break
                    if ch.acc != acc0:
                        all_equal_acc = False
                        break
                if all_children_uniform and all_equal_acc:
                    # Terms all equal to acc0?
                    if len(terms_tuple) == 0 or all(a == acc0 for a in terms_tuple):
                        # Create a uniform node with shape:
                        #   shape.terms = len(terms_tuple)
                        #   shape.children = {k: child.shape}
                        shape_children: Dict[T, Shape[T]] = {k: ch.shape for k, ch in pruned_children.items()}
                        shape = Shape(terms=len(terms_tuple), children=shape_children)
                        return _mk_uniform(acc0, shape)

            # Not suckable; return cleaned branch
            return BranchNode(terms_tuple, pruned_children)

    # No children; possibly only terms remain.
    # If terms exist but children don't, they are not suckable unless homogenous and some structure exists.
    # Keep them as a Branch.
    return BranchNode(terms_tuple, {})


def _shape_union(a: Shape[T], b: Shape[T]) -> Shape[T]:
    """Union of two shapes: sum of term counts and recursive union of children."""
    if a is b:
        return a
    terms = a.terms + b.terms
    if not a.children and not b.children:
        return Shape(terms, {})
    keys = set(a.children.keys()) | set(b.children.keys())
    merged: Dict[T, Shape[T]] = {}
    for k in keys:
        ca = a.children.get(k)
        cb = b.children.get(k)
        if ca is None:
            merged[k] = cb  # type: ignore[assignment]
        elif cb is None:
            merged[k] = ca
        else:
            merged[k] = _shape_union(ca, cb)
    return _shape_normalize(Shape(terms, merged))


def _b_union(x: Optional[BNode[T, Acc]], y: Optional[BNode[T, Acc]]) -> Optional[BNode[T, Acc]]:
    """Union of two BNode subtrees."""
    if x is None:
        return y
    if y is None:
        return x
    if isinstance(x, BranchNode) and isinstance(y, BranchNode):
        # Concatenate terms, merge children by key
        terms = x.terms + y.terms
        keys = set(x.children.keys()) | set(y.children.keys())
        children: Dict[T, BNode[T, Acc]] = {}
        for k in keys:
            cx = x.children.get(k)
            cy = y.children.get(k)
            if cx is None:
                c = cy
            elif cy is None:
                c = cx
            else:
                c = _b_union(cx, cy)
            if c is not None and not c.is_empty():
                children[k] = c
        return _canonicalize_branch(terms, children)

    if isinstance(x, UniformNode) and isinstance(y, UniformNode):
        if x.acc == y.acc:
            return _mk_uniform(x.acc, _shape_union(x.shape, y.shape))
        # Different accs at same level -> distribute both
        return _b_union(_unsuck(x), _unsuck(y))

    if isinstance(x, UniformNode) and isinstance(y, BranchNode):
        # Distribute x into y (efficient version of union(unsuck(x), y))
        terms = y.terms + (x.acc,) * x.shape.terms
        keys = set(y.children.keys()) | set(x.shape.children.keys())
        children: Dict[T, BNode[T, Acc]] = {}
        for k in keys:
            cy = y.children.get(k)
            sx = x.shape.children.get(k)
            if sx is None:
                children[k] = cy  # type: ignore[assignment]
            elif cy is None:
                children[k] = _mk_uniform(x.acc, sx)
            else:
                children[k] = _b_union(_mk_uniform(x.acc, sx), cy)  # type: ignore[assignment]
        return _canonicalize_branch(terms, children)

    if isinstance(x, BranchNode) and isinstance(y, UniformNode):
        return _b_union(y, x)

    raise AssertionError("Unreachable union case")


def _b_pop(node: BNode[T, Acc]) -> BNode[T, Acc]:
    """
    Pop: remove the first/top element from every non-empty stack.
    - Branch: union of its children (ignore terms since empty stacks can't be popped).
    - Uniform: union of a Uniform node built from each shape child.
    """
    if isinstance(node, BranchNode):
        res: Optional[BNode[T, Acc]] = None
        for child in node.children.values():
            res = _b_union(res, child)
        return res if res is not None else BranchNode((), {})

    # Uniform
    res: Optional[BNode[T, Acc]] = None
    for k, subshape in node.shape.children.items():
        res = _b_union(res, _mk_uniform(node.acc, subshape))
    return res if res is not None else BranchNode((), {})


def _b_isolate(node: BNode[T, Acc], value: T) -> BNode[T, Acc]:
    """
    Isolate: keep only stacks whose top equals 'value'.
    - Branch: return the child under 'value' (terms ignored).
    - Uniform: return Uniform(acc, shape_child[value]) if exists.
    """
    if isinstance(node, BranchNode):
        return node.children.get(value, BranchNode((), {}))
    # Uniform
    ch = node.shape.children.get(value)
    if ch is None:
        return BranchNode((), {})
    return _mk_uniform(node.acc, ch)


def _shape_insert(s: Shape[T], rev_path: Sequence[T]) -> Shape[T]:
    """Insert the reversed path (top-first) into an acc-free shape, increasing terminal counts."""
    if not rev_path:
        return Shape(s.terms + 1, s.children)
    head = rev_path[0]
    tail = rev_path[1:]
    child = s.children.get(head, Shape(0, {}))
    new_child = _shape_insert(child, tail)
    if new_child is child:
        return s
    new_children = dict(s.children)
    new_children[head] = new_child
    return _shape_normalize(Shape(s.terms, new_children))


def _b_insert(node: BNode[T, Acc], rev_path: Sequence[T], acc: Acc) -> BNode[T, Acc]:
    """
    Insert a stack (given as reversed path) with accumulator acc into the B-node.
    Maintains invariants, unsucking uniform nodes if a conflict appears.
    """
    if not rev_path:
        # End at this node: add a terminal here.
        if isinstance(node, BranchNode):
            return _canonicalize_branch(node.terms + (acc,), node.children)
        # Uniform
        if node.acc == acc:
            return _mk_uniform(acc, Shape(node.shape.terms + 1, node.shape.children))
        # Conflict: unsuck, then add term
        br = _unsuck(node)
        return _canonicalize_branch(br.terms + (acc,), br.children)

    head = rev_path[0]
    tail = rev_path[1:]

    if isinstance(node, BranchNode):
        child = node.children.get(head, BranchNode((), {}))
        new_child = _b_insert(child, tail, acc)
        if new_child is child:
            return node
        new_children = dict(node.children)
        if not new_child.is_empty():
            new_children[head] = new_child
        else:
            new_children.pop(head, None)
        return _canonicalize_branch(node.terms, new_children)

    # Uniform
    if node.acc == acc:
        # We can stay uniform and update the shape
        new_shape = _shape_insert(node.shape, rev_path)
        if new_shape is node.shape:
            return node
        return _mk_uniform(acc, new_shape)

    # Conflict: unsuck uniform to branch, then insert
    br = _unsuck(node)
    return _b_insert(br, rev_path, acc)


def _b_apply(node: BNode[T, Acc], func: Callable[[Acc], Acc]) -> BNode[T, Acc]:
    if isinstance(node, BranchNode):
        new_terms = tuple(func(a) for a in node.terms)
        new_children: Dict[T, BNode[T, Acc]] = {}
        for k, ch in node.children.items():
            new_children[k] = _b_apply(ch, func)
        return _canonicalize_branch(new_terms, new_children)
    # Uniform
    return _mk_uniform(func(node.acc), node.shape)


def _b_prune(node: BNode[T, Acc], predicate: Callable[[Acc], bool]) -> BNode[T, Acc]:
    if isinstance(node, BranchNode):
        kept_terms = tuple(a for a in node.terms if predicate(a))
        new_children: Dict[T, BNode[T, Acc]] = {}
        for k, ch in node.children.items():
            new_ch = _b_prune(ch, predicate)
            if not new_ch.is_empty():
                new_children[k] = new_ch
        return _canonicalize_branch(kept_terms, new_children)
    # Uniform
    if predicate(node.acc):
        return node
    return BranchNode((), {})


def _b_peek(node: BNode[T, Acc]) -> Set[T]:
    if isinstance(node, BranchNode):
        return set(node.children.keys())
    return set(node.shape.children.keys())


def _b_iter_entries(node: BNode[T, Acc], prefix_rev: List[T]) -> Iterator[Tuple[List[T], Acc]]:
    """
    Iterate all [stack_as_bottom_first, acc] pairs, including duplicates (by duplicating yields).
    prefix_rev holds the partial stack as top-first; we reverse when emitting.
    """
    if isinstance(node, BranchNode):
        # Terminals at this node
        if node.terms:
            stack = list(reversed(prefix_rev))
            for acc in node.terms:
                yield (stack, acc)
        # Children
        for k, ch in node.children.items():
            prefix_rev.append(k)
            yield from _b_iter_entries(ch, prefix_rev)
            prefix_rev.pop()
        return

    # Uniform node: enumerate from shape with uniform acc
    acc = node.acc

    def _shape_iter(s: Shape[T], pfx_rev: List[T]) -> Iterator[Tuple[List[T], Acc]]:
        if s.terms:
            stack = list(reversed(pfx_rev))
            for _ in range(s.terms):
                yield (stack, acc)
        for kk, sub in s.children.items():
            pfx_rev.append(kk)
            yield from _shape_iter(sub, pfx_rev)
            pfx_rev.pop()

    yield from _shape_iter(node.shape, prefix_rev)


def _b_count_total(node: BNode[T, Acc]) -> int:
    """Count total stacks (including duplicates)."""
    if isinstance(node, BranchNode):
        total = len(node.terms)
        for ch in node.children.values():
            total += _b_count_total(ch)
        return total
    # Uniform
    def _shape_count(s: Shape[T]) -> int:
        cnt = s.terms
        for sub in s.children.values():
            cnt += _shape_count(sub)
        return cnt
    return _shape_count(node.shape)


def _b_is_single_empty(node: BNode[T, Acc]) -> bool:
    """True iff the set contains exactly one stack and it is empty."""
    if isinstance(node, BranchNode):
        return len(node.terms) == 1 and not node.children
    if isinstance(node, UniformNode):
        return node.shape.terms == 1 and not node.shape.children
    return False


# ------------------------------------------------------------------------------
# Implementation of the GSS interface
# ------------------------------------------------------------------------------

class LeveledGSS(GSS[T, Acc]):
    """
    High-performance leveled GSS implementation with strict "acc-at-one-level" invariants.

    Invariants enforced:
    - Accumulators (Acc) exist only at one level at a time:
      * BranchNode: children may carry accs (via UniformNode) and/or terminal accs at terms.
      * UniformNode: holds a single acc for the entire subtree; below it there are no Acc values, only shape counts.
    - If at a BranchNode every child is a UniformNode with the same acc, and all terminal accs at that node
      equal that same acc, we "suck up" the acc into a parent UniformNode and drop accs from the children.
      As a result, if the children of a node have accs, there is at least one inequality between them.

    Orientation:
    - Stacks are represented top-first (i.e., the top of the stack is the first symbol in the path).
      This makes push/isolate/pop efficient.

    Notes:
    - This implementation preserves multiplicity of identical stacks (duplicates) via terminal lists or counts,
      so to_json output can reflect duplicates when they arise (e.g., after pop).
    - reduce_acc enumerates all stacks (including duplicates); results match reference when the merge
      function is order-independent (as requested).
    """

    __slots__ = ("_root",)

    def __init__(self, root: BNode[T, Acc]):
        self._root: BNode[T, Acc] = root

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        """
        Build a GSS from explicit stacks. If the same stack appears multiple times with different Acc values,
        they are stored as duplicates. If a conflict occurs at an already uniformized subtree, it is expanded
        ("unsucked") automatically.
        """
        root: BNode[T, Acc] = BranchNode((), {})
        for st, acc in stacks:
            rev = list(reversed(st))
            root = _b_insert(root, rev, acc)
        return cls(root)

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        """
        Push a value onto all active stack heads, returning a new GSS state.
        Implemented as: new_root = Branch({value: old_root}, terms=[]).
        """
        # Equivalent to pre-pending 'value' to every top-first path.
        new_root = _canonicalize_branch((), {value: self._root})
        return LeveledGSS(new_root)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        """
        Pop the top value from all non-empty stacks.
        """
        return LeveledGSS(_b_pop(self._root))

    def is_empty(self) -> bool:
        """
        Checks if the GSS contains only the initial empty stack.
        """
        # Exactly one stack and it is empty
        return _b_is_single_empty(self._root)

    def isolate(self, value: T) -> 'LeveledGSS[T, Acc]':
        """
        Keep only stacks that have `value` at the top.
        """
        return LeveledGSS(_b_isolate(self._root, value))

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        """
        Apply a function to each accumulator, returning a new GSS state.
        """
        return LeveledGSS(_b_apply(self._root, func))

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        """
        Removes stacks for which predicate(acc) is False.
        """
        return LeveledGSS(_b_prune(self._root, predicate))

    def peek(self) -> Set[T]:
        """
        Returns the set of all values at the top of any stack.
        """
        return _b_peek(self._root)

    def reduce_acc(self, merge_func: Callable[[Acc, Acc], Acc]) -> Optional[Acc]:
        """
        Merges the accumulators of all active stacks into a single optional value.
        Returns None if there are no active stacks.

        Note: We enumerate all (including duplicates) stacks and fold over the accs.
        The caller should ensure merge_func is associative/commutative if deterministic results
        are required across implementations (order of reduction is unspecified).
        """
        it = _b_iter_entries(self._root, [])
        try:
            _st0, acc0 = next(it)
        except StopIteration:
            return None
        res = acc0
        for _st, acc in it:
            res = merge_func(res, acc)
        return res

    @staticmethod
    def merge(gss_list: Iterable['LeveledGSS[T, Acc]'], merge_func: Callable[[Acc, Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        """
        Merges multiple GSS instances into one, combining accumulators for identical stacks.
        If the same stack appears multiple times, their accumulators are merged via `merge_func`.
        """
        # Accumulate by stack path (as tuple bottom-first) and merge using merge_func
        acc_by_stack: Dict[Tuple[T, ...], Acc] = {}
        for g in gss_list:
            for stack, acc in _b_iter_entries(g._root, []):
                key = tuple(stack)  # bottom-first
                if key in acc_by_stack:
                    acc_by_stack[key] = merge_func(acc_by_stack[key], acc)
                else:
                    acc_by_stack[key] = acc
        # Rebuild via from_stacks
        merged_stacks: List[Tuple[List[T], Acc]] = [([*key], acc) for key, acc in acc_by_stack.items()]
        return LeveledGSS.from_stacks(merged_stacks)

    def to_json_serializable(self) -> Any:
        """
        Returns a JSON-serializable, canonical representation of the GSS state for comparison:
        list of [stack_as_list_bottom_first, acc], sorted deterministically:
          - by stack length,
          - then by JSON of stack,
          - then by JSON of acc.
        """
        entries: List[Tuple[List[T], Acc]] = list(_b_iter_entries(self._root, []))

        def sort_key(item: Tuple[List[T], Acc]):
            st_list, acc = item
            st_json = json.dumps(st_list, sort_keys=True, separators=(",", ":"))
            acc_json = json.dumps(acc, sort_keys=True, separators=(",", ":"))
            return (len(st_list), st_json, acc_json)

        entries_sorted = sorted(entries, key=sort_key)
        return [[stack, acc] for stack, acc in entries_sorted]

    def __str__(self) -> str:
        try:
            data = self.to_json_serializable()
            return json.dumps(data, indent=2, sort_keys=True)
        except Exception:
            return super().__str__()

    def __eq__(self, other):
        if not isinstance(other, GSS):
            return NotImplemented
        return self.to_json_serializable() == other.to_json_serializable()

    def __hash__(self):
        data = self.to_json_serializable()
        s = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hash(s)
