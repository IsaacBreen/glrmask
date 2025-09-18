from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

from .interface import GSS, T, Acc
from .reference_impl import ReferenceGSS


@dataclass(frozen=True, eq=False)
class _StackNode(Generic[T]):
    """
    A single structural node representing one stack cell in a persistent linked structure.

    The stack for a head is represented by the chain:
      (prev --(value)-> this)
    where `value` is the symbol pushed at this step.

    Sharing:
      Nodes are interned via _NodeFactory to ensure identical (prev, value) pairs
      are represented by the same object, enabling a compact DAG.
    """
    prev: Optional["_StackNode[T]"]
    value: T

    def __repr__(self) -> str:
        return f"_StackNode(value={self.value!r}, prev_id={id(self.prev) if self.prev else None})"


class _NodeFactory(Generic[T]):
    """
    Interns _StackNode(prev, value) so identical pairs share the same object.
    """
    def __init__(self) -> None:
        self._table: Dict[Tuple[Optional[_StackNode[T]], T], _StackNode[T]] = {}

    def get(self, prev: Optional[_StackNode[T]], value: T) -> _StackNode[T]:
        key = (prev, value)
        node = self._table.get(key)
        if node is None:
            node = _StackNode(prev, value)
            self._table[key] = node
        return node


class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A Graph-Structured Stack (GSS) implementation using a persistent linked-DAG of stack cells.

    Design highlights:
    - Structure: each non-empty stack head is a _StackNode whose `prev` points to the remainder
      of the stack, and `value` is its top symbol. Identical suffixes are shared via interning.
    - Accumulators live only at heads: we keep a multiset (list) of accumulator values per head node.
      The empty stack(s) are tracked separately as a list of accumulators.
      This ensures per-stack accumulators are never scattered along a path (only at the head),
      which matches the "accs exist at one level" idea per path while preserving multiplicity.
    - Edges carry T and max depth: we maintain a map from (parent_node, T) -> max depth of any
      active stack that traverses that edge. Depth is measured as the length of the stack (1-based)
      at the child node. This metadata is recomputed after each operation.
    - Operations are persistent: each method returns a new LeveledGSS without mutating prior states.

    Equivalence:
    - All operations behave equivalently to ReferenceGSS in terms of observable results produced
      via to_reference_impl (which canonicalizes by merging accumulators for identical stacks).
    - reduce_acc will return the same result as the reference implementation when the accumulator
      merge is order-independent (commutative-associative), which is the documented caveat.
    """

    def __init__(
        self,
        node_factory: Optional[_NodeFactory[T]] = None,
        heads: Optional[Dict[_StackNode[T], List[Acc]]] = None,
        empty_accs: Optional[List[Acc]] = None,
    ) -> None:
        self._factory: _NodeFactory[T] = node_factory if node_factory is not None else _NodeFactory()
        # map: head_node -> list of Acc (multiplicity preserved)
        self._heads: Dict[_StackNode[T], List[Acc]] = heads if heads is not None else {}
        # accumulators for empty stacks (each entry is one stack's acc)
        self._empty_accs: List[Acc] = empty_accs if empty_accs is not None else []
        # Edge metadata: (parent_node, T) -> max depth (absolute depth from the implicit root)
        # parent_node is None for the first edge (i.e., edge to single-element stacks).
        self._edge_max_depth: Dict[Tuple[Optional[_StackNode[T]], T], int] = {}
        self._recompute_edge_max_depth()

    # ------------- Construction -------------

    @classmethod
    def from_stacks(cls: Type["LeveledGSS[T, Acc]"], stacks: List[Tuple[List[T], Acc]]) -> "LeveledGSS[T, Acc]":
        factory: _NodeFactory[T] = _NodeFactory()
        heads: Dict[_StackNode[T], List[Acc]] = {}
        empty_accs: List[Acc] = []
        # Build heads by inserting each explicit stack
        for vals, acc in stacks:
            if not vals:
                empty_accs.append(acc)
                continue
            prev: Optional[_StackNode[T]] = None
            for v in vals:
                prev = factory.get(prev, v)
            # prev now is the head node for this stack
            lst = heads.get(prev)
            if lst is None:
                heads[prev] = [acc]
            else:
                lst.append(acc)
        return cls(factory, heads, empty_accs)

    # ------------- Core helpers -------------

    def _recompute_edge_max_depth(self) -> None:
        """
        Recompute edge max-depths from the current multiset of heads.
        Edge depth is the absolute stack length at the child endpoint of that edge.
        """
        edge_depths: Dict[Tuple[Optional[_StackNode[T]], T], int] = {}

        # helper to compute depth (length) of a head node
        def head_depth(node: _StackNode[T]) -> int:
            d = 0
            cur: Optional[_StackNode[T]] = node
            while cur is not None:
                d += 1
                cur = cur.prev
            return d

        # For each head, walk up the chain and update edge depths
        for head in self._heads.keys():
            d = head_depth(head)
            cur: Optional[_StackNode[T]] = head
            while cur is not None:
                parent = cur.prev
                # edge parent --(cur.value)--> cur has depth d
                key = (parent, cur.value)
                prev_max = edge_depths.get(key)
                if prev_max is None or d > prev_max:
                    edge_depths[key] = d
                d -= 1
                cur = parent

        # Also consider single-step edges to first element from empty via empty heads if present
        # Actually empty stacks do not contribute any edges; they only exist as [].

        self._edge_max_depth = edge_depths

    def _clone_with(
        self,
        heads: Optional[Dict[_StackNode[T], List[Acc]]] = None,
        empty_accs: Optional[List[Acc]] = None,
    ) -> "LeveledGSS[T, Acc]":
        new_gss = LeveledGSS(self._factory, heads if heads is not None else dict(self._heads),
                              empty_accs if empty_accs is not None else list(self._empty_accs))
        # _recompute_edge_max_depth is called in __init__
        return new_gss

    # ------------- Interface implementation -------------

    def push(self, value: T) -> "LeveledGSS[T, Acc]":
        """
        Push `value` onto all active stack heads, returning a new GSS state.
        """
        new_heads: Dict[_StackNode[T], List[Acc]] = {}
        # Move all non-empty heads down one level by adding `value`
        for head, accs in self._heads.items():
            new_head = self._factory.get(head, value)
            lst = new_heads.get(new_head)
            if lst is None:
                new_heads[new_head] = list(accs)
            else:
                lst.extend(accs)
        # Move empty stacks as well
        if self._empty_accs:
            first_head = self._factory.get(None, value)
            lst = new_heads.get(first_head)
            if lst is None:
                new_heads[first_head] = list(self._empty_accs)
            else:
                lst.extend(self._empty_accs)
        # After push, there are no empty stacks
        return self._clone_with(heads=new_heads, empty_accs=[])

    def pop(self) -> "LeveledGSS[T, Acc]":
        """
        For all active stacks, create stacks by removing the top value.
        Returns a new GSS state containing the popped stacks.
        """
        new_heads: Dict[_StackNode[T], List[Acc]] = {}
        new_empty: List[Acc] = []

        for head, accs in self._heads.items():
            prev = head.prev
            if prev is None:
                # Popping a 1-element stack yields the empty stack
                new_empty.extend(accs)
            else:
                lst = new_heads.get(prev)
                if lst is None:
                    new_heads[prev] = list(accs)
                else:
                    lst.extend(accs)
        # Popping an empty stack yields nothing (as in reference implementation)
        return self._clone_with(heads=new_heads, empty_accs=new_empty)

    def is_empty(self) -> bool:
        """
        True iff there is exactly one active stack and that stack is the empty stack.
        """
        # count active non-empty stacks
        non_empty_count = sum(len(accs) for accs in self._heads.values())
        return non_empty_count == 0 and len(self._empty_accs) == 1

    def isolate(self, value: Optional[T]) -> "LeveledGSS[T, Acc]":
        """
        Keeps only stacks whose top equals `value`, or only the empty stacks if value is None.
        """
        if value is None:
            # Keep only empty stacks, drop all non-empty heads
            return self._clone_with(heads={}, empty_accs=list(self._empty_accs))

        # Keep only those heads whose top symbol equals `value`
        new_heads: Dict[_StackNode[T], List[Acc]] = {}
        for head, accs in self._heads.items():
            if head.value == value:
                new_heads[head] = list(accs)
        return self._clone_with(heads=new_heads, empty_accs=[])

    def apply(self, func: Callable[[Acc], Acc]) -> "LeveledGSS[T, Acc]":
        """
        Applies func to each accumulator independently; returns a new state.
        """
        new_heads: Dict[_StackNode[T], List[Acc]] = {
            head: [func(a) for a in accs] for head, accs in self._heads.items()
        }
        new_empty = [func(a) for a in self._empty_accs]
        return self._clone_with(heads=new_heads, empty_accs=new_empty)

    def prune(self, predicate: Callable[[Acc], bool]) -> "LeveledGSS[T, Acc]":
        """
        Removes stacks whose accumulator does not satisfy predicate.
        """
        new_heads: Dict[_StackNode[T], List[Acc]] = {}
        for head, accs in self._heads.items():
            kept = [a for a in accs if predicate(a)]
            if kept:
                new_heads[head] = kept
        new_empty = [a for a in self._empty_accs if predicate(a)]
        return self._clone_with(heads=new_heads, empty_accs=new_empty)

    def peek(self) -> Set[T]:
        """
        Returns the set of top values across all non-empty stacks.
        """
        return {head.value for head, accs in self._heads.items() if accs}

    def reduce_acc(self) -> Optional[Acc]:
        """
        Merges the accumulators of all active stacks into a single optional value.
        Returns None if there are no active stacks.
        """
        items: List[Acc] = []
        # Order is not guaranteed; only parity with ReferenceGSS is ensured for order-independent merges.
        for accs in self._heads.values():
            items.extend(accs)
        items.extend(self._empty_accs)
        if not items:
            return None
        return reduce(lambda a, b: a.merge(b), items)

    def to_reference_impl(self) -> "ReferenceGSS[T, Acc]":
        """
        Convert to canonical ReferenceGSS by enumerating all explicit stacks (with multiplicity),
        then merging accumulators for identical stacks.
        """
        stacks: List[Tuple[List[T], Acc]] = []

        # Emit non-empty stacks
        for head, accs in self._heads.items():
            # reconstruct the list of T from the head
            vals: List[T] = []
            cur: Optional[_StackNode[T]] = head
            while cur is not None:
                vals.append(cur.value)
                cur = cur.prev
            vals.reverse()
            for a in accs:
                stacks.append((list(vals), a))

        # Emit empty stacks
        for a in self._empty_accs:
            stacks.append(([], a))

        # Build a ReferenceGSS and then merge to canonical representation
        ref = ReferenceGSS.from_stacks(stacks)
        return ReferenceGSS.merge([ref])

    @staticmethod
    def merge(gss_list: Iterable["GSS[T, Acc]"]) -> "LeveledGSS[T, Acc]":
        """
        Merges multiple GSS instances, combining accumulators for identical stacks.
        Implementation strategy:
          - Convert each input to ReferenceGSS (or accept one directly),
          - Use ReferenceGSS.merge to coalesce duplicates,
          - Convert the merged result back into a LeveledGSS.
        """
        ref_inputs: List[ReferenceGSS[T, Acc]] = []
        for g in gss_list:
            if isinstance(g, ReferenceGSS):
                ref_inputs.append(g)
            else:
                ref_inputs.append(g.to_reference_impl())  # type: ignore[arg-type]
        merged_ref: ReferenceGSS[T, Acc] = ReferenceGSS.merge(ref_inputs)
        return LeveledGSS.from_stacks(merged_ref._stacks)  # _stacks are already canonical

    # ------------- Debug/Introspection helpers -------------

    def _edge_max_depth_map(self) -> Dict[Tuple[Optional[_StackNode[T]], T], int]:
        """
        Exposes the internal edge->max_depth map for debugging/introspection:
          key = (parent_node, symbol T); parent_node is None for first step from empty.
          value = max absolute depth of any active stack using that edge.
        """
        return dict(self._edge_max_depth)

    # Override core mutators to recompute edge depths after changes

    # The following overrides clone results to recompute edge metadata.
    # These are not part of the public API, but ensure the "edges carry max depth" invariant is maintained.

    def _clone_with(self, heads: Optional[Dict[_StackNode[T], List[Acc]]] = None,
                    empty_accs: Optional[List[Acc]] = None) -> "LeveledGSS[T, Acc]":
        new_heads = heads if heads is not None else dict(self._heads)
        new_empty = empty_accs if empty_accs is not None else list(self._empty_accs)
        new_inst = LeveledGSS(self._factory, new_heads, new_empty)
        # __init__ invokes _recompute_edge_max_depth
        return new_inst
