from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Set, Tuple, Type, TypeVar, Literal)

from ..interface import GSS, Acc, NewAcc, T
from .leveled_impl import LeveledGSS, LeveledGSSStats
from .reference_impl import ReferenceGSS


@dataclass(eq=False)
class FactoredLeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A GSS implementation that factors accumulators out of the core graph structure.

    - It uses a `LeveledGSS[T, int]` as its internal graph representation.
    - Accumulators (`Acc`) are stored in a separate map from integer IDs to `Acc` instances.
    - This design makes accumulator transformations (`apply`, `prune`) very fast, as they
      only need to modify the external map, not traverse the graph structure.
    - The cost of this factorization is paid during `merge` operations or when a
      canonical representation is required (`to_stacks`), as accumulator IDs may need
      to be remapped and the graph structure potentially updated to reflect merged
      accumulator values.
    - This implementation aims to combine the structural sharing of `LeveledGSS` with the
      fast accumulator operations of `LeveledPerAccGSS`.
    """
    _inner: LeveledGSS[T, int]
    _id_to_acc: Dict[int, Acc]
    _next_id: int
    _is_canonical: bool = field(repr=False, default=True)

    # ------------------------------
    # Constructors and serialization
    # ------------------------------
    @classmethod
    def empty(cls: Type["FactoredLeveledGSS[T, Acc]"]) -> "FactoredLeveledGSS[T, Acc]":
        return cls(
            _inner=LeveledGSS.empty(),
            _id_to_acc={},
            _next_id=0,
            _is_canonical=True,
        )

    @classmethod
    def from_stacks(cls: Type["FactoredLeveledGSS[T, Acc]"], stacks: List[Tuple[List[T], Acc]]) -> "FactoredLeveledGSS[T, Acc]":
        # Use ReferenceGSS to merge duplicates first, ensuring canonical input stacks.
        canonical_stacks = ReferenceGSS(stacks)._stacks

        acc_to_id: Dict[Acc, int] = {}
        id_to_acc: Dict[int, Acc] = {}
        next_id = 0
        inner_stacks: List[Tuple[List[T], int]] = []

        for vals, acc in canonical_stacks:
            if acc not in acc_to_id:
                acc_to_id[acc] = next_id
                id_to_acc[next_id] = acc
                next_id += 1
            inner_stacks.append((vals, acc_to_id[acc]))

        inner_gss = LeveledGSS.from_stacks(inner_stacks)
        return cls(
            _inner=inner_gss,
            _id_to_acc=id_to_acc,
            _next_id=next_id,
            _is_canonical=True,
        )

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        # This method must produce a canonical list, so it resolves the lazy mappings.
        result_stacks: List[Tuple[List[T], Acc]] = []
        for path, acc_id in self._inner.to_stacks():
            if acc_id in self._id_to_acc:
                acc = self._id_to_acc[acc_id]
                result_stacks.append((path, acc))

        # Use ReferenceGSS to merge stacks that might now have identical paths and
        # accumulators, and to sort into a canonical order.
        return ReferenceGSS.from_stacks(result_stacks).to_stacks()

    # ------------------------------
    # Core stack operations
    # ------------------------------
    def push(self, value: T) -> "FactoredLeveledGSS[T, Acc]":
        if self.is_empty():
            return self
        new_inner = self._inner.push(value)
        return FactoredLeveledGSS(new_inner, self._id_to_acc, self._next_id, self._is_canonical)

    def pop(self) -> "FactoredLeveledGSS[T, Acc]":
        if self.is_empty():
            return self
        new_inner = self._inner.pop()
        return FactoredLeveledGSS(new_inner, self._id_to_acc, self._next_id, self._is_canonical)

    def popn(self, n: int) -> "FactoredLeveledGSS[T, Acc]":
        if n <= 0 or self.is_empty():
            return self
        new_inner = self._inner.popn(n)
        return FactoredLeveledGSS(new_inner, self._id_to_acc, self._next_id, self._is_canonical)

    def is_empty(self) -> bool:
        # If the accumulator map is empty, no stacks can exist.
        # If the inner graph is empty, no stacks can exist.
        return not self._id_to_acc or self._inner.is_empty()

    def isolate(self, value: Optional[T]) -> "FactoredLeveledGSS[T, Acc]":
        if self.is_empty():
            return self
        new_inner = self._inner.isolate(value)
        return FactoredLeveledGSS(new_inner, self._id_to_acc, self._next_id, self._is_canonical)

    def isolate_many(self, values: Iterable[Optional[T]]) -> "FactoredLeveledGSS[T, Acc]":
        if self.is_empty():
            return self
        new_inner = self._inner.isolate_many(values)
        return FactoredLeveledGSS(new_inner, self._id_to_acc, self._next_id, self._is_canonical)

    # ------------------------------
    # Accumulator transforms (Lazy)
    # ------------------------------
    def apply(self, func: Callable[[Acc], NewAcc], memo: Optional[Dict[int, Any]] = None) -> GSS[T, NewAcc]:
        new_id_to_acc: Dict[int, NewAcc] = {
            id: func(acc) for id, acc in self._id_to_acc.items()
        }
        return FactoredLeveledGSS(
            _inner=self._inner,
            _id_to_acc=new_id_to_acc,
            _next_id=self._next_id,
            _is_canonical=False,
        )

    def prune(self, predicate: Callable[[Acc], bool], memo: Optional[Dict[int, Any]] = None) -> "FactoredLeveledGSS[T, Acc]":
        new_id_to_acc = {
            id: acc for id, acc in self._id_to_acc.items() if predicate(acc)
        }
        return FactoredLeveledGSS(
            _inner=self._inner,
            _id_to_acc=new_id_to_acc,
            _next_id=self._next_id,
            _is_canonical=False,
        )

    def apply_and_prune(self, mutator: Callable[[Acc], Optional[NewAcc]], memo: Optional[Dict[int, Any]] = None) -> GSS[T, NewAcc]:
        new_id_to_acc: Dict[int, NewAcc] = {}
        for id, acc in self._id_to_acc.items():
            new_acc = mutator(acc)
            if new_acc is not None:
                new_id_to_acc[id] = new_acc
        return FactoredLeveledGSS(
            _inner=self._inner,
            _id_to_acc=new_id_to_acc,
            _next_id=self._next_id,
            _is_canonical=False,
        )

    # ------------------------------
    # Merge operations
    # ------------------------------
    def merge(self, other: "FactoredLeveledGSS[T, Acc]") -> "FactoredLeveledGSS[T, Acc]":
        if not isinstance(other, FactoredLeveledGSS):
            # Fallback for merging with other GSS types
            return FactoredLeveledGSS.from_stacks(self.to_stacks() + other.to_stacks())

        if self.is_empty():
            return other
        if other.is_empty():
            return self

        # Lazy merge: create a disjoint union of ID spaces and merge the inner graphs.
        # The resulting accumulator map may be non-canonical.
        offset = self._next_id
        other_inner_remapped = other._inner.apply(lambda old_id: old_id + offset)
        new_inner = self._inner.merge(other_inner_remapped)

        new_id_to_acc = dict(self._id_to_acc)
        for other_id, other_acc in other._id_to_acc.items():
            new_id_to_acc[other_id + offset] = other_acc

        new_next_id = self._next_id + other._next_id

        return FactoredLeveledGSS(
            _inner=new_inner,
            _id_to_acc=new_id_to_acc,
            _next_id=new_next_id,
            _is_canonical=False,
        )

    # ------------------------------
    # Structure-level helpers
    # ------------------------------
    def peek(self) -> Set[T]:
        return self._inner.peek()

    def reduce_acc(self) -> Optional[Acc]:
        # This operates on the potentially non-canonical accumulator map.
        unique_accs = list(set(self._id_to_acc.values()))
        if not unique_accs:
            return None
        return reduce(lambda a, b: a.merge(b), unique_accs)

    def canonicalize(self) -> "FactoredLeveledGSS[T, Acc]":
        """
        Forces the GSS into a canonical state. This involves:
        1. Pruning any parts of the inner graph that refer to deleted accumulator IDs.
        2. Compacting the accumulator map so that each unique accumulator has exactly one ID.
        3. Applying the ID remapping to the inner graph.
        This can be an expensive operation and is performed lazily.
        """
        if self._is_canonical:
            return self

        # 1. Prune dead IDs from inner GSS
        surviving_old_ids = set(self._id_to_acc.keys())
        pruned_inner = self._inner.prune(lambda old_id: old_id in surviving_old_ids)

        # 2. Build remapping for compaction
        new_id_to_acc: Dict[int, Acc] = {}
        acc_to_new_id: Dict[Acc, int] = {}
        remapping: Dict[int, int] = {}
        next_id = 0

        # Sort for deterministic ID assignment
        sorted_old_ids = sorted(list(surviving_old_ids))

        for old_id in sorted_old_ids:
            acc = self._id_to_acc[old_id]
            if acc not in acc_to_new_id:
                new_id = next_id
                acc_to_new_id[acc] = new_id
                new_id_to_acc[new_id] = acc
                next_id += 1
            remapping[old_id] = acc_to_new_id[acc]

        # 3. Apply remapping to the inner graph
        remapped_inner = pruned_inner.apply(lambda old_id: remapping[old_id])

        return FactoredLeveledGSS(
            _inner=remapped_inner,
            _id_to_acc=new_id_to_acc,
            _next_id=next_id,
            _is_canonical=True,
        )

    # ------------------------------
    # LeveledGSS-specific (drop-in) extras
    # ------------------------------
    def fuse(self, levels: Optional[int | Literal["to_interface"]] = None, memo: Optional[Dict] = None) -> "FactoredLeveledGSS[T, Acc]":
        new_inner = self._inner.fuse(levels=levels, memo=memo)
        if new_inner is self._inner:
            return self
        return FactoredLeveledGSS(new_inner, self._id_to_acc, self._next_id, self._is_canonical)

    def stats(self) -> LeveledGSSStats[T, Acc]:
        """
        Computes statistics. Structural stats are from the inner graph, while
        accumulator stats are from the (potentially non-canonical) outer map.
        """
        struct_stats = self._inner.stats()
        
        # Override accumulator stats with info from the outer map
        unique_accs = set(self._id_to_acc.values())
        
        return LeveledGSSStats(
            top_values=struct_stats.top_values,
            num_upperbranch_nodes=struct_stats.num_upperbranch_nodes,
            num_interface_nodes=struct_stats.num_interface_nodes,
            num_lower_nodes=struct_stats.num_lower_nodes,
            total_unique_nodes=struct_stats.total_unique_nodes,
            upper_edges=struct_stats.upper_edges,
            interface_to_lower_edges=struct_stats.interface_to_lower_edges,
            lower_edges=struct_stats.lower_edges,
            total_edges=struct_stats.total_edges,
            max_upper_depth=struct_stats.max_upper_depth,
            max_lower_depth=struct_stats.max_lower_depth,
            distinct_values_count=struct_stats.distinct_values_count,
            distinct_values=struct_stats.distinct_values,
            unique_accumulators_count=len(unique_accs),
            unique_accumulators=unique_accs,
            total_accumulator_instances=struct_stats.total_accumulator_instances,
            num_upper_with_empty=struct_stats.num_upper_with_empty,
            num_interfaces_with_empty=struct_stats.num_interfaces_with_empty,
            num_lower_terminal_nodes=struct_stats.num_lower_terminal_nodes,
            num_interface_implicit_terminals=struct_stats.num_interface_implicit_terminals,
            num_multi_depth_slots_upper=struct_stats.num_multi_depth_slots_upper,
            num_multi_depth_slots_lower=struct_stats.num_multi_depth_slots_lower,
            max_multiplicity_per_value_upper=struct_stats.max_multiplicity_per_value_upper,
            max_multiplicity_per_value_lower=struct_stats.max_multiplicity_per_value_lower,
            average_in_degree=struct_stats.average_in_degree,
            max_in_degree=struct_stats.max_in_degree,
            structural_sharing_factor=struct_stats.structural_sharing_factor,
            promotable_upper_nodes=struct_stats.promotable_upper_nodes,
        )

    def to_graph_string(self, memo: Optional[Set[int]] = None, upper_only: bool = False) -> str:
        lines = []
        lines.append("--- FactoredLeveledGSS ---")
        lines.append(f"Canonical: {self._is_canonical}, Next ID: {self._next_id}")
        lines.append("Accumulator Map:")
        if not self._id_to_acc:
            lines.append("  (empty)")
        else:
            for id, acc in sorted(self._id_to_acc.items()):
                lines.append(f"  ID {id} -> {acc!r}")
        
        lines.append("\n--- Inner LeveledGSS[T, int] ---")
        inner_str = self._inner.to_graph_string(memo=memo, upper_only=upper_only)
        lines.append(inner_str)
        return "\n".join(lines)
