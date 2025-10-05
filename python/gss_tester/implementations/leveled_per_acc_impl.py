from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Dict, Generic, Iterable, List, Optional, Set, Tuple, Type, TypeVar, Callable, Any, Literal

from ..interface import GSS, T, Acc, NewAcc, Mergeable
from .leveled_impl import LeveledGSS, LeveledGSSStats
from .reference_impl import ReferenceGSS


# Internal Unit accumulator for inner LeveledGSS graphs.
# This accumulator carries no information and simply satisfies the Mergeable protocol.
@dataclass(frozen=True, eq=True)
class _UnitAcc(Mergeable):
    def merge(self, other: "_UnitAcc") -> "_UnitAcc":
        return self

    def __repr__(self) -> str:
        return "UnitAcc"


_UNIT: _UnitAcc = _UnitAcc()


@dataclass(eq=False)
class LeveledPerAccGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A drop-in GSS implementation that partitions the graph by accumulator.

    - Internally maintains a dict mapping each accumulator (Acc) to a LeveledGSS[T, _UnitAcc].
    - The inner LeveledGSS graphs do not store real Acc values in their nodes; they carry a
      single sentinel accumulator (_UnitAcc) everywhere.
    - All structure-changing operations (push/pop/isolate/merge/etc.) are delegated to the inner
      LeveledGSS instances and then re-assembled per accumulator partition.

    This can be more efficient when there are few distinct accumulators but large shared structure,
    as it avoids mixing (and propagating) Acc values into the graph.
    """

    # Mapping from accumulator to a leveled GSS that uses a unit accumulator internally.
    _parts: Dict[Acc, LeveledGSS[T, _UnitAcc]]

    def __post_init__(self):
        # Maintain class invariant: drop any empty inner graphs to keep the representation tight.
        self._parts = {a: g for a, g in self._parts.items() if not g.is_empty()}

    # ------------------------------
    # Constructors and serialization
    # ------------------------------
    @classmethod
    def empty(cls: Type["LeveledPerAccGSS[T, Acc]"]) -> "LeveledPerAccGSS[T, Acc]":
        return cls(_parts={})

    @classmethod
    def from_stacks(cls: Type["LeveledPerAccGSS[T, Acc]"], stacks: List[Tuple[List[T], Acc]]) -> "LeveledPerAccGSS[T, Acc]":
        # Group stacks by accumulator
        by_acc: Dict[Acc, List[List[T]]] = {}
        for vals, acc in stacks:
            by_acc.setdefault(acc, []).append(list(vals))

        parts: Dict[Acc, LeveledGSS[T, _UnitAcc]] = {}
        for acc, lists in by_acc.items():
            inner_stacks = [(v, _UNIT) for v in lists]
            parts[acc] = LeveledGSS.from_stacks(inner_stacks)

        return cls(_parts=parts)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        # Gather stacks from each partition and replace inner UnitAcc with the partition's Acc
        out: List[Tuple[List[T], Acc]] = []
        for acc, inner in self._parts.items():
            for vals, _unit in inner.to_stacks():
                out.append((list(vals), acc))
        # Canonicalize ordering and merge possible duplicates via ReferenceGSS
        return ReferenceGSS.from_stacks(out).to_stacks()

    # ------------------------------
    # Core stack operations
    # ------------------------------
    def push(self, value: T) -> "LeveledPerAccGSS[T, Acc]":
        if self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.push(value) for acc, g in self._parts.items()})

    def pop(self) -> "LeveledPerAccGSS[T, Acc]":
        if self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.pop() for acc, g in self._parts.items()})

    def popn(self, n: int) -> "LeveledPerAccGSS[T, Acc]":
        if n <= 0 or self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.popn(n) for acc, g in self._parts.items()})

    def is_empty(self) -> bool:
        return not self._parts

    def isolate(self, value: Optional[T]) -> "LeveledPerAccGSS[T, Acc]":
        if self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.isolate(value) for acc, g in self._parts.items()})

    def isolate_many(self, values: Iterable[Optional[T]]) -> "LeveledPerAccGSS[T, Acc]":
        valset = set(values)
        if not valset or self.is_empty():
            return LeveledPerAccGSS.empty()
        return LeveledPerAccGSS({acc: g.isolate_many(valset) for acc, g in self._parts.items()})

    def filter_by_length(self, min_len: Optional[int] = None, max_len: Optional[int] = None) -> "LeveledPerAccGSS[T, Acc]":
        if self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.filter_by_length(min_len, max_len) for acc, g in self._parts.items()})

    # ------------------------------
    # Accumulator transforms
    # ------------------------------
    def apply(self, func: Callable[[Acc], NewAcc], memo: Optional[Dict[int, Any]] = None) -> GSS[T, NewAcc]:
        # Map each partition's accumulator and merge partitions that map to the same new accumulator.
        new_parts: Dict[NewAcc, LeveledGSS[T, _UnitAcc]] = {}
        for acc, g in self._parts.items():
            new_acc = func(acc)
            if new_acc in new_parts:
                new_parts[new_acc] = new_parts[new_acc].merge(g)
            else:
                new_parts[new_acc] = g
        return LeveledPerAccGSS[T, NewAcc](new_parts)  # type: ignore[type-var]

    def prune(self, predicate: Callable[[Acc], bool], memo: Optional[Dict[int, Any]] = None) -> "LeveledPerAccGSS[T, Acc]":
        # Keep only partitions whose accumulator satisfies the predicate.
        return LeveledPerAccGSS({acc: g for acc, g in self._parts.items() if predicate(acc)})

    def apply_and_prune(self, mutator: Callable[[Acc], Optional[NewAcc]], memo: Optional[Dict[int, Any]] = None) -> GSS[T, NewAcc]:
        # Single pass over partitions: mutate acc, drop if None, merge graphs for identical results.
        cache: Dict[int, Optional[NewAcc]] = {} if memo is None else memo  # reuse provided memo if any

        def decide(a: Acc) -> Optional[NewAcc]:
            k = id(a)
            if k in cache:
                return cache[k]
            r = mutator(a)
            cache[k] = r
            return r

        new_parts: Dict[NewAcc, LeveledGSS[T, _UnitAcc]] = {}
        for acc, g in self._parts.items():
            new_acc = decide(acc)
            if new_acc is None:
                continue
            if new_acc in new_parts:
                new_parts[new_acc] = new_parts[new_acc].merge(g)
            else:
                new_parts[new_acc] = g
        return LeveledPerAccGSS[T, NewAcc](new_parts)  # type: ignore[type-var]

    # ------------------------------
    # Merge operations
    # ------------------------------
    def merge(self, other: "LeveledPerAccGSS[T, Acc]") -> "LeveledPerAccGSS[T, Acc]":
        if self is other:
            return self
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        merged: Dict[Acc, LeveledGSS[T, _UnitAcc]] = dict(self._parts)
        for acc, g in other._parts.items():
            if acc in merged:
                merged[acc] = merged[acc].merge(g)
            else:
                merged[acc] = g
        return LeveledPerAccGSS(merged)

    @classmethod
    def merge_many(cls: Type["LeveledPerAccGSS[T, Acc]"], gss_list: Iterable["LeveledPerAccGSS[T, Acc]"]) -> "LeveledPerAccGSS[T, Acc]":
        result: Dict[Acc, LeveledGSS[T, _UnitAcc]] = {}
        for g in gss_list:
            for acc, inner in g._parts.items():
                if acc in result:
                    result[acc] = result[acc].merge(inner)
                else:
                    result[acc] = inner
        return cls(result)

    # ------------------------------
    # Structure-level helpers
    # ------------------------------
    def peek(self) -> Set[T]:
        tops: Set[T] = set()
        for g in self._parts.values():
            tops |= g.peek()
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        if self.is_empty():
            return None
        it = iter(self._parts.keys())
        try:
            acc0 = next(it)
        except StopIteration:
            return None
        result = acc0
        for acc in it:
            if result is acc:
                continue
            result = result.merge(acc)
        return result

    def to_reference_impl(self) -> GSS[T, Acc]:
        # Build ReferenceGSS directly from the stacks of all partitions.
        return ReferenceGSS.from_stacks(self.to_stacks())

    # ------------------------------
    # LeveledGSS-specific (drop-in) extras
    # ------------------------------
    def fuse(self, levels: Optional[int | Literal["to_interface"]] = None, memo: Optional[Dict] = None) -> "LeveledPerAccGSS[T, Acc]":
        if self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.fuse(levels=levels, memo=memo) for acc, g in self._parts.items()})

    def stats(self) -> LeveledGSSStats[T, Acc]:
        """
        Aggregate structural stats across all partitions.
        Structural fields are computed by merging the inner LeveledGSS graphs into one,
        then calling stats() on the union. Accumulator coverage is reported using the
        outer partition keys.
        """
        if self.is_empty():
            # Reuse stats from an empty leveled GSS, then override accumulator coverage fields.
            empty_stats = LeveledGSS.from_stacks([]).stats()  # type: ignore[type-var]
            return LeveledGSSStats(
                top_values=empty_stats.top_values,
                num_upperbranch_nodes=empty_stats.num_upperbranch_nodes,
                num_interface_nodes=empty_stats.num_interface_nodes,
                num_lower_nodes=empty_stats.num_lower_nodes,
                total_unique_nodes=empty_stats.total_unique_nodes,
                upper_edges=empty_stats.upper_edges,
                interface_to_lower_edges=empty_stats.interface_to_lower_edges,
                lower_edges=empty_stats.lower_edges,
                total_edges=empty_stats.total_edges,
                max_upper_depth=empty_stats.max_upper_depth,
                max_lower_depth=empty_stats.max_lower_depth,
                distinct_values_count=empty_stats.distinct_values_count,
                distinct_values=empty_stats.distinct_values,
                unique_accumulators_count=0,
                unique_accumulators=set(),
                total_accumulator_instances=empty_stats.total_accumulator_instances,
                num_upper_with_empty=empty_stats.num_upper_with_empty,
                num_interfaces_with_empty=empty_stats.num_interfaces_with_empty,
                num_lower_terminal_nodes=empty_stats.num_lower_terminal_nodes,
                num_interface_implicit_terminals=empty_stats.num_interface_implicit_terminals,
                num_multi_depth_slots_upper=empty_stats.num_multi_depth_slots_upper,
                num_multi_depth_slots_lower=empty_stats.num_multi_depth_slots_lower,
                max_multiplicity_per_value_upper=empty_stats.max_multiplicity_per_value_upper,
                max_multiplicity_per_value_lower=empty_stats.max_multiplicity_per_value_lower,
                average_in_degree=empty_stats.average_in_degree,
                max_in_degree=empty_stats.max_in_degree,
                structural_sharing_factor=empty_stats.structural_sharing_factor,
                promotable_upper_nodes=empty_stats.promotable_upper_nodes,
            )

        # Merge all inner graphs into a single LeveledGSS over UnitAcc to compute structural stats.
        merged_inner: Optional[LeveledGSS[T, _UnitAcc]] = None
        for g in self._parts.values():
            merged_inner = g if merged_inner is None else merged_inner.merge(g)
        assert merged_inner is not None
        struct_stats = merged_inner.stats()

        # Acc coverage uses the outer keys (the true Acc domain for this GSS).
        acc_keys: Set[Acc] = set(self._parts.keys())
        acc_count = len(acc_keys)

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
            unique_accumulators_count=acc_count,
            unique_accumulators=acc_keys,
            # This counts physical accumulator storage in the merged inner graphs (UnitAcc slots).
            # It is not equal to unique_accumulators_count, which is keyed by outer Acc values.
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
        if memo is None:
            memo = set()
        if self.is_empty():
            return "--- Empty LeveledPerAccGSS ---"
        out_lines: List[str] = []
        for i, (acc, inner) in enumerate(sorted(self._parts.items(), key=lambda kv: repr(kv[0]))):
            prefix = "" if i == 0 else "\n"
            out_lines.append(prefix + f"=== Partition for Acc: {repr(acc)} ===")
            inner_str = inner.to_graph_string(memo=memo, upper_only=upper_only)
            # Indent inner graph for clarity
            for line in inner_str.splitlines():
                out_lines.append("  " + line)
        return "\n".join(out_lines)

    # ------------------------------
    # String representations
    # ------------------------------
    def __str__(self) -> str:
        """Human-readable string representation."""
        items = self.to_stacks()
        if not items:
            return f"{self.__class__.__name__}(empty)"
        lines = [f"{self.__class__.__name__}:"]
        for vals, acc in items:
            lines.append(f"  - Stack: {vals}, Acc: {acc!r}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Unambiguous string representation."""
        items = self.to_stacks()
        return f"{self.__class__.__name__}(_parts={[ (acc, '<inner>') for acc in self._parts.keys() ]!r})"


Leveled_per_accGSS = LeveledPerAccGSS