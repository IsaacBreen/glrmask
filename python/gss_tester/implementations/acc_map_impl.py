from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
)

from ..interface import GSS, T, Acc, NewAcc
from .leveled_impl import LeveledGSS, LeveledGSSStats
from .reference_impl import ReferenceGSS


@dataclass(frozen=True, eq=True)
class AccMapGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A GSS implementation that partitions the GSS by accumulator.

    The inner representation is a dictionary mapping each accumulator to a
    LeveledGSS instance. Each of these inner GSSs uses `NoneType` for its
    accumulator type, effectively only storing the stack structure.

    This approach may be more efficient when the number of unique accumulators
    is small, but the structural sharing between stacks with different
    accumulators is low.
    """

    inner: Dict[Acc, LeveledGSS[T, NoneType]]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> AccMapGSS[T, Acc]:
        stacks_by_acc: Dict[Acc, List[Tuple[List[T], NoneType]]] = defaultdict(list)
        for vals, acc in stacks:
            stacks_by_acc[acc].append((vals, None))

        inner: Dict[Acc, LeveledGSS[T, NoneType]] = {}
        for acc, acc_stacks in stacks_by_acc.items():
            inner[acc] = LeveledGSS.from_stacks(acc_stacks)

        return cls(inner)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        all_stacks: List[Tuple[List[T], Acc]] = []
        for acc, gss in self.inner.items():
            for vals, _ in gss.to_stacks():
                all_stacks.append((vals, acc))
        return ReferenceGSS(all_stacks).to_stacks()

    def push(self, value: T) -> AccMapGSS[T, Acc]:
        if self.is_empty():
            return self
        new_inner = {acc: gss.push(value) for acc, gss in self.inner.items()}
        return AccMapGSS(new_inner)

    def pop(self) -> AccMapGSS[T, Acc]:
        new_inner: Dict[Acc, LeveledGSS[T, NoneType]] = {}
        for acc, gss in self.inner.items():
            popped_gss = gss.pop()
            if not popped_gss.is_empty():
                new_inner[acc] = popped_gss
        return AccMapGSS(new_inner)

    def popn(self, n: int) -> AccMapGSS[T, Acc]:
        if n <= 0:
            return self
        new_inner: Dict[Acc, LeveledGSS[T, NoneType]] = {}
        for acc, gss in self.inner.items():
            popped_gss = gss.popn(n)
            if not popped_gss.is_empty():
                new_inner[acc] = popped_gss
        return AccMapGSS(new_inner)

    def is_empty(self) -> bool:
        return not self.inner

    def isolate(self, value: Optional[T]) -> AccMapGSS[T, Acc]:
        new_inner: Dict[Acc, LeveledGSS[T, NoneType]] = {}
        for acc, gss in self.inner.items():
            isolated_gss = gss.isolate(value)
            if not isolated_gss.is_empty():
                new_inner[acc] = isolated_gss
        return AccMapGSS(new_inner)

    def isolate_many(self, values: Iterable[Optional[T]]) -> AccMapGSS[T, Acc]:
        new_inner: Dict[Acc, LeveledGSS[T, NoneType]] = {}
        for acc, gss in self.inner.items():
            isolated_gss = gss.isolate_many(values)
            if not isolated_gss.is_empty():
                new_inner[acc] = isolated_gss
        return AccMapGSS(new_inner)

    def apply(
        self, func: Callable[[Acc], NewAcc], memo: Optional[Dict[int, Any]] = None
    ) -> AccMapGSS[T, NewAcc]:
        new_inner: Dict[NewAcc, LeveledGSS[T, NoneType]] = defaultdict(
            lambda: LeveledGSS.from_stacks([])
        )
        for old_acc, gss in self.inner.items():
            new_acc = func(old_acc)
            new_inner[new_acc] = new_inner[new_acc].merge(gss)
        return AccMapGSS(dict(new_inner))

    def prune(
        self, predicate: Callable[[Acc], bool], memo: Optional[Dict[int, Any]] = None
    ) -> AccMapGSS[T, Acc]:
        new_inner = {acc: gss for acc, gss in self.inner.items() if predicate(acc)}
        return AccMapGSS(new_inner)

    def apply_and_prune(
        self, mutator: Callable[[Acc], Optional[NewAcc]], memo: Optional[Dict[int, Any]] = None
    ) -> AccMapGSS[T, NewAcc]:
        new_inner: Dict[NewAcc, LeveledGSS[T, NoneType]] = defaultdict(
            lambda: LeveledGSS.from_stacks([])
        )
        for old_acc, gss in self.inner.items():
            new_acc_opt = mutator(old_acc)
            if new_acc_opt is not None:
                new_inner[new_acc_opt] = new_inner[new_acc_opt].merge(gss)
        return AccMapGSS(dict(new_inner))

    def merge(self, other: AccMapGSS[T, Acc]) -> AccMapGSS[T, Acc]:
        new_inner = self.inner.copy()
        for acc, other_gss in other.inner.items():
            if acc in new_inner:
                new_inner[acc] = new_inner[acc].merge(other_gss)
            else:
                new_inner[acc] = other_gss
        return AccMapGSS(new_inner)

    def peek(self) -> Set[T]:
        all_peeks: Set[T] = set()
        for gss in self.inner.values():
            all_peeks.update(gss.peek())
        return all_peeks

    def reduce_acc(self) -> Optional[Acc]:
        if not self.inner:
            return None
        accs = list(self.inner.keys())
        return reduce(lambda a, b: a.merge(b), accs)

    def stats(self) -> LeveledGSSStats[T, Acc]:
        if self.is_empty():
            empty_set: Set[Any] = set()
            return LeveledGSSStats(
                top_values=empty_set,
                num_upperbranch_nodes=0, num_interface_nodes=0, num_lower_nodes=0, total_unique_nodes=0,
                upper_edges=0, interface_to_lower_edges=0, lower_edges=0, total_edges=0,
                max_upper_depth=0, max_lower_depth=0,
                distinct_values_count=0, distinct_values=empty_set,
                unique_accumulators_count=0, unique_accumulators=empty_set, total_accumulator_instances=0,
                num_upper_with_empty=0, num_interfaces_with_empty=0, num_lower_terminal_nodes=0, num_interface_implicit_terminals=0,
                num_multi_depth_slots_upper=0, num_multi_depth_slots_lower=0,
                max_multiplicity_per_value_upper=0, max_multiplicity_per_value_lower=0,
                average_in_degree=0.0, max_in_degree=0, structural_sharing_factor=0.0,
                promotable_upper_nodes=0
            )

        all_stats = [gss.stats() for gss in self.inner.values()]

        total_unique_nodes = sum(s.total_unique_nodes for s in all_stats)
        total_edges = sum(s.total_edges for s in all_stats)

        if total_unique_nodes > 0:
            avg_in_degree = sum(s.average_in_degree * s.total_unique_nodes for s in all_stats) / total_unique_nodes
        else:
            avg_in_degree = 0.0

        return LeveledGSSStats(
            top_values=set().union(*(s.top_values for s in all_stats)),
            num_upperbranch_nodes=sum(s.num_upperbranch_nodes for s in all_stats),
            num_interface_nodes=sum(s.num_interface_nodes for s in all_stats),
            num_lower_nodes=sum(s.num_lower_nodes for s in all_stats),
            total_unique_nodes=total_unique_nodes,
            upper_edges=sum(s.upper_edges for s in all_stats),
            interface_to_lower_edges=sum(s.interface_to_lower_edges for s in all_stats),
            lower_edges=sum(s.lower_edges for s in all_stats),
            total_edges=total_edges,
            max_upper_depth=max((s.max_upper_depth for s in all_stats), default=0),
            max_lower_depth=max((s.max_lower_depth for s in all_stats), default=0),
            distinct_values=set().union(*(s.distinct_values for s in all_stats)),
            distinct_values_count=len(set().union(*(s.distinct_values for s in all_stats))),
            unique_accumulators_count=len(self.inner),
            unique_accumulators=set(self.inner.keys()),
            total_accumulator_instances=len(self.inner),
            num_upper_with_empty=sum(s.num_upper_with_empty for s in all_stats),
            num_interfaces_with_empty=sum(s.num_interfaces_with_empty for s in all_stats),
            num_lower_terminal_nodes=sum(s.num_lower_terminal_nodes for s in all_stats),
            num_interface_implicit_terminals=sum(s.num_interface_implicit_terminals for s in all_stats),
            num_multi_depth_slots_upper=sum(s.num_multi_depth_slots_upper for s in all_stats),
            num_multi_depth_slots_lower=sum(s.num_multi_depth_slots_lower for s in all_stats),
            max_multiplicity_per_value_upper=max((s.max_multiplicity_per_value_upper for s in all_stats), default=0),
            max_multiplicity_per_value_lower=max((s.max_multiplicity_per_value_lower for s in all_stats), default=0),
            average_in_degree=avg_in_degree,
            max_in_degree=max((s.max_in_degree for s in all_stats), default=0),
            structural_sharing_factor=total_edges / float(max(1, total_unique_nodes - len(self.inner))) if total_unique_nodes > len(self.inner) else 0.0,
            promotable_upper_nodes=sum(s.promotable_upper_nodes for s in all_stats),
        )

    def to_graph_string(self, memo: Optional[Set[int]] = None, upper_only: bool = False) -> str:
        output_lines: List[str] = []
        if memo is None:
            memo = set()

        # Sort by accumulator representation for deterministic output
        sorted_items = sorted(self.inner.items(), key=lambda item: repr(item[0]))

        for i, (acc, gss) in enumerate(sorted_items):
            if i > 0:
                output_lines.append("")
            output_lines.append(f"--- Graph for Acc: {acc!r} ---")
            output_lines.append(gss.to_graph_string(memo=memo, upper_only=upper_only))
        return "\n".join(output_lines)

    def fuse(
        self, levels: Optional[int | Literal["to_interface"]] = None, memo: Optional[Dict] = None
    ) -> AccMapGSS[T, Acc]:
        new_inner = {acc: gss.fuse(levels=levels) for acc, gss in self.inner.items()}
        return AccMapGSS(new_inner)
