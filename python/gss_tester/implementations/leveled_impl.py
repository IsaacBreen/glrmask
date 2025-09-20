from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, cast
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any
from collections import defaultdict

from ..interface import GSS, T, Acc


# ------------------------------
# Internal node classes
# ------------------------------

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    inner: UpperBranch[T, Acc] | Interface[T, Acc]


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    children: Dict[T, Dict[int, Upper[T, Acc]]]


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    node: Lower[T]
    acc: Acc


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    inner: LowerBranch[T] | Leaf


@dataclass(frozen=True, eq=True)
class LowerBranch(Generic[T]):
    children: Dict[T, Dict[int, Lower[T]]]


@dataclass(frozen=True, eq=True)
class Leaf:
    pass


# A shared, canonical leaf node
_LOWER_LEAF = Lower(Leaf())


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        # Canonicalize first using the reference implementation
        from .reference_impl import ReferenceGSS
        merged = ReferenceGSS(stacks).to_stacks()

        empty_acc: Optional[Acc] = None
        # A simple trie: { val: { "i": [acc, ...], "b": <subtrie> } }
        trie: Dict[T, Dict[str, Any]] = {}

        for vals, acc in merged:
            if not vals:
                empty_acc = acc
                continue
            node = trie
            for i, v in enumerate(vals):
                entry = node.setdefault(v, {"i": [], "b": {}})
                if i == len(vals) - 1:
                    entry["i"].append(acc)
                else:
                    node = entry["b"]

        def build(d: Dict[T, Dict[str, Any]]) -> Upper[T, Acc]:
            children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, e in d.items():
                nodes: List[Upper[T, Acc]] = [Upper(Interface(_LOWER_LEAF, a)) for a in e["i"]]
                # Always add a branch node (empty or not) to keep structure uniform and avoid edge invariants.
                branch_child = build(e["b"]) if e["b"] else Upper(UpperBranch({}))
                nodes.append(branch_child)
                children[v] = {i: n for i, n in enumerate(nodes)}
            return Upper(UpperBranch(children))

        return LeveledGSS(build(trie), empty_acc)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        res: List[Tuple[List[T], Acc]] = []
        if self.empty is not None:
            res.append(([], self.empty))

        def dfs(u: Upper[T, Acc], pref: List[T]) -> None:
            if isinstance(u.inner, Interface):
                res.append((pref, u.inner.acc))
                return
            for v, kids in u.inner.children.items():
                for child in kids.values():
                    dfs(child, pref + [v])

        dfs(self.inner, [])
        from .reference_impl import ReferenceGSS
        return ReferenceGSS(res).to_stacks()

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        """
        Efficiently push `value` onto all active stack heads by updating only the
        frontier where Interface nodes occur, reusing untouched substructure.
        Also handles the empty stack accumulator (if any) by producing [value].
        """
        # Transform inner by propagating push onto the frontier
        assert isinstance(self.inner.inner, UpperBranch)
        new_branch = _push_upper_branch(self.inner.inner, value)

        # Handle the empty stack case: [] -> [value]
        if self.empty is not None:
            new_branch = _ensure_add_interface_to_branch(new_branch, value, self.empty)
            new_empty: Optional[Acc] = None
        else:
            new_empty = None

        return LeveledGSS(Upper(new_branch), new_empty)

    def pop(self) -> LeveledGSS[T, Acc]:
        """
        Efficiently pop one element from all non-empty stacks by bubbling
        interface accumulators up one level and pruning empty subtrees.
        The result's `empty` is the merged accumulators of all stacks of length 1.
        """
        assert isinstance(self.inner.inner, UpperBranch)
        # Note: existing empty stacks are discarded by pop (as per ReferenceGSS).
        new_branch, up_acc = _pop_upper_branch(self.inner.inner)
        if new_branch is None:
            new_branch = UpperBranch({})
        return LeveledGSS(Upper(new_branch), up_acc)

    def is_empty(self) -> bool:
        # The GSS is empty if there's no accumulator for the empty stack, and
        # the inner trie has no children. from_stacks ensures inner is an UpperBranch.
        return self.empty is None and not self.inner.inner.children

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        if value is None:
            return LeveledGSS(Upper(UpperBranch({})), self.empty)

        def filter_node(u: Upper[T, Acc]) -> Optional[Upper[T, Acc]]:
            if isinstance(u.inner, Interface):
                return None

            # It's an UpperBranch
            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, children_map in u.inner.children.items():
                new_v_children: Dict[int, Upper[T, Acc]] = {}

                # If v is the value we're looking for, keep its interface children.
                if v == value:
                    for i, child in children_map.items():
                        if isinstance(child.inner, Interface):
                            new_v_children[i] = child

                # For all branch children, recurse.
                for i, child in children_map.items():
                    if isinstance(child.inner, UpperBranch):
                        filtered_child = filter_node(child)
                        if filtered_child:
                            new_v_children[i] = filtered_child

                if new_v_children:
                    new_children[v] = new_v_children

            if not new_children:
                return None
            return Upper(UpperBranch(new_children))

        new_inner = filter_node(self.inner)
        if not new_inner:
            new_inner = Upper(UpperBranch({}))

        return LeveledGSS(new_inner, None)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().apply(func).to_stacks())
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        return LeveledGSS.from_stacks(self.to_reference_impl().prune(predicate).to_stacks())
    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        """
        Merge two leveled structures structurally, combining interface accumulators
        for identical stacks and recursively merging branch subtrees. Reuses
        shared substructure aggressively.
        """
        assert isinstance(self.inner.inner, UpperBranch)
        assert isinstance(other.inner.inner, UpperBranch)
        merged_branch = _merge_upper_branch(self.inner.inner, other.inner.inner)
        merged_empty = _merge_opt_acc(self.empty, other.empty)
        return LeveledGSS(Upper(merged_branch), merged_empty)
    def peek(self) -> Set[T]:
        tops: Set[T] = set()

        def dfs(u: Upper[T, Acc]):
            if isinstance(u.inner, UpperBranch):
                for v, children_map in u.inner.children.items():
                    if any(isinstance(child.inner, Interface) for child in children_map.values()):
                        tops.add(v)

                    for child in children_map.values():
                        dfs(child)

        dfs(self.inner)
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        from functools import reduce
        accs: List[Acc] = []
        if self.empty is not None:
            accs.append(self.empty)

        def collect_accs(u: Upper[T, Acc]):
            if isinstance(u.inner, Interface):
                accs.append(u.inner.acc)
            elif isinstance(u.inner, UpperBranch):
                for children_map in u.inner.children.values():
                    for child in children_map.values():
                        collect_accs(child)

        collect_accs(self.inner)

        if not accs:
            return None

        return reduce(lambda a, b: a.merge(b), accs)


# ------------------------------
# Internal helpers for operations
# ------------------------------

def _merge_opt_acc(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
    if a is None:
        return b
    if b is None:
        return a
    return a.merge(b)


def _split_label_children(children_map: Dict[int, Upper[T, Acc]]) -> Tuple[Optional[Acc], Optional[UpperBranch[T, Acc]]]:
    """
    For a given per-label children map, return:
      - the merged interface accumulator (if any)
      - the merged/unique branch child (if any)
    Ensures at most a single interface acc and a single UpperBranch are returned.
    """
    iface_acc: Optional[Acc] = None
    branch: Optional[UpperBranch[T, Acc]] = None
    for child in children_map.values():
        inner = child.inner
        if isinstance(inner, Interface):
            iface_acc = _merge_opt_acc(iface_acc, inner.acc)
        else:
            # inner is UpperBranch
            if branch is None:
                branch = inner
            else:
                # Unexpected multiple branches for same label; merge them.
                branch = _merge_upper_branch(branch, inner)
    return iface_acc, branch


def _children_from_parts(iface_acc: Optional[Acc], branch: Optional[UpperBranch[T, Acc]]) -> Dict[int, Upper[T, Acc]]:
    """
    Build a per-label children map from an optional interface accumulator and optional branch child.
    At most two entries: interface first (index 0), then branch (index 1).
    """
    res: Dict[int, Upper[T, Acc]] = {}
    idx = 0
    if iface_acc is not None:
        res[idx] = Upper(Interface(_LOWER_LEAF, iface_acc))
        idx += 1
    if branch is not None:
        res[idx] = Upper(branch)
    return res


def _ensure_add_interface_to_branch(branch: UpperBranch[T, Acc], value: T, acc: Acc) -> UpperBranch[T, Acc]:
    """
    Ensure that within `branch`, under label `value`, there is an Interface with `acc`
    (merged with any existing one), and that a branch child exists (create empty if needed).
    Returns a (possibly new) UpperBranch; reuses existing objects when no change is needed.
    """
    children = branch.children
    existing = children.get(value)
    if existing is None:
        # Create fresh mapping: interface + empty branch
        new_children_map = _children_from_parts(acc, UpperBranch({}))
        new_children = dict(children)
        new_children[value] = new_children_map
        return UpperBranch(new_children)
    else:
        # Merge with existing mapping
        old_iface, old_branch = _split_label_children(existing)
        new_iface = _merge_opt_acc(old_iface, acc)
        new_branch = old_branch if old_branch is not None else UpperBranch({})
        # If nothing changed (old_iface merged with acc yielded same object by identity is unlikely),
        # still rebuild the per-label mapping to maintain invariant of single iface entry.
        new_children_map = _children_from_parts(new_iface, new_branch)
        if new_children_map is existing:
            return branch
        # Only replace the single label mapping; keep others identical for maximal sharing.
        new_children = dict(children)
        new_children[value] = new_children_map
        return UpperBranch(new_children)


def _push_upper_branch(branch: UpperBranch[T, Acc], value: T) -> UpperBranch[T, Acc]:
    """
    Recursively push `value` onto all stacks represented under `branch`.
    For each label 'v' at this level:
      - remove interface(s) at this level and insert them under the branch child
        at label 'value';
      - recursively push into the branch subtree.
    Reuses substructure where possible.
    """
    children_in = branch.children
    new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    changed = False

    for lbl, cmap in children_in.items():
        iface_acc, sub_branch = _split_label_children(cmap)

        # Recurse into the sub-branch if present
        new_sub_branch = _push_upper_branch(sub_branch, value) if sub_branch is not None else None

        # If there was an interface at this level, attach it under `value` in the sub-branch
        if iface_acc is not None:
            # Ensure sub-branch exists to host the new 'value' leaf
            host_branch = new_sub_branch if new_sub_branch is not None else UpperBranch({})
            updated_host = _ensure_add_interface_to_branch(host_branch, value, iface_acc)
            if updated_host is not host_branch:
                new_sub_branch = updated_host
            else:
                new_sub_branch = host_branch

        # Construct the new per-label children map:
        # After push, there are no interfaces at this level (they all moved deeper).
        if new_sub_branch is None:
            # If neither interface nor sub-branch, nothing remains (shouldn't happen unless cmap had only interfaces).
            # In that case, this label disappears entirely.
            # We simply skip adding this label to new_children.
            if iface_acc is not None:
                # We created at least one deeper node via _ensure_add_interface_to_branch; so not None here.
                # But if we end up here, ensure correctness by creating the deeper host now.
                host_branch = UpperBranch({})
                updated_host = _ensure_add_interface_to_branch(host_branch, value, iface_acc)
                new_children[lbl] = _children_from_parts(None, updated_host)
                changed = True
            else:
                # Nothing remains under this label.
                changed = True
                continue
        else:
            # Only the branch child remains at this level
            # Detect if unchanged for this label: no iface initially and sub-branch unchanged
            old_iface, old_sub = _split_label_children(cmap)
            if old_iface is None and old_sub is new_sub_branch:
                # Fully unchanged label mapping; reuse the existing cmap to maximize sharing.
                new_children[lbl] = cmap
            else:
                new_children[lbl] = _children_from_parts(None, new_sub_branch)
                changed = True

    if not changed:
        return branch
    return UpperBranch(new_children)


def _pop_upper_branch(branch: UpperBranch[T, Acc]) -> Tuple[Optional[UpperBranch[T, Acc]], Optional[Acc]]:
    """
    Pop one element from all stacks under `branch`.
    Returns:
      - A possibly new UpperBranch representing stacks with one fewer element.
      - An optional accumulator to bubble up to the parent (stacks that had length == 1 here).
    """
    children_in = branch.children
    if not children_in:
        # Nothing here; nothing to bubble up.
        return None, None

    new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    changed = False
    bubbled_up: Optional[Acc] = None

    for lbl, cmap in children_in.items():
        iface_acc, sub_branch = _split_label_children(cmap)

        # Recurse into the branch subtree to pop deeper stacks
        new_sub_branch, child_up = _pop_upper_branch(sub_branch) if sub_branch is not None else (None, None)

        # Stacks that ended exactly at this label (iface_acc) will bubble up to the parent.
        bubbled_up = _merge_opt_acc(bubbled_up, iface_acc)

        # Stacks that had one more element deeper (child_up) now end at this label -> new interface here
        new_iface_here = child_up

        # If nothing remains under this label after pop, omit it
        if new_iface_here is None and (new_sub_branch is None or not new_sub_branch.children):
            # Label disappears entirely
            if cmap in new_children.values():
                # Not expected; just mark change.
                pass
            changed = True
            continue

        # Build the new per-label children map
        old_iface, old_sub = _split_label_children(cmap)
        if new_iface_here == old_iface and (
            (old_sub is None and (new_sub_branch is None or not new_sub_branch.children)) or
            (old_sub is new_sub_branch)
        ):
            # Unchanged for this label: reuse old cmap
            new_children[lbl] = cmap
        else:
            # If new_sub_branch is empty (no children), treat it as None
            effective_branch = new_sub_branch if (new_sub_branch is not None and new_sub_branch.children) else None
            new_children[lbl] = _children_from_parts(new_iface_here, effective_branch)
            changed = True

    # If the entire node is now empty, return None (to prune)
    if not new_children:
        return None, bubbled_up

    if not changed:
        return branch, bubbled_up
    return UpperBranch(new_children), bubbled_up


def _merge_upper_branch(a: UpperBranch[T, Acc], b: UpperBranch[T, Acc]) -> UpperBranch[T, Acc]:
    """
    Merge two UpperBranch nodes:
      - For each label:
          * merge interface accumulators (if both present)
          * recursively merge branch subtrees (if both present)
      - Drop labels with neither interface nor branch (should not occur)
    Reuses substructure aggressively and early-outs on identity.
    """
    if a is b:
        return a

    children_a = a.children
    children_b = b.children

    # Fast-path: identical dictionaries object identity (rare but helps)
    if children_a is children_b:
        return a

    keys = set(children_a.keys()) | set(children_b.keys())
    new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    changed = False

    for lbl in keys:
        cmap_a = children_a.get(lbl)
        cmap_b = children_b.get(lbl)

        if cmap_a is cmap_b:
            if cmap_a is not None:
                new_children[lbl] = cmap_a
            continue

        iface_a, branch_a = (None, None)
        iface_b, branch_b = (None, None)
        if cmap_a is not None:
            iface_a, branch_a = _split_label_children(cmap_a)
        if cmap_b is not None:
            iface_b, branch_b = _split_label_children(cmap_b)

        merged_iface = _merge_opt_acc(iface_a, iface_b)
        if branch_a is None:
            merged_branch = branch_b
        elif branch_b is None:
            merged_branch = branch_a
        else:
            merged_branch = _merge_upper_branch(branch_a, branch_b)

        # If both None, label disappears
        if merged_iface is None and (merged_branch is None or not merged_branch.children):
            changed = True
            continue

        # Reuse existing cmap if possible
        if cmap_a is not None and iface_b is None and branch_b is None:
            # Only 'a' contributed; reuse original map
            new_children[lbl] = cmap_a
        elif cmap_b is not None and iface_a is None and branch_a is None:
            # Only 'b' contributed; reuse original map
            new_children[lbl] = cmap_b
        else:
            new_children[lbl] = _children_from_parts(merged_iface, merged_branch if (merged_branch is not None and merged_branch.children) else None)
            changed = True

    if not changed and len(new_children) == len(a.children):
        # Heuristic: if sizes match and nothing flagged changed, reuse 'a'
        # (It implies 'b' was a subset reusing objects from 'a')
        return a
    return UpperBranch(new_children)

def _get_upper_children(branch: UpperBranch[T, Acc]) -> List[Upper[T, Acc]]:
    """Helper to get all children from an UpperBranch."""
    return [child for children_by_val in branch.children.values() for child in children_by_val.values()]


def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Checks internal invariants of the LeveledGSS structure.
    Raises AssertionError if an invariant is violated.
    """
    def _validate_upper(node: Upper[T, Acc]):
        """Recursively validates invariants on Upper nodes."""
        if not isinstance(node.inner, UpperBranch):
            return  # Base case: node is an Interface.
        all_children = _get_upper_children(node.inner)
        # Invariant 1: If all children are interfaces, there must be more than one unique acc.
        if all_children and all(isinstance(child.inner, Interface) for child in all_children):
            if len({child.inner.acc for child in all_children}) > 1:
                raise AssertionError("Invariant violated: UpperBranch has children that are all Interfaces with duplicate accs.")
        # Recurse into children
        for child in all_children:
            _validate_upper(child)
    _validate_upper(gss.inner)
    # Invariant 2: If inner is an interface and empty exists, their accs must differ.
    if isinstance(gss.inner, Interface) and gss.empty is not None and gss.inner.acc == gss.empty:
        raise AssertionError("Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator.")
