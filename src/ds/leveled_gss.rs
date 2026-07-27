//! Compatibility adapter from GLRMask's historical GSS API to `weighted-gss`.
//!
//! This module intentionally contains no graph representation. It is a proving
//! adapter: GLRMask must be expressible using the standalone crate's public API.

use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use weighted_gss as wg;

/// Historical GLRMask name for the path-weight join operation.
pub trait Merge: Clone {
    fn merge(&self, other: &Self) -> Self;

    fn subsumes(&self, _other: &Self) -> bool {
        false
    }
}

impl Merge for () {
    fn merge(&self, _other: &Self) -> Self {}

    fn subsumes(&self, _other: &Self) -> bool {
        true
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct CompatWeight<A>(A);

impl<A: Merge + PartialEq> wg::Weight for CompatWeight<A> {
    #[inline]
    fn join(&self, other: &Self) -> Self {
        Self(self.0.merge(&other.0))
    }

}

/// Structural statistics retained for existing GLRMask profiling code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeveledGSSSummary {
    pub top_values_count: usize,
    pub upperbranch_nodes: usize,
    pub interface_nodes: usize,
    pub lower_nodes: usize,
    pub lower_general_nodes: usize,
    pub lower_segment_nodes: usize,
    pub total_unique_nodes: usize,
    pub total_edges: usize,
    pub accumulator_instances: usize,
    pub max_depth: u32,
}

/// Exact canonical key for the unweighted concrete stack language.
pub(crate) type GssSemanticKey = wg::engine::StackLanguageId;

/// Exact canonical interner for the unweighted concrete stack language.
pub(crate) struct GssSemanticKeyInterner<
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash,
> {
    inner: wg::engine::StackLanguageInterner<T>,
    marker: std::marker::PhantomData<fn() -> A>,
}

impl<T, A> GssSemanticKeyInterner<T, A>
where
    T: Clone + Eq + Hash + Ord,
    A: Merge + Clone + Eq + Hash,
{
    pub(crate) fn new() -> Self {
        Self {
            inner: wg::engine::StackLanguageInterner::new(),
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn key(&mut self, gss: &LeveledGSS<T, A>) -> GssSemanticKey {
        self.inner.key(&gss.inner)
    }

}

static NEXT_COMPAT_GSS_ID: AtomicUsize = AtomicUsize::new(1);

fn next_compat_gss_id() -> usize {
    let id = NEXT_COMPAT_GSS_ID.fetch_add(1, Ordering::Relaxed);
    assert_ne!(id, 0, "GLRMask compatibility GSS ID space exhausted");
    id
}

/// GLRMask compatibility wrapper over the standalone weighted GSS.
pub struct LeveledGSS<
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
> {
    inner: wg::WeightedGss<T, CompatWeight<A>>,
    identity: Arc<OnceLock<usize>>,
}

impl<T, A> Clone for LeveledGSS<T, A>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            identity: self.identity.clone(),
        }
    }
}

impl<T, A> PartialEq for LeveledGSS<T, A>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        if self.ptr_eq(other) {
            return true;
        }
        let Some(left) = self.to_stacks(4_096) else {
            return false;
        };
        let Some(right) = other.to_stacks(4_096) else {
            return false;
        };
        left.len() == right.len()
            && left.iter().all(|entry| right.contains(entry))
            && right.iter().all(|entry| left.contains(entry))
    }
}

impl<T, A> Eq for LeveledGSS<T, A>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
}

impl<T, A> fmt::Debug for LeveledGSS<T, A>
where
    T: Clone + Eq + Hash + fmt::Debug,
    A: Merge + Clone + Eq + Hash + fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(formatter)
    }
}

fn merge_weighted<T, A>(
    values: impl IntoIterator<Item = wg::WeightedGss<T, CompatWeight<A>>>,
) -> wg::WeightedGss<T, CompatWeight<A>>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let mut values: Vec<_> = values.into_iter().collect();
    if values.is_empty() {
        return wg::WeightedGss::new();
    }
    while values.len() > 1 {
        let mut merged = Vec::with_capacity(values.len().div_ceil(2));
        let mut iter = values.into_iter();
        while let Some(left) = iter.next() {
            merged.push(match iter.next() {
                Some(right) => left.merge(&right),
                None => left,
            });
        }
        values = merged;
    }
    values.pop().expect("non-empty merge reduction")
}

impl<T, A> LeveledGSS<T, A>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    fn from_inner(inner: wg::WeightedGss<T, CompatWeight<A>>) -> Self {
        Self {
            inner,
            identity: Arc::new(OnceLock::new()),
        }
    }

    pub fn empty() -> Self {
        Self::from_inner(wg::WeightedGss::new())
    }

    pub fn from_stacks(stacks: &[(Vec<T>, A)]) -> Self {
        let mut canonical = HashMap::<Vec<T>, A>::new();
        for (stack, weight) in stacks {
            canonical
                .entry(stack.clone())
                .and_modify(|current| *current = current.merge(weight))
                .or_insert_with(|| weight.clone());
        }

        let mut by_weight = HashMap::<A, Vec<Vec<T>>>::new();
        for (stack, weight) in canonical {
            by_weight.entry(weight).or_default().push(stack);
        }

        Self::from_inner(merge_weighted(by_weight.into_iter().map(|(weight, stacks)| {
            wg::WeightedGss::from_stacks_with_weight(stacks, CompatWeight(weight))
        })))
    }

    pub fn from_single_stack(values: Vec<T>, weight: A) -> Self {
        Self::from_inner(wg::WeightedGss::from_stack(values, CompatWeight(weight)))
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn max_depth(&self) -> u32 {
        self.inner.max_depth().min(u32::MAX as usize) as u32
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr_key() == other.ptr_key()
    }

    pub fn ptr_key(&self) -> usize {
        *self.identity.get_or_init(next_compat_gss_id)
    }

    pub(crate) fn single_interface_lower_id(&self) -> Option<usize> {
        (!self.is_empty()).then(|| self.ptr_key())
    }

    pub fn push(&self, value: T) -> Self {
        Self::from_inner(self.inner.push(value))
    }

    pub fn pop(&self) -> Self {
        Self::from_inner(self.inner.pop())
    }

    pub fn popn(&self, count: isize) -> Self {
        if count <= 0 {
            return self.clone();
        }
        Self::from_inner(self.inner.popn(count as usize))
    }

    pub fn isolate(&self, value: Option<T>) -> Self {
        match value {
            Some(value) => Self::from_inner(self.inner.retain_top(&value)),
            None => Self::from_inner(self.inner.retain_empty()),
        }
    }

    pub fn pop_top_value(&self, value: &T) -> Self {
        Self::from_inner(self.inner.pop_top(value))
    }

    pub fn merge(&self, other: &Self) -> Self {
        Self::from_inner(self.inner.merge(&other.inner))
    }

    pub fn merge_many(values: impl IntoIterator<Item = Self>) -> Self {
        Self::from_inner(merge_weighted(values.into_iter().map(|value| value.inner)))
    }

    pub fn fuse(&self, _levels: Option<isize>) -> Self {
        self.clone()
    }

    pub fn peek(&self) -> HashSet<T> {
        self.inner.tops().collect()
    }

    pub fn peek_values(&self) -> SmallVec<[T; 8]> {
        self.inner.tops().collect()
    }

    pub fn for_each_top_value(&self, mut visit: impl FnMut(T)) {
        for top in self.inner.tops() {
            visit(top);
        }
    }

    pub fn single_top_value(&self) -> Option<T> {
        let mut tops = self.inner.tops();
        let top = tops.next()?;
        tops.next().is_none().then_some(top)
    }

    pub fn single_exclusive_top_value(&self) -> Option<T> {
        self.inner.top()
    }

    pub fn path_count_at_most(&self, limit: usize) -> usize {
        let mut count = 0;
        if wg::engine::for_each_stack_top_first(&self.inner, limit, |_, _| count += 1).is_ok() {
            count
        } else {
            limit
        }
    }

    pub fn to_stacks(&self, max_stacks: usize) -> Option<Vec<(Vec<T>, A)>> {
        self.inner.to_stacks(max_stacks).ok().map(|stacks| {
            stacks
                .into_iter()
                .map(|(stack, weight)| (stack, weight.0))
                .collect()
        })
    }

    pub(crate) fn semantically_eq(&self, other: &Self, max_stacks: usize) -> Option<bool> {
        if self.ptr_eq(other) {
            return Some(true);
        }
        let left = self.to_stacks(max_stacks)?;
        let right = other.to_stacks(max_stacks)?;
        Some(
            left.len() == right.len()
                && left.iter().all(|entry| right.contains(entry))
                && right.iter().all(|entry| left.contains(entry)),
        )
    }

    pub(crate) fn for_each_stack_top_first_bounded(
        &self,
        limit: usize,
        mut visit: impl FnMut(&[T], &A),
    ) -> bool {
        wg::engine::for_each_stack_top_first(&self.inner, limit, |stack, weight| {
            visit(stack, &weight.0)
        })
        .is_ok()
    }

    pub(crate) fn for_each_stack_len_bounded(
        &self,
        limit: usize,
        mut visit: impl FnMut(usize, &A),
    ) -> bool {
        self.for_each_stack_top_first_bounded(limit, |stack, weight| {
            visit(stack.len(), weight)
        })
    }

    pub fn try_single_stack_bounded(&self, max_depth: usize) -> Option<(Vec<T>, A)> {
        if self.max_depth() as usize > max_depth {
            return None;
        }
        let mut top_first = SmallVec::<[T; 16]>::new();
        let weight = self.single_path_top_first_and_acc(&mut top_first)?;
        let mut stack = top_first.into_vec();
        stack.reverse();
        Some((stack, weight))
    }

    pub fn single_path_top_first_and_acc(
        &self,
        output: &mut SmallVec<[T; 16]>,
    ) -> Option<A> {
        let mut result = None;
        wg::engine::for_each_stack_top_first(&self.inner, 1, |stack, weight| {
            output.clear();
            output.extend(stack.iter().cloned());
            result = Some(weight.0.clone());
        })
        .ok()?;
        result
    }

    pub fn apply<B, F>(&self, mut map: F) -> LeveledGSS<T, B>
    where
        B: Merge + Clone + Eq + Hash,
        F: FnMut(&A) -> B,
    {
        LeveledGSS::from_inner(
            self.inner
                .map_weights(|weight| CompatWeight(map(&weight.0))),
        )
    }

    pub fn apply_and_prune<B, F>(&self, mut map: F) -> LeveledGSS<T, B>
    where
        B: Merge + Clone + Eq + Hash,
        F: FnMut(&A) -> Option<B>,
    {
        LeveledGSS::from_inner(
            self.inner
                .filter_map_weights(|weight| map(&weight.0).map(CompatWeight)),
        )
    }

    pub fn apply_and_prune_no_promote<B, F>(&self, map: F) -> LeveledGSS<T, B>
    where
        B: Merge + Clone + Eq + Hash,
        F: FnMut(&A) -> Option<B>,
    {
        self.apply_and_prune(map)
    }

    pub fn partition_by_accumulator(&self) -> Vec<(LeveledGSS<T, ()>, A)> {
        let mut weights = Vec::<A>::new();
        for weight in self.inner.weights() {
            if !weights.contains(&weight.0) {
                weights.push(weight.0.clone());
            }
        }
        weights
            .into_iter()
            .map(|weight| {
                let inner = self.inner.filter_map_weights(|candidate| {
                    (candidate.0 == weight).then_some(CompatWeight(()))
                });
                (LeveledGSS::from_inner(inner), weight)
            })
            .collect()
    }

    pub fn reduce_acc(&self) -> Option<A> {
        self.inner.joined_weight().map(|weight| weight.0)
    }

    pub fn join_weights(&self) -> Option<A> {
        self.reduce_acc()
    }

    pub fn for_each_acc(&self, mut visit: impl FnMut(&A)) {
        self.inner.weights().for_each(|weight| visit(&weight.0));
    }

    pub fn all_accs_satisfy(&self, predicate: impl Fn(&A) -> bool) -> bool {
        self.inner.weights().all(|weight| predicate(&weight.0))
    }

    pub fn for_each_decomposed(&self, mut visit: impl FnMut(T, Self)) {
        for top in self.inner.tops() {
            visit(top.clone(), Self::from_inner(self.inner.pop_top(&top)));
        }
    }

    pub fn remap_top_values<I>(&self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T)>,
    {
        Self::merge_many(
            shifts
                .into_iter()
                .map(|(from, to)| self.isolate(Some(from)).push(to)),
        )
    }

    pub fn remap_top_values_owned<I>(self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T)>,
    {
        self.remap_top_values(shifts)
    }

    pub fn apply_top_pure_shifts<I>(&self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T, bool)>,
    {
        Self::merge_many(shifts.into_iter().map(|(from, to, replace_top)| {
            let base = self.isolate(Some(from));
            if replace_top {
                base.pop().push(to)
            } else {
                base.push(to)
            }
        }))
    }

    pub fn try_apply_selective_top_pure_shifts<I>(&self, shifts: I) -> Option<Self>
    where
        I: IntoIterator<Item = (T, T, bool)>,
    {
        Some(self.apply_top_pure_shifts(shifts))
    }

    pub fn apply_stack_effects_to_single_concrete_path<'a, I>(
        &self,
        effects: I,
        _max_materialized_depth: usize,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = (usize, &'a [T])>,
        T: 'a,
    {
        Some(Self::merge_many(effects.into_iter().map(|(pop, push)| {
            let mut branch = self.popn(pop as isize);
            for value in push {
                branch = branch.push(value.clone());
            }
            branch
        })))
    }

    pub fn apply_guarded_stack_effects_to_single_concrete_path<'a, I, G>(
        &self,
        effects: I,
        max_materialized_depth: usize,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = (G, usize, &'a [T])>,
        G: IntoIterator<Item = (usize, &'a [T])>,
        T: 'a,
    {
        let (stack, weight) = self.try_single_stack_bounded(max_materialized_depth)?;
        let branches = effects.into_iter().filter_map(|(guards, pop, push)| {
            let allowed = guards.into_iter().all(|(depth, states)| {
                stack
                    .len()
                    .checked_sub(depth + 1)
                    .and_then(|index| stack.get(index))
                    .is_some_and(|state| states.contains(state))
            });
            if !allowed {
                return None;
            }
            let keep = stack.len().checked_sub(pop)?;
            let mut next = stack[..keep].to_vec();
            next.extend_from_slice(push);
            Some((next, weight.clone()))
        });
        Some(Self::from_stacks(&branches.collect::<Vec<_>>()))
    }

    pub fn apply_shared_pop_push_branches<'a, I>(
        &self,
        pop: usize,
        pushes: I,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = &'a [T]>,
        T: 'a,
    {
        self.try_virtual_stack()?
            .into_gss_after_popping_and_pushing_branches(pop, pushes)
    }

    pub fn apply_shared_pop_push_single_branches<'a, I>(
        &self,
        pop: usize,
        targets: I,
    ) -> Option<Self>
    where
        I: IntoIterator<Item = &'a T>,
        T: 'a,
    {
        self.try_virtual_stack()?
            .into_gss_after_popping_and_pushing_single_branches(pop, targets)
    }

    pub fn try_virtual_stack(&self) -> Option<VirtualStack<T, A>> {
        wg::engine::linear_prefix(&self.inner).map(|inner| VirtualStack { inner })
    }

    pub fn pop1_common_interface_base(&self) -> Option<Self> {
        if self.inner.has_empty_stack() {
            return None;
        }
        let tops: SmallVec<[T; 8]> = self.inner.tops().collect();
        if tops.len() < 2 {
            return None;
        }
        let first = self.inner.pop_top(&tops[0]);
        for top in &tops[1..] {
            let candidate = Self::from_inner(self.inner.pop_top(top));
            let expected = Self::from_inner(first.clone());
            if !candidate.semantically_eq(&expected, 4_096)? {
                return None;
            }
        }
        Some(Self::from_inner(first))
    }

    pub fn absorb_push_same_acc(self, value: T, base: &Self) -> Self {
        self.merge(&base.push(value))
    }

    pub fn absorb_vstack_same_acc_owned(self, stack: VirtualStack<T, A>) -> Self {
        self.merge(&stack.into_gss())
    }

    pub fn truncate(&self, max_len: isize) -> Self {
        if max_len < 0 {
            return Self::empty();
        }
        let max_len = max_len as usize;
        let Some(stacks) = self.to_stacks(1_000_000) else {
            return self.clone();
        };
        Self::from_stacks(
            &stacks
                .into_iter()
                .map(|(stack, weight)| {
                    let start = stack.len().saturating_sub(max_len);
                    (stack[start..].to_vec(), weight)
                })
                .collect::<Vec<_>>(),
        )
    }

    pub fn summary(&self) -> LeveledGSSSummary {
        LeveledGSSSummary {
            top_values_count: self.peek_values().len(),
            accumulator_instances: self.inner.weights().count(),
            max_depth: self.max_depth(),
            ..LeveledGSSSummary::default()
        }
    }
}

/// Compatibility wrapper for the standalone crate's linear-prefix fast path.
pub struct VirtualStack<
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
> {
    inner: wg::engine::LinearPrefix<T, CompatWeight<A>>,
}

impl<T, A> Clone for VirtualStack<T, A>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T, A> VirtualStack<T, A>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    pub fn top(&self) -> Option<&T> {
        self.inner.get(0)
    }

    pub fn top_after_popping(&self, count: usize) -> Option<&T> {
        self.inner.get(count)
    }

    pub fn parent_of_top(&self) -> Option<T> {
        self.inner.get(1).cloned()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }

    pub fn pop(&mut self, count: usize) -> usize {
        self.inner.popn(count)
    }

    pub fn replace_top(&mut self, value: T) -> bool {
        if self.inner.get(0).is_none() {
            return false;
        }
        let remaining = self.inner.popn(1);
        debug_assert_eq!(remaining, 0);
        self.inner.push(value);
        true
    }

    pub fn single_top_extension_of(&self, base: &Self) -> Option<T> {
        if self.len() != base.len() + 1 {
            return None;
        }
        let top = self.top()?.clone();
        let mut popped = self.clone();
        if popped.pop(1) != 0 {
            return None;
        }
        (popped.into_gss().ptr_eq(&base.clone().into_gss())).then_some(top)
    }

    pub(crate) fn has_hidden_floor_values(&self) -> bool {
        !self.inner.floor_is_empty()
    }

    pub fn into_gss(self) -> LeveledGSS<T, A> {
        LeveledGSS::from_inner(self.inner.into_gss())
    }

    pub fn into_gss_after_popping(mut self, count: usize) -> LeveledGSS<T, A> {
        let remaining = self.pop(count);
        self.into_gss().popn(remaining as isize)
    }

    pub fn into_gss_after_popping_and_pushing_branches<'a, I>(
        self,
        pop: usize,
        pushes: I,
    ) -> Option<LeveledGSS<T, A>>
    where
        I: IntoIterator<Item = &'a [T]>,
        T: 'a,
    {
        if pop > self.len() {
            let base = self.into_gss().popn(pop as isize);
            return Some(LeveledGSS::merge_many(pushes.into_iter().map(|push| {
                let mut branch = base.clone();
                for value in push {
                    branch = branch.push(value.clone());
                }
                branch
            })));
        }
        Some(LeveledGSS::merge_many(pushes.into_iter().map(|push| {
            let mut branch = self.clone();
            let remaining = branch.pop(pop);
            debug_assert_eq!(remaining, 0);
            for value in push {
                branch.push(value.clone());
            }
            branch.into_gss()
        })))
    }

    pub fn into_gss_after_popping_and_pushing_single_branches<'a, I>(
        self,
        pop: usize,
        targets: I,
    ) -> Option<LeveledGSS<T, A>>
    where
        I: IntoIterator<Item = &'a T>,
        T: 'a,
    {
        if pop > self.len() {
            let base = self.into_gss().popn(pop as isize);
            return Some(LeveledGSS::merge_many(targets.into_iter().map(|target| {
                base.push(target.clone())
            })));
        }
        Some(LeveledGSS::merge_many(targets.into_iter().map(|target| {
            let mut branch = self.clone();
            let remaining = branch.pop(pop);
            debug_assert_eq!(remaining, 0);
            branch.push(target.clone());
            branch.into_gss()
        })))
    }
}

#[cfg(test)]
mod engine_adapter_regression_tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct Bits(u8);

    impl Merge for Bits {
        fn merge(&self, other: &Self) -> Self {
            Self(self.0 | other.0)
        }
    }

    #[test]
    fn branch_root_single_language_pop_one_removes_old_top() {
        let merged = LeveledGSS::from_stacks(&[
            (vec![0_u32, 1, 6], Bits(1)),
            (vec![0_u32, 2, 7], Bits(2)),
        ]);
        let branch = merged.isolate(Some(6));
        assert_eq!(branch.to_stacks(8), Some(vec![(vec![0, 1, 6], Bits(1))]));
        assert!(branch.try_virtual_stack().is_none(), "branch-root probe should exercise fallback");
        let shifted = branch
            .apply_stack_effects_to_single_concrete_path(
                [(1_usize, [12_u32, 27].as_slice())],
                64,
            )
            .expect("effect should apply");
        assert_eq!(
            shifted.to_stacks(8),
            Some(vec![(vec![0, 1, 12, 27], Bits(1))])
        );
    }

    #[test]
    fn virtual_stack_replace_top_removes_old_top() {
        let gss = LeveledGSS::from_single_stack(vec![0_u32, 1, 6], ());
        let mut stack = gss.try_virtual_stack().expect("single stack should expose linear prefix");
        assert!(stack.replace_top(12));
        assert_eq!(stack.into_gss().to_stacks(8), Some(vec![(vec![0, 1, 12], ())]));
    }

    #[test]
    fn virtual_stack_pop_one_then_push_one_removes_old_top() {
        let gss = LeveledGSS::from_single_stack(vec![0_u32, 1, 6], ());
        let stack = gss.try_virtual_stack().expect("single stack should expose linear prefix");
        let targets = [12_u32];
        let shifted = stack
            .into_gss_after_popping_and_pushing_single_branches(1, targets.iter())
            .expect("effect should apply");
        assert_eq!(shifted.to_stacks(8), Some(vec![(vec![0, 1, 12], ())]));
    }

    #[test]
    fn virtual_stack_pop_one_then_push_two_removes_old_top() {
        let gss = LeveledGSS::from_single_stack(vec![0_u32, 1, 6], ());
        let stack = gss.try_virtual_stack().expect("single stack should expose linear prefix");
        let pushes = [vec![12_u32, 27]];
        let shifted = stack
            .into_gss_after_popping_and_pushing_branches(
                1,
                pushes.iter().map(Vec::as_slice),
            )
            .expect("effect should apply");
        assert_eq!(shifted.to_stacks(8), Some(vec![(vec![0, 1, 12, 27], ())]));
    }
}
