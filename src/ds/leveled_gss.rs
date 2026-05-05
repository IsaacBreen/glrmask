use smallvec::SmallVec;
use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Arc;

pub trait Merge: Clone {
    fn merge(&self, other: &Self) -> Self;
}

fn canonicalize_stacks<T, A>(stacks: impl IntoIterator<Item = (Vec<T>, A)>) -> Vec<(Vec<T>, A)>
where
    T: Clone + Eq + Hash,
    A: Merge + Clone + Eq + Hash,
{
    let mut merged: Vec<(Vec<T>, A)> = Vec::new();
    for (stack, acc) in stacks {
        if let Some((_, existing_acc)) = merged.iter_mut().find(|(existing_stack, _)| *existing_stack == stack) {
            *existing_acc = existing_acc.merge(&acc);
        } else {
            merged.push((stack, acc));
        }
    }
    merged
}

#[derive(Clone)]
pub struct LeveledGSS<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    stacks: Arc<Vec<(Vec<T>, A)>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainTail<T: Clone + Eq + Hash> {
    values: Arc<Vec<T>>,
}

#[derive(Clone)]
pub struct VirtualStack<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {
    values: Vec<T>,
    acc: A,
}

#[derive(Clone, Debug, Default)]
pub struct LeveledGssSummary {
    pub path_count: usize,
    pub total_edges: usize,
    pub max_depth: u32,
    pub segment_count: Option<usize>,
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> VirtualStack<T, A> {
    pub fn top(&self) -> Option<&T> {
        self.values.last()
    }

    pub fn pop(&mut self, count: usize) -> usize {
        let actual = count.min(self.values.len());
        let keep = self.values.len() - actual;
        self.values.truncate(keep);
        count - actual
    }

    pub fn push(&mut self, value: T) {
        self.values.push(value);
    }

    pub fn parent_of_top(&self) -> Option<T> {
        self.values
            .len()
            .checked_sub(2)
            .and_then(|index| self.values.get(index).cloned())
    }

    pub fn replace_top(&mut self, value: T) -> bool {
        let Some(top) = self.values.last_mut() else {
            return false;
        };
        *top = value;
        true
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn into_gss(self) -> LeveledGSS<T, A> {
        LeveledGSS::from_stacks(&[(self.values, self.acc)])
    }
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> PartialEq for LeveledGSS<T, A> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.stacks, &other.stacks) || *self.stacks == *other.stacks
    }
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> Eq for LeveledGSS<T, A> {}

impl<T: Clone + Eq + Hash + std::fmt::Debug, A: Merge + Clone + Eq + Hash + std::fmt::Debug> std::fmt::Debug
    for LeveledGSS<T, A>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeveledGSS")
            .field("num_stacks", &self.stacks.len())
            .field("max_depth", &self.max_depth())
            .finish()
    }
}

impl<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> LeveledGSS<T, A> {
    pub fn ptr_key(&self) -> usize {
        Arc::as_ptr(&self.stacks) as usize
    }

    pub fn empty() -> Self {
        Self {
            stacks: Arc::new(Vec::new()),
        }
    }

    pub fn from_stacks(stacks: &[(Vec<T>, A)]) -> Self {
        Self {
            stacks: Arc::new(canonicalize_stacks(stacks.iter().cloned())),
        }
    }

    pub fn to_stacks(&self) -> Vec<(Vec<T>, A)> {
        self.stacks.as_ref().clone()
    }

    pub fn merge(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        let combined = self
            .stacks
            .iter()
            .cloned()
            .chain(other.stacks.iter().cloned())
            .collect::<Vec<_>>();
        Self {
            stacks: Arc::new(canonicalize_stacks(combined)),
        }
    }

    pub fn merge_many(gsses: impl IntoIterator<Item = Self>) -> Self {
        let mut result = Self::empty();
        for gss in gsses {
            result = result.merge(&gss);
        }
        result
    }

    pub fn push(&self, value: T) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        Self {
            stacks: Arc::new(canonicalize_stacks(self.stacks.iter().map(|(stack, acc)| {
                let mut next = stack.clone();
                next.push(value.clone());
                (next, acc.clone())
            }))),
        }
    }

    pub fn pop(&self) -> Self {
        self.popn(1)
    }

    pub fn popn(&self, count: isize) -> Self {
        if count <= 0 || self.is_empty() {
            return self.clone();
        }
        let count = count as usize;
        Self {
            stacks: Arc::new(canonicalize_stacks(self.stacks.iter().filter_map(|(stack, acc)| {
                if stack.len() < count {
                    None
                } else {
                    let mut next = stack.clone();
                    next.truncate(next.len() - count);
                    Some((next, acc.clone()))
                }
            }))),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.stacks.is_empty()
    }

    pub fn max_depth(&self) -> u32 {
        self.stacks
            .iter()
            .map(|(stack, _)| stack.len() as u32)
            .max()
            .unwrap_or(0)
    }

    pub fn flattened_summary(&self) -> LeveledGssSummary {
        LeveledGssSummary {
            path_count: self.stacks.len(),
            total_edges: self
                .stacks
                .iter()
                .map(|(stack, _)| stack.len().saturating_sub(1))
                .sum(),
            max_depth: self.max_depth(),
            // The current LeveledGSS representation stores flat stacks of `T`
            // values and does not retain segment-node structure.
            segment_count: None,
        }
    }

    pub fn isolate(&self, value: Option<T>) -> Self {
        let filtered = self.stacks.iter().filter_map(|(stack, acc)| match (value.as_ref(), stack.last()) {
            (Some(expected), Some(actual)) if actual == expected => Some((stack.clone(), acc.clone())),
            (None, None) => Some((stack.clone(), acc.clone())),
            _ => None,
        });
        Self {
            stacks: Arc::new(canonicalize_stacks(filtered)),
        }
    }

    pub fn isolate_popn(&self, value: T, count: isize) -> Self {
        if count <= 0 {
            return self.isolate(Some(value));
        }
        self.isolate(Some(value)).popn(count)
    }

    pub fn isolate_pop_bases(&self, value: T, count: isize) -> SmallVec<[(T, Self); 4]> {
        let popped = self.isolate_popn(value, count);
        if popped.is_empty() {
            return SmallVec::new();
        }
        if let Some(single) = popped.single_top_value() {
            let mut result = SmallVec::new();
            result.push((single, popped));
            return result;
        }

        let mut groups: SmallVec<[(T, Vec<(Vec<T>, A)>); 4]> = SmallVec::new();
        for (stack, acc) in popped.stacks.iter() {
            let Some(top) = stack.last().cloned() else {
                continue;
            };
            if let Some((_, grouped)) = groups.iter_mut().find(|(existing, _)| *existing == top) {
                grouped.push((stack.clone(), acc.clone()));
            } else {
                groups.push((top, vec![(stack.clone(), acc.clone())]));
            }
        }

        groups
            .into_iter()
            .map(|(top, stacks)| (top, Self::from_stacks(&stacks)))
            .collect()
    }

    pub fn grouped_by_top(&self) -> SmallVec<[(T, Self); 4]> {
        if self.is_empty() {
            return SmallVec::new();
        }
        if let Some(single) = self.single_top_value() {
            let mut result = SmallVec::new();
            result.push((single, self.clone()));
            return result;
        }

        let mut groups: SmallVec<[(T, Vec<(Vec<T>, A)>); 4]> = SmallVec::new();
        for (stack, acc) in self.stacks.iter() {
            let Some(top) = stack.last().cloned() else {
                continue;
            };
            if let Some((_, grouped)) = groups.iter_mut().find(|(existing, _)| *existing == top) {
                grouped.push((stack.clone(), acc.clone()));
            } else {
                groups.push((top, vec![(stack.clone(), acc.clone())]));
            }
        }

        groups
            .into_iter()
            .map(|(top, stacks)| (top, Self::from_stacks(&stacks)))
            .collect()
    }

    pub fn remap_top_values<I>(&self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T)>,
    {
        let shifts: Vec<(T, T)> = shifts.into_iter().collect();
        if self.is_empty() || shifts.is_empty() {
            return Self::empty();
        }

        let remapped = self.stacks.iter().flat_map(|(stack, acc)| {
            let Some(top) = stack.last() else {
                return Vec::new().into_iter();
            };
            shifts
                .iter()
                .filter(move |(from, _)| from == top)
                .map(|(_, to)| {
                    let mut next = stack.clone();
                    *next.last_mut().unwrap() = to.clone();
                    (next, acc.clone())
                })
                .collect::<Vec<_>>()
                .into_iter()
        });

        Self {
            stacks: Arc::new(canonicalize_stacks(remapped)),
        }
    }

    pub fn remap_top_values_owned<I>(self, shifts: I) -> Self
    where
        I: IntoIterator<Item = (T, T)>,
    {
        self.remap_top_values(shifts)
    }

    pub fn absorb_push_same_acc(self, value: T, base: &Self) -> Self {
        self.merge(&base.push(value))
    }

    pub fn absorb_vstack_same_acc(self, stack: &VirtualStack<T, A>) -> Self {
        self.merge(&stack.clone().into_gss())
    }

    pub fn absorb_vstack_same_acc_owned(self, stack: VirtualStack<T, A>) -> Self {
        self.merge(&stack.into_gss())
    }

    pub fn for_each_decomposed(&self, mut f: impl FnMut(T, Self)) {
        let mut groups: SmallVec<[(T, Vec<(Vec<T>, A)>); 4]> = SmallVec::new();
        for (stack, acc) in self.stacks.iter() {
            let Some(top) = stack.last().cloned() else {
                continue;
            };
            let mut popped = stack.clone();
            popped.pop();
            if let Some((_, grouped)) = groups.iter_mut().find(|(existing, _)| *existing == top) {
                grouped.push((popped, acc.clone()));
            } else {
                groups.push((top, vec![(popped, acc.clone())]));
            }
        }

        for (top, grouped) in groups {
            f(top, Self::from_stacks(&grouped));
        }
    }

    pub fn extract_chain_and_tail(&self) -> Option<(SmallVec<[T; 16]>, &A, ChainTail<T>)> {
        let [(stack, acc)] = self.stacks.as_slice() else {
            return None;
        };
        if stack.len() < 2 {
            return None;
        }

        let mut chain = SmallVec::<[T; 16]>::new();
        for value in stack.iter().rev() {
            chain.push(value.clone());
        }

        Some((
            chain,
            acc,
            ChainTail {
                values: Arc::new(Vec::new()),
            },
        ))
    }

    pub fn try_virtual_stack(&self) -> Option<VirtualStack<T, A>> {
        let [(stack, acc)] = self.stacks.as_slice() else {
            return None;
        };
        if stack.is_empty() {
            return None;
        }
        Some(VirtualStack {
            values: stack.clone(),
            acc: acc.clone(),
        })
    }

    pub fn into_virtual_stack(self) -> Result<VirtualStack<T, A>, Self> {
        let [(stack, acc)] = self.stacks.as_slice() else {
            return Err(self);
        };
        if stack.is_empty() {
            return Err(self);
        }
        Ok(VirtualStack {
            values: stack.clone(),
            acc: acc.clone(),
        })
    }

    pub fn apply<B, F>(&self, mut func: F) -> LeveledGSS<T, B>
    where
        B: Merge + Clone + Eq + Hash,
        F: FnMut(&A) -> B,
    {
        LeveledGSS {
            stacks: Arc::new(canonicalize_stacks(self.stacks.iter().map(|(stack, acc)| {
                (stack.clone(), func(acc))
            }))),
        }
    }

    pub fn apply_transform_and_decompose<B, M>(&self, mut mutator: M) -> (Vec<(T, LeveledGSS<T, B>)>, Vec<B>)
    where
        B: Merge + Clone + Eq + Hash,
        M: FnMut(&A) -> Option<B>,
    {
        let mut root_accs = Vec::new();
        let mut groups: SmallVec<[(T, Vec<(Vec<T>, B)>); 4]> = SmallVec::new();

        for (stack, acc) in self.stacks.iter() {
            let Some(new_acc) = mutator(acc) else {
                continue;
            };

            if let Some(top) = stack.last().cloned() {
                let mut popped = stack.clone();
                popped.pop();
                if let Some((_, grouped)) = groups.iter_mut().find(|(existing, _)| *existing == top) {
                    grouped.push((popped, new_acc));
                } else {
                    groups.push((top, vec![(popped, new_acc)]));
                }
            } else {
                root_accs.push(new_acc);
            }
        }

        let decomposed = groups
            .into_iter()
            .map(|(top, grouped)| (top, LeveledGSS::from_stacks(&grouped)))
            .collect();

        (decomposed, root_accs)
    }

    pub fn apply_and_prune_no_promote(&self, mut mutator: impl FnMut(&A) -> Option<A>) -> Self {
        Self {
            stacks: Arc::new(canonicalize_stacks(self.stacks.iter().filter_map(|(stack, acc)| {
                mutator(acc).map(|new_acc| (stack.clone(), new_acc))
            }))),
        }
    }

    pub fn fuse(&self, _levels: Option<isize>) -> Self {
        self.clone()
    }

    pub fn peek(&self) -> HashSet<T> {
        self.peek_values().into_iter().collect()
    }

    pub fn peek_values(&self) -> SmallVec<[T; 8]> {
        let mut values = SmallVec::<[T; 8]>::new();
        for (stack, _) in self.stacks.iter() {
            let Some(top) = stack.last() else {
                continue;
            };
            if !values.iter().any(|existing| existing == top) {
                values.push(top.clone());
            }
        }
        values
    }

    pub fn single_top_value(&self) -> Option<T> {
        let values = self.peek_values();
        (values.len() == 1).then(|| values[0].clone())
    }

    pub fn single_exclusive_top_value(&self) -> Option<T> {
        if self.stacks.iter().any(|(stack, _)| stack.is_empty()) {
            return None;
        }
        self.single_top_value()
    }

    pub fn path_count_at_most(&self, limit: usize) -> usize {
        self.stacks.len().min(limit)
    }

    pub fn for_each_acc(&self, mut f: impl FnMut(&A)) {
        for (_, acc) in self.stacks.iter() {
            f(acc);
        }
    }

    pub fn all_accs_satisfy(&self, pred: impl Fn(&A) -> bool) -> bool {
        self.stacks.iter().all(|(_, acc)| pred(acc))
    }
}