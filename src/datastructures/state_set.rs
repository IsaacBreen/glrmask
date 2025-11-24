use std::hash::Hash;
use std::fmt::Debug;

/// A common interface for sets of states used in DFA construction.
/// Implementations allow swapping between sparse (CompressedStateSet)
/// and dense (BitSet) representations.
pub trait StateSet: Clone + Eq + Hash + Default + FromIterator<usize> + Debug {
    type Iter<'a>: Iterator<Item = usize> where Self: 'a;

    fn with_capacity(capacity_bits: usize) -> Self;
    fn insert(&mut self, state: usize) -> bool;
    fn contains(&self, state: usize) -> bool;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn clear(&mut self);
    fn iter<'a>(&'a self) -> Self::Iter<'a>;
    fn recompute_hash(&mut self) {}
}
