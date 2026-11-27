//! Bottom-up structural NWA quotient. Fingerprints only select candidates;
//! exact normalized signatures certify merges. No weighted-language
//! determinization or interpretation of DEFAULT occurs here.
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Item {
    Epsilon(u32, usize),
    Label(i32),
    Branch(i32, u32, usize),
}

#[derive(Default, PartialEq, Eq, Hash)]
pub struct RowSignature {
    final_weight: Option<usize>,
    items: Vec<Item>,
}

impl RowSignature {
    pub fn final_weight(&mut self, weight: Option<usize>) {
        self.final_weight = weight;
    }

    pub fn label(&mut self, label: i32) {
        self.items.push(Item::Label(label));
    }

    pub fn epsilon(&mut self, target: u32, weight: usize) {
        self.items.push(Item::Epsilon(target, weight));
    }

    pub fn branch(&mut self, label: i32, target: u32, weight: usize) {
        self.items.push(Item::Branch(label, target, weight));
    }

    fn reset(&mut self) {
        self.final_weight = None;
        self.items.clear();
    }

    fn normalize(&mut self) {
        self.items.sort_unstable();
    }

    fn fingerprint(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

pub struct StructuralQuotient {
    pub old_to_new: Vec<u32>,
    pub representatives: Vec<u32>,
}

/// The adapter must name exact weight identities (or use canonical weight
/// IDs), retain every label key and distinguish epsilon from label branches.
/// Nonzero successors must already have a class. A semantically inert zero
/// branch can name a fixed guard sentinel instead of its target, but only
/// where the caller's structural contract permits that.
pub fn classify_reverse(
    count: usize,
    reverse_order: impl IntoIterator<Item = u32>,
    mut signature: impl FnMut(u32, &[u32], &mut RowSignature) -> Option<()>,
) -> Option<StructuralQuotient> {
    let mut old_to_new = vec![u32::MAX; count];
    let mut representatives = Vec::<u32>::new();
    let mut buckets = FxHashMap::<u64, SmallVec<[u32; 2]>>::default();
    let mut current = RowSignature::default();
    let mut comparison = RowSignature::default();
    let mut visited = 0usize;

    for old in reverse_order {
        if *old_to_new.get(old as usize)? != u32::MAX {
            return None;
        }
        current.reset();
        signature(old, &old_to_new, &mut current)?;
        current.normalize();
        let fingerprint = current.fingerprint();
        let mut existing = None;
        if let Some(candidates) = buckets.get(&fingerprint) {
            for &candidate in candidates {
                comparison.reset();
                signature(representatives[candidate as usize], &old_to_new, &mut comparison)?;
                comparison.normalize();
                if comparison == current {
                    existing = Some(candidate);
                    break;
                }
            }
        }
        let class = if let Some(class) = existing {
            class
        } else {
            let class = u32::try_from(representatives.len()).ok()?;
            if class == u32::MAX {
                return None;
            }
            representatives.push(old);
            buckets.entry(fingerprint).or_default().push(class);
            class
        };
        old_to_new[old as usize] = class;
        visited += 1;
    }
    (visited == count).then_some(StructuralQuotient { old_to_new, representatives })
}

/// A live successor must precede its source in the checked reverse order.
pub fn checked_successor(classes: &[u32], target: u32) -> Option<u32> {
    let class = *classes.get(target as usize)?;
    (class != u32::MAX).then_some(class)
}
