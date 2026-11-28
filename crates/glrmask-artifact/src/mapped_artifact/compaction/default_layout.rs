//! Numeric layout of exact observation classes.
//! The profile is a set of contexts in which the class is observable.
//! Layout changes only numeric labels, never equivalence membership.
use super::DEFAULT_LAYOUT_SKETCH_WORDS;
use std::borrow::Cow;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ProfileOrder {
    FirstSeen,
    Lexical,
    Sketch,
}

fn bit_contexts(words: &[u64]) -> impl Iterator<Item = u32> + '_ {
    words.iter().enumerate().flat_map(|(word, &bits)| {
        let mut bits = bits;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let context = word * 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            Some(context as u32)
        })
    })
}

pub(super) trait ProfileContexts {
    fn contexts(&self) -> impl Iterator<Item = u32> + '_;
    fn lexical_key(&self) -> Cow<'_, [u32]> {
        Cow::Owned(self.contexts().collect())
    }
}

impl ProfileContexts for u64 {
    fn contexts(&self) -> impl Iterator<Item = u32> + '_ {
        bit_contexts(std::slice::from_ref(self))
    }
}

impl ProfileContexts for Box<[u64]> {
    fn contexts(&self) -> impl Iterator<Item = u32> + '_ {
        bit_contexts(self)
    }
}

impl ProfileContexts for Box<[u32]> {
    fn contexts(&self) -> impl Iterator<Item = u32> + '_ {
        self.iter().copied()
    }
    fn lexical_key(&self) -> Cow<'_, [u32]> {
        Cow::Borrowed(self)
    }
}

pub(super) fn order_profiles<K: ProfileContexts>(
    perm: &mut [u32], profiles: &[K], order: ProfileOrder,
) {
    if order == ProfileOrder::FirstSeen || profiles.len() <= 1 {
        return;
    }
    let mut layout: Vec<usize> = (0..profiles.len()).collect();
    match order {
        ProfileOrder::FirstSeen => unreachable!(),
        ProfileOrder::Lexical => {
            // Decode dense keys once, not on every comparison. Sparse
            // keys already are sorted context lists and can be borrowed.
            let keys: Vec<_> = profiles.iter().map(|profile| profile.lexical_key()).collect();
            layout.sort_unstable_by(|&left, &right| {
                keys[left].cmp(&keys[right]).then_with(|| left.cmp(&right))
            });
        }
        ProfileOrder::Sketch => {
            let sketches: Vec<_> = profiles.iter().map(|profile| {
                let mut sketch = [u64::MAX; DEFAULT_LAYOUT_SKETCH_WORDS];
                let mut degree = 0usize;
                for context in profile.contexts() {
                    degree += 1;
                    for (index, slot) in sketch.iter_mut().enumerate() {
                        let hash = mix64(u64::from(context) ^
                            (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                        *slot = (*slot).min(hash);
                    }
                }
                (sketch, std::cmp::Reverse(degree))
            }).collect();
            layout.sort_unstable_by_key(|&class| (sketches[class], class));
        }
    }
    let mut old_to_new = vec![0u32; profiles.len()];
    for (new, old) in layout.into_iter().enumerate() {
        old_to_new[old] = new as u32;
    }
    for class in perm {
        *class = old_to_new[*class as usize];
    }
}

pub(super) fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
