//! Generic weight compaction for mapped artifacts.
//!
//! This pass has two separate jobs:
//! - merge only tokenizer-state/token IDs that are provably equivalent for the
//!   entire supplied weight collection;
//! - choose deterministic numeric orders for the merged classes that tend to
//!   reduce `RangeMapBlaze` / `RangeSetBlaze` fragmentation.
//!
//! Ordering is heuristic. Merging is not: every many-to-one mapping below is
//! derived from exact membership profiles, so the rewritten weights remain a
//! valid representation of the same relations under the updated `InternalIdMap`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;

use crate::equiv_types::InternalIdMap;
use crate::ds::weight::Weight;
use super::reconcile::{RelationImage, TokenAxisMap, WeightBatch};

mod default_layout;

use default_layout::{ProfileOrder, order_profiles, mix64};

const DEFAULT_LAYOUT_SKETCH_WORDS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactLayout {
    MergeOnly,
    Sketch,
    Adaptive,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InternedRangeCounts {
    pub tsid_ranges: usize,
    pub token_ranges: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct UniqueStorageCounts {
    weight_ranges: usize,
    token_ranges: usize,
}

fn is_identity_perm(perm: &[u32]) -> bool {
    perm.iter().enumerate().all(|(id, &target)| target == id as u32)
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

/// Discover and publish a joint exact quotient in one operation. Neither the
/// permutation nor any pointer-indexed remap cache can outlive its sources.
pub(super) fn compact_weights_with_id_map(
    weights: &mut [&mut Weight],
    id_map: &mut InternalIdMap,
    layout: CompactLayout,
) {
    let profiling = env_flag("GLRMASK_PROFILE_COMPILE");
    let started = profiling.then(Instant::now);
    let num_tsids = id_map.num_tsids() as usize;
    let num_tokens = id_map.num_internal_tokens() as usize;

    let batch = WeightBatch::new(weights);
    let unique_count = batch.unique.len();
    let unique_ms = started.map_or(0.0, elapsed_ms);
    let order = match layout {
        CompactLayout::MergeOnly => ProfileOrder::FirstSeen,
        CompactLayout::Adaptive if weights.len() > unique_count.saturating_mul(4) =>
            ProfileOrder::Lexical,
        _ => ProfileOrder::Sketch,
    };
    let refs = weight_refs(&batch.unique);
    let token_started = profiling.then(Instant::now);
    let (token_perm, token_count) = build_token_partition(&refs, num_tokens, order);
    let token_ms = token_started.map_or(0.0, elapsed_ms);
    let state_started = profiling.then(Instant::now);
    let (tsid_perm, tsid_count) = build_state_partition(&refs, num_tsids, order);
    let state_ms = state_started.map_or(0.0, elapsed_ms);

    let apply_started = profiling.then(Instant::now);
    let tsid_identity = is_identity_perm(&tsid_perm);
    let token_identity = is_identity_perm(&token_perm);
    if !tsid_identity || !token_identity {
        batch.publish(weights, RelationImage::new(
            TokenAxisMap::Quotient(&tsid_perm), TokenAxisMap::Quotient(&token_perm),
            tsid_count, token_count,
        ));
    }
    let apply_ms = apply_started.map_or(0.0, elapsed_ms);
    if !tsid_identity {
        id_map.tokenizer_states.relabel_classes(&tsid_perm, tsid_count);
    }
    if !token_identity {
        id_map.vocab_tokens.relabel_classes(&token_perm, token_count);
    }
    if let Some(started) = started {
        eprintln!(
            "[glrmask/profile][mapped_compaction] weights={} unique_weights={} tsids_before={} tsids_after={} tokens_before={} tokens_after={} layout={layout:?} unique_ms={:.3} token_partition_ms={:.3} state_partition_ms={:.3} apply_ms={:.3} total_ms={:.3}",
            weights.len(), unique_count, num_tsids, tsid_count, num_tokens, token_count,
            unique_ms, token_ms, state_ms, apply_ms, elapsed_ms(started),
        );
    }
}

pub(super) fn count_interned_ranges_for_weight_refs(weights: &[&Weight]) -> InternedRangeCounts {
    let counts = count_unique_storage_for_weight_refs(weights);
    InternedRangeCounts {
        tsid_ranges: counts.weight_ranges,
        token_ranges: counts.token_ranges,
    }
}

/// Two coordinates are equivalent precisely when they belong to the same
/// observation contexts. A context is a union of disjoint inclusive runs.
/// Membership therefore changes only at run boundaries; adjacent runs of
/// the same context cancel at their common boundary.
#[derive(Default)]
struct ObservationEvents(Vec<u64>);

impl ObservationEvents {
    fn push_range(&mut self, start: u32, end: u32, context: u32, size: usize) {
        if start as usize >= size || start > end {
            return;
        }
        self.0.push((u64::from(start) << 32) | u64::from(context));
        if let Some(after) = end.checked_add(1)
            && (after as usize) < size
        {
            self.0.push((u64::from(after) << 32) | u64::from(context));
        }
    }

    fn sort(&mut self, size: usize) {
        if self.0.len() <= size.saturating_mul(2) {
            self.0.sort_unstable();
            return;
        }
        // Counting by coordinate keeps the dense-boundary case linear.
        // All toggles at one boundary commute.
        let mut offsets = vec![0usize; size];
        for &event in &self.0 {
            offsets[(event >> 32) as usize] += 1;
        }
        let mut next = 0;
        for slot in &mut offsets {
            let count = *slot;
            *slot = next;
            next += count;
        }
        let mut ordered = vec![0; self.0.len()];
        for &event in &self.0 {
            let slot = &mut offsets[(event >> 32) as usize];
            ordered[*slot] = event;
            *slot += 1;
        }
        self.0 = ordered;
    }

    fn partition<P: ActiveMembership>(
        mut self,
        size: usize,
        mut active: P,
    ) -> (Vec<u32>, Vec<P::Key>) {
        self.sort(size);
        let mut profiles = ProfileClasses::<P::Key>::default();
        let mut perm = vec![0; size];
        let mut cursor = 0;
        let mut events = self.0.into_iter().peekable();
        while let Some(&event) = events.peek() {
            let pos = (event >> 32) as usize;
            if cursor < pos {
                perm[cursor..pos].fill(profiles.intern(&active));
                cursor = pos;
            }
            while events.peek().is_some_and(|event| (*event >> 32) as usize == pos) {
                active.toggle(events.next().unwrap() as u32);
            }
        }
        if cursor < size {
            perm[cursor..].fill(profiles.intern(&active));
        }
        (perm, profiles.keys)
    }
}

fn context_fingerprint(context: u32) -> u64 {
    mix64(u64::from(context).wrapping_add(0x9e37_79b9_7f4a_7c15))
}
/// The fingerprint only selects candidates. Complete active membership
/// determines identity, even when adversarial fingerprints collide.
trait ActiveMembership {
    type Key;
    fn toggle(&mut self, context: u32);
    fn fingerprint(&self) -> u64;
    fn equals(&self, key: &Self::Key) -> bool;
    fn snapshot(&self) -> Self::Key;
}

struct ProfileClasses<K> {
    heads: HashMap<u64, u32>,
    links: Vec<Option<u32>>,
    keys: Vec<K>,
}

impl<K> Default for ProfileClasses<K> {
    fn default() -> Self {
        Self { heads: HashMap::new(), links: Vec::new(), keys: Vec::new() }
    }
}

impl<K> ProfileClasses<K> {
    fn intern<P: ActiveMembership<Key = K>>(&mut self, profile: &P) -> u32 {
        let hash = profile.fingerprint();
        let mut candidate = self.heads.get(&hash).copied();
        while let Some(id) = candidate {
            if profile.equals(&self.keys[id as usize]) {
                return id;
            }
            candidate = self.links[id as usize];
        }
        let group = u32::try_from(self.keys.len()).expect("observation class ID overflow");
        self.keys.push(profile.snapshot());
        self.links.push(self.heads.insert(hash, group));
        group
    }
}

impl ActiveMembership for u64 {
    type Key = u64;

    #[inline]
    fn toggle(&mut self, context: u32) { *self ^= 1 << context; }
    fn fingerprint(&self) -> u64 { *self }
    fn equals(&self, key: &u64) -> bool { self == key }
    fn snapshot(&self) -> u64 { *self }
}

struct DenseMembership {
    words: Vec<u64>,
    fingerprint: u64,
}

impl ActiveMembership for DenseMembership {
    type Key = Box<[u64]>;

    fn toggle(&mut self, context: u32) {
        self.words[context as usize / 64] ^= 1 << (context % 64);
        self.fingerprint ^= context_fingerprint(context);
    }
    fn fingerprint(&self) -> u64 { self.fingerprint }
    fn equals(&self, key: &Self::Key) -> bool { self.words.as_slice() == key.as_ref() }
    fn snapshot(&self) -> Self::Key { self.words.clone().into_boxed_slice() }
}

#[derive(Default)]
struct SparseMembership {
    contexts: std::collections::BTreeSet<u32>,
    fingerprint: u64,
}

impl ActiveMembership for SparseMembership {
    type Key = Box<[u32]>;

    fn toggle(&mut self, context: u32) {
        if !self.contexts.remove(&context) {
            self.contexts.insert(context);
        }
        self.fingerprint ^= context_fingerprint(context);
    }
    fn fingerprint(&self) -> u64 { self.fingerprint }
    fn equals(&self, key: &Self::Key) -> bool {
        self.contexts.len() == key.len() && self.contexts.iter().eq(key.iter())
    }
    fn snapshot(&self) -> Self::Key { self.contexts.iter().copied().collect() }
}

fn build_token_partition(
    weights: &[&Weight],
    num_tokens: usize,
    order: ProfileOrder,
) -> (Vec<u32>, usize) {
    if num_tokens == 0 {
        return (Vec::new(), 0);
    }

    let mut seen = HashSet::new();
    let mut seen_contents = HashSet::new();
    let mut contexts = 0usize;
    let mut events = ObservationEvents::default();
    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_, tokens) in weight.raw_range_values() {
            let ptr = Arc::as_ptr(tokens) as usize;
            if !seen.insert(ptr) {
                continue;
            }
            // Equal contexts need not count twice in a layout heuristic.
            if order != ProfileOrder::FirstSeen && !seen_contents.insert(rangeset_key(tokens)) {
                continue;
            }
            let context = u32::try_from(contexts).expect("token observation context overflow");
            contexts += 1;
            for range in tokens.ranges() {
                events.push_range(*range.start(), *range.end(), context, num_tokens);
            }
        }
    }
    if contexts <= 64 {
        let (mut perm, profiles) = events.partition(num_tokens, 0u64);
        order_profiles(&mut perm, &profiles, order);
        return (perm, profiles.len());
    }

    let (mut perm, profiles) = events.partition(num_tokens, DenseMembership {
        words: vec![0; contexts.div_ceil(64)],
        fingerprint: 0,
    });
    let count = profiles.len();
    order_profiles(&mut perm, &profiles, order);
    (perm, count)
}

/// Every observed token set is a union of exact token-equivalence classes.
/// Passing to that quotient is injective on these sets, so it cannot change
/// which TSIDs have equal rows. No token remap is needed to discover the
/// exact state partition.
fn build_state_partition(
    weights: &[&Weight], num_tsids: usize, order: ProfileOrder,
) -> (Vec<u32>, usize) {
    let mut events = ObservationEvents::default();
    let mut next_context = 0u32;
    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        let mut by_contents = HashMap::<Vec<(u32, u32)>, u32>::new();
        let mut by_pointer = HashMap::<usize, u32>::new();
        for (range, tokens) in weight.raw_range_values() {
            let ptr = Arc::as_ptr(tokens) as usize;
            let context = *by_pointer.entry(ptr).or_insert_with(|| {
                *by_contents.entry(rangeset_key(tokens)).or_insert_with(|| {
                    let id = next_context;
                    next_context = next_context.checked_add(1).expect("state observation context overflow");
                    id
                })
            });
            events.push_range(*range.start(), *range.end(), context, num_tsids);
        }
    }

    let (mut perm, profiles) = events.partition(num_tsids, SparseMembership::default());
    let count = profiles.len();
    // Unmerged states retain the original lexical-profile numbering.
    let order = if count < num_tsids && order == ProfileOrder::Sketch {
        ProfileOrder::Sketch
    } else {
        ProfileOrder::Lexical
    };
    order_profiles(&mut perm, &profiles, order);
    (perm, count)
}

fn count_unique_storage_for_weight_refs(weights: &[&Weight]) -> UniqueStorageCounts {
    let mut seen_weights = HashSet::new();
    let mut seen_token_sets = HashSet::new();
    let mut storage = UniqueStorageCounts::default();

    for weight in weights {
        if seen_weights.insert(weight.ptr_key()) {
            storage.weight_ranges += weight.num_ranges();
        }
        for (_tsid_range, token_set) in weight.raw_range_values() {
            if seen_token_sets.insert(Arc::as_ptr(token_set) as usize) {
                storage.token_ranges += token_set.ranges().count();
            }
        }
    }

    storage
}

fn weight_refs(weights: &[Weight]) -> Vec<&Weight> {
    weights.iter().collect()
}

#[cfg(test)]
fn identity_perm(size: usize) -> Vec<u32> {
    (0..size as u32).collect()
}

fn rangeset_key(set: &RangeSetBlaze<u32>) -> Vec<(u32, u32)> {
    set.ranges()
        .map(|range| (*range.start(), *range.end()))
        .collect()
}

#[cfg(test)]
// Tests intentionally build one-range sets; clippy's suggested rewrite would
// replace the range element with the individual range members.
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    fn token_set(ranges: &[(u32, u32)]) -> Arc<RangeSetBlaze<u32>> {
        Arc::new(ranges.iter().map(|&(start, end)| start..=end).collect())
    }

    #[test]
    fn sparse_weight_quotient_matches_literal_image() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([2..=2])),
            (3, RangeSetBlaze::from_iter([1..=3])),
            (4, RangeSetBlaze::from_iter([0..=0])),
        ]);
        for states in [[3, 0, 4, 1, 2], [1, 0, 1, 2, 0], [0, 1, 2, 3, 4]] {
            for tokens in [[1, 4, 0, 3, 2], [1, 0, 1, 2, 0], [0, 1, 2, 3, 4]] {
                let state_count = states.iter().copied().max().unwrap() as usize + 1;
                let token_count = tokens.iter().copied().max().unwrap() as usize + 1;
                let mut actual = weight.clone();
                WeightBatch::new(&[&mut actual]).publish(
                    &mut [&mut actual], RelationImage::new(TokenAxisMap::Quotient(&states),
                        TokenAxisMap::Quotient(&tokens), state_count, token_count),
                );
                for dest_state in 0..5 {
                    for dest_token in 0..5 {
                        let expected = (0..5).any(|state| {
                            states[state] == dest_state && (0..5).any(|token| {
                                tokens[token] == dest_token
                                    && weight.tokens_for_tsid(state as u32).contains(token as u32)
                            })
                        });
                        assert_eq!(actual.tokens_for_tsid(dest_state).contains(dest_token), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn exact_observation_partitions_match_literal_memberships() {
        const TOKENS: usize = 257;
        const STATES: usize = 29;
        let weights: Vec<_> = (0..137u32).map(|context| {
            let first = context.wrapping_mul(29) % TOKENS as u32;
            let second = context.wrapping_mul(47).wrapping_add(11) % TOKENS as u32;
            let tokens = token_set(&[
                (first, (first + context % 19).min(TOKENS as u32 - 1)),
                (second, (second + context % 7).min(TOKENS as u32 - 1)),
            ]);
            let states = context % (STATES as u32 - 1);
            Weight::from_per_tsid_token_sets([
                (states, tokens.as_ref().clone()),
                (states + 1, tokens.as_ref().clone()),
            ])
        }).chain([Weight::all(), Weight::empty()]).collect();

        for count in [0, 1, 9, weights.len()] {
            let refs = weight_refs(&weights[..count]);
            let (token_perm, token_count) = build_token_partition(&refs, TOKENS, ProfileOrder::FirstSeen);
            let (tsid_perm, tsid_count) = build_state_partition(&refs, STATES, ProfileOrder::Lexical);

            let token_signature = |token: usize| {
                refs.iter().flat_map(|weight| {
                    (0..STATES).map(move |state| {
                        weight.is_full() || weight.tokens_for_tsid(state as u32).contains(token as u32)
                    })
                }).collect::<Vec<_>>()
            };
            let state_signature = |state: usize| {
                refs.iter().filter(|weight| !weight.is_full())
                    .map(|weight| weight.tokens_for_tsid(state as u32).clone())
                    .collect::<Vec<_>>()
            };
            for order in [ProfileOrder::Lexical, ProfileOrder::Sketch] {
                let (ordered, count) = build_token_partition(&refs, TOKENS, order);
                assert_eq!(count, token_count);
                let mut image = vec![None; count];
                for (&class, &mapped) in token_perm.iter().zip(&ordered) {
                    let slot = &mut image[class as usize];
                    assert!(slot.is_none_or(|old| old == mapped));
                    *slot = Some(mapped);
                }
                assert_eq!(image.into_iter().flatten().collect::<HashSet<_>>().len(), count);
                let (ordered, count) = build_state_partition(&refs, STATES, order);
                assert_eq!(count, tsid_count);
                for left in 0..STATES {
                    for right in 0..STATES {
                        assert_eq!(ordered[left] == ordered[right], tsid_perm[left] == tsid_perm[right]);
                    }
                }
            }
            let token_profiles = (0..TOKENS).map(token_signature).collect::<Vec<_>>();
            let state_profiles = (0..STATES).map(state_signature).collect::<Vec<_>>();
            for left in 0..TOKENS {
                for right in 0..TOKENS {
                    assert_eq!(token_perm[left] == token_perm[right], token_profiles[left] == token_profiles[right]);
                }
            }
            for left in 0..STATES {
                for right in 0..STATES {
                    assert_eq!(tsid_perm[left] == tsid_perm[right], state_profiles[left] == state_profiles[right]);
                }
            }
            assert_eq!(token_perm.iter().copied().collect::<HashSet<_>>().len(), token_count);
            assert_eq!(tsid_perm.iter().copied().collect::<HashSet<_>>().len(), tsid_count);
            let mut materialized = weights[..count].to_vec();
            let mut refs = materialized.iter_mut().collect::<Vec<_>>();
            WeightBatch::new(&refs).publish(&mut refs, RelationImage::new(TokenAxisMap::Quotient(&identity_perm(STATES)),
                TokenAxisMap::Quotient(&token_perm), STATES, token_count));
            let (remapped, _) = build_state_partition(&weight_refs(&materialized), STATES, ProfileOrder::Lexical);
            for left in 0..STATES {
                for right in 0..STATES {
                    assert_eq!(tsid_perm[left] == tsid_perm[right], remapped[left] == remapped[right]);
                }
            }
        }
    }

    #[test]
    fn adjacent_observation_runs_toggle_once_at_their_outer_boundaries() {
        let mut events = ObservationEvents::default();
        events.push_range(1, 2, 0, 8);
        events.push_range(3, 4, 0, 8);
        events.push_range(2, 3, 1, 8);
        events.push_range(6, u32::MAX, 0, 8);
        let (perm, _) = events.partition(8, 0u64);
        assert_eq!(perm, vec![0, 1, 2, 2, 1, 0, 1, 1]);
    }
}
