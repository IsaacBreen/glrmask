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
use std::sync::{Arc, Weak};
use std::time::Instant;

use rayon::prelude::*;
use range_set_blaze::RangeSetBlaze;

use crate::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::{SharedTokenSet, Weight, shared_rangeset};

mod default_layout;

use default_layout::{ProfileOrder, ProfileContexts, order_profiles, mix64};

const DEFAULT_LAYOUT_SKETCH_WORDS: usize = 4;

/// Keep the source allocation address reserved for the cache lifetime.
/// Weak ownership does not retain the source token-set contents after their
/// last real owner disappears, but prevents a later allocation from
/// spuriously matching a stale pointer key.
#[derive(Clone, Debug)]
struct CachedTokenRemap {
    _source: Weak<RangeSetBlaze<u32>>,
    mapped: SharedTokenSet,
}

impl CachedTokenRemap {
    fn entry(source: &SharedTokenSet, mapped: SharedTokenSet) -> (usize, Self) {
        (Arc::as_ptr(source) as usize, Self {
            _source: Arc::downgrade(source),
            mapped,
        })
    }
}

type TokenRemapCache = HashMap<usize, CachedTokenRemap>;

#[derive(Clone, Debug)]
pub struct CompactReport {
    pub tsid_perm: Vec<u32>,
    pub token_perm: Vec<u32>,
    pub profile_stats: Option<CompactProfileStats>,
}

#[derive(Clone, Debug)]
pub struct CompactPlan {
    tsid_perm: Vec<u32>,
    ordered_num_tsids: usize,
    token_perm: Vec<u32>,
    ordered_num_tokens: usize,
    token_remaps: Arc<TokenRemapCache>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InternedRangeCounts {
    pub tsid_ranges: usize,
    pub token_ranges: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompactProfileStats {
    pub tsids_before: usize,
    pub tsids_after: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub token_ranges_before: usize,
    pub token_ranges_after: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct UniqueStorageCounts {
    weight_ranges: usize,
    token_ranges: usize,
}

impl CompactPlan {
    fn is_identity(&self, num_tsids: usize, num_tokens: usize) -> bool {
        self.tsid_perm_is_identity(num_tsids) && self.token_perm_is_identity(num_tokens)
    }

    fn tsid_perm_is_identity(&self, num_tsids: usize) -> bool {
        self.ordered_num_tsids == num_tsids
            && self
                .tsid_perm
                .iter()
                .enumerate()
                .all(|(idx, &mapped)| mapped == idx as u32)
    }

    fn token_perm_is_identity(&self, num_tokens: usize) -> bool {
        self.ordered_num_tokens == num_tokens
            && self
                .token_perm
                .iter()
                .enumerate()
                .all(|(idx, &mapped)| mapped == idx as u32)
    }
}

#[derive(Clone, Debug)]
struct PermRun {
    start: u32,
    end: u32,
    mapped: u32,
}

#[derive(Clone, Debug)]
struct TokenPermutation {
    runs: Vec<PermRun>,
    identity: bool,
}

impl TokenPermutation {
    fn new(perm: &[u32]) -> Self {
        let identity = is_identity_perm(perm);
        Self {
            runs: if identity { Vec::new() } else { permutation_runs(perm) },
            identity,
        }
    }
}

fn is_identity_perm(perm: &[u32]) -> bool {
    perm.iter().enumerate().all(|(id, &target)| target == id as u32)
}

#[derive(Clone, Debug)]
struct PermutationContext {
    tsid_runs: Vec<PermRun>,
    tsid_perm_is_identity: bool,
    tsid_perm_is_bijection: bool,
    tokens: TokenPermutation,
}

impl PermutationContext {
    fn new(tsid_perm: &[u32], token_perm: &[u32]) -> Self {
        let tsid_perm_is_identity = is_identity_perm(tsid_perm);
        let tsid_perm_is_bijection = tsid_perm_is_identity || {
            let mut seen = vec![false; tsid_perm.len()];
            tsid_perm.iter().copied().all(|mapped| {
                seen.get_mut(mapped as usize).is_some_and(|slot| {
                    !std::mem::replace(slot, true)
                })
            })
        };
        Self {
            tsid_runs: if tsid_perm_is_identity { Vec::new() } else { permutation_runs(tsid_perm) },
            tsid_perm_is_identity,
            tsid_perm_is_bijection,
            tokens: TokenPermutation::new(token_perm),
        }
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn compaction_thread_count() -> Option<usize> {
    if let Some(value) = std::env::var("GLRMASK_COMPACTION_THREADS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
    {
        return (value > 1).then_some(value);
    }

    std::thread::available_parallelism()
        .ok()
        .map(|parallelism| parallelism.get().min(16))
        .filter(|&value| value > 1)
}

fn run_with_compaction_thread_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    // Compilation already runs inside a bounded Rayon pool. Installing a
    // second pool from one of its workers oversubscribes the machine while
    // partition compactions overlap. Nested Rayon work can use the enclosing
    // pool directly; retain the dedicated pool for standalone callers.
    if rayon::current_thread_index().is_some() {
        return f();
    }

    static COMPACTION_THREAD_POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> =
        std::sync::OnceLock::new();
    let pool = COMPACTION_THREAD_POOL.get_or_init(|| {
        let thread_count = compaction_thread_count()?;
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .thread_name(|index| format!("glrmask-compact-{index}"))
            .build()
            .ok()
    });

    match pool {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

pub(super) fn compact_weights_with_id_map(
    weights: &mut [&mut Weight],
    id_map: &mut InternalIdMap,
    collect_profile_stats: bool,
    allow_expensive_layout: bool,
    use_default_layout: bool,
) -> CompactReport {
    let weight_refs = weight_ref_slice(weights);
    let plan = plan_compaction_for_weight_refs(
        &weight_refs,
        id_map,
        allow_expensive_layout,
        use_default_layout,
        false,
        false,
    );
    apply_compaction_plan_to_weight_refs(weights, id_map, collect_profile_stats, &plan)
}

pub(super) fn plan_compaction_for_weight_refs(
    weights: &[&Weight],
    id_map: &InternalIdMap,
    allow_expensive_layout: bool,
    use_default_layout: bool,
    keep_unmerged_tsid_identity: bool,
    tsids_proven_irredundant: bool,
) -> CompactPlan {
    let profile_compaction = env_flag("GLRMASK_PROFILE_COMPILE");
    let total_started_at = profile_compaction.then(Instant::now);
    let num_tsids = id_map.num_tsids() as usize;
    let num_tokens = id_map.num_internal_tokens() as usize;

    let unique_started_at = profile_compaction.then(Instant::now);
    let unique_weights = collect_unique_weights_from_weight_refs(weights);
    let unique_ms = unique_started_at.map_or(0.0, elapsed_ms);
    let build_started_at = profile_compaction.then(Instant::now);
    let use_thread_pool = allow_expensive_layout;
    let build_compaction = || {
        build_dimension_compaction(
            &unique_weights,
            weights.len(),
            num_tsids,
            num_tokens,
            allow_expensive_layout,
            use_default_layout,
            keep_unmerged_tsid_identity,
            tsids_proven_irredundant,
        )
    };
    let compaction = if use_thread_pool {
        run_with_compaction_thread_pool(build_compaction)
    } else {
        build_compaction()
    };
    let build_ms = build_started_at.map_or(0.0, elapsed_ms);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][mapped_compaction_plan] weights={} unique_weights={} tsids_before={} tsids_after={} tokens_before={} tokens_after={} expensive_layout={} unique_ms={:.3} build_ms={:.3} total_ms={:.3}",
            weights.len(),
            unique_weights.len(),
            num_tsids,
            compaction.ordered_num_tsids,
            num_tokens,
            compaction.ordered_num_tokens,
            allow_expensive_layout,
            unique_ms,
            build_ms,
            elapsed_ms(total_started_at),
        );
    }

    compaction
}

pub(super) fn apply_compaction_plan_to_weight_refs(
    weights: &mut [&mut Weight],
    id_map: &mut InternalIdMap,
    collect_profile_stats: bool,
    plan: &CompactPlan,
) -> CompactReport {
    let profile_compaction = env_flag("GLRMASK_PROFILE_COMPILE");
    let total_started_at = profile_compaction.then(Instant::now);
    let num_tsids = id_map.num_tsids() as usize;
    let num_tokens = id_map.num_internal_tokens() as usize;
    let storage_before_started_at = profile_compaction.then(Instant::now);
    let storage_before = if collect_profile_stats && weights.len() <= 4096 {
        Some(count_unique_storage_for_weight_refs(&weight_ref_slice(weights)))
    } else {
        None
    };
    let storage_before_ms = storage_before_started_at.map_or(0.0, elapsed_ms);

    let unique_started_at = profile_compaction.then(Instant::now);
    let unique_weights = collect_unique_weights_from_refs(weights);
    let unique_ms = unique_started_at.map_or(0.0, elapsed_ms);

    let apply_weights_started_at = profile_compaction.then(Instant::now);
    if !plan.is_identity(num_tsids, num_tokens) {
        apply_permutations_to_weight_refs(
            weights, &unique_weights, &plan.tsid_perm, &plan.token_perm,
            &plan.token_remaps,
        );
    }
    let apply_weights_ms = apply_weights_started_at.map_or(0.0, elapsed_ms);
    let apply_id_map_started_at = profile_compaction.then(Instant::now);
    if !plan.tsid_perm_is_identity(num_tsids) {
        apply_perm_to_id_map(
            &mut id_map.tokenizer_states,
            &plan.tsid_perm,
            plan.ordered_num_tsids,
        );
    }
    if !plan.token_perm_is_identity(num_tokens) {
        if id_map.deferred_vocab_singleton_original_ids.is_some() {
            apply_perm_to_deferred_vocab_singletons(
                id_map,
                &plan.token_perm,
                plan.ordered_num_tokens,
            );
        } else {
            apply_perm_to_id_map(
                &mut id_map.vocab_tokens,
                &plan.token_perm,
                plan.ordered_num_tokens,
            );
        }
    }
    let apply_id_map_ms = apply_id_map_started_at.map_or(0.0, elapsed_ms);

    let storage_after_started_at = profile_compaction.then(Instant::now);
    let profile_stats = storage_before.map(|storage_before| {
        let storage_after = count_unique_storage_for_weight_refs(&weight_ref_slice(weights));
        CompactProfileStats {
            tsids_before: num_tsids,
            tsids_after: plan.ordered_num_tsids,
            tokens_before: num_tokens,
            tokens_after: plan.ordered_num_tokens,
            token_ranges_before: storage_before.token_ranges,
            token_ranges_after: storage_after.token_ranges,
        }
    });
    let storage_after_ms = storage_after_started_at.map_or(0.0, elapsed_ms);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][mapped_compaction_apply] weights={} unique_weights={} tsids_before={} tsids_after={} tokens_before={} tokens_after={} storage_before_ms={:.3} unique_ms={:.3} apply_weights_ms={:.3} apply_id_map_ms={:.3} storage_after_ms={:.3} total_ms={:.3}",
            weights.len(),
            unique_weights.len(),
            num_tsids,
            plan.ordered_num_tsids,
            num_tokens,
            plan.ordered_num_tokens,
            storage_before_ms,
            unique_ms,
            apply_weights_ms,
            apply_id_map_ms,
            storage_after_ms,
            elapsed_ms(total_started_at),
        );
    }

    CompactReport {
        tsid_perm: plan.tsid_perm.clone(),
        token_perm: plan.token_perm.clone(),
        profile_stats,
    }
}

pub(super) fn count_interned_ranges_for_weight_refs(weights: &[&Weight]) -> InternedRangeCounts {
    let counts = count_unique_storage_for_weight_refs(weights);
    InternedRangeCounts {
        tsid_ranges: counts.weight_ranges,
        token_ranges: counts.token_ranges,
    }
}

fn build_dimension_compaction(
    unique_weights: &[Weight],
    weight_ref_count: usize,
    num_tsids: usize,
    num_tokens: usize,
    allow_expensive_layout: bool,
    use_default_layout: bool,
    keep_unmerged_tsid_identity: bool,
    tsids_proven_irredundant: bool,
) -> CompactPlan {
    let use_exact_profile_layout = use_default_layout
        && allow_expensive_layout
        && weight_ref_count > unique_weights.len().saturating_mul(4);
    build_default_dimension_compaction(
        unique_weights,
        num_tsids,
        num_tokens,
        use_default_layout,
        use_exact_profile_layout,
        keep_unmerged_tsid_identity,
        tsids_proven_irredundant,
    )
}

fn build_default_dimension_compaction(
    unique_weights: &[Weight],
    num_tsids: usize,
    num_tokens: usize,
    use_default_layout: bool,
    adaptive_layout: bool,
    keep_unmerged_tsid_identity: bool,
    tsids_proven_irredundant: bool,
) -> CompactPlan {
    let profile_compaction = env_flag("GLRMASK_PROFILE_COMPILE");
    let total_started_at = profile_compaction.then(Instant::now);
    let original_weight_refs = weight_refs(unique_weights);

    let order = if !use_default_layout {
        ProfileOrder::FirstSeen
    } else if adaptive_layout {
        ProfileOrder::Lexical
    } else {
        ProfileOrder::Sketch
    };
    let token_merge_started_at = profile_compaction.then(Instant::now);
    let fused_token_compaction = tsids_proven_irredundant
        && keep_unmerged_tsid_identity && !use_default_layout;
    let (token_perm, merged_num_tokens, fused_token_remaps) =
        build_token_partition(&original_weight_refs, num_tokens, order, fused_token_compaction);
    let token_merge_ms = token_merge_started_at.map_or(0.0, elapsed_ms);

    let token_remap_started_at = profile_compaction.then(Instant::now);
    let token_remaps = fused_token_remaps
        .unwrap_or_else(|| build_global_permuted_token_cache(unique_weights, &TokenPermutation::new(&token_perm)));
    let token_remap_ms = token_remap_started_at.map_or(0.0, elapsed_ms);
    let tsid_merge_started_at = profile_compaction.then(Instant::now);
    let (tsid_perm, merged_num_tsids) = if tsids_proven_irredundant {
        (identity_perm(num_tsids), num_tsids)
    } else {
        build_state_partition(
            &original_weight_refs, num_tsids, order, keep_unmerged_tsid_identity,
        )
    };
    let tsid_merge_ms = tsid_merge_started_at.map_or(0.0, elapsed_ms);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][mapped_compaction_default_detail] unique_weights={} num_tsids={} num_tokens={} merged_tokens={} merged_tsids={} fused_token_compaction={} token_merge_ms={:.3} token_remap_ms={:.3} tsid_merge_ms={:.3} total_ms={:.3}",
            unique_weights.len(), num_tsids, num_tokens, merged_num_tokens, merged_num_tsids,
            fused_token_compaction, token_merge_ms, token_remap_ms, tsid_merge_ms,
            elapsed_ms(total_started_at),
        );
    }
    CompactPlan {
        tsid_perm,
        ordered_num_tsids: merged_num_tsids,
        token_perm,
        ordered_num_tokens: merged_num_tokens,
        token_remaps: Arc::new(token_remaps),
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
    precompute_remaps: bool,
) -> (Vec<u32>, usize, Option<TokenRemapCache>) {
    if num_tokens == 0 {
        return (Vec::new(), 0, None);
    }

    let mut seen = HashSet::new();
    let mut seen_contents = HashSet::new();
    let mut sets = Vec::new();
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
            // Preserve every source pointer in the fused remap route.
            if order != ProfileOrder::FirstSeen && !seen_contents.insert(rangeset_key(tokens)) {
                continue;
            }
            let context = u32::try_from(sets.len()).expect("token observation context overflow");
            sets.push(tokens);
            for range in tokens.ranges() {
                events.push_range(*range.start(), *range.end(), context, num_tokens);
            }
        }
    }
    if sets.len() <= 64 {
        let (mut perm, profiles) = events.partition(num_tokens, 0u64);
        order_profiles(&mut perm, &profiles, order);
        return (perm, profiles.len(), None);
    }

    let (mut perm, profiles) = events.partition(num_tokens, DenseMembership {
        words: vec![0; sets.len().div_ceil(64)],
        fingerprint: 0,
    });
    let count = profiles.len();
    let remaps = (precompute_remaps && order == ProfileOrder::FirstSeen && count < num_tokens).then(|| {
        // Invert the exact class profiles directly. Every profile has just
        // one allocation, also used for collision resolution in the sweep.
        let mut groups = vec![Vec::<u32>::new(); sets.len()];
        for (group, profile) in profiles.iter().enumerate() {
            for context in profile.contexts() {
                groups[context as usize].push(group as u32);
            }
        }
        sets.into_iter().zip(groups).map(|(source, ids)| {
            let mapped = shared_rangeset(ids.into_iter().map(|id| id..=id).collect());
            CachedTokenRemap::entry(source, mapped)
        }).collect()
    });
    order_profiles(&mut perm, &profiles, order);
    (perm, count, remaps)
}

/// Every observed token set is a union of exact token-equivalence classes.
/// Passing to that quotient is injective on these sets, so it cannot change
/// which TSIDs have equal rows. No token remap is needed to discover the
/// exact state partition.
fn build_state_partition(
    weights: &[&Weight], num_tsids: usize, order: ProfileOrder, keep_unmerged_identity: bool,
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
    if count == num_tsids && keep_unmerged_identity {
        return (identity_perm(num_tsids), count);
    }
    // Unmerged states retain the original lexical-profile numbering.
    let order = if count < num_tsids && order == ProfileOrder::Sketch {
        ProfileOrder::Sketch
    } else {
        ProfileOrder::Lexical
    };
    order_profiles(&mut perm, &profiles, order);
    (perm, count)
}

fn apply_permutations_to_weight_refs(
    weights: &mut [&mut Weight],
    unique_weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
    precomputed_token_remaps: &TokenRemapCache,
) {
    let profile_compaction = env_flag("GLRMASK_PROFILE_COMPILE");
    let total_started_at = profile_compaction.then(Instant::now);
    let perm_context = PermutationContext::new(tsid_perm, token_perm);
    let cache_started_at = profile_compaction.then(Instant::now);
    let fallback_token_cache = build_missing_global_permuted_token_cache(
        unique_weights, &perm_context.tokens, precomputed_token_remaps,
    );
    let fallback_cache_ms = cache_started_at.map_or(0.0, elapsed_ms);

    let transform_started_at = profile_compaction.then(Instant::now);
    let permute_entry = |weight: &Weight| {
        let new_weight = permute_weight_with_caches(
            weight, &perm_context, precomputed_token_remaps, &fallback_token_cache,
        );
        (weight.ptr_key(), new_weight)
    };
    let weight_entries: Vec<(usize, Weight)> = if rayon::current_num_threads() == 1 {
        unique_weights.iter().map(permute_entry).collect()
    } else {
        unique_weights.par_iter().map(permute_entry).collect()
    };
    let transform_ms = transform_started_at.map_or(0.0, elapsed_ms);
    let map_started_at = profile_compaction.then(Instant::now);
    let weight_map: HashMap<usize, Weight> = weight_entries.into_iter().collect();
    let map_ms = map_started_at.map_or(0.0, elapsed_ms);

    let writeback_started_at = profile_compaction.then(Instant::now);
    for weight in weights.iter_mut() {
        let ptr = weight.ptr_key();
        if let Some(new_weight) = weight_map.get(&ptr) {
            **weight = new_weight.clone();
        }
    }
    let writeback_ms = writeback_started_at.map_or(0.0, elapsed_ms);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][mapped_compaction_apply_detail] weights={} unique_weights={} planned_token_sets={} fallback_token_sets={} fallback_cache_ms={:.3} transform_ms={:.3} map_ms={:.3} writeback_ms={:.3} total_ms={:.3}",
            weights.len(),
            unique_weights.len(),
            precomputed_token_remaps.len(),
            fallback_token_cache.len(),
            fallback_cache_ms,
            transform_ms,
            map_ms,
            writeback_ms,
            elapsed_ms(total_started_at),
        );
    }
}

fn permute_weight_with_caches(
    weight: &Weight,
    perm_context: &PermutationContext,
    precomputed_token_remaps: &TokenRemapCache,
    fallback_token_cache: &TokenRemapCache,
) -> Weight {
    if weight.is_empty() {
        return Weight::empty();
    }
    if weight.is_full() {
        return Weight::all();
    }
    let mapped_tokens = |token_set: &SharedTokenSet| {
        if perm_context.tokens.identity {
            Arc::clone(token_set)
        } else {
            lookup_permuted_token_set(
                token_set, Arc::as_ptr(token_set) as usize, perm_context,
                precomputed_token_remaps, fallback_token_cache,
            )
        }
    };
    if perm_context.tsid_perm_is_identity {
        if perm_context.tokens.identity {
            return weight.clone();
        }
        return Weight::from_tsid_ranges_shared(weight.raw_range_values().map(|(range, tokens)| {
            (*range.start(), *range.end(), mapped_tokens(tokens))
        }));
    }

    // Visit only permutation runs covered by the source coefficient.
    // Sorting by destination admits one common sparse publication; a
    // bijection has no collisions and needs no union operation at all.
    let mut entries = Vec::<(u32, SharedTokenSet)>::new();
    for (range, tokens) in weight.raw_range_values() {
        let mapped = mapped_tokens(tokens);
        for run in overlapping_perm_runs(&perm_context.tsid_runs, *range.start(), *range.end()) {
            entries.push((run.mapped, Arc::clone(&mapped)));
        }
    }
    entries.sort_unstable_by_key(|(tsid, _)| *tsid);
    if perm_context.tsid_perm_is_bijection {
        Weight::from_per_tsid_shared(entries)
    } else {
        Weight::union_sorted_point_entries(entries)
    }
}

fn lookup_permuted_token_set(
    token_set: &RangeSetBlaze<u32>,
    token_set_ptr: usize,
    perm_context: &PermutationContext,
    precomputed_token_remaps: &TokenRemapCache,
    fallback_token_cache: &TokenRemapCache,
) -> SharedTokenSet {
    precomputed_token_remaps
        .get(&token_set_ptr)
        .or_else(|| fallback_token_cache.get(&token_set_ptr))
        .map(|entry| Arc::clone(&entry.mapped))
        .unwrap_or_else(|| shared_rangeset(permute_rangeset_with_runs(token_set, &perm_context.tokens.runs)))
}

fn build_global_permuted_token_cache(
    weights: &[Weight],
    permutation: &TokenPermutation,
) -> TokenRemapCache {
    build_missing_global_permuted_token_cache(weights, permutation, &TokenRemapCache::new())
}

fn build_missing_global_permuted_token_cache(
    weights: &[Weight],
    permutation: &TokenPermutation,
    precomputed_token_remaps: &TokenRemapCache,
) -> TokenRemapCache {
    if permutation.identity {
        return TokenRemapCache::new();
    }

    let mut seen = HashSet::new();
    let mut token_sets = Vec::new();
    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_tsid_range, token_set) in weight.raw_range_values() {
            let ptr = Arc::as_ptr(token_set) as usize;
            if precomputed_token_remaps.contains_key(&ptr) {
                continue;
            }
            if seen.insert(ptr) {
                token_sets.push((ptr, Arc::clone(token_set)));
            }
        }
    }

    let profile_compaction = env_flag("GLRMASK_PROFILE_COMPILE");
    let source_range_count = profile_compaction.then(|| {
        token_sets
            .iter()
            .map(|(_, token_set)| token_set.ranges().count())
            .sum::<usize>()
    });
    let overlap_run_count = profile_compaction.then(|| {
        token_sets
            .iter()
            .flat_map(|(_, token_set)| token_set.ranges())
            .map(|range| {
                overlapping_perm_runs(
                    &permutation.runs,
                    *range.start(),
                    *range.end(),
                )
                .len()
            })
            .sum::<usize>()
    });
    let remap_started_at = profile_compaction.then(Instant::now);
    let permute_token_set = |(_, token_set): &(usize, SharedTokenSet)| {
        CachedTokenRemap::entry(
            token_set,
            shared_rangeset(permute_rangeset_with_runs(token_set, &permutation.runs)),
        )
    };
    let remaps: TokenRemapCache = if rayon::current_num_threads() == 1 {
        token_sets.iter().map(permute_token_set).collect()
    } else {
        token_sets.par_iter().map(permute_token_set).collect()
    };
    if let Some(remap_started_at) = remap_started_at {
        let output_range_count: usize = remaps.values().map(|entry| entry.mapped.ranges().count()).sum();
        eprintln!(
            "[glrmask/profile][mapped_compaction_token_remap] token_sets={} source_ranges={} perm_runs={} overlap_runs={} output_ranges={} remap_ms={:.3}",
            token_sets.len(),
            source_range_count.unwrap_or(0),
            permutation.runs.len(),
            overlap_run_count.unwrap_or(0),
            output_range_count,
            elapsed_ms(remap_started_at),
        );
    }
    remaps
}

fn permutation_runs(perm: &[u32]) -> Vec<PermRun> {
    let mut runs = Vec::new();
    let Some((&first, rest)) = perm.split_first() else {
        return runs;
    };

    let mut start = 0u32;
    let mut end = 0u32;
    let mut mapped = first;
    for (offset, &value) in rest.iter().enumerate() {
        let index = offset as u32 + 1;
        if value == mapped {
            end = index;
        } else {
            runs.push(PermRun { start, end, mapped });
            start = index;
            end = index;
            mapped = value;
        }
    }
    runs.push(PermRun { start, end, mapped });
    runs
}

fn overlapping_perm_runs(runs: &[PermRun], start: u32, end: u32) -> &[PermRun] {
    if runs.is_empty() || start > end {
        return &[];
    }

    let mut first = 0usize;
    let mut step = runs.len();
    while step > 0 {
        let half = step / 2;
        let mid = first + half;
        if runs[mid].end < start {
            first = mid + 1;
            step -= half + 1;
        } else {
            step = half;
        }
    }

    let mut last = first;
    while last < runs.len() && runs[last].start <= end {
        last += 1;
    }
    &runs[first..last]
}

fn permute_rangeset_with_runs(set: &RangeSetBlaze<u32>, runs: &[PermRun]) -> RangeSetBlaze<u32> {
    let mut mapped: Vec<u32> = set
        .ranges()
        .flat_map(|range| {
            overlapping_perm_runs(runs, *range.start(), *range.end())
                .iter()
                .map(|run| run.mapped)
        })
        .collect();
    mapped.sort_unstable();
    mapped.dedup();

    let mut ranges = Vec::new();
    let Some((&first, rest)) = mapped.split_first() else {
        return RangeSetBlaze::new();
    };
    let mut start = first;
    let mut end = first;
    for &token in rest {
        if token == end + 1 {
            end = token;
        } else {
            ranges.push(start..=end);
            start = token;
            end = token;
        }
    }
    ranges.push(start..=end);
    RangeSetBlaze::from_iter(ranges)
}

fn apply_perm_to_id_map(id_map: &mut ManyToOneIdMap, perm: &[u32], new_count: usize) {
    let old_internal_to_originals = std::mem::take(&mut id_map.internal_to_originals);
    let old_representatives = std::mem::take(&mut id_map.representative_original_ids);
    let old_classes_are_singletons =
        old_internal_to_originals.len() == old_representatives.len()
            && old_internal_to_originals
                .iter()
                .all(|originals| originals.len() <= 1);

    for internal in &mut id_map.original_to_internal {
        if *internal == u32::MAX {
            continue;
        }
        if let Some(&new_id) = perm.get(*internal as usize) {
            *internal = new_id;
        }
    }

    if old_classes_are_singletons {
        let mut new_sizes = vec![0usize; new_count];
        for (old_internal, originals) in old_internal_to_originals.iter().enumerate() {
            if originals.is_empty() {
                continue;
            }
            let Some(&new_internal) = perm.get(old_internal) else {
                continue;
            };
            let new_internal = new_internal as usize;
            if new_internal < new_count {
                new_sizes[new_internal] += 1;
            }
        }

        let mut new_internal_to_originals: Vec<Vec<u32>> =
            new_sizes.into_iter().map(Vec::with_capacity).collect();
        let mut new_representatives = vec![u32::MAX; new_count];
        for (old_internal, originals) in old_internal_to_originals.into_iter().enumerate() {
            let Some(&original) = originals.first() else {
                continue;
            };
            let Some(&new_internal) = perm.get(old_internal) else {
                continue;
            };
            let new_internal = new_internal as usize;
            if new_internal >= new_count {
                continue;
            }
            new_internal_to_originals[new_internal].push(original);
            if new_representatives[new_internal] == u32::MAX {
                new_representatives[new_internal] = old_representatives[old_internal];
            }
        }

        id_map.internal_to_originals = new_internal_to_originals;
        id_map.representative_original_ids = new_representatives;
        return;
    }

    let mut new_sizes = vec![0usize; new_count];
    for (old_internal, originals) in old_internal_to_originals.iter().enumerate() {
        let Some(&new_internal) = perm.get(old_internal) else {
            continue;
        };
        let new_internal = new_internal as usize;
        if new_internal < new_count {
            new_sizes[new_internal] += originals.len();
        }
    }

    let mut new_internal_to_originals: Vec<Vec<u32>> =
        new_sizes.into_iter().map(Vec::with_capacity).collect();
    let mut new_representatives = vec![u32::MAX; new_count];
    for (old_internal, originals) in old_internal_to_originals.into_iter().enumerate() {
        let Some(&new_internal) = perm.get(old_internal) else {
            continue;
        };
        let new_internal = new_internal as usize;
        if new_internal >= new_count {
            continue;
        }
        new_internal_to_originals[new_internal].extend(originals);
        if new_representatives[new_internal] == u32::MAX {
            new_representatives[new_internal] = old_representatives[old_internal];
        }
    }

    id_map.internal_to_originals = new_internal_to_originals;
    id_map.representative_original_ids = new_representatives;
}

fn apply_perm_to_deferred_vocab_singletons(
    id_map: &mut InternalIdMap,
    perm: &[u32],
    new_count: usize,
) {
    let original_ids = id_map
        .deferred_vocab_singleton_original_ids
        .take()
        .expect("deferred vocabulary singleton order disappeared before compaction");
    let original_count = original_ids
        .iter()
        .copied()
        .max()
        .map_or(0, |max_original| max_original as usize + 1);
    let mut original_to_internal = vec![u32::MAX; original_count];
    let mut sizes = vec![0usize; new_count];
    for old_internal in 0..original_ids.len() {
        let Some(&new_internal) = perm.get(old_internal) else {
            continue;
        };
        if (new_internal as usize) < new_count {
            sizes[new_internal as usize] += 1;
        }
    }
    let mut internal_to_originals: Vec<Vec<u32>> =
        sizes.into_iter().map(Vec::with_capacity).collect();
    let mut representative_original_ids = vec![u32::MAX; new_count];
    for (old_internal, &original) in original_ids.iter().enumerate() {
        let Some(&new_internal) = perm.get(old_internal) else {
            continue;
        };
        let new_internal = new_internal as usize;
        if new_internal >= new_count {
            continue;
        }
        original_to_internal[original as usize] = new_internal as u32;
        internal_to_originals[new_internal].push(original);
        if representative_original_ids[new_internal] == u32::MAX {
            representative_original_ids[new_internal] = original;
        }
    }
    id_map.vocab_tokens = ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    };
}

fn collect_unique_weights_from_refs(weights: &[&mut Weight]) -> Vec<Weight> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for weight in weights {
        if seen.insert(weight.ptr_key()) {
            unique.push((**weight).clone());
        }
    }
    unique
}

fn collect_unique_weights_from_weight_refs(weights: &[&Weight]) -> Vec<Weight> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for weight in weights {
        if seen.insert(weight.ptr_key()) {
            unique.push((**weight).clone());
        }
    }
    unique
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

fn weight_ref_slice<'a>(weights: &'a [&'a mut Weight]) -> Vec<&'a Weight> {
    weights.iter().map(|weight| &**weight).collect()
}

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
    fn sparse_weight_permutation_matches_literal_image() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([2..=2])),
            (3, RangeSetBlaze::from_iter([1..=3])),
            (4, RangeSetBlaze::from_iter([0..=0])),
        ]);
        for states in [[3, 0, 4, 1, 2], [1, 0, 1, 2, 0], [0, 1, 2, 3, 4]] {
            for tokens in [[1, 4, 0, 3, 2], [1, 0, 1, 2, 0], [0, 1, 2, 3, 4]] {
                let context = PermutationContext::new(&states, &tokens);
                let actual = permute_weight_with_caches(
                    &weight, &context, &TokenRemapCache::new(), &TokenRemapCache::new(),
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
    fn shared_token_remap_cache_matches_fallback_and_reuses_storage() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=0])),
            (1, RangeSetBlaze::from_iter([1..=1])),
            (2, RangeSetBlaze::from_iter([2..=2])),
        ]);
        let context = PermutationContext::new(&[0, 1, 2], &[2, 0, 1]);
        let cache = build_global_permuted_token_cache(std::slice::from_ref(&weight), &context.tokens);
        let cached = permute_weight_with_caches(
            &weight,
            &context,
            &cache,
            &TokenRemapCache::new(),
        );
        let fallback = permute_weight_with_caches(
            &weight,
            &context,
            &TokenRemapCache::new(),
            &TokenRemapCache::new(),
        );

        assert_eq!(cached, fallback);
        for ((_, source_tokens), (_, cached_tokens)) in
            weight.raw_range_values().zip(cached.raw_range_values())
        {
            let source_ptr = Arc::as_ptr(source_tokens) as usize;
            assert!(Arc::ptr_eq(cached_tokens, &cache.get(&source_ptr).unwrap().mapped));
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
            let (token_perm, token_count, fused) = build_token_partition(&refs, TOKENS, ProfileOrder::FirstSeen, true);
            let (tsid_perm, tsid_count) = build_state_partition(&refs, STATES, ProfileOrder::Lexical, false);

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
                let (ordered, count, _) = build_token_partition(&refs, TOKENS, order, false);
                assert_eq!(count, token_count);
                let mut image = vec![None; count];
                for (&class, &mapped) in token_perm.iter().zip(&ordered) {
                    let slot = &mut image[class as usize];
                    assert!(slot.is_none_or(|old| old == mapped));
                    *slot = Some(mapped);
                }
                assert_eq!(image.into_iter().flatten().collect::<HashSet<_>>().len(), count);
                let (ordered, count) = build_state_partition(&refs, STATES, order, false);
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
            if let Some(fused) = fused {
                let expected = build_global_permuted_token_cache(&weights[..count],
                    &TokenPermutation::new(&token_perm));
                assert_eq!(fused.len(), expected.len());
                for (ptr, entry) in fused {
                    assert_eq!(entry.mapped, expected[&ptr].mapped);
                    assert_eq!(entry._source.as_ptr(), expected[&ptr]._source.as_ptr());
                }
            }
            let context = PermutationContext::new(&identity_perm(STATES), &token_perm);
            let materialized: Vec<_> = weights[..count].iter().map(|weight| {
                permute_weight_with_caches(
                    weight, &context, &TokenRemapCache::new(), &TokenRemapCache::new(),
                )
            }).collect();
            let (remapped, _) = build_state_partition(&weight_refs(&materialized), STATES, ProfileOrder::Lexical, false);
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
