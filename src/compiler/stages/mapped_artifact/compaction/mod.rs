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

use rayon::prelude::*;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};

use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::{Weight, finalize_weight_map, shared_rangeset};

mod almost_optimal_layout;
mod default_layout;
mod exact_layout;

use default_layout::order_token_groups;
use exact_layout::{order_token_groups_globally_exact, order_tsid_groups_globally_exact};

const EXACT_LAYOUT_MAX_GROUPS: usize = 20;
const GLOBALLY_EXACT_COMPONENT_MAX_GROUPS_DEFAULT: usize = EXACT_LAYOUT_MAX_GROUPS;
const LARGE_ALMOST_OPTIMAL_COMPONENT_GROUPS: usize = 512;
const LARGE_ALMOST_OPTIMAL_GREEDY_STARTS: usize = 64;
const LARGE_ALMOST_OPTIMAL_NEIGHBORS: usize = 384;
const LARGE_ALMOST_OPTIMAL_RANDOM_WINDOW: usize = 16;
const LARGE_ALMOST_OPTIMAL_2OPT_WINDOW: usize = 64;
const DEFAULT_LAYOUT_SKETCH_WORDS: usize = 4;

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

struct DimensionCompaction {
    tsid_perm: Vec<u32>,
    ordered_num_tsids: usize,
    token_perm: Vec<u32>,
    ordered_num_tokens: usize,
}

impl DimensionCompaction {
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

impl CompactPlan {
    fn from_dimension_compaction(compaction: DimensionCompaction) -> Self {
        Self {
            tsid_perm: compaction.tsid_perm,
            ordered_num_tsids: compaction.ordered_num_tsids,
            token_perm: compaction.token_perm,
            ordered_num_tokens: compaction.ordered_num_tokens,
        }
    }

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
struct PermutationContext {
    tsid_runs: Vec<PermRun>,
    new_tsid_count: usize,
    token_runs: Vec<PermRun>,
    token_perm_is_identity: bool,
}

impl PermutationContext {
    fn new(tsid_perm: &[u32], token_perm: &[u32]) -> Self {
        Self {
            tsid_runs: permutation_runs(tsid_perm),
            new_tsid_count: tsid_perm
                .iter()
                .copied()
                .max()
                .map_or(0, |max| max as usize + 1),
            token_runs: permutation_runs(token_perm),
            token_perm_is_identity: token_perm
                .iter()
                .enumerate()
                .all(|(idx, &value)| value == idx as u32),
        }
    }
}

fn legacy_exact_adjacency_proxy_enabled() -> bool {
    // Historical name retained for compatibility.  This is *not* globally
    // exact compaction; it only solves the old adjacency-proxy layout when the
    // number of groups is tiny enough for Held-Karp DP.
    env_flag("GLRMASK_EXACT_COMPACTION")
}

fn globally_exact_compaction_enabled() -> bool {
    env_flag("GLRMASK_GLOBALLY_EXACT_COMPACTION")
}

fn almost_optimal_compaction_enabled() -> bool {
    env_flag("GLRMASK_ALMOST_OPTIMAL_COMPACTION")
}

fn globally_exact_component_max_groups() -> usize {
    static MAX_GROUPS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX_GROUPS.get_or_init(|| {
        std::env::var("GLRMASK_GLOBALLY_EXACT_MAX_COMPONENT_GROUPS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(GLOBALLY_EXACT_COMPONENT_MAX_GROUPS_DEFAULT)
    })
}

fn almost_optimal_passes() -> usize {
    static PASSES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PASSES.get_or_init(|| {
        std::env::var("GLRMASK_ALMOST_OPTIMAL_COMPACTION_PASSES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(16)
    })
}

fn almost_optimal_seed() -> u64 {
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SEED.get_or_init(|| {
        std::env::var("GLRMASK_ALMOST_OPTIMAL_COMPACTION_SEED")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(0x9e37_79b9_7f4a_7c15)
    })
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
    );
    apply_compaction_plan_to_weight_refs(weights, id_map, collect_profile_stats, &plan)
}

pub(super) fn plan_compaction_for_weight_refs(
    weights: &[&Weight],
    id_map: &InternalIdMap,
    allow_expensive_layout: bool,
    use_default_layout: bool,
) -> CompactPlan {
    let profile_compaction = env_flag("GLRMASK_PROFILE_COMPILE");
    let total_started_at = profile_compaction.then(Instant::now);
    let num_tsids = id_map.num_tsids() as usize;
    let num_tokens = id_map.num_internal_tokens() as usize;

    let unique_started_at = profile_compaction.then(Instant::now);
    let unique_weights = collect_unique_weights_from_weight_refs(weights);
    let unique_ms = unique_started_at.map_or(0.0, elapsed_ms);
    let build_started_at = profile_compaction.then(Instant::now);
    let use_thread_pool = allow_expensive_layout || use_default_layout;
    let build_compaction = || {
        build_dimension_compaction(
            &unique_weights,
            num_tsids,
            num_tokens,
            allow_expensive_layout,
            use_default_layout,
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

    CompactPlan::from_dimension_compaction(compaction)
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
    let storage_before = collect_profile_stats.then(|| {
        count_unique_storage_for_weight_refs(&weight_ref_slice(weights))
    });
    let storage_before_ms = storage_before_started_at.map_or(0.0, elapsed_ms);

    let unique_started_at = profile_compaction.then(Instant::now);
    let unique_weights = collect_unique_weights_from_refs(weights);
    let unique_ms = unique_started_at.map_or(0.0, elapsed_ms);

    let apply_weights_started_at = profile_compaction.then(Instant::now);
    let use_thread_pool = true;
    if !plan.is_identity(num_tsids, num_tokens) {
        let mut apply_compaction = || {
            apply_permutations_to_weight_refs(
                weights,
                &unique_weights,
                &plan.tsid_perm,
                &plan.token_perm,
            );
        };
        if use_thread_pool {
            run_with_compaction_thread_pool(apply_compaction);
        } else {
            apply_compaction();
        }
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
        apply_perm_to_id_map(
            &mut id_map.vocab_tokens,
            &plan.token_perm,
            plan.ordered_num_tokens,
        );
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
    num_tsids: usize,
    num_tokens: usize,
    allow_expensive_layout: bool,
    use_default_layout: bool,
) -> DimensionCompaction {
    if allow_expensive_layout
        && (globally_exact_compaction_enabled() || almost_optimal_compaction_enabled())
    {
        return build_global_objective_dimension_compaction(unique_weights, num_tsids, num_tokens);
    }

    let original_weight_refs = weight_refs(unique_weights);

    let (token_merge_perm, merged_num_tokens) =
        build_exact_token_merge_permutation(&original_weight_refs, num_tokens);
    let token_perm = if use_default_layout {
        order_token_groups(unique_weights, token_merge_perm, merged_num_tokens)
    } else {
        token_merge_perm
    };

    DimensionCompaction {
        tsid_perm: identity_perm(num_tsids),
        ordered_num_tsids: num_tsids,
        token_perm,
        ordered_num_tokens: merged_num_tokens,
    }
}

fn build_global_objective_dimension_compaction(
    unique_weights: &[Weight],
    num_tsids: usize,
    num_tokens: usize,
) -> DimensionCompaction {
    let original_weight_refs = weight_refs(unique_weights);

    let (token_merge_perm, merged_num_tokens) =
        build_exact_token_merge_permutation(&original_weight_refs, num_tokens);
    let token_perm = order_token_groups_globally_exact(
        unique_weights,
        token_merge_perm,
        merged_num_tokens,
    );

    let token_compacted_weights = apply_permutations_to_weight_set(
        unique_weights,
        &identity_perm(num_tsids),
        &token_perm,
    );
    let token_compacted_refs = weight_refs(&token_compacted_weights);
    let (tsid_merge_perm, merged_num_tsids) =
        build_exact_tsid_merge_permutation(&token_compacted_refs, num_tsids);
    let tsid_perm = order_tsid_groups_globally_exact(
        &token_compacted_weights,
        tsid_merge_perm,
        merged_num_tsids,
        merged_num_tokens,
    );

    DimensionCompaction {
        tsid_perm,
        ordered_num_tsids: merged_num_tsids,
        token_perm,
        ordered_num_tokens: merged_num_tokens,
    }
}

fn build_exact_token_merge_permutation(weights: &[&Weight], num_tokens: usize) -> (Vec<u32>, usize) {
    if num_tokens == 0 {
        return (Vec::new(), 0);
    }

    let mut seen_token_sets = HashSet::new();
    let mut token_sets = Vec::new();
    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
            let ptr = Arc::as_ptr(token_set) as usize;
            if seen_token_sets.insert(ptr) {
                token_sets.push(Arc::clone(token_set));
            }
        }
    }

    if token_sets.is_empty() {
        return (vec![0; num_tokens], 1);
    }

    let profile_words = token_sets.len().div_ceil(64);
    let mut profiles = vec![0u64; num_tokens * profile_words];
    for (context, token_set) in token_sets.iter().enumerate() {
        let word = context / 64;
        let bit = 1u64 << (context % 64);
        for token_range in token_set.ranges() {
            let start = (*token_range.start() as usize).min(num_tokens);
            let end = (*token_range.end() as usize).min(num_tokens.saturating_sub(1));
            if start > end {
                continue;
            }
            for token in start..=end {
                profiles[token * profile_words + word] |= bit;
            }
        }
    }

    let mut profile_to_group = HashMap::<Vec<u64>, u32>::new();
    let mut perm = vec![0u32; num_tokens];
    let mut next_group = 0u32;
    for token in 0..num_tokens {
        let profile_start = token * profile_words;
        let profile = profiles[profile_start..profile_start + profile_words].to_vec();
        perm[token] = *profile_to_group.entry(profile).or_insert_with(|| {
            let group = next_group;
            next_group += 1;
            group
        });
    }

    (perm, next_group as usize)
}

fn build_exact_tsid_merge_permutation(weights: &[&Weight], num_tsids: usize) -> (Vec<u32>, usize) {
    if num_tsids == 0 {
        return (Vec::new(), 0);
    }

    let mut profiles = vec![Vec::<u32>::new(); num_tsids];
    let mut context = 0u32;
    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        let mut contexts_by_token_set = HashMap::<Vec<(u32, u32)>, u32>::new();
        for (tsid_range, token_set) in weight.0.range_values() {
            let key = rangeset_key(token_set);
            let token_set_context = *contexts_by_token_set.entry(key).or_insert_with(|| {
                let current = context;
                context += 1;
                current
            });
            let start = *tsid_range.start();
            let end = (*tsid_range.end()).min(num_tsids.saturating_sub(1) as u32);
            for tsid in start..=end {
                profiles[tsid as usize].push(token_set_context);
            }
        }
    }

    build_profile_merge_permutation(&profiles)
}

fn compose_group_layout(initial_perm: Vec<u32>, layout: &[usize]) -> Vec<u32> {
    let mut group_to_position = vec![0u32; layout.len()];
    for (position, &group) in layout.iter().enumerate() {
        group_to_position[group] = position as u32;
    }
    initial_perm
        .into_iter()
        .map(|group| group_to_position[group as usize])
        .collect()
}

fn group_for_profile(profile: Vec<u32>, profile_to_group: &mut HashMap<Vec<u32>, u32>) -> u32 {
    let next_group = profile_to_group.len() as u32;
    *profile_to_group.entry(profile).or_insert(next_group)
}

fn build_profile_merge_permutation<P: Ord>(profiles: &[P]) -> (Vec<u32>, usize) {
    if profiles.is_empty() {
        return (Vec::new(), 0);
    }

    let mut indices: Vec<usize> = (0..profiles.len()).collect();
    indices.sort_by(|&left, &right| profiles[left].cmp(&profiles[right]));

    let mut perm = vec![0u32; profiles.len()];
    let mut group = 0u32;
    perm[indices[0]] = group;
    for pair in indices.windows(2) {
        if profiles[pair[0]] != profiles[pair[1]] {
            group += 1;
        }
        perm[pair[1]] = group;
    }

    (perm, group as usize + 1)
}

fn densify_used_group_ids(perm: Vec<u32>) -> (Vec<u32>, usize) {
    let mut remap = HashMap::<u32, u32>::new();
    let mut next_group = 0u32;
    let dense_perm = perm
        .into_iter()
        .map(|group| {
            *remap.entry(group).or_insert_with(|| {
                let dense = next_group;
                next_group += 1;
                dense
            })
        })
        .collect();

    (dense_perm, next_group as usize)
}

fn apply_permutations_to_weight_set(
    weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
) -> Vec<Weight> {
    let perm_context = PermutationContext::new(tsid_perm, token_perm);
    let token_cache = build_global_permuted_token_cache(weights, &perm_context);
    dedup_weights_by_storage_ptr(
        weights
            .iter()
            .map(|weight| permute_weight_with_cache(weight, &perm_context, &token_cache))
            .collect(),
    )
}

fn apply_permutations_to_weight_refs(
    weights: &mut [&mut Weight],
    unique_weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
) {
    let perm_context = PermutationContext::new(tsid_perm, token_perm);
    let token_cache = build_global_permuted_token_cache(unique_weights, &perm_context);
    let weight_entries: Vec<(usize, Weight)> = unique_weights
        .par_iter()
        .map(|weight| {
            let new_weight = permute_weight_with_cache(weight, &perm_context, &token_cache);
            (Arc::as_ptr(&weight.0) as usize, new_weight)
        })
        .collect();
    let weight_map: HashMap<usize, Weight> = weight_entries.into_iter().collect();

    for weight in weights.iter_mut() {
        let ptr = Arc::as_ptr(&weight.0) as usize;
        if let Some(new_weight) = weight_map.get(&ptr) {
            **weight = new_weight.clone();
        }
    }
}

fn permute_weight_with_cache(
    weight: &Weight,
    perm_context: &PermutationContext,
    permuted_token_cache: &HashMap<usize, RangeSetBlaze<u32>>,
) -> Weight {
    if weight.is_empty() {
        return Weight::empty();
    }
    if weight.is_full() {
        return Weight::all();
    }

    let mut tokens_by_new_tsid = vec![None::<RangeSetBlaze<u32>>; perm_context.new_tsid_count];

    for (tsid_range, token_set) in weight.0.range_values() {
        let token_set_ptr = Arc::as_ptr(token_set) as usize;
        let mapped_tokens = if perm_context.token_perm_is_identity {
            (**token_set).clone()
        } else {
            permuted_token_cache
                .get(&token_set_ptr)
                .cloned()
                .unwrap_or_else(|| permute_rangeset_with_runs(token_set, &perm_context.token_runs))
        };

        for run in overlapping_perm_runs(
            &perm_context.tsid_runs,
            *tsid_range.start(),
            *tsid_range.end(),
        ) {
            let new_tsid = run.mapped;
            let slot = &mut tokens_by_new_tsid[new_tsid as usize];
            match slot {
                Some(existing) => *existing |= mapped_tokens.clone(),
                None => *slot = Some(mapped_tokens.clone()),
            }
        }
    }

    finalize_weight_map(build_weight_map_from_tsid_tokens(tokens_by_new_tsid))
}

fn build_global_permuted_token_cache(
    weights: &[Weight],
    perm_context: &PermutationContext,
) -> HashMap<usize, RangeSetBlaze<u32>> {
    if perm_context.token_perm_is_identity {
        return HashMap::new();
    }

    let mut seen = HashSet::new();
    let mut token_sets = Vec::new();
    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
            let ptr = Arc::as_ptr(token_set) as usize;
            if seen.insert(ptr) {
                token_sets.push((ptr, Arc::clone(token_set)));
            }
        }
    }

    token_sets
        .par_iter()
        .map(|(ptr, token_set)| {
            (
                *ptr,
                permute_rangeset_with_runs(token_set, &perm_context.token_runs),
            )
        })
        .collect()
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

fn build_weight_map_from_tsid_tokens(
    tokens_by_tsid: Vec<Option<RangeSetBlaze<u32>>>,
) -> RangeMapBlaze<u32, Arc<RangeSetBlaze<u32>>> {
    let mut map = RangeMapBlaze::new();
    let mut run: Option<(u32, u32, RangeSetBlaze<u32>)> = None;

    for (tsid, tokens) in tokens_by_tsid.into_iter().enumerate() {
        let Some(tokens) = tokens else {
            if let Some((start, end, run_tokens)) = run.take() {
                map.extend_simple(std::iter::once((start..=end, shared_rangeset(run_tokens))));
            }
            continue;
        };

        let tsid = tsid as u32;
        match run.as_mut() {
            Some((_start, end, run_tokens)) if *end + 1 == tsid && *run_tokens == tokens => {
                *end = tsid;
            }
            Some(_) => {
                let (start, end, run_tokens) = run.take().unwrap();
                map.extend_simple(std::iter::once((start..=end, shared_rangeset(run_tokens))));
                run = Some((tsid, tsid, tokens));
            }
            None => run = Some((tsid, tsid, tokens)),
        }
    }

    if let Some((start, end, run_tokens)) = run {
        map.extend_simple(std::iter::once((start..=end, shared_rangeset(run_tokens))));
    }

    map
}

fn permute_rangeset(set: &RangeSetBlaze<u32>, perm: &[u32]) -> RangeSetBlaze<u32> {
    let runs = permutation_runs(perm);
    permute_rangeset_with_runs(set, &runs)
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

    for internal in &mut id_map.original_to_internal {
        if *internal == u32::MAX {
            continue;
        }
        if let Some(&new_id) = perm.get(*internal as usize) {
            *internal = new_id;
        }
    }

    let mut new_internal_to_originals = vec![Vec::new(); new_count];
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

fn collect_token_sets_after_permutation(
    weights: &[Weight],
    token_perm: &[u32],
) -> Vec<RangeSetBlaze<u32>> {
    let mut cache = HashMap::<usize, RangeSetBlaze<u32>>::new();
    let mut seen = HashSet::<Vec<(u32, u32)>>::new();
    let mut unique_sets = Vec::new();

    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
            let ptr = Arc::as_ptr(token_set) as usize;
            let mapped = cache
                .entry(ptr)
                .or_insert_with(|| permute_rangeset(token_set, token_perm));
            if seen.insert(rangeset_key(mapped)) {
                unique_sets.push(mapped.clone());
            }
        }
    }

    unique_sets
}

fn collect_unique_weights_from_refs(weights: &[&mut Weight]) -> Vec<Weight> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for weight in weights {
        if seen.insert(Arc::as_ptr(&weight.0) as usize) {
            unique.push((**weight).clone());
        }
    }
    unique
}

fn collect_unique_weights_from_weight_refs(weights: &[&Weight]) -> Vec<Weight> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for weight in weights {
        if seen.insert(Arc::as_ptr(&weight.0) as usize) {
            unique.push((**weight).clone());
        }
    }
    unique
}

fn dedup_weights_by_storage_ptr(weights: Vec<Weight>) -> Vec<Weight> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for weight in weights {
        if seen.insert(Arc::as_ptr(&weight.0) as usize) {
            unique.push(weight);
        }
    }
    unique
}

fn count_unique_storage_for_weight_refs(weights: &[&Weight]) -> UniqueStorageCounts {
    let mut seen_weights = HashSet::new();
    let mut seen_token_sets = HashSet::new();
    let mut storage = UniqueStorageCounts::default();

    for weight in weights {
        if seen_weights.insert(Arc::as_ptr(&weight.0) as usize) {
            storage.weight_ranges += weight.num_ranges();
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
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
