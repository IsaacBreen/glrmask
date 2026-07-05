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
use crate::ds::weight::{SharedTokenSet, Weight, finalize_weight_map, shared_rangeset};

mod almost_optimal_layout;
mod default_layout;
mod exact_layout;

use default_layout::{
    order_token_groups_exact_profile, order_token_groups_sketch,
    order_tsid_groups_exact_profile, order_tsid_groups_sketch,
};
use exact_layout::{order_token_groups_globally_exact, order_tsid_groups_globally_exact};

const EXACT_LAYOUT_MAX_GROUPS: usize = 20;
const GLOBALLY_EXACT_COMPONENT_MAX_GROUPS_DEFAULT: usize = EXACT_LAYOUT_MAX_GROUPS;
const LARGE_ALMOST_OPTIMAL_COMPONENT_GROUPS: usize = 512;
const LARGE_ALMOST_OPTIMAL_GREEDY_STARTS: usize = 64;
const LARGE_ALMOST_OPTIMAL_NEIGHBORS: usize = 384;
const LARGE_ALMOST_OPTIMAL_RANDOM_WINDOW: usize = 16;
const LARGE_ALMOST_OPTIMAL_2OPT_WINDOW: usize = 64;
const DEFAULT_LAYOUT_SKETCH_WORDS: usize = 4;
const MIN_UNIQUE_WEIGHTS_FOR_COMPACTION_POOL: usize = 1024;

type TokenRemapCache = HashMap<usize, SharedTokenSet>;

#[derive(Debug, Default)]
struct PrecomputedTokenCompaction {
    token_remaps: TokenRemapCache,
    // Original weight storage pointer -> token-remapped Weight. This is final
    // whenever the plan preserves the TSID dimension.
    weight_remaps: HashMap<usize, Weight>,
}

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
    // Token-set remaps already computed while constructing the plan's
    // token-compacted weights. Pointer keys are intentionally advisory:
    // applying the plan to a related artifact falls back for cache misses.
    precomputed_token_compaction: Arc<PrecomputedTokenCompaction>,
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
    precomputed_token_compaction: Arc<PrecomputedTokenCompaction>,
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
            precomputed_token_compaction: compaction.precomputed_token_compaction,
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
    tsid_perm_is_identity: bool,
    /// True when the TSID map is a pure permutation. In that case weights can
    /// be rebuilt from their sparse entries directly, without allocating a
    /// dense `new_tsid_count` scratch vector per weight.
    tsid_perm_is_bijection: bool,
    new_tsid_count: usize,
    token_runs: Vec<PermRun>,
    token_perm_is_identity: bool,
}

impl PermutationContext {
    fn new(tsid_perm: &[u32], token_perm: &[u32]) -> Self {
        let new_tsid_count = tsid_perm
            .iter()
            .copied()
            .max()
            .map_or(0, |max| max as usize + 1);
        let tsid_perm_is_bijection = if new_tsid_count != tsid_perm.len() {
            false
        } else {
            let mut seen = vec![false; new_tsid_count];
            tsid_perm.iter().copied().all(|mapped| {
                let slot = &mut seen[mapped as usize];
                let was_seen = *slot;
                *slot = true;
                !was_seen
            })
        };
        Self {
            tsid_runs: permutation_runs(tsid_perm),
            tsid_perm_is_identity: tsid_perm
                .iter()
                .enumerate()
                .all(|(idx, &value)| value == idx as u32),
            tsid_perm_is_bijection,
            new_tsid_count,
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
    let use_thread_pool = false;
    if !plan.is_identity(num_tsids, num_tokens) {
        let mut apply_compaction = || {
            apply_permutations_to_weight_refs(
                weights,
                &unique_weights,
                &plan.tsid_perm,
                &plan.token_perm,
                plan.precomputed_token_compaction.as_ref(),
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
    weight_ref_count: usize,
    num_tsids: usize,
    num_tokens: usize,
    allow_expensive_layout: bool,
    use_default_layout: bool,
    keep_unmerged_tsid_identity: bool,
    tsids_proven_irredundant: bool,
) -> DimensionCompaction {
    if allow_expensive_layout
        && (globally_exact_compaction_enabled() || almost_optimal_compaction_enabled())
    {
        return build_global_objective_dimension_compaction(unique_weights, num_tsids, num_tokens);
    }

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
) -> DimensionCompaction {
    let profile_compaction = env_flag("GLRMASK_PROFILE_COMPILE");
    let total_started_at = profile_compaction.then(Instant::now);
    let original_weight_refs = weight_refs(unique_weights);

    let token_merge_started_at = profile_compaction.then(Instant::now);
    let (token_merge_perm, merged_num_tokens) =
        build_exact_token_merge_permutation(&original_weight_refs, num_tokens);
    let token_merge_ms = token_merge_started_at.map_or(0.0, elapsed_ms);
    let token_order_started_at = profile_compaction.then(Instant::now);
    let token_perm = if use_default_layout {
        if adaptive_layout {
            order_token_groups_exact_profile(unique_weights, token_merge_perm, merged_num_tokens)
        } else {
            order_token_groups_sketch(unique_weights, token_merge_perm, merged_num_tokens)
        }
    } else {
        token_merge_perm
    };
    let token_order_ms = token_order_started_at.map_or(0.0, elapsed_ms);

    let token_context = PermutationContext::new(&identity_perm(num_tsids), &token_perm);
    let token_remap_started_at = profile_compaction.then(Instant::now);
    let token_remaps = build_global_permuted_token_cache(unique_weights, &token_context);
    let token_remap_ms = token_remap_started_at.map_or(0.0, elapsed_ms);
    let tsid_merge_started_at = profile_compaction.then(Instant::now);
    let (tsid_merge_perm, merged_num_tsids) = if tsids_proven_irredundant {
        (identity_perm(num_tsids), num_tsids)
    } else {
        build_exact_tsid_merge_permutation_with_token_remaps(
            &original_weight_refs,
            num_tsids,
            &token_context,
            &token_remaps,
        )
    };
    let tsid_merge_ms = tsid_merge_started_at.map_or(0.0, elapsed_ms);

    // Only ordering merged TSID groups needs token-remapped Weight objects.
    // When there are no TSID merges, the plan can carry its token remap cache
    // straight to the final rewrite and materialize every weight exactly once.
    let needs_token_compacted_weights = merged_num_tsids != num_tsids && use_default_layout;
    let tsid_order_started_at = profile_compaction.then(Instant::now);
    let (tsid_perm, precomputed_token_compaction) = if needs_token_compacted_weights {
        let (token_compacted_weights, precomputed) =
            apply_permutations_to_weight_set_with_token_remaps(
                unique_weights,
                &identity_perm(num_tsids),
                &token_perm,
                token_remaps,
            );
        let tsid_perm = if adaptive_layout {
            order_tsid_groups_exact_profile(
                &token_compacted_weights,
                tsid_merge_perm,
                merged_num_tsids,
                merged_num_tokens,
            )
        } else {
            order_tsid_groups_sketch(
                &token_compacted_weights,
                tsid_merge_perm,
                merged_num_tsids,
                merged_num_tokens,
            )
        };
        (tsid_perm, precomputed)
    } else {
        let tsid_perm = if merged_num_tsids == num_tsids && keep_unmerged_tsid_identity {
            identity_perm(num_tsids)
        } else {
            tsid_merge_perm
        };
        (
            tsid_perm,
            PrecomputedTokenCompaction {
                token_remaps,
                weight_remaps: HashMap::new(),
            },
        )
    };
    let tsid_order_ms = tsid_order_started_at.map_or(0.0, elapsed_ms);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][mapped_compaction_default_detail] unique_weights={} num_tsids={} num_tokens={} merged_tokens={} merged_tsids={} token_merge_ms={:.3} token_order_ms={:.3} token_remap_ms={:.3} tsid_merge_ms={:.3} tsid_order_ms={:.3} materialized_token_weights={} total_ms={:.3}",
            unique_weights.len(),
            num_tsids,
            num_tokens,
            merged_num_tokens,
            merged_num_tsids,
            token_merge_ms,
            token_order_ms,
            token_remap_ms,
            tsid_merge_ms,
            tsid_order_ms,
            needs_token_compacted_weights,
            elapsed_ms(total_started_at),
        );
    }

    DimensionCompaction {
        tsid_perm,
        ordered_num_tsids: merged_num_tsids,
        token_perm,
        ordered_num_tokens: merged_num_tokens,
        precomputed_token_compaction: Arc::new(precomputed_token_compaction),
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

    let (token_compacted_weights, precomputed_token_compaction) =
        apply_permutations_to_weight_set_with_cache(
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
        precomputed_token_compaction: Arc::new(precomputed_token_compaction),
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
    if profile_words == 1 {
        return build_exact_token_merge_permutation_one_word(&token_sets, num_tokens);
    }
    if exact_token_merge_sweep_enabled() {
        build_exact_token_merge_permutation_multiword_sweep(&token_sets, num_tokens, profile_words)
    } else {
        build_exact_token_merge_permutation_multiword_dense(&token_sets, num_tokens, profile_words)
    }
}

fn exact_token_merge_sweep_enabled() -> bool {
    std::env::var("GLRMASK_EXACT_TOKEN_MERGE_SWEEP")
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true)
}

fn build_exact_token_merge_permutation_multiword_dense(
    token_sets: &[Arc<RangeSetBlaze<u32>>],
    num_tokens: usize,
    profile_words: usize,
) -> (Vec<u32>, usize) {
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

    let mut profile_hash_to_groups = HashMap::<u64, Vec<(Vec<u64>, u32)>>::new();
    let mut perm = vec![0u32; num_tokens];
    let mut next_group = 0u32;
    for token in 0..num_tokens {
        let profile_start = token * profile_words;
        let profile = &profiles[profile_start..profile_start + profile_words];
        let profile_hash = hash_profile_words(profile);
        let groups = profile_hash_to_groups.entry(profile_hash).or_default();
        if let Some((_, group)) = groups
            .iter()
            .find(|(existing_profile, _)| existing_profile.as_slice() == profile)
        {
            perm[token] = *group;
            continue;
        }

        let group = next_group;
        groups.push((profile.to_vec(), group));
        perm[token] = group;
        next_group += 1;
    }

    (perm, next_group as usize)
}

#[derive(Clone, Copy, Debug)]
struct TokenProfileWordEvent {
    pos: usize,
    word: usize,
    bit: u64,
    fingerprint: u64,
    add: bool,
}

#[inline]
fn token_profile_context_fingerprint(word: usize, bit: u64) -> u64 {
    let mut value = ((word as u64) << 6) | bit.trailing_zeros() as u64;
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn intern_active_token_profile(
    active_profile: &[u64],
    profile_fingerprint: u64,
    profile_groups: &mut HashMap<u64, Vec<(Vec<u64>, u32)>>,
    next_group: &mut u32,
) -> u32 {
    let candidates = profile_groups.entry(profile_fingerprint).or_default();
    if let Some((_, group)) = candidates
        .iter()
        .find(|(existing_profile, _)| existing_profile.as_slice() == active_profile)
    {
        return *group;
    }

    let group = *next_group;
    *next_group += 1;
    candidates.push((active_profile.to_vec(), group));
    group
}

fn sort_token_profile_word_events(
    events: &mut Vec<TokenProfileWordEvent>,
    num_tokens: usize,
) {
    // Counting by token position avoids comparison sorting for dense boundary
    // sets. Event order within a position is intentionally irrelevant: the
    // sweep processes all removals, then all additions, and bit operations
    // commute within either phase.
    let use_counting_sort = events.len() > num_tokens.saturating_mul(2);
    if !use_counting_sort {
        events.sort_unstable_by_key(|event| (event.pos, event.add, event.word, event.bit));
        return;
    }

    let mut counts = vec![0usize; num_tokens];
    for event in events.iter() {
        counts[event.pos] += 1;
    }
    let mut offsets = vec![0usize; num_tokens + 1];
    for position in 0..num_tokens {
        offsets[position + 1] = offsets[position] + counts[position];
    }
    let mut next = offsets[..num_tokens].to_vec();
    let mut ordered = vec![
        TokenProfileWordEvent {
            pos: 0,
            word: 0,
            bit: 0,
            fingerprint: 0,
            add: false,
        };
        events.len()
    ];
    for &event in events.iter() {
        let slot = next[event.pos];
        ordered[slot] = event;
        next[event.pos] += 1;
    }
    *events = ordered;
}

fn build_exact_token_merge_permutation_multiword_sweep(
    token_sets: &[Arc<RangeSetBlaze<u32>>],
    num_tokens: usize,
    profile_words: usize,
) -> (Vec<u32>, usize) {
    let mut events = Vec::<TokenProfileWordEvent>::new();
    for (context, token_set) in token_sets.iter().enumerate() {
        let word = context / 64;
        let bit = 1u64 << (context % 64);
        let fingerprint = token_profile_context_fingerprint(word, bit);
        for token_range in token_set.ranges() {
            let start = (*token_range.start() as usize).min(num_tokens);
            let end = (*token_range.end() as usize).min(num_tokens.saturating_sub(1));
            if start > end {
                continue;
            }
            events.push(TokenProfileWordEvent {
                pos: start,
                word,
                bit,
                fingerprint,
                add: true,
            });
            let remove_pos = end + 1;
            if remove_pos < num_tokens {
                events.push(TokenProfileWordEvent {
                    pos: remove_pos,
                    word,
                    bit,
                    fingerprint,
                    add: false,
                });
            }
        }
    }

    if events.is_empty() {
        return (vec![0; num_tokens], 1);
    }

    // At a boundary, remove old membership before adding new membership. This
    // also makes the behavior match the one-word sweep implementation.
    sort_token_profile_word_events(&mut events, num_tokens);

    let mut active_profile = vec![0u64; profile_words];
    let mut active_profile_fingerprint = 0u64;
    let mut profile_groups = HashMap::<u64, Vec<(Vec<u64>, u32)>>::new();
    let mut perm = vec![0u32; num_tokens];
    let mut next_group = 0u32;
    let mut cursor = 0usize;
    let mut idx = 0usize;

    while idx < events.len() {
        let pos = events[idx].pos;
        if cursor < pos {
            let group = intern_active_token_profile(
                &active_profile,
                active_profile_fingerprint,
                &mut profile_groups,
                &mut next_group,
            );
            perm[cursor..pos].fill(group);
            cursor = pos;
        }

        let bucket_start = idx;
        while idx < events.len() && events[idx].pos == pos {
            idx += 1;
        }
        for event in &events[bucket_start..idx] {
            if !event.add {
                active_profile[event.word] &= !event.bit;
                active_profile_fingerprint ^= event.fingerprint;
            }
        }
        for event in &events[bucket_start..idx] {
            if event.add {
                active_profile[event.word] |= event.bit;
                active_profile_fingerprint ^= event.fingerprint;
            }
        }
    }

    if cursor < num_tokens {
        let group = intern_active_token_profile(
            &active_profile,
            active_profile_fingerprint,
            &mut profile_groups,
            &mut next_group,
        );
        perm[cursor..num_tokens].fill(group);
    }

    (perm, next_group as usize)
}

#[derive(Clone, Copy, Debug)]
struct TokenProfileEvent {
    pos: usize,
    bit: u64,
    add: bool,
}

fn build_exact_token_merge_permutation_one_word(
    token_sets: &[Arc<RangeSetBlaze<u32>>],
    num_tokens: usize,
) -> (Vec<u32>, usize) {
    let mut events = Vec::new();
    for (context, token_set) in token_sets.iter().enumerate() {
        let bit = 1u64 << context;
        for token_range in token_set.ranges() {
            let start = (*token_range.start() as usize).min(num_tokens);
            let end = (*token_range.end() as usize).min(num_tokens.saturating_sub(1));
            if start > end {
                continue;
            }
            events.push(TokenProfileEvent {
                pos: start,
                bit,
                add: true,
            });
            let remove_pos = end + 1;
            if remove_pos < num_tokens {
                events.push(TokenProfileEvent {
                    pos: remove_pos,
                    bit,
                    add: false,
                });
            }
        }
    }

    if events.is_empty() {
        return (vec![0; num_tokens], 1);
    }

    events.sort_unstable_by_key(|event| (event.pos, event.add));

    let mut profile_to_group = HashMap::<u64, u32>::new();
    let mut perm = vec![0u32; num_tokens];
    let mut next_group = 0u32;
    let mut active_profile = 0u64;
    let mut cursor = 0usize;
    let mut idx = 0usize;

    while idx < events.len() {
        let pos = events[idx].pos;
        if cursor < pos {
            let group = *profile_to_group.entry(active_profile).or_insert_with(|| {
                let group = next_group;
                next_group += 1;
                group
            });
            perm[cursor..pos].fill(group);
            cursor = pos;
        }

        while idx < events.len() && events[idx].pos == pos && !events[idx].add {
            active_profile &= !events[idx].bit;
            idx += 1;
        }
        while idx < events.len() && events[idx].pos == pos && events[idx].add {
            active_profile |= events[idx].bit;
            idx += 1;
        }
    }

    if cursor < num_tokens {
        let group = *profile_to_group.entry(active_profile).or_insert_with(|| {
            let group = next_group;
            next_group += 1;
            group
        });
        perm[cursor..num_tokens].fill(group);
    }

    (perm, next_group as usize)
}

fn hash_profile_words(profile: &[u64]) -> u64 {
    let mut hash = 0x9e37_79b9_7f4a_7c15 ^ profile.len() as u64;
    for &word in profile {
        hash ^= mix_profile_word(word.wrapping_add(0x9e37_79b9_7f4a_7c15));
        hash = hash.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    hash
}

fn mix_profile_word(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Compute the exact TSID merge profile after a token permutation without
/// first materializing a token-remapped copy of every weight. The materialized
/// path below exists for TSID ordering; this path is sufficient for exact
/// merging and preserves the same per-weight context numbering.
fn build_exact_tsid_merge_permutation_with_token_remaps(
    weights: &[&Weight],
    num_tsids: usize,
    token_context: &PermutationContext,
    token_remaps: &TokenRemapCache,
) -> (Vec<u32>, usize) {
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
        // Inner token-set rangesets are interned, so an identical `Arc` reached
        // at several tsid ranges within this weight always denotes the same
        // (remapped) token content. Cache the resolved context per source `Arc`
        // pointer to skip recomputing the remap key for those repeats; this is
        // exact and only avoids redundant `rangeset_key` allocations.
        let mut context_by_source_ptr = HashMap::<usize, u32>::new();
        for (tsid_range, token_set) in weight.0.range_values() {
            let token_set_ptr = Arc::as_ptr(token_set) as usize;
            let token_set_context = match context_by_source_ptr.entry(token_set_ptr) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let remapped_tokens = if token_context.token_perm_is_identity {
                        token_set.as_ref()
                    } else {
                        token_remaps
                            .get(&token_set_ptr)
                            .expect("token plan must remap every source token set")
                    };
                    let key = rangeset_key(remapped_tokens);
                    let resolved = *contexts_by_token_set.entry(key).or_insert_with(|| {
                        let current = context;
                        context += 1;
                        current
                    });
                    entry.insert(resolved);
                    resolved
                }
            };
            let start = *tsid_range.start();
            let end = (*tsid_range.end()).min(num_tsids.saturating_sub(1) as u32);
            for tsid in start..=end {
                profiles[tsid as usize].push(token_set_context);
            }
        }
    }

    build_profile_merge_permutation(&profiles)
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
    apply_permutations_to_weight_set_with_cache(weights, tsid_perm, token_perm).0
}

fn apply_permutations_to_weight_set_with_cache(
    weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
) -> (Vec<Weight>, PrecomputedTokenCompaction) {
    let perm_context = PermutationContext::new(tsid_perm, token_perm);
    let token_remaps = build_global_permuted_token_cache(weights, &perm_context);
    apply_permutations_to_weight_set_with_token_remaps(
        weights,
        tsid_perm,
        token_perm,
        token_remaps,
    )
}

fn apply_permutations_to_weight_set_with_token_remaps(
    weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
    token_remaps: TokenRemapCache,
) -> (Vec<Weight>, PrecomputedTokenCompaction) {
    let perm_context = PermutationContext::new(tsid_perm, token_perm);
    let empty_cache = TokenRemapCache::new();
    let weight_entries: Vec<(usize, Weight)> = weights
        .iter()
        .map(|weight| {
            (
                Arc::as_ptr(&weight.0) as usize,
                permute_weight_with_caches(weight, &perm_context, &empty_cache, &token_remaps),
            )
        })
        .collect();
    let weight_remaps: HashMap<usize, Weight> = weight_entries.iter().cloned().collect();
    let compacted = dedup_weights_by_storage_ptr(
        weight_entries.into_iter().map(|(_, weight)| weight).collect(),
    );
    (
        compacted,
        PrecomputedTokenCompaction {
            token_remaps,
            weight_remaps,
        },
    )
}

fn apply_permutations_to_weight_refs(
    weights: &mut [&mut Weight],
    unique_weights: &[Weight],
    tsid_perm: &[u32],
    token_perm: &[u32],
    precomputed_token_compaction: &PrecomputedTokenCompaction,
) {
    let profile_compaction = env_flag("GLRMASK_PROFILE_COMPILE");
    let total_started_at = profile_compaction.then(Instant::now);
    let perm_context = PermutationContext::new(tsid_perm, token_perm);
    let direct_weight_remaps = perm_context
        .tsid_perm_is_identity
        .then_some(&precomputed_token_compaction.weight_remaps);
    // The plan itself applied the token permutation to these exact source
    // weights in order to derive the TSID compaction. When TSIDs still need
    // reordering, reuse that materialized intermediate instead of remapping
    // every token set a second time.
    let tsid_only_context = (!perm_context.tsid_perm_is_identity
        && !perm_context.token_perm_is_identity)
        .then(|| PermutationContext::new(tsid_perm, &identity_perm(token_perm.len())));
    let needs_token_fallback = !perm_context.token_perm_is_identity
        && unique_weights.iter().any(|weight| {
            let ptr = Arc::as_ptr(&weight.0) as usize;
            direct_weight_remaps
                .is_none_or(|remaps| !remaps.contains_key(&ptr))
                && tsid_only_context
                    .as_ref()
                    .is_none_or(|_| !precomputed_token_compaction.weight_remaps.contains_key(&ptr))
        });

    let cache_started_at = profile_compaction.then(Instant::now);
    let fallback_token_cache = if needs_token_fallback {
        build_missing_global_permuted_token_cache(
            unique_weights,
            &perm_context,
            &precomputed_token_compaction.token_remaps,
        )
    } else {
        TokenRemapCache::new()
    };
    let fallback_cache_ms = cache_started_at.map_or(0.0, elapsed_ms);

    let transform_started_at = profile_compaction.then(Instant::now);
    let empty_token_cache = TokenRemapCache::new();
    let permute_entry = |weight: &Weight| {
        let ptr = Arc::as_ptr(&weight.0) as usize;
        let new_weight = if let Some(remaps) = direct_weight_remaps
            && let Some(remapped) = remaps.get(&ptr)
        {
            remapped.clone()
        } else if let Some(tsid_only_context) = &tsid_only_context
            && let Some(token_remapped) = precomputed_token_compaction.weight_remaps.get(&ptr)
        {
            permute_weight_with_caches(
                token_remapped,
                tsid_only_context,
                &empty_token_cache,
                &empty_token_cache,
            )
        } else {
            permute_weight_with_caches(
                weight,
                &perm_context,
                &precomputed_token_compaction.token_remaps,
                &fallback_token_cache,
            )
        };
        (ptr, new_weight)
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
        let ptr = Arc::as_ptr(&weight.0) as usize;
        if let Some(new_weight) = weight_map.get(&ptr) {
            **weight = new_weight.clone();
        }
    }
    let writeback_ms = writeback_started_at.map_or(0.0, elapsed_ms);

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][mapped_compaction_apply_detail] weights={} unique_weights={} planned_token_sets={} planned_weight_remaps={} direct_weight_remaps={} fallback_token_sets={} fallback_cache_ms={:.3} transform_ms={:.3} map_ms={:.3} writeback_ms={:.3} total_ms={:.3}",
            weights.len(),
            unique_weights.len(),
            precomputed_token_compaction.token_remaps.len(),
            precomputed_token_compaction.weight_remaps.len(),
            direct_weight_remaps.is_some(),
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
    if perm_context.tsid_perm_is_identity {
        if perm_context.token_perm_is_identity {
            return weight.clone();
        }

        let mut map = RangeMapBlaze::new();
        for (tsid_range, token_set) in weight.0.range_values() {
            let token_set_ptr = Arc::as_ptr(token_set) as usize;
            let mapped_tokens = lookup_permuted_token_set(
                token_set,
                token_set_ptr,
                perm_context,
                precomputed_token_remaps,
                fallback_token_cache,
            );
            map.extend_simple(std::iter::once((
                *tsid_range.start()..=*tsid_range.end(),
                mapped_tokens,
            )));
        }
        return finalize_weight_map(map);
    }

    if perm_context.tsid_perm_is_bijection {
        return permute_weight_with_bijective_tsid_perm(
            weight,
            perm_context,
            precomputed_token_remaps,
            fallback_token_cache,
        );
    }

    permute_weight_with_general_tsid_remap(
        weight,
        perm_context,
        precomputed_token_remaps,
        fallback_token_cache,
    )
}

/// Fast exact path for a pure TSID permutation. The generic implementation
/// uses a dense output slot for every TSID because it must union colliding
/// source IDs. A bijection cannot collide, so rebuilding from the sparse
/// weight entries gives the same RangeMap without the O(total-TSIDs) scratch
/// allocation per weight.
fn permute_weight_with_bijective_tsid_perm(
    weight: &Weight,
    perm_context: &PermutationContext,
    precomputed_token_remaps: &TokenRemapCache,
    fallback_token_cache: &TokenRemapCache,
) -> Weight {
    debug_assert!(perm_context.tsid_perm_is_bijection);
    let mut entries = Vec::<(u32, Arc<RangeSetBlaze<u32>>)>::new();
    for (tsid_range, token_set) in weight.0.range_values() {
        let token_set_ptr = Arc::as_ptr(token_set) as usize;
        let mapped_tokens = if perm_context.token_perm_is_identity {
            Arc::clone(token_set)
        } else {
            lookup_permuted_token_set(
                token_set,
                token_set_ptr,
                perm_context,
                precomputed_token_remaps,
                fallback_token_cache,
            )
        };
        for run in overlapping_perm_runs(
            &perm_context.tsid_runs,
            *tsid_range.start(),
            *tsid_range.end(),
        ) {
            debug_assert_eq!(run.start, run.end);
            entries.push((run.mapped, Arc::clone(&mapped_tokens)));
        }
    }
    entries.sort_unstable_by_key(|(tsid, _)| *tsid);

    let mut map = RangeMapBlaze::new();
    let mut current: Option<(u32, u32, Arc<RangeSetBlaze<u32>>)> = None;
    for (tsid, tokens) in entries {
        match current.as_mut() {
            Some((_, end, current_tokens))
                if *end != u32::MAX
                    && *end + 1 == tsid
                    && (Arc::ptr_eq(current_tokens, &tokens)
                        || current_tokens.as_ref() == tokens.as_ref()) =>
            {
                *end = tsid;
            }
            Some(_) => {
                let (start, end, previous_tokens) = current.take().unwrap();
                map.extend_simple(std::iter::once((start..=end, previous_tokens)));
                current = Some((tsid, tsid, tokens));
            }
            None => current = Some((tsid, tsid, tokens)),
        }
    }
    if let Some((start, end, tokens)) = current {
        map.extend_simple(std::iter::once((start..=end, tokens)));
    }
    finalize_weight_map(map)
}

fn permute_weight_with_general_tsid_remap(
    weight: &Weight,
    perm_context: &PermutationContext,
    precomputed_token_remaps: &TokenRemapCache,
    fallback_token_cache: &TokenRemapCache,
) -> Weight {
    // A non-bijective TSID map can collide source coordinates, but each weight
    // normally covers only a sparse set of permutation runs. Collect those
    // point contributions and use Weight's exact sorted reducer instead of
    // clearing a dense output slot for every destination TSID per weight.
    let mut entries = Vec::<(u32, SharedTokenSet)>::new();
    for (tsid_range, token_set) in weight.0.range_values() {
        let token_set_ptr = Arc::as_ptr(token_set) as usize;
        let mapped_tokens = if perm_context.token_perm_is_identity {
            Arc::clone(token_set)
        } else {
            lookup_permuted_token_set(
                token_set,
                token_set_ptr,
                perm_context,
                precomputed_token_remaps,
                fallback_token_cache,
            )
        };
        for run in overlapping_perm_runs(
            &perm_context.tsid_runs,
            *tsid_range.start(),
            *tsid_range.end(),
        ) {
            entries.push((run.mapped, Arc::clone(&mapped_tokens)));
        }
    }
    entries.sort_unstable_by_key(|(tsid, _)| *tsid);
    Weight::union_sorted_point_entries(entries)
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
        .cloned()
        .unwrap_or_else(|| shared_rangeset(permute_rangeset_with_runs(token_set, &perm_context.token_runs)))
}

fn build_global_permuted_token_cache(
    weights: &[Weight],
    perm_context: &PermutationContext,
) -> TokenRemapCache {
    build_missing_global_permuted_token_cache(weights, perm_context, &TokenRemapCache::new())
}

fn build_missing_global_permuted_token_cache(
    weights: &[Weight],
    perm_context: &PermutationContext,
    precomputed_token_remaps: &TokenRemapCache,
) -> TokenRemapCache {
    if perm_context.token_perm_is_identity {
        return TokenRemapCache::new();
    }

    let mut seen = HashSet::new();
    let mut token_sets = Vec::new();
    for weight in weights {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        for (_tsid_range, token_set) in weight.0.range_values() {
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
                    &perm_context.token_runs,
                    *range.start(),
                    *range.end(),
                )
                .len()
            })
            .sum::<usize>()
    });
    let remap_started_at = profile_compaction.then(Instant::now);
    let permute_token_set = |(ptr, token_set): &(usize, Arc<RangeSetBlaze<u32>>)| {
        (
            *ptr,
            shared_rangeset(permute_rangeset_with_runs(token_set, &perm_context.token_runs)),
        )
    };
    let remaps: TokenRemapCache = if rayon::current_num_threads() == 1 {
        token_sets.iter().map(permute_token_set).collect()
    } else {
        token_sets.par_iter().map(permute_token_set).collect()
    };
    if let Some(remap_started_at) = remap_started_at {
        let output_range_count: usize = remaps.values().map(|set| set.as_ref().ranges().count()).sum();
        eprintln!(
            "[glrmask/profile][mapped_compaction_token_remap] token_sets={} source_ranges={} perm_runs={} overlap_runs={} output_ranges={} remap_ms={:.3}",
            token_sets.len(),
            source_range_count.unwrap_or(0),
            perm_context.token_runs.len(),
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

fn build_weight_map_from_tsid_tokens(
    tokens_by_tsid: Vec<Option<RangeSetBlaze<u32>>>,
) -> RangeMapBlaze<u32, SharedTokenSet> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn token_set(ranges: &[(u32, u32)]) -> Arc<RangeSetBlaze<u32>> {
        Arc::new(ranges.iter().map(|&(start, end)| start..=end).collect())
    }

    #[test]
    fn sparse_bijective_tsid_permutation_matches_general_remap() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([2..=2])),
            (3, RangeSetBlaze::from_iter([1..=3])),
            (4, RangeSetBlaze::from_iter([0..=0])),
        ]);
        let context = PermutationContext::new(&[3, 0, 4, 1, 2], &[1, 4, 0, 3, 2]);
        assert!(context.tsid_perm_is_bijection);
        let direct = permute_weight_with_bijective_tsid_perm(
            &weight,
            &context,
            &TokenRemapCache::new(),
            &TokenRemapCache::new(),
        );
        let general = permute_weight_with_general_tsid_remap(
            &weight,
            &context,
            &TokenRemapCache::new(),
            &TokenRemapCache::new(),
        );
        assert_eq!(direct, general);
    }

    #[test]
    fn shared_token_remap_cache_matches_fallback_and_reuses_storage() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=0])),
            (1, RangeSetBlaze::from_iter([1..=1])),
            (2, RangeSetBlaze::from_iter([2..=2])),
        ]);
        let context = PermutationContext::new(&[0, 1, 2], &[2, 0, 1]);
        let cache = build_global_permuted_token_cache(&[weight.clone()], &context);
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
            weight.0.range_values().zip(cached.0.range_values())
        {
            let source_ptr = Arc::as_ptr(source_tokens) as usize;
            assert!(Arc::ptr_eq(cached_tokens, cache.get(&source_ptr).unwrap()));
        }
    }

    #[test]
    fn direct_tsid_profiles_match_materialized_token_compaction() {
        let weights = vec![
            Weight::from_per_tsid_token_sets([
                (0, RangeSetBlaze::from_iter([0..=0])),
                (1, RangeSetBlaze::from_iter([1..=1])),
                (2, RangeSetBlaze::from_iter([2..=2])),
                (3, RangeSetBlaze::from_iter([3..=3])),
            ]),
            Weight::from_per_tsid_token_sets([
                (0, RangeSetBlaze::from_iter([0..=1])),
                (1, RangeSetBlaze::from_iter([0..=1])),
                (2, RangeSetBlaze::from_iter([2..=3])),
                (3, RangeSetBlaze::from_iter([2..=3])),
            ]),
        ];
        let tsid_perm = identity_perm(4);
        let token_perm = vec![0, 0, 1, 1];
        let token_context = PermutationContext::new(&tsid_perm, &token_perm);
        let original_refs = weight_refs(&weights);
        let token_remaps = build_global_permuted_token_cache(&weights, &token_context);
        let direct = build_exact_tsid_merge_permutation_with_token_remaps(
            &original_refs,
            4,
            &token_context,
            &token_remaps,
        );

        let (materialized, _) = apply_permutations_to_weight_set_with_cache(
            &weights,
            &tsid_perm,
            &token_perm,
        );
        let materialized_refs = weight_refs(&materialized);
        let expected = build_exact_tsid_merge_permutation(&materialized_refs, 4);

        assert_eq!(direct, expected);
    }

    #[test]
    fn multiword_token_profile_sweep_matches_dense_permutation() {
        const NUM_TOKENS: usize = 257;
        let token_sets: Vec<Arc<RangeSetBlaze<u32>>> = (0..137u32)
            .map(|context| {
                let first = (context.wrapping_mul(29) % NUM_TOKENS as u32) as u32;
                let second = (context.wrapping_mul(47).wrapping_add(11) % NUM_TOKENS as u32) as u32;
                token_set(&[
                    (first, (first + context % 19).min(NUM_TOKENS as u32 - 1)),
                    (second, (second + context % 7).min(NUM_TOKENS as u32 - 1)),
                ])
            })
            .collect();
        let profile_words = token_sets.len().div_ceil(64);

        let dense = build_exact_token_merge_permutation_multiword_dense(
            &token_sets,
            NUM_TOKENS,
            profile_words,
        );
        let sweep = build_exact_token_merge_permutation_multiword_sweep(
            &token_sets,
            NUM_TOKENS,
            profile_words,
        );

        assert_eq!(sweep, dense);
    }
}
