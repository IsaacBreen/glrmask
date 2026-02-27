use crate::constraint::{GrammarConstraintState, TerminalAllowanceCheckMode};
use crate::datastructures::hybrid_bitset::RangeSet;
use crate::datastructures::leveled_gss::LeveledGSS;
use crate::glr::parser::{GLRParserState, ParseStateEdgeContent};
use crate::glr::table::TerminalID;
use crate::dwa_i32::common::{Label, StateID as WAStateID};

use crate::datastructures::abstract_weight::AbstractWeight;
use crate::dfa_u8::TokenizerStateID;
use profiler_macro::time_it;
use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use crate::datastructures::bitset::Bitset;
use crate::datastructures::gss_acc::TerminalsDisallowed;

type ParserGSS = LeveledGSS<ParseStateEdgeContent, TerminalsDisallowed>;

// Benchmark mode for capturing Rust-native timings without Python overhead
static BENCHMARK_MODE: AtomicBool = AtomicBool::new(false);
static LAST_MASK_TIME_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_COMPUTE_TIME_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_CONVERT_TIME_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_EOS_TIME_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_SEED_TIME_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WORKLIST_TIME_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WORKLIST_ITER_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_EXPAND_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_INTERSECT_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_GSS_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_MERGE_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_FINAL_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_EXPAND_COUNT: AtomicU64 = AtomicU64::new(0);

// Fine-grained sub-counters
static LAST_MASK_WL_FINAL_INTERSECT_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_FINAL_COLLAPSE_NS: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_FINAL_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_INTERSECT_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_MAX_WEIGHT_RANGES: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_TOTAL_WEIGHT_RANGES: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_MAX_DWA_WEIGHT_RANGES: AtomicU64 = AtomicU64::new(0);
static LAST_MASK_WL_TOTAL_DWA_WEIGHT_RANGES: AtomicU64 = AtomicU64::new(0);

/// Enable benchmark mode which captures precise timing inside Rust.
/// Call get_last_mask_time_ns() after each fill_mask_i32 call.
pub fn set_benchmark_mode(enabled: bool) {
    BENCHMARK_MODE.store(enabled, Ordering::Relaxed);
    LAST_MASK_TIME_NS.store(0, Ordering::Relaxed);
    LAST_MASK_COMPUTE_TIME_NS.store(0, Ordering::Relaxed);
    LAST_MASK_CONVERT_TIME_NS.store(0, Ordering::Relaxed);
    LAST_MASK_EOS_TIME_NS.store(0, Ordering::Relaxed);
    LAST_MASK_SEED_TIME_NS.store(0, Ordering::Relaxed);
    LAST_MASK_WORKLIST_TIME_NS.store(0, Ordering::Relaxed);
    LAST_MASK_WORKLIST_ITER_COUNT.store(0, Ordering::Relaxed);
    LAST_MASK_WL_EXPAND_NS.store(0, Ordering::Relaxed);
    LAST_MASK_WL_INTERSECT_NS.store(0, Ordering::Relaxed);
    LAST_MASK_WL_GSS_NS.store(0, Ordering::Relaxed);
    LAST_MASK_WL_MERGE_NS.store(0, Ordering::Relaxed);
    LAST_MASK_WL_FINAL_NS.store(0, Ordering::Relaxed);
    LAST_MASK_WL_EXPAND_COUNT.store(0, Ordering::Relaxed);
    LAST_MASK_WL_FINAL_INTERSECT_NS.store(0, Ordering::Relaxed);
    LAST_MASK_WL_FINAL_COLLAPSE_NS.store(0, Ordering::Relaxed);
    LAST_MASK_WL_FINAL_COUNT.store(0, Ordering::Relaxed);
    LAST_MASK_WL_INTERSECT_COUNT.store(0, Ordering::Relaxed);
    LAST_MASK_WL_MAX_WEIGHT_RANGES.store(0, Ordering::Relaxed);
    LAST_MASK_WL_TOTAL_WEIGHT_RANGES.store(0, Ordering::Relaxed);
    LAST_MASK_WL_MAX_DWA_WEIGHT_RANGES.store(0, Ordering::Relaxed);
    LAST_MASK_WL_TOTAL_DWA_WEIGHT_RANGES.store(0, Ordering::Relaxed);
}

/// Get the last total mask computation time in nanoseconds.
/// Only valid if benchmark mode is enabled.
pub fn get_last_mask_time_ns() -> u64 {
    LAST_MASK_TIME_NS.load(Ordering::Relaxed)
}

/// Get the last `compute_internal_mask*` phase time in nanoseconds.
pub fn get_last_mask_compute_time_ns() -> u64 {
    LAST_MASK_COMPUTE_TIME_NS.load(Ordering::Relaxed)
}

/// Get the last `fill_internal_bv_to_original_i32` phase time in nanoseconds.
pub fn get_last_mask_convert_time_ns() -> u64 {
    LAST_MASK_CONVERT_TIME_NS.load(Ordering::Relaxed)
}

/// Get the last EOS post-processing phase time in nanoseconds.
pub fn get_last_mask_eos_time_ns() -> u64 {
    LAST_MASK_EOS_TIME_NS.load(Ordering::Relaxed)
}

/// Get the last seed phase time in nanoseconds.
pub fn get_last_mask_seed_time_ns() -> u64 {
    LAST_MASK_SEED_TIME_NS.load(Ordering::Relaxed)
}

/// Get the last main worklist phase time in nanoseconds.
pub fn get_last_mask_worklist_time_ns() -> u64 {
    LAST_MASK_WORKLIST_TIME_NS.load(Ordering::Relaxed)
}

/// Get the last main worklist iteration count.
pub fn get_last_mask_worklist_iter_count() -> u64 {
    LAST_MASK_WORKLIST_ITER_COUNT.load(Ordering::Relaxed)
}

/// Get the last worklist expand time in nanoseconds.
pub fn get_last_mask_wl_expand_ns() -> u64 {
    LAST_MASK_WL_EXPAND_NS.load(Ordering::Relaxed)
}

/// Get the last worklist intersect time in nanoseconds.
pub fn get_last_mask_wl_intersect_ns() -> u64 {
    LAST_MASK_WL_INTERSECT_NS.load(Ordering::Relaxed)
}

/// Get the last worklist GSS ops time in nanoseconds.
pub fn get_last_mask_wl_gss_ns() -> u64 {
    LAST_MASK_WL_GSS_NS.load(Ordering::Relaxed)
}

/// Get the last worklist merge time in nanoseconds.
pub fn get_last_mask_wl_merge_ns() -> u64 {
    LAST_MASK_WL_MERGE_NS.load(Ordering::Relaxed)
}

/// Get the last worklist final weight time in nanoseconds.
pub fn get_last_mask_wl_final_ns() -> u64 {
    LAST_MASK_WL_FINAL_NS.load(Ordering::Relaxed)
}

/// Get the count of expand operations in last worklist.
pub fn get_last_mask_wl_expand_count() -> u64 {
    LAST_MASK_WL_EXPAND_COUNT.load(Ordering::Relaxed)
}

/// Get the final-weight intersection time (subset of wl_final).
pub fn get_last_mask_wl_final_intersect_ns() -> u64 {
    LAST_MASK_WL_FINAL_INTERSECT_NS.load(Ordering::Relaxed)
}

/// Get the final-weight collapse time (subset of wl_final).
pub fn get_last_mask_wl_final_collapse_ns() -> u64 {
    LAST_MASK_WL_FINAL_COLLAPSE_NS.load(Ordering::Relaxed)
}

/// Get the number of final-weight computations in last worklist.
pub fn get_last_mask_wl_final_count() -> u64 {
    LAST_MASK_WL_FINAL_COUNT.load(Ordering::Relaxed)
}

/// Get the count of intersect (apply_and_prune) calls in the worklist.
pub fn get_last_mask_wl_intersect_count() -> u64 {
    LAST_MASK_WL_INTERSECT_COUNT.load(Ordering::Relaxed)
}

/// Get the maximum weight size (in ranges) encountered during the worklist.
pub fn get_last_mask_wl_max_weight_ranges() -> u64 {
    LAST_MASK_WL_MAX_WEIGHT_RANGES.load(Ordering::Relaxed)
}

/// Get the total weight ranges processed during the worklist.
pub fn get_last_mask_wl_total_weight_ranges() -> u64 {
    LAST_MASK_WL_TOTAL_WEIGHT_RANGES.load(Ordering::Relaxed)
}
pub fn get_last_mask_wl_max_dwa_weight_ranges() -> u64 {
    LAST_MASK_WL_MAX_DWA_WEIGHT_RANGES.load(Ordering::Relaxed)
}
pub fn get_last_mask_wl_total_dwa_weight_ranges() -> u64 {
    LAST_MASK_WL_TOTAL_DWA_WEIGHT_RANGES.load(Ordering::Relaxed)
}

impl<'a> GrammarConstraintState<'a> {
    /// Expose compute_internal_mask for testing/debugging.
    #[cfg(test)]
    pub fn compute_internal_mask_debug(&self) -> RangeSet {
        let (mask, _) = self.compute_internal_mask();
        mask
    }
    
    /// Compute the internal mask (RangeSet of internal token IDs) for the current state.
    /// This is the core computation shared by get_mask and fill_mask_i32.
    fn compute_internal_mask(&self) -> (RangeSet, bool) {
        let benchmark_enabled = BENCHMARK_MODE.load(Ordering::Relaxed);
        let mut final_mask_internal = RangeSet::zeros();
        let mut has_accepting = false;
        if self.state.is_empty() {
            crate::debug!(7, "compute_internal_mask: state is empty");
            if benchmark_enabled {
                LAST_MASK_SEED_TIME_NS.store(0, Ordering::Relaxed);
                LAST_MASK_WORKLIST_TIME_NS.store(0, Ordering::Relaxed);
                LAST_MASK_WORKLIST_ITER_COUNT.store(0, Ordering::Relaxed);
            }
            return (final_mask_internal, has_accepting);
        }

        let mut queue: BTreeMap<isize, BTreeMap<WAStateID, LeveledGSS<ParseStateEdgeContent, RangeSetBlaze<usize>>>> = BTreeMap::new();
        let dwa = &self.parent.parser_dwa;
        let dwa_start_state = &dwa.states[dwa.body.start_state];

        crate::debug!(5, "compute_internal_mask: {} tokenizer states in self.state", self.state.len());
        for (&tsid, glr_state) in &self.state {
            crate::debug!(6, "  tsid={}, stack_empty={}", tsid.0, glr_state.stack.is_empty());
        }

        crate::debug!(5, ">>> Seeding initial states");
        let disable_disallowed_filter = std::env::var("DISABLE_TERMINALS_DISALLOWED_FILTER").is_ok();
        let seed_start = if benchmark_enabled {
            Some(Instant::now())
        } else {
            None
        };
        // 1. Seed initial states
        for (&tokenizer_state_id, glr_state) in &self.state {
            if glr_state.stack.is_empty() {
                continue;
            }

            let gss = glr_state.stack.clone();

            // In symbol-heavy mode, tsid labels are offset by terminals_count
            // to avoid collision with terminal labels (0 to terminals_count-1).
            // This matches the labeling in precompute1.
            let terminals_count = self.parent.parser.terminal_map.len();
            let tsid_label = (tokenizer_state_id.0 + terminals_count) as Label;

            crate::debug!(6, "  Looking for tsid transition: tokenizer_state_id={}, tsid_label={}, {} available transitions",
                tokenizer_state_id.0, tsid_label,
                dwa_start_state.transitions.len());
            if let Some((target_wa_state_id, weight)) = dwa_start_state.get_transition(tsid_label) {
                crate::debug!(6, "    Found transition to state {} with weight {:?}", target_wa_state_id, weight);
                let possible_matches = &self.parent.possible_matches;

                // Convert TerminalsDisallowed -> RangeSetBlaze<usize> (LLM tokens allowed)
                // by computing forbidden tokens and subtracting from weight
                let f = |terminals_disallowed: &TerminalsDisallowed| {
                    // Start with all tokens allowed by the weight
                    let mut allowed = weight.to_rsb_allow_expansion();

                    if !disable_disallowed_filter {
                        // Subtract forbidden tokens based on disallowed terminals
                        for (&tsid, disallowed_in_state) in terminals_disallowed {
                            if disallowed_in_state.is_empty() { continue; }
                            if let Some(state_matches) = possible_matches.get(&TokenizerStateID(tsid)) {
                                for (terminal_id, llm_tokens) in state_matches {
                                    if disallowed_in_state.contains(&terminal_id.0) {
                                        allowed = &allowed - llm_tokens.inner.as_ref();
                                    }
                                }
                            }
                        }
                    }

                    if allowed.is_empty() { None } else { Some(allowed) }
                };
                let weighted_gss = gss.apply_and_prune(f);

                if !weighted_gss.is_empty() {
                    queue
                        .entry(weighted_gss.max_depth())
                        .or_default()
                        .entry(target_wa_state_id)
                        .and_modify(|existing| *existing = existing.merge(&weighted_gss))
                        .or_insert(weighted_gss);
                }
            } else {
                crate::debug!(6, "    NO transition found for tsid_label={}", tsid_label);
            }
        }
        if let Some(start) = seed_start {
            LAST_MASK_SEED_TIME_NS.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        let worklist_start = if benchmark_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let mut worklist_iters: u64 = 0;
        let mut wl_expand_ns: u64 = 0;
        let mut wl_intersect_ns: u64 = 0;
        let mut wl_gss_ns: u64 = 0;
        let mut wl_merge_ns: u64 = 0;
        let mut wl_final_ns: u64 = 0;
        let mut wl_expand_count: u64 = 0;
        // 2. Main worklist loop
        while let Some((depth, states_at_depth)) = queue.pop_last() {
            for (current_wa_state_id, gss) in states_at_depth {
                worklist_iters += 1;
                let dwa_state = &dwa.states[current_wa_state_id];

                // Check for final state
                if let Some(final_weight) = &dwa_state.final_weight {
                    if let Some(reduced_acc) = gss.reduce_acc() {
                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        let final_rsb = final_weight.to_rsb_allow_expansion();
                        if let Some(t0) = t0 { wl_expand_ns += t0.elapsed().as_nanos() as u64; wl_expand_count += 1; }

                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        let final_tokens = &reduced_acc & &final_rsb;
                        if let Some(t0) = t0 { wl_final_ns += t0.elapsed().as_nanos() as u64; }

                        if !final_tokens.is_empty() {
                            has_accepting = true;
                            crate::debug!(7, "Adding {} tokens from final state {}", final_tokens.ranges_len(), current_wa_state_id);
                            final_mask_internal |= RangeSet::from(final_tokens);
                        }
                    }
                }

                // Process transitions
                for peeked_edge in gss.peek() {
                    let parser_state_id = peeked_edge.state_id.0 as Label;
                    if let Some((target_wa_state_id, trans_weight)) = dwa_state.get_transition(parser_state_id) {
                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        let isolated_gss = gss.isolate(Some(peeked_edge));
                        let popped_gss = isolated_gss.pop();
                        if let Some(t0) = t0 { wl_gss_ns += t0.elapsed().as_nanos() as u64; }
                        if popped_gss.is_empty() { continue; }

                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        let expanded = trans_weight.to_rsb_allow_expansion();
                        if let Some(t0) = t0 { wl_expand_ns += t0.elapsed().as_nanos() as u64; wl_expand_count += 1; }

                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        let f = |rsb: &RangeSetBlaze<usize>| {
                            let new_rsb = rsb & &expanded;
                            if new_rsb.is_empty() { None } else { Some(new_rsb) }
                        };
                        let final_gss = popped_gss.apply_and_prune(f);
                        if let Some(t0) = t0 { wl_intersect_ns += t0.elapsed().as_nanos() as u64; }

                        if !final_gss.is_empty() {
                            let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                            queue
                                .entry(final_gss.max_depth())
                                .or_default()
                                .entry(target_wa_state_id)
                                .and_modify(|existing| *existing = existing.merge(&final_gss))
                                .or_insert(final_gss);
                            if let Some(t0) = t0 { wl_merge_ns += t0.elapsed().as_nanos() as u64; }
                        }
                    }

                    if let Some((target_wa_state_id, trans_weight)) = dwa_state.get_transition(crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL) {
                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        let isolated_gss = gss.isolate(Some(peeked_edge));
                        let popped_gss = isolated_gss.pop();
                        if let Some(t0) = t0 { wl_gss_ns += t0.elapsed().as_nanos() as u64; }
                        if popped_gss.is_empty() { continue; }

                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        let expanded = trans_weight.to_rsb_allow_expansion();
                        if let Some(t0) = t0 { wl_expand_ns += t0.elapsed().as_nanos() as u64; wl_expand_count += 1; }

                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        let f = |rsb: &RangeSetBlaze<usize>| {
                            let new_rsb = rsb & &expanded;
                            if new_rsb.is_empty() { None } else { Some(new_rsb) }
                        };
                        let final_gss = popped_gss.apply_and_prune(f);
                        if let Some(t0) = t0 { wl_intersect_ns += t0.elapsed().as_nanos() as u64; }

                        if !final_gss.is_empty() {
                            let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                            queue
                                .entry(final_gss.max_depth())
                                .or_default()
                                .entry(target_wa_state_id)
                                .and_modify(|existing| *existing = existing.merge(&final_gss))
                                .or_insert(final_gss);
                            if let Some(t0) = t0 { wl_merge_ns += t0.elapsed().as_nanos() as u64; }
                        }
                    }
                }
            }
        }
        if let Some(start) = worklist_start {
            LAST_MASK_WORKLIST_TIME_NS.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if benchmark_enabled {
            LAST_MASK_WORKLIST_ITER_COUNT.store(worklist_iters, Ordering::Relaxed);
            LAST_MASK_WL_EXPAND_NS.store(wl_expand_ns, Ordering::Relaxed);
            LAST_MASK_WL_INTERSECT_NS.store(wl_intersect_ns, Ordering::Relaxed);
            LAST_MASK_WL_GSS_NS.store(wl_gss_ns, Ordering::Relaxed);
            LAST_MASK_WL_MERGE_NS.store(wl_merge_ns, Ordering::Relaxed);
            LAST_MASK_WL_FINAL_NS.store(wl_final_ns, Ordering::Relaxed);
            LAST_MASK_WL_EXPAND_COUNT.store(wl_expand_count, Ordering::Relaxed);
        }

        (final_mask_internal, has_accepting)
    }

    /// Compute the internal mask using weight-heavy encoding.
    /// 
    /// In weight-heavy mode:
    /// - num_tsids > 0 indicates weight-heavy encoding
    /// - Weights are in N×M space where position = llm_token * M + tsid
    /// - No tsid transitions at DWA start (all items start at the same start state)
    /// - GSS accumulators carry N×M-space Weight values throughout
    /// - Only collapsed to N-space (vocab) when collecting final mask
    fn compute_internal_mask_weight_heavy(&self) -> (RangeSet, bool) {
        use crate::dwa_i32::common::Weight;

        let benchmark_enabled = BENCHMARK_MODE.load(Ordering::Relaxed);
        let num_tsids = self.parent.num_tsids;
        let max_llm_token = self.parent.parser_dwa_vocab.internal_max_llm_token;
        let mut final_mask_internal = RangeSet::zeros();
        let mut has_accepting = false;
        if self.state.is_empty() {
            if benchmark_enabled {
                LAST_MASK_SEED_TIME_NS.store(0, Ordering::Relaxed);
                LAST_MASK_WORKLIST_TIME_NS.store(0, Ordering::Relaxed);
                LAST_MASK_WORKLIST_ITER_COUNT.store(0, Ordering::Relaxed);
                LAST_MASK_WL_EXPAND_NS.store(0, Ordering::Relaxed);
                LAST_MASK_WL_INTERSECT_NS.store(0, Ordering::Relaxed);
                LAST_MASK_WL_GSS_NS.store(0, Ordering::Relaxed);
                LAST_MASK_WL_MERGE_NS.store(0, Ordering::Relaxed);
                LAST_MASK_WL_FINAL_NS.store(0, Ordering::Relaxed);
                LAST_MASK_WL_EXPAND_COUNT.store(0, Ordering::Relaxed);
            }
            return (final_mask_internal, has_accepting);
        }

        let dwa = &self.parent.parser_dwa;
        let dwa_start_state_id = dwa.body.start_state;

        let disable_disallowed_filter = std::env::var("DISABLE_TERMINALS_DISALLOWED_FILTER").is_ok();
        let seed_start = if benchmark_enabled {
            Some(Instant::now())
        } else {
            None
        };

        // Queue: depth -> (dwa_state -> GSS with N×M-space Weight accumulators)
        let mut queue: BTreeMap<isize, BTreeMap<WAStateID, LeveledGSS<ParseStateEdgeContent, Weight>>> = BTreeMap::new();

        // 1. Seed: For each tokenizer state, compute vocab-space allowed set
        //    (same disallowed-terminal logic as symbol-heavy), expand to N×M,
        //    intersect with per-tsid mask, and seed at DWA start state.
        for (&tokenizer_state_id, glr_state) in &self.state {
            if glr_state.stack.is_empty() {
                continue;
            }

            let gss = glr_state.stack.clone();
            let possible_matches = &self.parent.possible_matches;

            // Build tsid mask for this tokenizer state in N×M space
            // Use internal tsid if mapping is available, otherwise raw state ID
            let internal_tsid = if !self.parent.state_to_internal_tsid.is_empty() {
                self.parent.state_to_internal_tsid[tokenizer_state_id.0]
            } else {
                tokenizer_state_id.0
            };
            let tsid_mask = crate::dwa_i32::weight_expansion::create_tsid_set_mask(
                std::iter::once(internal_tsid),
                num_tsids,
                max_llm_token,
            );

            // Convert GSS accumulator (TerminalsDisallowed) to N×M-space Weight.
            // Compute vocab-space allowed (same as symbol-heavy), expand to N×M,
            // intersect with this entry's tsid mask.
            let f = |terminals_disallowed: &TerminalsDisallowed| {
                // Start with all LLM tokens in vocab-space (same as symbol-heavy)
                let mut allowed: RangeSetBlaze<usize> = RangeSetBlaze::from_iter([0..=max_llm_token]);

                if !disable_disallowed_filter {
                    for (&ts_id, disallowed_in_state) in terminals_disallowed {
                        if disallowed_in_state.is_empty() { continue; }
                        if let Some(state_matches) = possible_matches.get(&TokenizerStateID(ts_id)) {
                            for (terminal_id, llm_tokens) in state_matches {
                                if disallowed_in_state.contains(&terminal_id.0) {
                                    allowed = &allowed - llm_tokens.inner.as_ref();
                                }
                            }
                        }
                    }
                }
                if allowed.is_empty() { return None; }

                // Expand from vocab-space (N) to N×M space
                let expanded = crate::dwa_i32::weight_expansion::expand_rsb(&allowed, num_tsids);
                // Convert to Weight and intersect with tsid mask
                let weight = Weight::from_rsb(expanded) & &tsid_mask;
                if weight.is_empty() { None } else { Some(weight) }
            };
            let weighted_gss = gss.apply_and_prune(f);

            if !weighted_gss.is_empty() {
                queue
                    .entry(weighted_gss.max_depth())
                    .or_default()
                    .entry(dwa_start_state_id)
                    .and_modify(|existing| *existing = existing.merge(&weighted_gss))
                    .or_insert(weighted_gss);
            }
        }
        if let Some(start) = seed_start {
            LAST_MASK_SEED_TIME_NS.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        let worklist_start = if benchmark_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let mut worklist_iters: u64 = 0;
        let mut wl_expand_ns: u64 = 0;
        let mut wl_intersect_ns: u64 = 0;
        let mut wl_gss_ns: u64 = 0;
        let mut wl_merge_ns: u64 = 0;
        let mut wl_final_ns: u64 = 0;
        let mut wl_expand_count: u64 = 0;
        let mut wl_final_intersect_ns: u64 = 0;
        let mut wl_final_collapse_ns: u64 = 0;
        let mut wl_final_count: u64 = 0;
        let mut wl_intersect_count: u64 = 0;
        let mut wl_max_weight_ranges: u64 = 0;
        let mut wl_total_weight_ranges: u64 = 0;
        let mut wl_max_dwa_weight_ranges: u64 = 0;
        let mut wl_total_dwa_weight_ranges: u64 = 0;

        // 2. Main worklist loop — all intersections in N×M space (Weight)
        while let Some((_depth, states_at_depth)) = queue.pop_last() {
            for (current_wa_state_id, gss) in states_at_depth {
                worklist_iters += 1;
                let dwa_state = &dwa.states[current_wa_state_id];

                // Check for final state
                if let Some(final_weight) = &dwa_state.final_weight {
                    if let Some(reduced_acc) = gss.reduce_acc() {
                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        // Intersect accumulated N×M weight with final weight (also N×M)
                        let final_nxm = &reduced_acc & final_weight;
                        if let Some(t0) = t0 {
                            wl_final_intersect_ns += t0.elapsed().as_nanos() as u64;
                            wl_final_count += 1;
                            let nr = reduced_acc.num_ranges() as u64;
                            wl_total_weight_ranges += nr;
                            if nr > wl_max_weight_ranges { wl_max_weight_ranges = nr; }
                            let dwa_nr = final_weight.num_ranges() as u64;
                            wl_total_dwa_weight_ranges += dwa_nr;
                            if dwa_nr > wl_max_dwa_weight_ranges { wl_max_dwa_weight_ranges = dwa_nr; }
                        }
                        if !final_nxm.is_empty() {
                            has_accepting = true;
                            let t1 = if benchmark_enabled { Some(Instant::now()) } else { None };
                            // Collapse from N×M to N-space by unioning along tsid dimension
                            let collapsed = crate::dwa_i32::weight_expansion::collapse_weight(&final_nxm, num_tsids);
                            final_mask_internal |= &RangeSet::from(collapsed.to_rsb_allow_expansion());
                            if let Some(t1) = t1 {
                                wl_final_collapse_ns += t1.elapsed().as_nanos() as u64;
                            }
                        }
                        if let Some(t0) = t0 {
                            wl_final_ns += t0.elapsed().as_nanos() as u64;
                        }
                    }
                }

                // Process transitions
                for peeked_edge in gss.peek() {
                    let parser_state_id = peeked_edge.state_id.0 as Label;

                    // Helper: process a single transition in N×M space
                    let mut process_transition = |target_wa_state_id: WAStateID, trans_weight: &AbstractWeight| {
                        wl_expand_count += 1;

                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        let isolated_gss = gss.isolate(Some(peeked_edge));
                        let popped_gss = isolated_gss.pop();
                        if let Some(t0) = t0 {
                            wl_gss_ns += t0.elapsed().as_nanos() as u64;
                        }
                        if popped_gss.is_empty() { return; }

                        // Intersect GSS weights with transition weight (both N×M)
                        let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                        let f = |acc: &Weight| {
                            let new_acc = acc & trans_weight;
                            if new_acc.is_empty() { None } else { Some(new_acc) }
                        };
                        let final_gss = popped_gss.apply_and_prune(f);
                        if let Some(t0) = t0 {
                            wl_intersect_ns += t0.elapsed().as_nanos() as u64;
                            wl_intersect_count += 1;
                            let dwa_nr = trans_weight.num_ranges() as u64;
                            wl_total_dwa_weight_ranges += dwa_nr;
                            if dwa_nr > wl_max_dwa_weight_ranges { wl_max_dwa_weight_ranges = dwa_nr; }
                        }

                        if !final_gss.is_empty() {
                            let t0 = if benchmark_enabled { Some(Instant::now()) } else { None };
                            queue
                                .entry(final_gss.max_depth())
                                .or_default()
                                .entry(target_wa_state_id)
                                .and_modify(|existing| *existing = existing.merge(&final_gss))
                                .or_insert(final_gss);
                            if let Some(t0) = t0 {
                                wl_merge_ns += t0.elapsed().as_nanos() as u64;
                            }
                        }
                    };

                    if let Some((target_wa_state_id, trans_weight)) = dwa_state.get_transition(parser_state_id) {
                        process_transition(target_wa_state_id, trans_weight);
                    }

                    if let Some((target_wa_state_id, trans_weight)) = dwa_state.get_transition(crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL) {
                        process_transition(target_wa_state_id, trans_weight);
                    }
                }
            }
        }
        if let Some(start) = worklist_start {
            LAST_MASK_WORKLIST_TIME_NS.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if benchmark_enabled {
            LAST_MASK_WORKLIST_ITER_COUNT.store(worklist_iters, Ordering::Relaxed);
            LAST_MASK_WL_EXPAND_NS.store(wl_expand_ns, Ordering::Relaxed);
            LAST_MASK_WL_INTERSECT_NS.store(wl_intersect_ns, Ordering::Relaxed);
            LAST_MASK_WL_GSS_NS.store(wl_gss_ns, Ordering::Relaxed);
            LAST_MASK_WL_MERGE_NS.store(wl_merge_ns, Ordering::Relaxed);
            LAST_MASK_WL_FINAL_NS.store(wl_final_ns, Ordering::Relaxed);
            LAST_MASK_WL_EXPAND_COUNT.store(wl_expand_count, Ordering::Relaxed);
            LAST_MASK_WL_FINAL_INTERSECT_NS.store(wl_final_intersect_ns, Ordering::Relaxed);
            LAST_MASK_WL_FINAL_COLLAPSE_NS.store(wl_final_collapse_ns, Ordering::Relaxed);
            LAST_MASK_WL_FINAL_COUNT.store(wl_final_count, Ordering::Relaxed);
            LAST_MASK_WL_INTERSECT_COUNT.store(wl_intersect_count, Ordering::Relaxed);
            LAST_MASK_WL_MAX_WEIGHT_RANGES.store(wl_max_weight_ranges, Ordering::Relaxed);
            LAST_MASK_WL_TOTAL_WEIGHT_RANGES.store(wl_total_weight_ranges, Ordering::Relaxed);
            LAST_MASK_WL_MAX_DWA_WEIGHT_RANGES.store(wl_max_dwa_weight_ranges, Ordering::Relaxed);
            LAST_MASK_WL_TOTAL_DWA_WEIGHT_RANGES.store(wl_total_dwa_weight_ranges, Ordering::Relaxed);
        }

        (final_mask_internal, has_accepting)
    }

    /// Get the allowed token mask as a dense bitvector.
    ///
    /// This is the main method for getting the allowed tokens mask. It returns
    /// a dense `Bitset` which can be efficiently converted to formats used by
    /// ML frameworks (numpy arrays, torch tensors, etc.).
    ///
    /// For zero-allocation mask filling, see `fill_mask_i32` and `fill_mask_i32_ptr`.
    pub fn get_mask(&self) -> Bitset {
        let (final_mask_internal, _has_accepting) = if self.parent.num_tsids > 0 {
            // Weight-heavy mode: tsid encoded in N×M weights
            self.compute_internal_mask_weight_heavy()
        } else {
            // Symbol-heavy mode: tsid as initial transition labels
            self.compute_internal_mask()
        };
        let mut mask = self.parent.parser_dwa_vocab.internal_bv_to_original(&final_mask_internal);
        if let Some(eos_id) = self.parent.eos_token_id {
            // Treat EOS as a reserved token: only allow it when the parse is complete.
            mask.remove(eos_id);
            if self.is_complete() {
                mask.insert(eos_id);
            }
        }
        mask
    }

    /// Fill an i32 slice with the token mask (compatible with llguidance format).
    ///
    /// This is a zero-allocation version that writes directly to the provided buffer.
    /// The output slice should have length `(vocab_size + 31) / 32`.
    ///
    /// This is the most efficient way to get the mask when you have a pre-allocated
    /// buffer (e.g., numpy array, torch tensor, or reused buffer).
    #[inline]
    pub fn fill_mask_i32(&self, out: &mut [i32]) {
        let benchmark_enabled = BENCHMARK_MODE.load(Ordering::Relaxed);
        if benchmark_enabled {
            LAST_MASK_SEED_TIME_NS.store(0, Ordering::Relaxed);
            LAST_MASK_WORKLIST_TIME_NS.store(0, Ordering::Relaxed);
            LAST_MASK_WORKLIST_ITER_COUNT.store(0, Ordering::Relaxed);
            LAST_MASK_WL_EXPAND_NS.store(0, Ordering::Relaxed);
            LAST_MASK_WL_INTERSECT_NS.store(0, Ordering::Relaxed);
            LAST_MASK_WL_GSS_NS.store(0, Ordering::Relaxed);
            LAST_MASK_WL_MERGE_NS.store(0, Ordering::Relaxed);
            LAST_MASK_WL_FINAL_NS.store(0, Ordering::Relaxed);
            LAST_MASK_WL_EXPAND_COUNT.store(0, Ordering::Relaxed);
            LAST_MASK_WL_FINAL_INTERSECT_NS.store(0, Ordering::Relaxed);
            LAST_MASK_WL_FINAL_COLLAPSE_NS.store(0, Ordering::Relaxed);
            LAST_MASK_WL_FINAL_COUNT.store(0, Ordering::Relaxed);
            LAST_MASK_WL_INTERSECT_COUNT.store(0, Ordering::Relaxed);
            LAST_MASK_WL_MAX_WEIGHT_RANGES.store(0, Ordering::Relaxed);
            LAST_MASK_WL_TOTAL_WEIGHT_RANGES.store(0, Ordering::Relaxed);
            LAST_MASK_WL_MAX_DWA_WEIGHT_RANGES.store(0, Ordering::Relaxed);
            LAST_MASK_WL_TOTAL_DWA_WEIGHT_RANGES.store(0, Ordering::Relaxed);
        }
        let total_start = if benchmark_enabled {
            Some(Instant::now())
        } else {
            None
        };

        let compute_start = if benchmark_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let (final_mask_internal, _has_accepting) = if self.parent.num_tsids > 0 {
            self.compute_internal_mask_weight_heavy()
        } else {
            self.compute_internal_mask()
        };
        if let Some(start) = compute_start {
            LAST_MASK_COMPUTE_TIME_NS.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        let convert_start = if benchmark_enabled {
            Some(Instant::now())
        } else {
            None
        };
        self.parent
            .parser_dwa_vocab
            .fill_internal_bv_to_original_i32(&final_mask_internal, out);
        if let Some(start) = convert_start {
            LAST_MASK_CONVERT_TIME_NS.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        let eos_start = if benchmark_enabled {
            Some(Instant::now())
        } else {
            None
        };
        if let Some(eos_id) = self.parent.eos_token_id {
            let word_idx = eos_id / 32;
            let bit_idx = eos_id % 32;
            if word_idx < out.len() {
                out[word_idx] &= !(1i32 << bit_idx);
                if self.is_complete() {
                    out[word_idx] |= 1i32 << bit_idx;
                }
            }
        }
        if let Some(start) = eos_start {
            LAST_MASK_EOS_TIME_NS.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        if let Some(start) = total_start {
            LAST_MASK_TIME_NS.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    /// Fill an i32 slice with the token mask via a raw pointer.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `ptr` points to at least `len` i32 values of valid, writable memory
    /// - The memory is properly aligned for i32
    /// - No other references to this memory exist during the call
    #[inline]
    pub unsafe fn fill_mask_i32_ptr(&self, ptr: *mut i32, len: usize) {
        let out = std::slice::from_raw_parts_mut(ptr, len);
        self.fill_mask_i32(out);
    }

    /// Returns the required buffer size in i32 elements for the mask.
    #[inline]
    pub fn mask_buffer_size_i32(&self) -> usize {
        self.parent.parser_dwa_vocab.mask_buffer_size_i32()
    }

    #[time_it]
    pub fn commit_bytes(&mut self, llm_token_bytes: &[u8]) {
        if llm_token_bytes.is_empty() {
            return;
        }
        crate::debug!(8, "Committing bytes: {:?}", String::from_utf8_lossy(llm_token_bytes));
        crate::debug!(8, "  Current state tokenizer IDs: {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        crate::debug!(8, "  Current state stacks empty?: {:?}", self.state.iter().map(|(k, v)| (k.0, v.stack.is_empty())).collect::<Vec<_>>());

        let (state_map, terminals_map) = self.compute_commit_maps(llm_token_bytes);
        crate::debug!(8, "  state_map: {:?}", state_map.iter().map(|(k, v)| (k.0, v.0)).collect::<Vec<_>>());
        crate::debug!(8, "  terminals_map: {:?}", terminals_map.iter().map(|(k, v)| (k.0, format!("{:?}", v))).collect::<Vec<_>>());

        // Prune stacks based on matched terminals and remap tokenizer state constraints.
        for glr_state in self.state.values_mut() {
            let mut gss = glr_state.stack.clone();
            // Prune based on matched terminals
            gss = gss.apply_and_prune(|terminals_disallowed: &TerminalsDisallowed| {
                for (sid, matched_terminals) in &terminals_map {
                    if let Some(disallowed) = terminals_disallowed.get(&sid.0) {
                        // Check if any matched terminal is in the disallowed set
                        for t in matched_terminals.iter_indices() {
                            if disallowed.contains(&t) {
                                return None;
                            }
                        }
                    }
                }
                Some(terminals_disallowed.clone())
            });
            // Remap tokenizer states
            gss = gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
                let mut new_terminals_union: TerminalsDisallowed = BTreeMap::new();
                for (old, new) in &state_map {
                    if let Some(disallowed) = terminals_disallowed.get(&old.0) {
                        new_terminals_union.entry(new.0).or_default().extend(disallowed.iter().cloned());
                    }
                }
                new_terminals_union
            });
            glr_state.stack = gss;
        }
        crate::debug!(8, "  After pruning/remapping, state tokenizer IDs: {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        self.state.retain(|_, s| !s.stack.is_empty());
        crate::debug!(8, "  After retain, state tokenizer IDs: {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());

        let mut new_overall_state: BTreeMap<TokenizerStateID, GLRParserState<'a>> = BTreeMap::new();
        let mut processing_queue: BTreeMap<usize, BTreeMap<TokenizerStateID, ParserGSS>> = BTreeMap::new();

        let initial_states: BTreeMap<_,_> = self.state.iter().map(|(sid, s)| (*sid, s.stack.clone())).collect();
        processing_queue.insert(0, initial_states);
        crate::debug!(8, "  Processing queue initial: {:?}", processing_queue.keys().collect::<Vec<_>>());

        while let Some((offset, states_to_process)) = processing_queue.pop_first() {
            crate::debug!(8, "    Processing offset {}, tokenizer states: {:?}", offset, states_to_process.keys().map(|k| k.0).collect::<Vec<_>>());
            for (tokenizer_s_id_at_offset, gss_at_offset) in states_to_process {
                let exec_result = self.parent.tokenizer.execute_from_state(&llm_token_bytes[offset..], tokenizer_s_id_at_offset);
                crate::debug!(8, "      exec_result for tsid {}: end_state={:?}, matches={:?}", tokenizer_s_id_at_offset.0, exec_result.end_state, exec_result.matches.iter().map(|m| (m.id, m.width)).collect::<Vec<_>>());

                for match_info in &exec_result.matches {
                    let mut gss = gss_at_offset.clone();
                    let terminal_id = TerminalID(match_info.id);
                    crate::debug!(8, "        Processing terminal_id={}, width={}", terminal_id.0, match_info.width);

                    gss = self.parent.parser.process_token_gss(&gss, terminal_id);
                    crate::debug!(8, "        After process_token_gss, gss.is_empty()={}", gss.is_empty());

                    if !gss.is_empty() {
                        if let Some(end_state_id) = exec_result.end_state {
                            if self.parent.tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state_id)).contains(&terminal_id) {
                                let terminal_to_disallow = match_info.id;
                                gss = gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
                                    let mut new_td = terminals_disallowed.clone();
                                    new_td.entry(end_state_id).or_default().insert(terminal_to_disallow);
                                    new_td
                                });
                            }
                        }

                        if !gss.is_empty() {
                            let new_offset = offset + match_info.width;
                            let next_tsid = self.parent.tokenizer.initial_state_id();
                            if new_offset == llm_token_bytes.len() {
                                new_overall_state.entry(next_tsid).and_modify(|s| s.stack = s.stack.merge(&gss)).or_insert_with(|| GLRParserState { parser: &self.parent.parser, stack: gss });
                            } else {
                                processing_queue.entry(new_offset).or_default().entry(next_tsid).and_modify(|s| *s = s.merge(&gss)).or_insert(gss);
                            }
                        }
                    }
                }

                if let Some(end_state_id) = exec_result.end_state {
                    let final_tsid = TokenizerStateID(end_state_id);
                    new_overall_state.entry(final_tsid).and_modify(|s| s.stack = s.stack.merge(&gss_at_offset)).or_insert_with(|| GLRParserState { parser: &self.parent.parser, stack: gss_at_offset });
                }
            }
        }

        self.state = new_overall_state;

        // Fuse GSS levels - no longer need to reset llm_tokens_union since we compute it on-the-fly
        for glr_state in self.state.values_mut() {
            glr_state.stack = glr_state.stack.fuse(Some(1));
        }
        self.state.retain(|_, glr_parser_state| glr_parser_state.is_ok());

        crate::debug!(9, "Active tokenizer states after committing text (bytes {:?}): {:?}", llm_token_bytes, self.state.keys().map(|k| k.0).collect::<Vec<_>>());
    }
}
