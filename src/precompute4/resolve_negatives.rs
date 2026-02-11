use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::dwa_i32::common::Label;
use crate::dwa_i32::{DWA, NWA, NWAStateID, NWAStates, Weight};

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;
use once_cell::sync::Lazy;

type Code = Label;
type QueryKey = (NWAStateID, Code);

static PROFILE_PASS2_CANCELLATIONS: Lazy<bool> = Lazy::new(|| {
    std::env::var("PROFILE_PASS2_CANCELLATIONS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
});

static CANCELLATIONS_RANGE_CALLS: AtomicU64 = AtomicU64::new(0);
static CANCELLATIONS_RANGE_TIME_US: AtomicU64 = AtomicU64::new(0);
static CANCELLATIONS_RANGE_STEPS: AtomicU64 = AtomicU64::new(0);
static CANCELLATIONS_RANGE_SEEDS: AtomicU64 = AtomicU64::new(0);
static CANCELLATIONS_RANGE_EPS_ADDED: AtomicU64 = AtomicU64::new(0);

#[inline]
fn cancellations_range_profile_enabled() -> bool {
    *PROFILE_PASS2_CANCELLATIONS
}

pub fn log_cancellations_range_profile() {
    if !cancellations_range_profile_enabled() {
        return;
    }
    let calls = CANCELLATIONS_RANGE_CALLS.load(AtomicOrdering::Relaxed).max(1);
    let total_us = CANCELLATIONS_RANGE_TIME_US.load(AtomicOrdering::Relaxed);
    let steps = CANCELLATIONS_RANGE_STEPS.load(AtomicOrdering::Relaxed);
    let seeds = CANCELLATIONS_RANGE_SEEDS.load(AtomicOrdering::Relaxed);
    let eps_added = CANCELLATIONS_RANGE_EPS_ADDED.load(AtomicOrdering::Relaxed);
    eprintln!(
        "PASS2 cancellations profile: calls={}, total={:?}, avg={:?}, avg_steps={}, avg_seeds={}, avg_eps_added={}",
        calls,
        std::time::Duration::from_micros(total_us),
        std::time::Duration::from_micros(total_us / calls),
        steps / calls,
        seeds / calls,
        eps_added / calls,
    );
}

#[inline]
pub fn is_negative_symbol(label: Code) -> bool { label < 0 && label != DEFAULT_TRANSITION_SYMBOL }

fn progress_step(_step: u64, msg: &str) {
    crate::debug!(5, "{}", msg);
}

pub fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    let all_states: HashSet<NWAStateID> = (0..nwa.states.len()).collect();

    progress_step(1, "Compute cancellations");
    apply_cancellations(&mut nwa.states, &all_states);

    progress_step(2, "Propagate finality");
    apply_finality_fixpoint(&mut nwa.states, &all_states);

    progress_step(3, "Apply changes & remove negatives");
    remove_negative_transitions(&mut nwa.states, &all_states);
    
    progress_step(4, "Shortcut default transitions to terminal states");
    remove_redundant_default_transitions(&mut nwa.states, &all_states);
    
    crate::debug!(6, "Applied changes to NWA.");

    crate::debug!(5, "Resolve negatives in NWA: Done");
}

pub fn apply_cancellations(states: &mut NWAStates, source_states_filter: &HashSet<NWAStateID>) {
    let epsilons_to_add = compute_cancellations(states, source_states_filter);
    crate::debug!(8, "Computed {} new epsilon transitions from cancellations.", epsilons_to_add.len());
    for (from, to, w) in epsilons_to_add {
        states.add_epsilon(from, to, w);
    }
}

/// Range-based version for contiguous state ranges - avoids HashSet allocation
pub fn apply_cancellations_range(states: &mut NWAStates, range: std::ops::Range<NWAStateID>) {
    let profile = cancellations_range_profile_enabled();
    let start = profile.then(Instant::now);
    let epsilons_to_add = compute_cancellations_range(states, range);
    if profile {
        CANCELLATIONS_RANGE_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
        if let Some(start) = start {
            CANCELLATIONS_RANGE_TIME_US
                .fetch_add(start.elapsed().as_micros() as u64, AtomicOrdering::Relaxed);
        }
        CANCELLATIONS_RANGE_EPS_ADDED
            .fetch_add(epsilons_to_add.len() as u64, AtomicOrdering::Relaxed);
    }
    crate::debug!(8, "Computed {} new epsilon transitions from cancellations.", epsilons_to_add.len());
    for (from, to, w) in epsilons_to_add {
        states.add_epsilon(from, to, w);
    }
}

/// Parallel per-range cancellation: runs each range's cancellation independently
/// using rayon, then applies all results. Each range is processed on the unmodified
/// arena (no intermediate epsilons from other ranges), producing the same result
/// as sequential per-range processing because cancellation epsilons don't cross
/// between template ranges.
pub fn apply_cancellations_multi_range(states: &mut NWAStates, ranges: &[std::ops::Range<NWAStateID>]) {
    use rayon::prelude::*;
    
    if ranges.is_empty() {
        return;
    }
    let start = Instant::now();
    
    // Run all cancellations in parallel on the immutable arena.
    // Each range's cancellation is independent — cancellation epsilons go from
    // template states to right_body/template states, and template ranges don't
    // overlap or interact.
    let all_epsilons: Vec<Vec<(NWAStateID, NWAStateID, Weight)>> = ranges
        .par_iter()
        .map(|range| compute_cancellations_range(states, range.clone()))
        .collect();
    
    let total_eps: usize = all_epsilons.iter().map(|v| v.len()).sum();
    let elapsed = start.elapsed();
    eprintln!(
        "TIMING: apply_cancellations_multi_range ranges={} epsilons={} {:?}",
        ranges.len(), total_eps, elapsed
    );
    
    // Apply all epsilons to the arena
    for eps_batch in all_epsilons {
        for (from, to, w) in eps_batch {
            states.add_epsilon(from, to, w);
        }
    }
}

pub fn apply_finality_fixpoint(
    states: &mut NWAStates,
    source_states_filter: &HashSet<NWAStateID>,
) {
    let final_fix = compute_finality_fixpoint(states, source_states_filter);
    for &sid in source_states_filter {
        if let Some(add) = final_fix.get(&sid) {
            let st = &mut states.0[sid];
            if let Some(fw) = &mut st.final_weight {
                *fw |= add;
            } else {
                st.final_weight = Some(add.clone());
            }
        }
    }
}

/// Range-based version for contiguous state ranges - avoids HashSet allocation
pub fn apply_finality_fixpoint_range(
    states: &mut NWAStates,
    range: std::ops::Range<NWAStateID>,
) {
    let final_fix = compute_finality_fixpoint_range(states, range.clone());
    for sid in range {
        if let Some(add) = final_fix.get(&sid) {
            let st = &mut states.0[sid];
            if let Some(fw) = &mut st.final_weight {
                *fw |= add;
            } else {
                st.final_weight = Some(add.clone());
            }
        }
    }
}

pub fn remove_negative_transitions(states: &mut NWAStates, source_states_filter: &HashSet<NWAStateID>) {
    for &sid in source_states_filter {
        states.0[sid].transitions.retain(|&label, _| !is_negative_symbol(label));
    }
}

/// Range-based version for contiguous state ranges - avoids HashSet allocation
pub fn remove_negative_transitions_range(states: &mut NWAStates, range: std::ops::Range<NWAStateID>) {
    for sid in range {
        states.0[sid].transitions.retain(|&label, _| !is_negative_symbol(label));
    }
}

/// Remove default transitions that point to "terminal" states.
/// 
/// A "terminal" state is one that:
/// 1. Has no outgoing transitions (other than default transitions to other terminal states)
/// 2. Has no epsilon transitions
/// 3. Is final (after finality propagation)
///
/// After finality fixpoint, finality reachable via default transitions has been propagated back,
/// so these default transitions are redundant and can be safely removed.
///
/// This is an optimization that reduces the number of states/transitions in the resulting DWA.
pub fn remove_redundant_default_transitions(states: &mut NWAStates, source_states_filter: &HashSet<NWAStateID>) {
    let n = states.len();
    
    // First, identify "terminal" states: states with no non-default outgoing transitions,
    // no epsilons, and that are final.
    // We use a fixpoint because a state is terminal if it only has defaults to terminal states.
    let mut is_terminal = vec![false; n];
    
    // Initial pass: mark states with no outgoing transitions at all as terminal (if final)
    for sid in 0..n {
        let st = &states.0[sid];
        let has_non_default_transitions = st.transitions.iter().any(|(label, targets)| {
            *label != DEFAULT_TRANSITION_SYMBOL && !targets.is_empty()
        });
        let has_epsilons = !st.epsilons.is_empty();
        let is_final = st.final_weight.as_ref().map_or(false, |w| !w.is_empty());
        
        if !has_non_default_transitions && !has_epsilons && is_final {
            is_terminal[sid] = true;
        }
    }
    
    // Fixpoint: a state is terminal if all its default transitions point to terminal states
    // (and it meets the other criteria)
    let mut changed = true;
    while changed {
        changed = false;
        for sid in 0..n {
            if is_terminal[sid] {
                continue;
            }
            let st = &states.0[sid];
            
            // Check if this state only has default transitions
            let has_non_default_transitions = st.transitions.iter().any(|(label, targets)| {
                *label != DEFAULT_TRANSITION_SYMBOL && !targets.is_empty()
            });
            let has_epsilons = !st.epsilons.is_empty();
            let is_final = st.final_weight.as_ref().map_or(false, |w| !w.is_empty());
            
            if has_non_default_transitions || has_epsilons || !is_final {
                continue;
            }
            
            // Check if all default targets are terminal
            let all_default_targets_terminal = st.transitions
                .get(&DEFAULT_TRANSITION_SYMBOL)
                .map_or(true, |targets| {
                    targets.iter().all(|(target, _)| *target < n && is_terminal[*target])
                });
            
            if all_default_targets_terminal {
                is_terminal[sid] = true;
                changed = true;
            }
        }
    }
    
    let num_terminal = is_terminal.iter().filter(|&&t| t).count();
    if num_terminal > 0 {
        crate::debug!(6, "Found {} terminal states for default transition shortcutting", num_terminal);
    }
    
    // Now remove default transitions that point to terminal states
    let mut removed_count = 0;
    for &sid in source_states_filter {
        if sid >= n {
            continue;
        }
        if let Some(default_targets) = states.0[sid].transitions.get_mut(&DEFAULT_TRANSITION_SYMBOL) {
            let old_len = default_targets.len();
            default_targets.retain(|(target, _)| !(*target < n && is_terminal[*target]));
            removed_count += old_len - default_targets.len();
        }
        // Clean up empty entries
        states.0[sid].transitions.retain(|_, targets| !targets.is_empty());
    }
    
    if removed_count > 0 {
        crate::debug!(6, "Removed {} redundant default transitions to terminal states", removed_count);
    }
}

/// Range-based version for contiguous state ranges
pub fn remove_redundant_default_transitions_range(states: &mut NWAStates, range: std::ops::Range<NWAStateID>) {
    let n = states.len();
    
    // First, identify "terminal" states
    let mut is_terminal = vec![false; n];
    
    // Initial pass
    for sid in 0..n {
        let st = &states.0[sid];
        let has_non_default_transitions = st.transitions.iter().any(|(label, targets)| {
            *label != DEFAULT_TRANSITION_SYMBOL && !targets.is_empty()
        });
        let has_epsilons = !st.epsilons.is_empty();
        let is_final = st.final_weight.as_ref().map_or(false, |w| !w.is_empty());
        
        if !has_non_default_transitions && !has_epsilons && is_final {
            is_terminal[sid] = true;
        }
    }
    
    // Fixpoint
    let mut changed = true;
    while changed {
        changed = false;
        for sid in 0..n {
            if is_terminal[sid] {
                continue;
            }
            let st = &states.0[sid];
            
            let has_non_default_transitions = st.transitions.iter().any(|(label, targets)| {
                *label != DEFAULT_TRANSITION_SYMBOL && !targets.is_empty()
            });
            let has_epsilons = !st.epsilons.is_empty();
            let is_final = st.final_weight.as_ref().map_or(false, |w| !w.is_empty());
            
            if has_non_default_transitions || has_epsilons || !is_final {
                continue;
            }
            
            let all_default_targets_terminal = st.transitions
                .get(&DEFAULT_TRANSITION_SYMBOL)
                .map_or(true, |targets| {
                    targets.iter().all(|(target, _)| *target < n && is_terminal[*target])
                });
            
            if all_default_targets_terminal {
                is_terminal[sid] = true;
                changed = true;
            }
        }
    }
    
    // Remove default transitions to terminal states
    for sid in range {
        if sid >= n {
            continue;
        }
        if let Some(default_targets) = states.0[sid].transitions.get_mut(&DEFAULT_TRANSITION_SYMBOL) {
            default_targets.retain(|(target, _)| !(*target < n && is_terminal[*target]));
        }
        states.0[sid].transitions.retain(|_, targets| !targets.is_empty());
    }
}

/// Resolve negative codes in a DWA by a single, high-performance, semantics-preserving NWA rewrite.
pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let now = Instant::now();
    crate::debug!(6, "Resolving negative codes in DWA with {}...", dwa.stats());

    progress_step(1, "DWA -> NWA");
    let mut nwa = NWA::from_dwa(dwa);
    crate::debug!(6, "Converted to NWA with {}.", nwa.stats());
    crate::debug!(6, "Stats for NWA from DWA:\n{}", nwa.stats());

    progress_step(2, "Resolve negatives in NWA");
    resolve_negative_codes_in_nwa(&mut nwa);
    crate::debug!(6, "Applied changes, NWA has {} before determinization.", nwa.stats());
    crate::debug!(6, "Stats for NWA after negative resolution:\n{}", nwa.stats());

    progress_step(3, "Determinize");
    let mut result = nwa.determinize();

    progress_step(4, "Minimize");
    result.minimize();
    *dwa = result;
    crate::debug!(6, "Stats for parser DWA after negative resolution:\n{}", dwa.stats());

    crate::debug!(5, "Resolve negative codes in DWA: Done");
    crate::debug!(6, "resolve_negative_codes_in_dwa took: {:?}", now.elapsed());
}

fn compute_cancellations(states: &NWAStates, source_states_filter: &HashSet<NWAStateID>) -> Vec<(NWAStateID, NWAStateID, Weight)> {
    let n = states.len();
    let profile = cancellations_range_profile_enabled();
    let mut seed_count: u64 = 0;
    let mut steps: u64 = 0;
    let mut prop_eps_us: u64 = 0;
    let mut pos_us: u64 = 0;
    let mut default_us: u64 = 0;
    let mut eps_us: u64 = 0;
    let mut prop_eps_edges: u64 = 0;
    let mut pos_edges: u64 = 0;
    let mut default_edges: u64 = 0;
    let mut eps_edges: u64 = 0;
    let mut eps_create_events: u64 = 0;
    let mut queries_at_source_total: u64 = 0;
    let mut queries_at_source_max: u64 = 0;
    let mut eps_created: u64 = 0;
    let mut eps_grown: u64 = 0;

    let mut neg_source_states: FxHashSet<NWAStateID> = FxHashSet::default();
    if profile {
        for &a in source_states_filter {
            if states[a]
                .transitions
                .iter()
                .any(|(label, _)| is_negative_symbol(*label))
            {
                neg_source_states.insert(a);
            }
        }
    }

    let mut queries: FxHashMap<NWAStateID, FxHashMap<QueryKey, Weight>> = FxHashMap::default();
    let mut worklist: VecDeque<(NWAStateID, NWAStateID, Code)> = VecDeque::new();
    let mut in_queue: Vec<FxHashSet<QueryKey>> = Vec::with_capacity(n);
    in_queue.resize_with(n, FxHashSet::default);

    // new_eps_from[from][to] = weight of the cancellation epsilon from `from` to `to`.
    let mut new_eps_from: HashMap<NWAStateID, HashMap<NWAStateID, Weight>> = HashMap::new();

    // Seed from negative transitions.
    for &a in source_states_filter {
        for (&label, targets) in &states[a].transitions {
            if !is_negative_symbol(label) {
                continue;
            }
            let c = label.wrapping_sub(Code::MIN);
            for (b, w_ab) in targets {
                if *b >= n {
                    continue;
                }
                let query_key: QueryKey = (a, c);
                let query_weight = queries.entry(*b).or_default().entry(query_key).or_default();
                let old_w = query_weight.clone();
                *query_weight |= w_ab;
                if *query_weight != old_w {
                    let key: QueryKey = (a, c);
                    if in_queue[*b].insert(key) {
                        worklist.push_back((*b, a, c));
                        if profile {
                            seed_count = seed_count.saturating_add(1);
                        }
                    }
                }
            }
        }
    }

    while let Some((s, a, c)) = worklist.pop_front() {
        in_queue[s].remove(&(a, c));
        if profile {
            steps = steps.saturating_add(1);
        }
        let w_as = match queries.get(&s).and_then(|m| m.get(&(a, c))) {
            Some(w) => w.clone(),
            None => continue,
        };
        // First, propagate this query through any already-known cancellation epsilons
        // originating from the current location `s`.
        let prop_start = if profile { Some(Instant::now()) } else { None };
        if let Some(epsilons_from_s) = new_eps_from.get(&s) {
            if profile {
                prop_eps_edges = prop_eps_edges.saturating_add(epsilons_from_s.len() as u64);
            }
            for (&target, eps_w) in epsilons_from_s {
                let prop_w = &w_as & eps_w;
                if prop_w.is_empty() {
                    continue;
                }
                let query_key: QueryKey = (a, c);
                let query_weight = queries.entry(target).or_default().entry(query_key).or_default();
                let old_qw = query_weight.clone();
                *query_weight |= &prop_w;
                if *query_weight != old_qw {
                    let key: QueryKey = (a, c);
                    if in_queue[target].insert(key) {
                        worklist.push_back((target, a, c));
                    }
                }
            }
        }
        if let Some(start) = prop_start {
            prop_eps_us = prop_eps_us.saturating_add(start.elapsed().as_micros() as u64);
        }

        let mut check_cancellations = |target: NWAStateID,
                                       w_st: &Weight,
                                       worklist: &mut VecDeque<(NWAStateID, NWAStateID, Code)>,
                                       in_queue: &mut Vec<FxHashSet<QueryKey>>| {
            let new_eps_w = &w_as & w_st;
            if new_eps_w.is_empty() {
                return;
            }

            // Epsilon summarizing a cancellation from `a` (the negative's source) to `target`.
            let eps_from_a = new_eps_from.entry(a).or_default();
            let eps_weight = eps_from_a.entry(target).or_default();
            let old_eps_w = eps_weight.clone();
            *eps_weight |= &new_eps_w;

            if *eps_weight != old_eps_w {
                if profile {
                    eps_create_events = eps_create_events.saturating_add(1);
                    if old_eps_w.is_empty() {
                        eps_created = eps_created.saturating_add(1);
                    } else {
                        eps_grown = eps_grown.saturating_add(1);
                    }
                    let queries_at_a = queries.get(&a).map_or(0, |m| m.len()) as u64;
                    queries_at_source_total = queries_at_source_total.saturating_add(queries_at_a);
                    if queries_at_a > queries_at_source_max {
                        queries_at_source_max = queries_at_a;
                    }
                }
                // When this epsilon grows, any existing queries that are currently at `a`
                // can also traverse it.
                if let Some(queries_at_a) = queries.get(&a) {
                    for (&(a_prime, c_prime), w_a_prime_a) in &queries_at_a.clone() {
                        let prop_w = w_a_prime_a & &*eps_weight;
                        if prop_w.is_empty() {
                            continue;
                        }
                        let query_key: QueryKey = (a_prime, c_prime);
                        let query_weight = queries.entry(target).or_default().entry(query_key).or_default();
                        let old_qw = query_weight.clone();
                        *query_weight |= &prop_w;
                        if *query_weight != old_qw {
                            let key: QueryKey = (a_prime, c_prime);
                            if in_queue[target].insert(key) {
                                worklist.push_back((target, a_prime, c_prime));
                            }
                        }
                    }
                }
            }
        };

        let pos_start = if profile { Some(Instant::now()) } else { None };
        if let Some(pos_targets) = states[s].transitions.get(&c) {
            for (t, w_st) in pos_targets {
                if *t < n {
                    if profile {
                        pos_edges = pos_edges.saturating_add(1);
                    }
                    check_cancellations(*t, w_st, &mut worklist, &mut in_queue);
                }
            }
        }
        if let Some(start) = pos_start {
            pos_us = pos_us.saturating_add(start.elapsed().as_micros() as u64);
        }

        let default_start = if profile { Some(Instant::now()) } else { None };
        if let Some(default_targets) = states[s].transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            for (target, weight) in default_targets {
                if profile {
                    default_edges = default_edges.saturating_add(1);
                }
                check_cancellations(*target, weight, &mut worklist, &mut in_queue);
            }
        }
        if let Some(start) = default_start {
            default_us = default_us.saturating_add(start.elapsed().as_micros() as u64);
        }

        let eps_start = if profile { Some(Instant::now()) } else { None };
        for (t, w_st) in &states[s].epsilons {
            if *t >= n {
                continue;
            }
            let prop_w = &w_as & w_st;
            if prop_w.is_empty() {
                continue;
            }
            if profile {
                eps_edges = eps_edges.saturating_add(1);
            }
            let query_key: QueryKey = (a, c);
            let query_weight = queries.entry(*t).or_default().entry(query_key).or_default();
            let old_qw = query_weight.clone();
            *query_weight |= &prop_w;
            if *query_weight != old_qw {
                let key: QueryKey = (a, c);
                if in_queue[*t].insert(key) {
                    worklist.push_back((*t, a, c));
                }
            }
        }
        if let Some(start) = eps_start {
            eps_us = eps_us.saturating_add(start.elapsed().as_micros() as u64);
        }
    }

    let mut result = Vec::new();
    let mut eps_added: u64 = 0;
    let mut eps_fanout_sources: u64 = 0;
    let mut eps_fanout_total: u64 = 0;
    let mut eps_fanout_max: u64 = 0;
    for (from, targets) in new_eps_from {
        eps_added = eps_added.saturating_add(targets.len() as u64);
        if profile {
            let fanout = targets.len() as u64;
            if fanout > 0 {
                eps_fanout_sources = eps_fanout_sources.saturating_add(1);
                eps_fanout_total = eps_fanout_total.saturating_add(fanout);
                if fanout > eps_fanout_max {
                    eps_fanout_max = fanout;
                }
            }
        }
        for (to, w) in targets {
            result.push((from, to, w));
        }
    }
    if profile {
        eprintln!(
            "PASS2 cancellations profile (hashset): seeds={}, steps={}, eps_added={}",
            seed_count,
            steps,
            eps_added,
        );
        eprintln!(
            "PASS2 cancellations breakdown: prop_eps_us={} pos_us={} default_us={} eps_us={} prop_eps_edges={} pos_edges={} default_edges={} eps_edges={}",
            prop_eps_us,
            pos_us,
            default_us,
            eps_us,
            prop_eps_edges,
            pos_edges,
            default_edges,
            eps_edges,
        );
        eprintln!("PASS2 cancellations neg_source_states={}", neg_source_states.len());
        let queries_at_source_avg = if eps_create_events > 0 {
            queries_at_source_total / eps_create_events
        } else {
            0
        };
        eprintln!(
            "PASS2 cancellations queries_at_source: events={} total={} max={} avg={}",
            eps_create_events,
            queries_at_source_total,
            queries_at_source_max,
            queries_at_source_avg,
        );
        let eps_fanout_avg = if eps_fanout_sources > 0 {
            eps_fanout_total / eps_fanout_sources
        } else {
            0
        };
        eprintln!(
            "PASS2 cancellations eps_fanout: sources={} total={} max={} avg={}",
            eps_fanout_sources,
            eps_fanout_total,
            eps_fanout_max,
            eps_fanout_avg,
        );
        eprintln!(
            "PASS2 cancellations eps_updates: created={} grown={}",
            eps_created,
            eps_grown,
        );
    }
    result
}

/// Range-based version of compute_cancellations for contiguous state ranges
pub(crate) fn compute_cancellations_range(states: &NWAStates, range: std::ops::Range<NWAStateID>) -> Vec<(NWAStateID, NWAStateID, Weight)> {
    let n = states.len();
    let profile = cancellations_range_profile_enabled();
    let mut seed_count: u64 = 0;

    // Use HashMaps instead of Vec to avoid O(arena_size) allocations.
    // Only states that actually receive queries/epsilons take memory.
    let mut queries: FxHashMap<NWAStateID, FxHashMap<QueryKey, Weight>> = FxHashMap::default();
    let mut worklist: VecDeque<(NWAStateID, NWAStateID, Code)> = VecDeque::new();
    let mut in_queue: FxHashSet<(NWAStateID, NWAStateID, Code)> = FxHashSet::default();
    let mut new_eps_from: FxHashMap<NWAStateID, FxHashMap<NWAStateID, Weight>> = FxHashMap::default();
    let mut enqueue = |worklist: &mut VecDeque<(NWAStateID, NWAStateID, Code)>,
                       in_queue: &mut FxHashSet<(NWAStateID, NWAStateID, Code)>,
                       s: NWAStateID,
                       a: NWAStateID,
                       c: Code| {
        let key = (s, a, c);
        if in_queue.insert(key) {
            worklist.push_back(key);
        }
    };

    // Seed from negative transitions in the range
    for a in range.clone() {
        for (&label, targets) in &states[a].transitions {
            if !is_negative_symbol(label) {
                continue;
            }
            let c = label.wrapping_sub(Code::MIN);
            for (b, w_ab) in targets {
                if *b >= n {
                    continue;
                }
                let query_key: QueryKey = (a, c);
                let query_weight = queries.entry(*b).or_default().entry(query_key).or_default();
                if !w_ab.is_subset_of(query_weight) {
                    *query_weight |= w_ab;
                    enqueue(&mut worklist, &mut in_queue, *b, a, c);
                    seed_count = seed_count.saturating_add(1);
                }
            }
        }
    }

    // Early exit if no negative transitions were found
    if worklist.is_empty() {
        if profile {
            CANCELLATIONS_RANGE_SEEDS.fetch_add(seed_count, AtomicOrdering::Relaxed);
            CANCELLATIONS_RANGE_STEPS.fetch_add(0, AtomicOrdering::Relaxed);
        }
        return Vec::new();
    }

    static MAX_STEPS: Lazy<usize> = Lazy::new(|| {
        std::env::var("NWA_PASS2_CANCELLATIONS_MAX_STEPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    });
    let max_steps = *MAX_STEPS;
    let mut steps = 0usize;

    while let Some((s, a, c)) = worklist.pop_front() {
        in_queue.remove(&(s, a, c));
        if max_steps > 0 && steps >= max_steps {
            crate::debug!(4, "Pass2 cancellations: reached max steps {}, truncating", max_steps);
            break;
        }
        steps += 1;
        let w_as = match queries.get(&s).and_then(|m| m.get(&(a, c))) {
            Some(w) => w.clone(),
            None => continue,
        };
        
        // Propagate through existing cancellation epsilons from s
        if let Some(epsilons_from_s) = new_eps_from.get(&s) {
            if !epsilons_from_s.is_empty() {
                let propagations: Vec<(NWAStateID, Weight)> = epsilons_from_s.iter()
                    .filter_map(|(&target, eps_w)| {
                        let prop_w = &w_as & eps_w;
                        if prop_w.is_empty() { None } else { Some((target, prop_w)) }
                    })
                    .collect();
                for (target, prop_w) in propagations {
                    let query_key: QueryKey = (a, c);
                    let query_weight = queries.entry(target).or_default().entry(query_key).or_default();
                    if !prop_w.is_subset_of(query_weight) {
                        *query_weight |= &prop_w;
                        enqueue(&mut worklist, &mut in_queue, target, a, c);
                    }
                }
            }
        }

        // Collect cancellation targets, then apply mutations
        let mut cancellation_updates: Vec<(NWAStateID, Weight)> = Vec::new();

        if let Some(pos_targets) = states[s].transitions.get(&c) {
            for (t, w_st) in pos_targets {
                if *t < n {
                    let new_eps_w = &w_as & w_st;
                    if !new_eps_w.is_empty() {
                        cancellation_updates.push((*t, new_eps_w));
                    }
                }
            }
        }
        if let Some(default_targets) = states[s].transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            for (target, weight) in default_targets {
                let new_eps_w = &w_as & weight;
                if !new_eps_w.is_empty() {
                    cancellation_updates.push((*target, new_eps_w));
                }
            }
        }

        // Apply cancellation updates
        for (target, new_eps_w) in cancellation_updates {
            let eps_from_a = new_eps_from.entry(a).or_default();
            let eps_weight = eps_from_a.entry(target).or_default();
            if !new_eps_w.is_subset_of(eps_weight) {
                *eps_weight |= &new_eps_w;
                let combined_eps_w = eps_weight.clone();

                // Propagate existing queries at `a` through the new/updated epsilon
                let queries_at_a: Vec<(QueryKey, Weight)> = queries.get(&a)
                    .map(|m| m.iter().map(|(k, v)| (*k, v.clone())).collect())
                    .unwrap_or_default();
                for ((a_prime, c_prime), w_a_prime_a) in queries_at_a {
                    let prop_w = w_a_prime_a & &combined_eps_w;
                    if prop_w.is_empty() {
                        continue;
                    }
                    let query_weight = queries.entry(target).or_default().entry((a_prime, c_prime)).or_default();
                    if !prop_w.is_subset_of(query_weight) {
                        *query_weight |= &prop_w;
                        enqueue(&mut worklist, &mut in_queue, target, a_prime, c_prime);
                    }
                }
            }
        }

        // Propagate through epsilon transitions from s
        for (t, w_st) in &states[s].epsilons {
            if *t >= n {
                continue;
            }
            let prop_w = &w_as & w_st;
            if prop_w.is_empty() {
                continue;
            }
            let query_key: QueryKey = (a, c);
            let query_weight = queries.entry(*t).or_default().entry(query_key).or_default();
            if !prop_w.is_subset_of(query_weight) {
                *query_weight |= &prop_w;
                enqueue(&mut worklist, &mut in_queue, *t, a, c);
            }
        }
    }

    if profile {
        CANCELLATIONS_RANGE_SEEDS.fetch_add(seed_count, AtomicOrdering::Relaxed);
        CANCELLATIONS_RANGE_STEPS.fetch_add(steps as u64, AtomicOrdering::Relaxed);
    }

    let mut result = Vec::new();
    for (from, targets) in &new_eps_from {
        for (_to, _w) in targets {
            result.push((*from, *_to, _w.clone()));
        }
    }
    result
}

fn compute_finality_fixpoint(
    states: &NWAStates,
    source_states_filter: &HashSet<NWAStateID>,
) -> HashMap<NWAStateID, Weight> {
    let n = states.len();
    if n == 0 || source_states_filter.is_empty() {
        return HashMap::new();
    }

    #[derive(Clone, Copy)]
    enum PredEdge {
        Epsilon { from: NWAStateID, eps_idx: usize },
        Negative { from: NWAStateID, label: Code, trans_idx: usize },
        Default { from: NWAStateID, trans_idx: usize },
    }

    // Phase 1: forward exploration from the filtered sources,
    // restricted to epsilon edges everywhere, negative edges
    // only from states in `source_states_filter`, and default transitions
    // (which can be shortcut because they match any symbol). While exploring we
    // also build the reverse adjacency (predecessor lists) restricted
    // to this reachable subgraph.
    let mut visited = vec![false; n];
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();
    let mut reachable_states: Vec<NWAStateID> = Vec::new();
    let mut preds: HashMap<NWAStateID, Vec<PredEdge>> = HashMap::new();

    // Seed BFS from the filtered source states.
    for &a in source_states_filter {
        if a >= n {
            continue;
        }
        if !visited[a] {
            visited[a] = true;
            queue.push_back(a);
            reachable_states.push(a);
        }
    }

    while let Some(s) = queue.pop_front() {
        let state = &states[s];

        // Epsilon edges from `s` are always allowed.
        for (eps_idx, &(target, ref w)) in state.epsilons.iter().enumerate() {
            if target >= n || w.is_empty() {
                continue;
            }
            preds
                .entry(target)
                .or_insert_with(Vec::new)
                .push(PredEdge::Epsilon { from: s, eps_idx });
            if !visited[target] {
                visited[target] = true;
                queue.push_back(target);
                reachable_states.push(target);
            }
        }

        // Negative transitions from `s` are only allowed when `s` is in the filter.
        if source_states_filter.contains(&s) {
            for (&label, targets) in &state.transitions {
                if !is_negative_symbol(label) {
                    continue;
                }
                for (trans_idx, &(target, ref w)) in targets.iter().enumerate() {
                    if target >= n || w.is_empty() {
                        continue;
                    }
                    preds
                        .entry(target)
                        .or_insert_with(Vec::new)
                        .push(PredEdge::Negative {
                            from: s,
                            label,
                            trans_idx,
                        });
                    if !visited[target] {
                        visited[target] = true;
                        queue.push_back(target);
                        reachable_states.push(target);
                    }
                }
            }
        }
        
        // Default transitions can be shortcut: if we can reach a final state via a default
        // transition, we can propagate the finality backwards. This is valid because
        // default transitions match "any symbol not explicitly listed", so they are
        // always traversable.
        if let Some(default_targets) = state.transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            for (trans_idx, &(target, ref w)) in default_targets.iter().enumerate() {
                if target >= n || w.is_empty() {
                    continue;
                }
                preds
                    .entry(target)
                    .or_insert_with(Vec::new)
                    .push(PredEdge::Default { from: s, trans_idx });
                if !visited[target] {
                    visited[target] = true;
                    queue.push_back(target);
                    reachable_states.push(target);
                }
            }
        }
    }

    // Phase 2: backward fixpoint over the reachable subgraph.
    //
    // future_final_all[s] = summary weight of all allowed epsilon/negative/default
    // paths from `s` to any final state (including the zero-length path
    // when `s` itself is final).
    let mut future_final_all: HashMap<NWAStateID, Weight> = HashMap::new();
    let mut worklist: VecDeque<NWAStateID> = VecDeque::new();
    let mut in_queue = vec![false; n];

    // Initialize from final states in the reachable subgraph.
    for &s in &reachable_states {
        if let Some(ref fw) = states[s].final_weight {
            if !fw.is_empty() {
                future_final_all.insert(s, fw.clone());
                if !in_queue[s] {
                    in_queue[s] = true;
                    worklist.push_back(s);
                }
            }
        }
    }

    while let Some(s) = worklist.pop_front() {
        in_queue[s] = false;
        let f_s = match future_final_all.get(&s) {
            Some(w) if !w.is_empty() => w.clone(),
            _ => continue,
        };

        if let Some(pred_edges) = preds.get(&s) {
            for edge in pred_edges {
                let (pred_state, edge_w): (NWAStateID, &Weight) = match *edge {
                    PredEdge::Epsilon { from, eps_idx } => {
                        let &(target, ref w) = &states[from].epsilons[eps_idx];
                        debug_assert_eq!(target, s);
                        (from, w)
                    }
                    PredEdge::Negative {
                        from,
                        label,
                        trans_idx,
                    } => {
                        let targets = states[from]
                            .transitions
                            .get(&label)
                            .expect("stored negative edge must exist");
                        let &(target, ref w) = &targets[trans_idx];
                        debug_assert_eq!(target, s);
                        (from, w)
                    }
                    PredEdge::Default { from, trans_idx } => {
                        let targets = states[from]
                            .transitions
                            .get(&DEFAULT_TRANSITION_SYMBOL)
                            .expect("stored default edge must exist");
                        let &(target, ref w) = &targets[trans_idx];
                        debug_assert_eq!(target, s);
                        (from, w)
                    }
                };

                let add = &f_s & edge_w;
                if add.is_empty() {
                    continue;
                }

                let entry = future_final_all
                    .entry(pred_state)
                    .or_insert_with(Weight::zeros);
                let old = entry.clone();
                *entry |= &add;
                if *entry != old {
                    if !in_queue[pred_state] {
                        in_queue[pred_state] = true;
                        worklist.push_back(pred_state);
                    }
                }
            }
        }
    }

    // We only need entries for states in `source_states_filter`.
    let mut result: HashMap<NWAStateID, Weight> = HashMap::new();
    for &a in source_states_filter {
        if a >= n {
            continue;
        }
        if let Some(w) = future_final_all.get(&a) {
            if !w.is_empty() {
                result.insert(a, w.clone());
            }
        }
    }

    result
}

/// Range-based version of compute_finality_fixpoint for contiguous state ranges
fn compute_finality_fixpoint_range(
    states: &NWAStates,
    range: std::ops::Range<NWAStateID>,
) -> FxHashMap<NWAStateID, Weight> {
    let n = states.len();
    if n == 0 || range.is_empty() {
        return FxHashMap::default();
    }

    let phase1_start = std::time::Instant::now();

    // Use packed representation: for each state, store offset+length into a flat edge buffer
    // This avoids per-state Vec allocations (1.75M vecs) and improves cache locality.
    #[derive(Clone, Copy)]
    enum PredEdge {
        Epsilon { from: NWAStateID, eps_idx: usize },
        Negative { from: NWAStateID, label: Code, trans_idx: usize },
        Default { from: NWAStateID, trans_idx: usize },
    }

    // Phase 1: Build pred graph via BFS from seed range.
    // Use Vec<bool> for visited (much faster than FxHashSet for dense ranges).
    let mut visited = vec![false; n];
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();
    // Pre-count edges to allocate pred_edges flat buffer once
    let mut preds_offset = vec![0u32; n + 1]; // offset into flat pred_edges
    let mut total_pred_edges: usize = 0;

    // First pass: BFS to find reachable states and count pred edges
    let mut reachable_states: Vec<NWAStateID> = Vec::new();
    
    for a in range.clone() {
        if a < n && !visited[a] {
            visited[a] = true;
            queue.push_back(a);
            reachable_states.push(a);
        }
    }

    // BFS to discover reachable states and count pred edges per target
    while let Some(s) = queue.pop_front() {
        let state = &states[s];

        for &(target, ref w) in &state.epsilons {
            if target >= n || w.is_empty() { continue; }
            preds_offset[target + 1] += 1;
            total_pred_edges += 1;
            if !visited[target] {
                visited[target] = true;
                queue.push_back(target);
                reachable_states.push(target);
            }
        }

        if range.contains(&s) {
            for (&label, targets) in &state.transitions {
                if !is_negative_symbol(label) { continue; }
                for &(target, ref w) in targets {
                    if target >= n || w.is_empty() { continue; }
                    preds_offset[target + 1] += 1;
                    total_pred_edges += 1;
                    if !visited[target] {
                        visited[target] = true;
                        queue.push_back(target);
                        reachable_states.push(target);
                    }
                }
            }
        }

        if let Some(default_targets) = state.transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            for &(target, ref w) in default_targets {
                if target >= n || w.is_empty() { continue; }
                preds_offset[target + 1] += 1;
                total_pred_edges += 1;
                if !visited[target] {
                    visited[target] = true;
                    queue.push_back(target);
                    reachable_states.push(target);
                }
            }
        }
    }

    // Compute prefix sums for flat pred edge storage
    for i in 1..=n {
        preds_offset[i] += preds_offset[i - 1];
    }
    debug_assert_eq!(preds_offset[n] as usize, total_pred_edges);
    
    // Second pass: populate flat pred_edges array
    let mut pred_edges: Vec<PredEdge> = Vec::with_capacity(total_pred_edges);
    unsafe { pred_edges.set_len(total_pred_edges); }
    let mut write_pos = vec![0u32; n]; // current write position per target
    
    // Re-BFS to fill pred edges (same traversal order)
    let mut visited2 = vec![false; n];
    let mut queue2: VecDeque<NWAStateID> = VecDeque::new();
    for a in range.clone() {
        if a < n && !visited2[a] {
            visited2[a] = true;
            queue2.push_back(a);
        }
    }

    while let Some(s) = queue2.pop_front() {
        let state = &states[s];

        for (eps_idx, &(target, ref w)) in state.epsilons.iter().enumerate() {
            if target >= n || w.is_empty() { continue; }
            let pos = preds_offset[target] as usize + write_pos[target] as usize;
            pred_edges[pos] = PredEdge::Epsilon { from: s, eps_idx };
            write_pos[target] += 1;
            if !visited2[target] {
                visited2[target] = true;
                queue2.push_back(target);
            }
        }

        if range.contains(&s) {
            for (&label, targets) in &state.transitions {
                if !is_negative_symbol(label) { continue; }
                for (trans_idx, &(target, ref w)) in targets.iter().enumerate() {
                    if target >= n || w.is_empty() { continue; }
                    let pos = preds_offset[target] as usize + write_pos[target] as usize;
                    pred_edges[pos] = PredEdge::Negative { from: s, label, trans_idx };
                    write_pos[target] += 1;
                    if !visited2[target] {
                        visited2[target] = true;
                        queue2.push_back(target);
                    }
                }
            }
        }

        if let Some(default_targets) = state.transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            for (trans_idx, &(target, ref w)) in default_targets.iter().enumerate() {
                if target >= n || w.is_empty() { continue; }
                let pos = preds_offset[target] as usize + write_pos[target] as usize;
                pred_edges[pos] = PredEdge::Default { from: s, trans_idx };
                write_pos[target] += 1;
                if !visited2[target] {
                    visited2[target] = true;
                    queue2.push_back(target);
                }
            }
        }
    }

    drop(visited);
    drop(visited2);
    drop(write_pos);

    let phase1_time = phase1_start.elapsed();
    let phase2_start = std::time::Instant::now();

    // Phase 2: Backward fixpoint using Vec-backed storage for future_final.
    // Use Option<Weight> vector indexed by state ID for O(1) access.
    let mut future_final: Vec<Option<Weight>> = vec![None; n];
    let mut worklist: VecDeque<NWAStateID> = VecDeque::new();
    // Track which states are already in the worklist to avoid redundant enqueues.
    // Without this, the same state can be enqueued multiple times, causing redundant
    // processing of all its predecessor edges on each dequeue.
    let mut in_worklist: Vec<bool> = vec![false; n];

    for &s in &reachable_states {
        if let Some(ref fw) = states[s].final_weight {
            if !fw.is_empty() {
                future_final[s] = Some(fw.clone());
                worklist.push_back(s);
                in_worklist[s] = true;
            }
        }
    }

    let max_steps = std::env::var("NWA_PASS2_FINALITY_FIXPOINT_MAX_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut steps = 0usize;

    while let Some(s) = worklist.pop_front() {
        in_worklist[s] = false;
        if max_steps > 0 && steps >= max_steps {
            crate::debug!(4, "Pass2 finality fixpoint: reached max steps {}, truncating", max_steps);
            break;
        }
        steps += 1;
        let f_s = match &future_final[s] {
            Some(w) if !w.is_empty() => w.clone(),
            _ => continue,
        };

        let start = preds_offset[s] as usize;
        let end = preds_offset[s + 1] as usize;
        for edge in &pred_edges[start..end] {
            let (pred_state, edge_w): (NWAStateID, &Weight) = match *edge {
                PredEdge::Epsilon { from, eps_idx } => {
                    let &(target, ref w) = &states[from].epsilons[eps_idx];
                    debug_assert_eq!(target, s);
                    (from, w)
                }
                PredEdge::Negative { from, label, trans_idx } => {
                    let targets = states[from]
                        .transitions
                        .get(&label)
                        .expect("stored negative edge must exist");
                    let &(target, ref w) = &targets[trans_idx];
                    debug_assert_eq!(target, s);
                    (from, w)
                }
                PredEdge::Default { from, trans_idx } => {
                    let targets = states[from]
                        .transitions
                        .get(&DEFAULT_TRANSITION_SYMBOL)
                        .expect("stored default edge must exist");
                    let &(target, ref w) = &targets[trans_idx];
                    debug_assert_eq!(target, s);
                    (from, w)
                }
            };

            let add = &f_s & edge_w;
            if add.is_empty() {
                continue;
            }

            let entry = future_final[pred_state].get_or_insert_with(Weight::zeros);
            // Use is_subset_of check instead of clone+compare to avoid expensive clone
            if !add.is_subset_of(entry) {
                *entry |= &add;
                if !in_worklist[pred_state] {
                    worklist.push_back(pred_state);
                    in_worklist[pred_state] = true;
                }
            }
        }
    }

    let phase2_time = phase2_start.elapsed();
    eprintln!("TIMING: finality_fixpoint phase1={:?} phase2={:?} reachable={} pred_edges={} steps={}", 
        phase1_time, phase2_time, reachable_states.len(), total_pred_edges, steps);

    let mut result: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
    for a in range {
        if a >= n { continue; }
        if let Some(w) = &future_final[a] {
            if !w.is_empty() {
                result.insert(a, w.clone());
            }
        }
    }

    result
}
