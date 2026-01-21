use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::dwa_i32::common::Label;
use crate::dwa_i32::{DWA, NWA, NWAStateID, NWAStates, Weight};

use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

type Code = Label;
type QueryKey = (NWAStateID, Code);

#[inline]
fn is_negative_symbol(label: Code) -> bool { label < 0 && label != DEFAULT_TRANSITION_SYMBOL }

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
    let epsilons_to_add = compute_cancellations_range(states, range);
    crate::debug!(8, "Computed {} new epsilon transitions from cancellations.", epsilons_to_add.len());
    for (from, to, w) in epsilons_to_add {
        states.add_epsilon(from, to, w);
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

    let mut queries: HashMap<NWAStateID, HashMap<QueryKey, Weight>> = HashMap::new();
    let mut worklist: VecDeque<(NWAStateID, NWAStateID, Code, Weight)> = VecDeque::new();

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
                    worklist.push_back((*b, a, c, query_weight.clone()));
                }
            }
        }
    }

    while let Some((s, a, c, w_as)) = worklist.pop_front() {
        // First, propagate this query through any already-known cancellation epsilons
        // originating from the current location `s`.
        if let Some(epsilons_from_s) = new_eps_from.get(&s) {
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
                    worklist.push_back((target, a, c, query_weight.clone()));
                }
            }
        }

        let mut check_cancellations = |target: NWAStateID,
                                       w_st: &Weight,
                                       worklist: &mut VecDeque<(NWAStateID, NWAStateID, Code, Weight)>| {
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
                            worklist.push_back((target, a_prime, c_prime, query_weight.clone()));
                        }
                    }
                }
            }
        };

        if let Some(pos_targets) = states[s].transitions.get(&c) {
            for (t, w_st) in pos_targets {
                if *t < n {
                    check_cancellations(*t, w_st, &mut worklist);
                }
            }
        }
        if let Some(default_targets) = states[s].transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            for (target, weight) in default_targets {
                check_cancellations(*target, weight, &mut worklist);
            }
        }

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
            let old_qw = query_weight.clone();
            *query_weight |= &prop_w;
            if *query_weight != old_qw {
                worklist.push_back((*t, a, c, query_weight.clone()));
            }
        }
    }

    let mut result = Vec::new();
    for (from, targets) in new_eps_from {
        for (to, w) in targets {
            result.push((from, to, w));
        }
    }
    result
}

/// Range-based version of compute_cancellations for contiguous state ranges
fn compute_cancellations_range(states: &NWAStates, range: std::ops::Range<NWAStateID>) -> Vec<(NWAStateID, NWAStateID, Weight)> {
    let n = states.len();
    
    let mut queries: FxHashMap<NWAStateID, FxHashMap<QueryKey, Weight>> = FxHashMap::default();
    let mut worklist: VecDeque<(NWAStateID, NWAStateID, Code, Weight)> = VecDeque::new();
    let mut new_eps_from: FxHashMap<NWAStateID, FxHashMap<NWAStateID, Weight>> = FxHashMap::default();

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
                let old_w = query_weight.clone();
                *query_weight |= w_ab;
                if *query_weight != old_w {
                    worklist.push_back((*b, a, c, query_weight.clone()));
                }
            }
        }
    }
    
    // Early exit if no negative transitions were found
    if worklist.is_empty() {
        return Vec::new();
    }

    let max_steps = std::env::var("NWA_PASS2_CANCELLATIONS_MAX_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut steps = 0usize;

    while let Some((s, a, c, w_as)) = worklist.pop_front() {
        if max_steps > 0 && steps >= max_steps {
            crate::debug!(4, "Pass2 cancellations: reached max steps {}, truncating", max_steps);
            break;
        }
        steps += 1;
        if let Some(epsilons_from_s) = new_eps_from.get(&s) {
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
                    worklist.push_back((target, a, c, query_weight.clone()));
                }
            }
        }

        let mut check_cancellations = |target: NWAStateID,
                                       w_st: &Weight,
                                       worklist: &mut VecDeque<(NWAStateID, NWAStateID, Code, Weight)>| {
            let new_eps_w = &w_as & w_st;
            if new_eps_w.is_empty() {
                return;
            }

            let eps_from_a = new_eps_from.entry(a).or_default();
            let eps_weight = eps_from_a.entry(target).or_default();
            let old_eps_w = eps_weight.clone();
            *eps_weight |= &new_eps_w;

            if *eps_weight != old_eps_w {
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
                            worklist.push_back((target, a_prime, c_prime, query_weight.clone()));
                        }
                    }
                }
            }
        };

        if let Some(pos_targets) = states[s].transitions.get(&c) {
            for (t, w_st) in pos_targets {
                if *t < n {
                    check_cancellations(*t, w_st, &mut worklist);
                }
            }
        }
        if let Some(default_targets) = states[s].transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            for (target, weight) in default_targets {
                check_cancellations(*target, weight, &mut worklist);
            }
        }

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
            let old_qw = query_weight.clone();
            *query_weight |= &prop_w;
            if *query_weight != old_qw {
                worklist.push_back((*t, a, c, query_weight.clone()));
            }
        }
    }

    let mut result = Vec::new();
    for (from, targets) in new_eps_from {
        for (to, w) in targets {
            result.push((from, to, w));
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

    // Initialize from final states in the reachable subgraph.
    for &s in &reachable_states {
        if let Some(ref fw) = states[s].final_weight {
            if !fw.is_empty() {
                future_final_all.insert(s, fw.clone());
                worklist.push_back(s);
            }
        }
    }

    while let Some(s) = worklist.pop_front() {
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
                    worklist.push_back(pred_state);
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

    #[derive(Clone, Copy)]
    enum PredEdge {
        Epsilon { from: NWAStateID, eps_idx: usize },
        Negative { from: NWAStateID, label: Code, trans_idx: usize },
        Default { from: NWAStateID, trans_idx: usize },
    }

    let mut visited = vec![false; n];
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();
    let mut reachable_states: Vec<NWAStateID> = Vec::new();
    let mut preds: FxHashMap<NWAStateID, Vec<PredEdge>> = FxHashMap::default();

    // Seed BFS from the range
    for a in range.clone() {
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

        // Negative transitions only from states in the range
        if range.contains(&s) {
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
        
        // Default transitions can be shortcut
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

    let mut future_final_all: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
    let mut worklist: VecDeque<NWAStateID> = VecDeque::new();

    for &s in &reachable_states {
        if let Some(ref fw) = states[s].final_weight {
            if !fw.is_empty() {
                future_final_all.insert(s, fw.clone());
                worklist.push_back(s);
            }
        }
    }

    let max_steps = std::env::var("NWA_PASS2_FINALITY_FIXPOINT_MAX_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut steps = 0usize;

    while let Some(s) = worklist.pop_front() {
        if max_steps > 0 && steps >= max_steps {
            crate::debug!(4, "Pass2 finality fixpoint: reached max steps {}, truncating", max_steps);
            break;
        }
        steps += 1;
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
                    worklist.push_back(pred_state);
                }
            }
        }
    }

    let mut result: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
    for a in range {
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
