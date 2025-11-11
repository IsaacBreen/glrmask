// src/precompute4/weighted_automata/simplification.rs
//
// Simplified structure with the same core passes and behavior:
// - One core simplification loop parameterized for small/large DWAs.
// - Helpers to normalize edges, prune unreachable, relax edge weights, and minimize via partition refinement.
// - NWA simplification kept feature-complete with light refactoring for clarity.

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{StateID, Weight, STOCHASTIC_DEBUG};
use super::dwa::{DWABody, DWAState, DWAStates, DWA};
use super::nwa::{NWAState, NWADefaultTransition, NWAStates, NWA};
use crate::precompute4::test_weighted_automata;
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::Instant;

/// For very large DWAs, we skip heavy fixpoint/minimization passes to guarantee fast simplification.
/// This is semantics-preserving; it only reduces the amount of compression performed.
const LARGE_AUTOMATON_THRESHOLD: usize = 20_000_000;

thread_local! {
    static IN_SIMPLIFY_CHECK: Cell<bool> = Cell::new(false);
}

impl DWA {
    pub fn simplify(&mut self) {
        let is_checking = IN_SIMPLIFY_CHECK.with(|c| c.get());
        let before_simplify = if !is_checking && STOCHASTIC_DEBUG { Some(self.clone()) } else { None };

        Self::simplify_components(&mut self.states, &mut self.body);

        if let Some(before) = before_simplify {
            IN_SIMPLIFY_CHECK.with(|c| c.set(true));
            test_weighted_automata::stochastic_equivalence_test(before, self.clone());
            IN_SIMPLIFY_CHECK.with(|c| c.set(false));
        }
    }

    pub fn simplify_components(states: &mut DWAStates, body: &mut DWABody) {
        let now = Instant::now();
        let initial_len = states.len();
        if states.0.is_empty() {
            return;
        }
        let large = states.len() > LARGE_AUTOMATON_THRESHOLD;
        Self::simplify_core(states, body, large);
        crate::debug!(3, "DWA::simplify_components ({} states -> {} states) took: {:?}", initial_len, states.len(), now.elapsed());
    }

    fn run_pass_with_test<F>(
        states: &mut DWAStates,
        body: &mut DWABody,
        pass_name: &str,
        mut pass: F,
    ) -> bool
    where
        F: FnMut(&mut DWAStates, &mut DWABody) -> bool,
    {
        let is_checking = IN_SIMPLIFY_CHECK.with(|c| c.get());
        let before_dwa = if !is_checking && STOCHASTIC_DEBUG {
            Some(DWA { states: states.clone(), body: body.clone() })
        } else {
            None
        };

        let now = Instant::now();
        let changed = pass(states, body);
        let elapsed = now.elapsed();
        crate::debug!(3, "DWA simplify pass '{}' took {:?} (changed: {})", pass_name, elapsed, changed);

        if let Some(before) = before_dwa {
            if changed {
                let after_dwa = DWA { states: states.clone(), body: body.clone() };
                IN_SIMPLIFY_CHECK.with(|c| c.set(true));
                crate::debug!(1, "Stochastic testing DWA after pass: {}", pass_name);
                test_weighted_automata::stochastic_equivalence_test(before, after_dwa);
                IN_SIMPLIFY_CHECK.with(|c| c.set(false));
            }
        }
        changed
    }

    fn simplify_core(states: &mut DWAStates, body: &mut DWABody, large: bool) {
        let max_passes: usize = if large { 2 } else { 50 };
        let pb = if PROGRESS_BAR_ENABLED {
            let title = if large { "Simplifying DWA (large)" } else { "Simplifying DWA (small)" };
            let p = ProgressBar::new(max_passes as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template(&("{spinner:.green} [".to_owned() + title + ": {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} passes ({msg})"))
                    .expect("progress-bar"),
            );
            Some(p)
        } else {
            None
        };

        // Initial normalize + prune + constrain finals
        if let Some(p) = &pb { p.set_message("normalize/prune".to_string()); }
        let _ = Self::run_pass_with_test(states, body, "initial normalize_edges_inplace", |s, _b| {
            Self::normalize_edges_inplace(s)
        });
        let _ = Self::run_pass_with_test(states, body, "initial prune_dead_ends", |s, _b| {
            Self::prune_dead_ends(s)
        });
        let _ = Self::run_pass_with_test(states, body, "initial prune_unreachable", |s, b| {
            Self::prune_unreachable(s, b)
        });
        let _ = Self::run_pass_with_test(states, body, "initial constrain finals", |s, b| {
            Self::propagate_and_constrain_weights(s, b)
        });

        let mut changed_any = true;
        let mut passes = 0usize;
        while changed_any && passes < max_passes {
            passes += 1;
            if let Some(p) = &pb { p.inc(1); }
            changed_any = false;

            if let Some(p) = &pb { p.set_message("normalize".to_string()); }
            changed_any |= Self::run_pass_with_test(states, body, "normalize_edges_inplace", |s, _b| {
                Self::normalize_edges_inplace(s)
            });

            if let Some(p) = &pb { p.set_message("absorb sink finals".to_string()); }
            changed_any |=
                Self::run_pass_with_test(states, body, "absorb_sink_finals_into_incoming", |s, _b| {
                    Self::absorb_sink_finals_into_incoming(s)
                });

            // Relax edges locally (cheap) to unlock structure equality.
            if let Some(p) = &pb { p.set_message("relax local future".to_string()); }
            changed_any |=
                Self::run_pass_with_test(states, body, "relax_weights_by_local_future", |s, _b| {
                    Self::relax_weights_by_local_future(s)
                });

            // Propagate full future (heavier) for small automata only.
            if !large {
                if let Some(p) = &pb { p.set_message("propagate future weights".to_string()); }
                changed_any |=
                    Self::run_pass_with_test(states, body, "propagate_future_weights", |s, _b| {
                        Self::propagate_future_weights(s)
                    });
            }

            if !large {
                if let Some(p) = &pb { p.set_message("minimize".to_string()); }
                changed_any |=
                    Self::run_pass_with_test(states, body, "minimize_partition_refinement", |s, b| {
                        Self::minimize_partition_refinement(s, b)
                    });
            }

            if let Some(p) = &pb { p.set_message("normalize".to_string()); }
            changed_any |= Self::run_pass_with_test(states, body, "normalize_edges_inplace", |s, _b| {
                Self::normalize_edges_inplace(s)
            });

            // Constrain finals again with updated reachability (cheap, helps prune/minimize)
            if let Some(p) = &pb { p.set_message("constrain finals".to_string()); }
            changed_any |= Self::run_pass_with_test(states, body, "propagate_and_constrain_weights", |s, b| {
                Self::propagate_and_constrain_weights(s, b)
            });

            if let Some(p) = &pb { p.set_message("prune dead".to_string()); }
            changed_any |= Self::run_pass_with_test(states, body, "prune_dead_ends", |s, _b| {
                Self::prune_dead_ends(s)
            });

            if let Some(p) = &pb { p.set_message("prune unreachable".to_string()); }
            changed_any |= Self::run_pass_with_test(states, body, "prune_unreachable", |s, b| {
                Self::prune_unreachable(s, b)
            });
        }

        if let Some(p) = &pb {
            p.finish_with_message(format!("Simplified to {} states", states.len()));
        }
    }

    pub fn propagate_and_constrain_weights(states: &mut DWAStates, body: &mut DWABody) -> bool {
        let n = states.len();
        if n == 0 {
            return false;
        }

        let mut reachable_weights = vec![Weight::zeros(); n];
        let mut worklist = VecDeque::new();

        if body.start_state >= n {
            return false;
        }
        reachable_weights[body.start_state] = Weight::all();
        worklist.push_back(body.start_state);

        while let Some(u) = worklist.pop_front() {
            let u_rw = reachable_weights[u].clone();
            if u_rw.is_empty() {
                continue;
            }
            let u_state = &states[u];

            // State entry weight gates outgoing paths too
            let gated_u_rw = if let Some(sw) = &u_state.state_weight {
                &u_rw & sw
            } else {
                u_rw
            };

            for (_lbl, v, edge_w) in u_state.iter_edges() {
                if v >= n { continue; }
                let new_v_rw = &gated_u_rw & edge_w;
                let old_v_rw = &reachable_weights[v];

                if !new_v_rw.is_subset_of(old_v_rw) {
                    reachable_weights[v] |= &new_v_rw;
                    worklist.push_back(v);
                }
            }
        }

        // Now apply constraints
        let mut changed = false;
        for i in 0..n {
            let old_fw = states[i].final_weight.clone();
            if let Some(fw) = states[i].final_weight.as_mut() {
                *fw &= &reachable_weights[i];
                if fw.is_empty() {
                    states[i].final_weight = None;
                }
            }
            if states[i].final_weight != old_fw {
                changed = true;
            }
        }
        changed
    }

    pub fn propagate_future_weights(states: &mut DWAStates) -> bool {
        let n = states.len();
        if n == 0 {
            return false;
        }

        // Reverse adjacency (unique preds)
        let mut rev_adj: Vec<Vec<StateID>> = vec![vec![]; n];
        for i in 0..n {
            for (_lbl, v, _w) in states[i].iter_edges() {
                if v < n { rev_adj[v].push(i); }
            }
        }
        for preds in rev_adj.iter_mut() {
            preds.sort_unstable();
            preds.dedup();
        }

        let mut future_weights = vec![Weight::zeros(); n];
        let mut worklist = VecDeque::new();
        for i in 0..n {
            if let Some(fw) = &states[i].final_weight {
                if !fw.is_empty() {
                    future_weights[i] = fw.clone();
                    for &pred in &rev_adj[i] {
                        worklist.push_back(pred);
                    }
                }
            }
        }

        while let Some(u) = worklist.pop_front() {
            let mut u_new_fw = states[u].final_weight.clone().unwrap_or_else(Weight::zeros);

            if let Some(sw) = &states[u].state_weight {
                u_new_fw |= sw;
            }

            for (_lbl, v, edge_w) in states[u].iter_edges() {
                if v < n {
                    u_new_fw |= &(edge_w & &future_weights[v]);
                }
            }

            if u_new_fw != future_weights[u] {
                future_weights[u] = u_new_fw;
                for &pred in &rev_adj[u] {
                    worklist.push_back(pred);
                }
            }
        }

        // Precompute complements to avoid recomputing !future_weights[v] many times
        let not_future: Vec<Weight> = future_weights.iter().map(|w| !w).collect();

        // 2. Update edge weights. An edge weight W can be relaxed to W | !future_weight(target)
        // because any bits not in future_weight(target) would be filtered out anyway.
        let mut any_weight_changed = false;
        for i in 0..n {
            let st = &mut states[i];
            // Default
            if let Some(v) = st.transitions.default {
                if v < n {
                    if let Some(w) = st.trans_weight_default.as_mut() {
                        let new_w = &*w | &not_future[v];
                        if new_w != *w {
                            *w = new_w;
                            any_weight_changed = true;
                        }
                    }
                }
            }
            // Exceptions
            let keys: Vec<i16> = st.transitions.exceptions.keys().copied().collect();
            for ch in keys {
                if let Some(&v) = st.transitions.exceptions.get(&ch) {
                    if v < n {
                        if let Some(w) = st.trans_weights_exceptions.get_mut(&ch) {
                            let new_w = &*w | &not_future[v];
                            if new_w != *w {
                                *w = new_w;
                                any_weight_changed = true;
                            }
                        }
                    }
                }
            }
        }

        any_weight_changed
    }

    /// Fast, single-pass local relaxation:
    /// For each state v, compute an upper bound U[v] on future acceptance:
    ///     U[v] = (state_weight[v] if present else ALL) ∧ (final_weight[v] ∪ default_weight[v] ∪ ⋃ exception_weights[v])
    /// Then S[v] = ¬U[v] ⊆ ¬F[v], so for each edge u→v we can safely set weight(u→v) := weight(u→v) ∪ S[v].
    /// This preserves semantics and is O(E) per pass.
    pub fn relax_weights_by_local_future(states: &mut DWAStates) -> bool {
        let n = states.len();
        if n == 0 {
            return false;
        }
        // 1) Compute U[v] for all v
        let mut upper: Vec<Weight> = Vec::with_capacity(n);
        for i in 0..n {
            let mut u = states[i].final_weight.clone().unwrap_or_else(Weight::zeros);
            if let Some(w) = states[i].trans_weight_default.as_ref() {
                u |= w;
            }
            for w in states[i].trans_weights_exceptions.values() {
                u |= w;
            }
            if let Some(sw) = states[i].state_weight.as_ref() {
                u &= sw;
            }
            upper.push(u);
        }
        // 2) Precompute complements S[v] = ¬U[v]
        let not_upper: Vec<Weight> = upper.iter().map(|w| !w).collect();
        // 3) Relax all edges using S[target]
        let mut any_weight_changed = false;
        for i in 0..n {
            let st = &mut states[i];
            if let Some(v) = st.transitions.default {
                if let Some(w) = st.trans_weight_default.as_mut() {
                    let new_w = &*w | &not_upper[v];
                    if new_w != *w {
                        *w = new_w;
                        any_weight_changed = true;
                    }
                }
            }
            let keys: Vec<i16> = st.transitions.exceptions.keys().copied().collect();
            for ch in keys {
                if let Some(&v) = st.transitions.exceptions.get(&ch) {
                    if let Some(w) = st.trans_weights_exceptions.get_mut(&ch) {
                        let new_w = &*w | &not_upper[v];
                        if new_w != *w {
                            *w = new_w;
                            any_weight_changed = true;
                        }
                    }
                }
            }
        }
        any_weight_changed
    }

    /// Absorb final weights of sink states (no outgoing edges) into incoming edges.
    /// For each sink v with final_weight F (non-empty, not ALL):
    ///   - For every incoming edge e: u -> v, set weight(e) := weight(e) ∧ F.
    ///   - Set final_weight[v] := ALL.
    /// This preserves semantics (any accepted word ending in v has its weight gated by F either way)
    /// and enables merging of multiple structurally identical sinks that previously differed only by final_weight.
    pub fn absorb_sink_finals_into_incoming(states: &mut DWAStates) -> bool {
        let n = states.len();
        if n == 0 {
            return false;
        }
        // Build reverse adjacency: for each target v, collect predecessors
        let mut def_preds: Vec<Vec<StateID>> = vec![Vec::new(); n];
        let mut ex_preds: Vec<Vec<(StateID, i16)>> = vec![Vec::new(); n];
        for p in 0..n {
            if let Some(d) = states[p].transitions.default {
                if d < n {
                    def_preds[d].push(p);
                }
            }
            for (ch, &tgt) in states[p].transitions.exceptions.iter() {
                if tgt < n {
                    ex_preds[tgt].push((p, *ch));
                }
            }
        }

        let mut changed = false;
        for v in 0..n {
            // Sink = no outgoing edges (no default and no exceptions)
            if states[v].transitions.default.is_some() || !states[v].transitions.exceptions.is_empty() {
                continue;
            }

            // The effective weight is the intersection of final_weight and state_weight.
            let mut effective_weight = Weight::all();
            let mut has_restriction = false;

            if let Some(fw) = &states[v].final_weight {
                if !fw.is_all_fast() {
                    effective_weight &= fw;
                    has_restriction = true;
                }
            }
            if let Some(sw) = &states[v].state_weight {
                if !sw.is_all_fast() {
                    effective_weight &= sw;
                    has_restriction = true;
                }
            }

            if !has_restriction {
                continue;
            }

            // This transformation is only valid if there are incoming edges to absorb the weight.
            if def_preds[v].is_empty() && ex_preds[v].is_empty() {
                continue;
            }

            // Intersect incoming default edges
            for &p in &def_preds[v] {
                if let Some(w) = states[p].trans_weight_default.as_mut() {
                    let old_w = w.clone();
                    *w &= &effective_weight;
                    if *w != old_w {
                        changed = true;
                    }
                }
            }
            // Intersect incoming exception edges
            for &(p, ch) in &ex_preds[v] {
                if let Some(w) = states[p].trans_weights_exceptions.get_mut(&ch) {
                    let old_w = w.clone();
                    *w &= &effective_weight;
                    if *w != old_w {
                        changed = true;
                    }
                }
            }
            // Make the sink final weight ALL and clear state weight to enable merging
            if states[v].final_weight.as_ref().map_or(false, |w| !w.is_all_fast()) {
                states[v].final_weight = Some(Weight::all());
                changed = true;
            }
            if states[v].state_weight.is_some() {
                states[v].state_weight = None;
                changed = true;
            }
        }
        changed
    }

    pub fn normalize_edges_inplace(states: &mut DWAStates) -> bool {
        let mut changed = false;
        for st in &mut states.0 {
            let before = st.transitions.exceptions.len();
            if let (Some(def_tgt), Some(def_w)) = (st.transitions.default, &st.trans_weight_default) {
                st.transitions.exceptions.retain(|ch, &mut tgt| {
                    if tgt != def_tgt {
                        return true; // Different target, keep.
                    }
                    // Same target, check weight. An exception is redundant only if its weight matches the default.
                    st.trans_weights_exceptions.get(ch) != Some(def_w)
                });
            } else if let Some(def_tgt) = st.transitions.default {
                // Default transition exists but has no weight. This is an inconsistent state.
                // For safety, just retain based on target.
                st.transitions.exceptions.retain(|_, &mut tgt| tgt != def_tgt);
            }
            changed |= st.transitions.exceptions.len() != before;

            let before_w = st.trans_weights_exceptions.len();
            st.trans_weights_exceptions.retain(|ch, _| st.transitions.exceptions.contains_key(ch));
            changed |= st.trans_weights_exceptions.len() != before_w;
        }
        changed
    }

    /// Partition-refinement minimization (structure-only), aggregating weights by union.
    pub fn minimize_partition_refinement(states: &mut DWAStates, body: &mut DWABody) -> bool {
        let n = states.0.len();
        if n <= 1 {
            return false;
        }

        // Initial partition by outputs (state_weight, final_weight).
        let mut part: Vec<usize> = vec![0; n];
        let mut canon0: HashMap<(Option<Weight>, Option<Weight>), usize> = HashMap::new();
        for i in 0..n {
            let key = (states[i].state_weight.clone(), states[i].final_weight.clone());
            let next_id = canon0.len();
            part[i] = *canon0.entry(key).or_insert(next_id);
        }

        // Refine until stable
        let mut changed = true;
        let mut rounds = 0usize;
        while changed && rounds < 30 {
            rounds += 1;
            changed = false;
            let mut next_part: Vec<usize> = vec![0; n];
            let mut sig2pid: HashMap<(
                Option<Weight>,
                Option<Weight>,
                Option<(usize, Weight)>,
                Vec<(i16, (usize, Weight))>,
            ), usize> = HashMap::new();

            for i in 0..n {
                let st = &states[i];
                let def_sig = st.transitions.default.map(|d| {
                    (part[d], st.trans_weight_default.as_ref().cloned().unwrap_or_else(Weight::all))
                });
                // Keep only exceptions that structurally differ from default (dest or weight).
                let ex_sig: Vec<_> = st.transitions.exceptions.iter()
                    .map(|(ch, &tgt)| (*ch, (part[tgt], st.trans_weights_exceptions.get(ch).cloned().unwrap_or_else(Weight::all))))
                    .filter(|(_, (cls, w))| def_sig.as_ref().map_or(true, |(dc, dw)| *dc != *cls || dw != w))
                    .collect();
                let sig = (st.state_weight.clone(), st.final_weight.clone(), def_sig, ex_sig);
                let next_pid = sig2pid.len();
                next_part[i] = *sig2pid.entry(sig).or_insert(next_pid);
            }
            if next_part != part {
                part = next_part;
                changed = true;
            }
        }

        // Build groups
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in part.iter().enumerate() {
            groups.entry(*p).or_default().push(i);
        }
        if groups.len() == n {
            return false;
        }

        // Map partition id -> new state id
        let mut pid_to_new: HashMap<usize, usize> = HashMap::new();
        let mut new_states: Vec<DWAState> = vec![DWAState::default(); groups.len()];

        // Pre-assign ids
        for (pid, _) in &groups {
            let new_id = pid_to_new.len();
            pid_to_new.insert(*pid, new_id);
        }

        // Rebuild states by copying representative weights and remapping targets by partition.
        for (pid, members) in &groups {
            let rep = members[0];
            let rep_state = &states[rep];
            let new_id = *pid_to_new.get(pid).unwrap();

            let mut st = DWAState::default();
            st.state_weight = rep_state.state_weight.clone();
            st.final_weight = rep_state.final_weight.clone();

            // Default
            if let Some(d) = rep_state.transitions.default {
                let cls = part[d];
                st.transitions.default = Some(*pid_to_new.get(&cls).unwrap());
                st.trans_weight_default = rep_state.trans_weight_default.clone();
            }

            // Exceptions (copy rep structure; members have the same by construction)
            for (ch, tgt) in &rep_state.transitions.exceptions {
                let cls = part[*tgt];
                st.transitions.exceptions.insert(*ch, *pid_to_new.get(&cls).unwrap());
                if let Some(w) = rep_state.trans_weights_exceptions.get(ch) {
                    st.trans_weights_exceptions.insert(*ch, w.clone());
                }
            }

            new_states[new_id] = st;
        }

        states.0 = new_states;
        let _ = Self::normalize_edges_inplace(states);
        let start_pid = part[body.start_state];
        body.start_state = *pid_to_new.get(&start_pid).unwrap();

        true
    }

    pub fn prune_dead_ends(states: &mut DWAStates) -> bool {
        let n = states.len();
        if n == 0 {
            return false;
        }

        // 1. Backward reachability from final states to find "live" states.
        let mut live = vec![false; n];
        let mut q_live: VecDeque<usize> = VecDeque::new();
        let mut rev_adj: Vec<Vec<usize>> = vec![vec![]; n];
        for i in 0..n {
            if states[i].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                live[i] = true;
                q_live.push_back(i);
            }
            for (_lbl, v, _w) in states[i].iter_edges() {
                if v < n {
                    rev_adj[v].push(i);
                }
            }
        }
        while let Some(u) = q_live.pop_front() {
            for &v in &rev_adj[u] {
                if !live[v] {
                    live[v] = true;
                    q_live.push_back(v);
                }
            }
        }

        // 2. Remove transitions to non-live states, preserving correctness.
        let mut changed = false;
        for i in 0..n {
            let st = &mut states[i];

            // Check if default transition goes to a live state.
            // This check must happen before we modify the default transition.
            let default_goes_live = st.transitions.default.map_or(false, |d| d < n && live[d]);

            // A default transition to a dead state can be removed. This is equivalent to
            // transitioning to an implicit sink state.
            if let Some(d) = st.transitions.default {
                if d < n && !live[d] {
                    st.transitions.default = None;
                    st.trans_weight_default = None;
                    changed = true;
                }
            }

            let before = st.transitions.exceptions.len();
            st.transitions.exceptions.retain(|_, tgt| {
                // Keep an exception if its target is live.
                // Also keep it if it overrides a default transition that goes to a live state,
                // even if the exception's own target is dead.
                (*tgt < n && live[*tgt]) || default_goes_live
            });

            if st.transitions.exceptions.len() != before {
                changed = true;
                // Clean up corresponding weights for removed exception transitions.
                st.trans_weights_exceptions.retain(|ch, _| st.transitions.exceptions.contains_key(ch));
            }
        }
        changed
    }

    pub fn prune_unreachable(states: &mut DWAStates, body: &mut DWABody) -> bool {
        if states.0.is_empty() {
            return false;
        }
        let n = states.0.len();

        // 1. Backward reachability from final states to find "live" states.
        let mut visited = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        if body.start_state < n {
            visited[body.start_state] = true;
            q.push_back(body.start_state);
        } else {
            // Start state is out of bounds, everything is unreachable.
            if n > 0 {
                states.0.clear();
                body.start_state = states.add_state();
                return true;
            }
            return false;
        }
        while let Some(u) = q.pop_front() {
            for (_lbl, v, _w) in states[u].iter_edges() {
                if v < n && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }

        let num_reachable = visited.iter().filter(|&&b| b).count();
        if num_reachable == n {
            return false;
        }

        // 2. Remap kept states.
        let mut map = vec![usize::MAX; n];
        let mut new_states: Vec<DWAState> = Vec::with_capacity(num_reachable);
        for i in 0..n {
            if visited[i] {
                map[i] = new_states.len();
                new_states.push(states[i].clone());
            }
        }

        for st in &mut new_states {
            if let Some(d) = st.transitions.default.as_mut() {
                *d = map[*d];
            }
            for tgt in st.transitions.exceptions.values_mut() {
                *tgt = map[*tgt];
            }
        }
        states.0 = new_states;
        if num_reachable > 0 {
            body.start_state = map[body.start_state];
        } else {
            // This case should be handled by the start_state check above, but for safety:
            states.0.clear();
            body.start_state = states.add_state();
        }
        true
    }
}

impl NWA {
    fn run_pass(
        pb: &Option<ProgressBar>,
        msg: &str,
        changed_any: &mut bool,
        mut pass: impl FnMut() -> bool,
    ) {
        if let Some(p) = pb {
            p.set_message(msg.to_string());
        }
        let now = Instant::now();
        let changed = pass();
        let elapsed = now.elapsed();
        crate::debug!(3, "NWA simplify pass '{}' took {:?} (changed: {})", msg, elapsed, changed);
        if changed {
            *changed_any = true;
        }
    }

    pub fn simplify(&mut self) -> bool {
        let now = Instant::now();
        let initial_n = self.states.len();
        let max_passes = 12;
        let pb = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(max_passes as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Simplifying NWA: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} passes ({msg})")
                    .expect("progress-bar"),
            );
            Some(p)
        } else {
            None
        };

        let mut changed = true;
        let mut passes = 0;
        while changed && passes < max_passes {
            passes += 1;
            if let Some(p) = &pb { p.inc(1); }
            changed = false;

            Self::run_pass(&pb, "normalize", &mut changed, || self.normalize_edges_inplace());
            Self::run_pass(&pb, "dedup labeled", &mut changed, || self.dedup_labeled_edges());
            Self::run_pass(&pb, "dedup epsilons", &mut changed, || self.dedup_epsilon_edges());
            Self::run_pass(&pb, "unify defaults", &mut changed, || self.unify_default_transitions());

            Self::run_pass(&pb, "unify final states", &mut changed, || self.unify_final_states());
            Self::run_pass(&pb, "bypass ε-chains", &mut changed, || self.bypass_trivial_epsilon_chains());
            Self::run_pass(&pb, "collapse SCCs", &mut changed, || self.collapse_all_weight_epsilon_sccs());
            Self::run_pass(&pb, "prune unreachable", &mut changed, || self.prune_unreachable());
            Self::run_pass(&pb, "prune dead ends", &mut changed, || self.prune_dead_ends());
            Self::run_pass(&pb, "merge equivalent", &mut changed, || self.merge_equivalent_states_partition());
            Self::run_pass(&pb, "normalize", &mut changed, || self.normalize_edges_inplace());
        }
        if let Some(p) = &pb { p.finish_with_message(format!("Simplified to {} states", self.states.len())); }
        crate::debug!(3, "NWA::simplify ({} states -> {} states) took: {:?}", initial_n, self.states.len(), now.elapsed());
        self.states.len() != initial_n
    }

    fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }

        let mut reachable = vec![false; n];
        let mut q = VecDeque::new();

        if self.body.start_state < n {
            reachable[self.body.start_state] = true;
            q.push_back(self.body.start_state);
        } else {
            let changed = n > 0;
            if changed {
                self.states.0.clear();
                self.body.start_state = self.states.add_state();
            }
            return changed;
        }

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];
            for (v, _) in &st.epsilons {
                if *v < n && !reachable[*v] { reachable[*v] = true; q.push_back(*v); }
            }
            for (_, targets) in &st.transitions {
                for (v, _) in targets {
                    if *v < n && !reachable[*v] { reachable[*v] = true; q.push_back(*v); }
                }
            }
            for def in &st.default {
                if def.target < n && !reachable[def.target] {
                    reachable[def.target] = true; q.push_back(def.target);
                }
            }
        }

        let num_reachable = reachable.iter().filter(|&&b| b).count();
        if num_reachable == n { return false; }

        let mut remap = vec![usize::MAX; n];
        let mut new_states_vec = Vec::with_capacity(num_reachable);
        for i in 0..n {
            if reachable[i] {
                remap[i] = new_states_vec.len();
                new_states_vec.push(self.states[i].clone());
            }
        }

        for st in &mut new_states_vec {
            st.epsilons.iter_mut().for_each(|(v, _)| *v = remap[*v]);
            st.transitions.values_mut().for_each(|targets| {
                for (v, _) in targets {
                    *v = remap[*v];
                }
            });
            for def in &mut st.default {
                def.target = remap[def.target]
            }
        }

        self.states.0 = new_states_vec;
        self.body.start_state = remap[self.body.start_state];

        true
    }

    fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }

        let fut = self.compute_future_weights();
        let live: Vec<bool> = fut.iter().map(|w| !w.is_empty()).collect();

        if self.body.start_state >= n || !live[self.body.start_state] {
            let changed = n > 0;
            self.states.0.clear();
            self.body.start_state = self.states.add_state();
            return changed;
        }

        // Relax edges using future weights (sound: (w | !F[v]) & F[v] == w & F[v])
        let not_fut: Vec<Weight> = fut.iter().map(|w| !w).collect();
        let mut changed_weights = false;
        for i in 0..n {
            let st = &mut self.states[i];
            for (v, w) in &mut st.epsilons {
                let new_w = &*w | &not_fut[*v];
                if new_w != *w { *w = new_w; changed_weights = true; }
            }
            for (_, targets) in &mut st.transitions {
                for (v, w) in targets {
                    let new_w = &*w | &not_fut[*v];
                    if new_w != *w { *w = new_w; changed_weights = true; }
                }
            }
            for def in &mut st.default {
                let new_w = &def.weight | &not_fut[def.target];
                if new_w != def.weight { def.weight = new_w; changed_weights = true; }
            }
        }

        let num_live = live.iter().filter(|&&b| b).count();
        if num_live == n {
            return changed_weights;
        }

        let mut remap = vec![usize::MAX; n];
        let mut new_states_vec = Vec::with_capacity(num_live);
        for i in 0..n {
            if live[i] {
                remap[i] = new_states_vec.len();
                new_states_vec.push(self.states[i].clone());
            }
        }

        for st in &mut new_states_vec {
            st.epsilons.retain(|(v, _)| live[*v]);
            st.epsilons.iter_mut().for_each(|(v, _)| *v = remap[*v]);

            st.transitions.values_mut().for_each(|targets| {
                targets.retain(|(v, _)| live[*v]);
                targets.iter_mut().for_each(|(v, _)| *v = remap[*v]);
            });
            st.transitions.retain(|_, targets| !targets.is_empty());

            st.default.retain(|def| live[def.target]);
            st.default.iter_mut().for_each(|def| def.target = remap[def.target]);
        }

        self.states.0 = new_states_vec;
        self.body.start_state = remap[self.body.start_state];

        true
    }

    fn collapse_all_weight_epsilon_sccs(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }

        let mut index = 0;
        let mut indices = vec![usize::MAX; n];
        let mut lowlink = vec![0; n];
        let mut on_stack = vec![false; n];
        let mut stack = Vec::new();
        let mut comp_of = vec![0; n];
        let mut comps: Vec<Vec<NWAStateID>> = Vec::new();

        for i in 0..n {
            if indices[i] == usize::MAX {
                self.strongconnect(i, &mut index, &mut indices, &mut lowlink, &mut on_stack, &mut stack, &mut comp_of, &mut comps);
            }
        }

        if comps.len() == n { return false; }

        let mut new_states_vec = Vec::with_capacity(comps.len());
        for (cid, comp) in comps.iter().enumerate() {
            let mut new_state = NWAState::default();

            let mut final_weight: Option<Weight> = None;
            for &sid in comp {
                if let Some(fw) = &self.states[sid].final_weight {
                    if let Some(acc) = &mut final_weight { *acc |= fw; } else { final_weight = Some(fw.clone()); }
                }
            }
            new_state.final_weight = final_weight;

            for &sid in comp {
                for (&lbl, targets) in &self.states[sid].transitions {
                    for &(to, ref w) in targets {
                        let to_comp = comp_of[to];
                        Self::add_transition_to_state(&mut new_state, lbl, to_comp, w.clone());
                    }
                }
                for def in &self.states[sid].default {
                    Self::add_default_transition_to_state(&mut new_state, comp_of[def.target], def.weight.clone(), def.exceptions.clone());
                }
                for &(to, ref w) in &self.states[sid].epsilons {
                    let to_comp = comp_of[to];
                    if cid != to_comp || !w.is_all_fast() {
                        new_state.epsilons.push((to_comp, w.clone()));
                    }
                }
            }
            new_states_vec.push(new_state);
        }

        self.states.0 = new_states_vec;
        self.body.start_state = comp_of[self.body.start_state];

        true
    }

    fn strongconnect(&self, v: NWAStateID, index: &mut usize, indices: &mut [usize], lowlink: &mut [usize], on_stack: &mut [bool], stack: &mut Vec<NWAStateID>, comp_of: &mut [usize], comps: &mut Vec<Vec<NWAStateID>>) {
        indices[v] = *index;
        lowlink[v] = *index;
        *index += 1;
        stack.push(v);
        on_stack[v] = true;

        for (w, weight) in &self.states[v].epsilons {
            if weight.is_all_fast() {
                if indices[*w] == usize::MAX {
                    self.strongconnect(*w, index, indices, lowlink, on_stack, stack, comp_of, comps);
                    lowlink[v] = lowlink[v].min(lowlink[*w]);
                } else if on_stack[*w] {
                    lowlink[v] = lowlink[v].min(indices[*w]);
                }
            }
        }

        if lowlink[v] == indices[v] {
            let mut comp = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack[w] = false;
                comp_of[w] = comps.len();
                comp.push(w);
                if w == v { break; }
            }
            comps.push(comp);
        }
    }

    fn add_transition_to_state(state: &mut NWAState, on: i16, to: NWAStateID, w: Weight) {
        let targets = state.transitions.entry(on).or_default();
        if let Some((_, existing_w)) = targets.iter_mut().find(|(t, _)| *t == to) {
            *existing_w |= &w;
        } else {
            targets.push((to, w));
        }
    }

    fn add_default_transition_to_state(state: &mut NWAState, to: NWAStateID, w: Weight, exceptions: BTreeSet<i16>) {
        if let Some(old_def) = state.default.iter_mut().find(|d| d.target == to && d.exceptions == exceptions) {
            old_def.weight |= &w;
        } else {
            state.default.push(NWADefaultTransition {
                target: to,
                weight: w,
                exceptions,
            });
        }
    }

    /// New: Merge duplicate labeled edges by unioning weights per (label, target).
    fn dedup_labeled_edges(&mut self) -> bool {
        let mut changed = false;
        for st in &mut self.states.0 {
            for targets in st.transitions.values_mut() {
                if targets.len() <= 1 {
                    continue;
                }
                let old_targets = targets.clone();

                let mut acc: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
                for (to, w) in targets.iter() {
                    *acc.entry(*to).or_insert_with(Weight::zeros) |= w;
                }

                let new_targets: Vec<_> = acc.into_iter().collect();
                if new_targets != old_targets {
                    *targets = new_targets;
                    changed = true;
                }
            }
        }
        changed
    }

    /// New: Merge multiple default transitions with identical (target, weight) by intersecting their exception sets.
    /// Proof: Presence condition across defaults is (∃i: l ∉ Ei) for label l. A single default with exceptions E=⋂Ei
    /// has presence l ∉ E, which is equivalent.
    fn unify_default_transitions(&mut self) -> bool {
        let mut changed = false;
        for st in &mut self.states.0 {
            if st.default.len() <= 1 {
                continue;
            }
            let mut merged: Vec<NWADefaultTransition> = Vec::new();
            for def in st.default.clone() {
                if let Some(ex) = merged.iter_mut().find(|d| d.target == def.target && d.weight == def.weight) {
                    // Intersect exception sets
                    let a = std::mem::take(&mut ex.exceptions);
                    let b = def.exceptions;
                    let inter: BTreeSet<i16> = a.intersection(&b).copied().collect();
                    ex.exceptions = inter;
                    changed = true;
                } else {
                    merged.push(def);
                }
            }
            // Keep canonical ordering
            if merged != st.default {
                st.default = merged;
                changed = true;
            }
        }
        changed
    }
}

impl NWA {
    /// Normalize in-place:
    /// - Remove empty-weight edges (ε, labeled, and default)
    /// - Drop labeled transitions that are identical to the default (same target and weight)
    fn normalize_edges_inplace(&mut self) -> bool {
        let mut changed = false;
        for st in &mut self.states.0 {
            // Remove empty-weight epsilons
            let before_eps = st.epsilons.len();
            st.epsilons.retain(|(_, w)| !w.is_empty());
            if st.epsilons.len() != before_eps { changed = true; }

            // Remove empty-weight labeled transitions
            let before_lbl = st.transitions.len();
            st.transitions.values_mut().for_each(|targets| targets.retain(|(_, w)| !w.is_empty()));
            st.transitions.retain(|_, targets| !targets.is_empty());
            if st.transitions.len() != before_lbl { changed = true; }

            // Remove empty-weight default
            let before_def = st.default.len();
            st.default.retain(|def| !def.weight.is_empty());
            if st.default.len() != before_def { changed = true; }

            // NOTE: The old logic for removing labeled transitions identical to default is no longer valid,
            // as there can be multiple default transitions with different exception sets. This was an
            // optimization, so removing it preserves correctness.
        }
        changed
    }

    /// Deduplicate epsilon edges to the same target by unioning their weights.
    fn dedup_epsilon_edges(&mut self) -> bool {
        let mut changed = false;
        for st in &mut self.states.0 {
            if st.epsilons.len() <= 1 {
                continue;
            }
            let mut acc: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
            for (to, w) in &st.epsilons {
                let e = acc.entry(*to).or_insert_with(Weight::zeros);
                *e |= w;
            }
            let new_eps: Vec<(NWAStateID, Weight)> = acc.into_iter().collect();
            if new_eps != st.epsilons {
                st.epsilons = new_eps;
                changed = true;
            }
        }
        changed
    }

    /// Consolidate all final states into a single new final state.
    /// For each state `s` with `final_weight` `w`, add an ε-transition from `s` to a new
    /// final state `F` with weight `w`, and then clear `s.final_weight`. `F` will have
    /// `final_weight = ALL`.
    fn unify_final_states(&mut self) -> bool {
        let final_states: Vec<(NWAStateID, Weight)> = self
            .states
            .0
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.final_weight.clone().map(|w| (i, w)))
            .collect();

        if final_states.len() <= 1 {
            return false;
        }

        let new_final_id = self.states.add_state();
        self.states[new_final_id].final_weight = Some(Weight::all());

        for (sid, weight) in final_states {
            if !weight.is_empty() {
                self.states.add_epsilon(sid, new_final_id, weight);
            }
            self.states[sid].final_weight = None;
        }

        true
    }

    /// Bypass trivial ε-chains: if a state s
    /// - has no labeled transitions
    /// - has no default
    /// - has no final weight
    /// - has exactly one ε edge with ALL weight to t (t != s)
    /// then redirect all incoming edges targeting s to t and (after a prune) drop s.
    fn bypass_trivial_epsilon_chains(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }
        let mut bypass: Vec<Option<NWAStateID>> = vec![None; n];
        for s in 0..n {
            let st = &self.states[s];
            if st.final_weight.is_some() { continue; }
            if !st.transitions.is_empty() { continue; }
            if !st.default.is_empty() { continue; }
            if st.epsilons.len() != 1 { continue; }
            let (t, w) = &st.epsilons[0];
            if !w.is_all_fast() { continue; }
            if *t == s { continue; }
            bypass[s] = Some(*t);
        }
        if bypass.iter().all(|o| o.is_none()) {
            return false;
        }

        // Compute ultimate mapping along bypass chains (path compression-like).
        let mut ultimate: Vec<NWAStateID> = (0..n).collect();
        for i in 0..n {
            let mut v = i;
            while let Some(t) = bypass[v] {
                if t == v { break; }
                v = t;
            }
            ultimate[i] = v;
        }

        let mut changed = false;
        for st in &mut self.states.0 {
            for (v, _) in &mut st.epsilons {
                let nv = ultimate[*v];
                if nv != *v { *v = nv; changed = true; }
            }
            for (_, targets) in &mut st.transitions {
                for (v, _) in targets {
                    let nv = ultimate[*v];
                    if nv != *v { *v = nv; changed = true; }
                }
            }
            for def in &mut st.default {
                let nv = ultimate[def.target];
                if nv != def.target {
                    def.target = nv;
                    changed = true;
                }
            }
        }
        let new_start = ultimate[self.body.start_state];
        if new_start != self.body.start_state {
            self.body.start_state = new_start;
            changed = true;
        }
        if changed {
            changed |= self.prune_unreachable();
        }
        changed
    }

    /// Merge equivalent NWA states by partition refinement.
    /// Signature per state includes:
    ///  - final_weight
    ///  - default: Option<(class(target), weight)>
    ///  - epsilons: vector of (class(target), UNION(weight to that class)), sorted by class
    ///  - labeled transitions: vector of (label, class(target), weight), sorted by label
    fn merge_equivalent_states_partition(&mut self) -> bool {
        let n = self.states.len();
        if n <= 1 {
            return false;
        }
        // Initial partition by final_weight only (coarse).
        let mut part: Vec<usize> = vec![0; n];
        let mut canon0: HashMap<Option<Weight>, usize> = HashMap::new();
        for i in 0..n {
            let key = self.states[i].final_weight.clone();
            let next_id = canon0.len();
            part[i] = *canon0.entry(key).or_insert(next_id);
        }

        // Iteratively refine using structural signatures keyed by current partitions.
        let mut changed = true;
        let mut rounds = 0usize;
        while changed && rounds < 30 {
            rounds += 1;
            changed = false;
            let mut next_part: Vec<usize> = vec![0; n];
            let mut sig2pid: HashMap<
                (
                    Option<Weight>,
                    Vec<(usize, Weight, BTreeSet<i16>)>,
                    Vec<(usize, Weight)>,
                    Vec<(i16, usize, Weight)>,
                ),
                usize,
            > = HashMap::new();

            for i in 0..n {
                let st = &self.states[i];
                let mut def_sig: Vec<_> = st.default.iter().map(|def| (part[def.target], def.weight.clone(), def.exceptions.clone())).collect();
                def_sig.sort_unstable();

                let mut eps_map: BTreeMap<usize, Weight> = BTreeMap::new();
                for (to, w) in &st.epsilons { *eps_map.entry(part[*to]).or_default() |= w; }
                let eps_sig: Vec<(usize, Weight)> = eps_map.into_iter().collect();

                let mut lbl_map: BTreeMap<(i16, usize), Weight> = BTreeMap::new();
                for (lbl, targets) in &st.transitions { for (to, w) in targets { *lbl_map.entry((*lbl, part[*to])).or_default() |= w; } }
                let lbl_sig: Vec<(i16, usize, Weight)> = lbl_map.into_iter().map(|((lbl, p), w)| (lbl, p, w)).collect();

                let sig = (st.final_weight.clone(), def_sig, eps_sig, lbl_sig);
                let pid_next = sig2pid.len();
                next_part[i] = *sig2pid.entry(sig).or_insert(pid_next);
            }
            if next_part != part {
                part = next_part;
                changed = true;
            }
        }

        // Group states by final partition id
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in part.iter().enumerate() {
            groups.entry(*p).or_default().push(i);
        }
        if groups.len() == n {
            return false;
        }

        // Map partition id -> new state id
        let mut pid_to_new: HashMap<usize, usize> = HashMap::new();
        let mut new_states: Vec<NWAState> = Vec::with_capacity(groups.len());
        for (pid, members) in &groups {
            let rep = members[0];
            let mut st = self.states[rep].clone();

            // Fix targets to new partition ids
            // Default
            let mut new_default = Vec::new();
            for def in st.default {
                let cls = part[def.target];
                new_default.push(NWADefaultTransition {
                    target: cls,
                    weight: def.weight,
                    exceptions: def.exceptions,
                });
            }
            st.default = new_default;
            // Labeled transitions
            let trans = st.transitions.clone();
            st.transitions.clear();
            for (lbl, targets) in trans {
                let mut new_targets = Vec::new();
                for (to, w) in targets {
                    let cls = part[to];
                    new_targets.push((cls, w));
                }
                st.transitions.insert(lbl, new_targets);
            }
            // Epsilons (aggregate after class remap)
            let eps = st.epsilons.clone();
            st.epsilons.clear();
            for (to, w) in eps {
                let cls = part[to];
                st.epsilons.push((cls, w));
            }

            pid_to_new.insert(*pid, new_states.len());
            new_states.push(st);
        }

        // Rewrite class ids in edges to actual new state ids and deduplicate ε-edges
        for st in &mut new_states {
            // Default
            let mut new_default = Vec::new();
            for def in &st.default {
                new_default.push(NWADefaultTransition {
                    target: *pid_to_new.get(&def.target).expect("missing class"),
                    weight: def.weight.clone(),
                    exceptions: def.exceptions.clone(),
                });
            }
            st.default = new_default;
            // Labeled
            let trans = st.transitions.clone();
            st.transitions.clear();
            for (lbl, targets) in trans {
                let mut new_targets = Vec::new();
                for (cls, w) in targets {
                    let to_new = *pid_to_new.get(&cls).expect("missing class");
                    new_targets.push((to_new, w));
                }
                st.transitions.insert(lbl, new_targets);
            }
            // Epsilons
            let eps = st.epsilons.clone();
            st.epsilons.clear();
            let mut acc: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
            for (cls, w) in eps {
                let to_new = *pid_to_new.get(&cls).expect("missing class");
                let e = acc.entry(to_new).or_insert_with(Weight::zeros);
                *e |= &w;
            }
            st.epsilons = acc.into_iter().collect();
        }

        self.states.0 = new_states;
        self.body.start_state = *pid_to_new.get(&part[self.body.start_state]).expect("missing start class");
        let _ = self.normalize_edges_inplace();
        let _ = self.dedup_epsilon_edges();

        true
    }
}
