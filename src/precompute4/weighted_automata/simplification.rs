// src/precompute4/weighted_automata/simplification.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{StateID, Weight, STOCHASTIC_DEBUG};
use super::dwa::{DWABody, DWAState, DWAStates, DWA};
use super::nwa::{NWAState, NWAStates, NWA};
use crate::precompute4::test_weighted_automata;
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Instant};

/// For very large DWAs, we skip heavy fixpoint/minimization passes to guarantee fast simplification.
/// This is semantics-preserving; it only reduces the amount of compression performed.
const LARGE_AUTOMATON_THRESHOLD: usize = 200_000;

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
            test_weighted_automata::assert_dwa_equivalent(before, self.clone());
            IN_SIMPLIFY_CHECK.with(|c| c.set(false));
        }
    }

    pub fn simplify_components(states: &mut DWAStates, body: &mut DWABody) {
        let now = Instant::now();
        let initial_len = states.len();
        if states.0.is_empty() {
            return;
        }
        if states.len() > LARGE_AUTOMATON_THRESHOLD {
            Self::simplify_large(states, body);
        } else {
            Self::simplify_small(states, body);
        }
        crate::debug!(3, "DWA::simplify_components ({} states -> {} states) took: {:?}", initial_len, states.len(), now.elapsed());
    }

    fn run_pass(
        pb: &Option<ProgressBar>,
        msg: &str,
        changed_any: &mut bool,
        mut pass: impl FnMut() -> bool,
    ) {
        if let Some(p) = pb {
            p.set_message(msg.to_string());
        }
        if pass() {
            *changed_any = true;
        }
    }

    fn simplify_small(states: &mut DWAStates, body: &mut DWABody) {
        let max_passes: usize = 10;
        let pb = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(max_passes as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Simplifying DWA (small): {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} passes ({msg})")
                    .expect("progress-bar"),
            );
            Some(p)
        } else {
            None
        };
        if let Some(p) = &pb { p.set_message("Initial normalize/prune"); }
        Self::normalize_edges_inplace(states);
        Self::prune_unreachable(states, body);
        let mut changed_any = true;
        let mut passes = 0usize;
        while changed_any && passes < max_passes {
            passes += 1;
            if let Some(p) = &pb { p.inc(1); }
            changed_any = false;
            Self::run_pass(&pb, "normalize", &mut changed_any, || Self::normalize_edges_inplace(states));
            // Self::run_pass(&pb, "propagate constraints", &mut changed_any, || Self::propagate_and_constrain_weights(states, body));
            // Self::run_pass(&pb, "propagate future", &mut changed_any, || Self::propagate_future_weights(states));
            Self::run_pass(&pb, "minimize", &mut changed_any, || Self::minimize_partition_refinement(states, body));
            Self::run_pass(&pb, "normalize", &mut changed_any, || Self::normalize_edges_inplace(states));
            Self::run_pass(&pb, "prune", &mut changed_any, || Self::prune_unreachable(states, body));
        }
        if let Some(p) = &pb {
            p.finish_with_message(format!("Simplified to {} states", states.len()));
        }
    }

    fn simplify_large(states: &mut DWAStates, body: &mut DWABody) {
        let max_passes: usize = 2;
        let pb = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(max_passes as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Simplifying DWA (large): {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} passes ({msg})")
                    .expect("progress-bar"),
            );
            Some(p)
        } else {
            None
        };
        if let Some(p) = &pb { p.set_message("Initial normalize/prune"); }
        Self::normalize_edges_inplace(states);
        Self::prune_unreachable(states, body);
        let mut changed_any = true;
        let mut passes = 0usize;
        while changed_any && passes < max_passes {
            passes += 1;
            if let Some(p) = &pb { p.inc(1); }
            changed_any = false;
            Self::run_pass(&pb, "normalize", &mut changed_any, || Self::normalize_edges_inplace(states));
            // Self::run_pass(&pb, "relax local future", &mut changed_any, || Self::relax_weights_by_local_future(states));
            // Self::run_pass(&pb, "normalize", &mut changed_any, || Self::normalize_edges_inplace(states));
            Self::run_pass(&pb, "prune", &mut changed_any, || Self::prune_unreachable(states, body));
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
            // Start with final weight (or zeros)
            let mut u = states[i].final_weight.clone().unwrap_or_else(Weight::zeros);
            // Include default outgoing weight if present
            if let Some(w) = states[i].trans_weight_default.as_ref() {
                u |= w;
            }
            // Include all exception outgoing weights
            for w in states[i].trans_weights_exceptions.values() {
                u |= w;
            }
            // Gate by state-weight (applied on state entry)
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
            // Default
            if let Some(v) = st.transitions.default {
                if let Some(w) = st.trans_weight_default.as_mut() {
                    let new_w = &*w | &not_upper[v];
                    if new_w != *w {
                        *w = new_w;
                        any_weight_changed = true;
                    }
                }
            }
            // Exceptions
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

    pub fn normalize_edges_inplace(states: &mut DWAStates) -> bool {
        let mut changed = false;
        for st in &mut states.0 {
            let before = st.transitions.exceptions.len();
            if let Some(def) = st.transitions.default {
                st.transitions.exceptions.retain(|_, &mut tgt| tgt != def);
            }
            changed |= st.transitions.exceptions.len() != before;

            let before_w = st.trans_weights_exceptions.len();
            st.trans_weights_exceptions.retain(|ch, _| st.transitions.exceptions.contains_key(ch));
            changed |= st.trans_weights_exceptions.len() != before_w;
        }
        changed
    }

    pub fn minimize_partition_refinement(states: &mut DWAStates, body: &mut DWABody) -> bool {
        let n = states.0.len();
        if n <= 1 {
            return false;
        }
        let sink_pid: usize = n;

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
            let mut sig2pid: HashMap<(Option<Weight>, Option<Weight>, usize, Vec<(i16, usize)>), usize> = HashMap::new();

            for i in 0..n {
                let st = &states[i];
                let def_cls = st.transitions.default.map(|d| part[d]).unwrap_or(sink_pid);
                let mut ex: Vec<(i16, usize)> = Vec::with_capacity(st.transitions.exceptions.len());
                for (ch, tgt) in &st.transitions.exceptions {
                    let cls = part[*tgt];
                    if cls != def_cls {
                        ex.push((*ch, cls));
                    }
                }
                let sig = (st.state_weight.clone(), st.final_weight.clone(), def_cls, ex);
                let next_pid = sig2pid.len();
                next_part[i] = *sig2pid.entry(sig).or_insert(next_pid);
            }
            if next_part != part {
                part = next_part;
                changed = true;
            }
        }

        // Early exit if nothing to merge
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in part.iter().enumerate() {
            groups.entry(*p).or_default().push(i);
        }
        if groups.len() == n {
            return false;
        }

        // Build representatives
        let mut pid_to_new: HashMap<usize, usize> = HashMap::new();
        let mut new_states: Vec<DWAState> = Vec::with_capacity(groups.len());
        for (pid, members) in &groups {
            let rep = members[0];
            let rep_state = &states[rep];
            let def_cls = rep_state.transitions.default.map(|d| part[d]).unwrap_or(sink_pid);

            let mut st = DWAState::default();
            st.state_weight = rep_state.state_weight.clone();
            st.final_weight = rep_state.final_weight.clone();
            st.transitions.default = if def_cls == sink_pid { None } else { Some(0) };
            for (ch, tgt) in &rep_state.transitions.exceptions {
                let cls = part[*tgt];
                if cls != def_cls {
                    st.transitions.exceptions.insert(*ch, 0);
                }
            }

            // Aggregate per-edge weights across members (join).
            if st.transitions.default.is_some() {
                let mut agg_def: Option<Weight> = None;
                for &old in members {
                    if let Some(w) = states[old].trans_weight_default.as_ref() {
                        if let Some(ref mut a) = agg_def {
                            *a |= w;
                        } else {
                            agg_def = Some(w.clone());
                        }
                    }
                }
                st.trans_weight_default = agg_def;
            }
            let ex_keys: Vec<i16> = st.transitions.exceptions.keys().cloned().collect();
            for ch in ex_keys {
                let mut agg: Option<Weight> = None;
                for &old in members {
                    if let Some(w) = states[old].trans_weights_exceptions.get(&ch) {
                        if let Some(ref mut a) = agg {
                            *a |= w;
                        } else {
                            agg = Some(w.clone());
                        }
                    }
                }
                if let Some(w) = agg {
                    st.trans_weights_exceptions.insert(ch, w);
                }
            }

            let new_id = new_states.len();
            pid_to_new.insert(*pid, new_id);
            new_states.push(st);
        }

        // Fix transition targets
        for (pid, members) in &groups {
            let new_id = *pid_to_new.get(pid).unwrap();
            let rep = members[0];
            let rep_state = &states[rep];

            let def_cls = rep_state.transitions.default.map(|d| part[d]).unwrap_or(sink_pid);
            if let Some(ref mut d) = new_states[new_id].transitions.default {
                *d = *pid_to_new.get(&def_cls).unwrap();
            }
            let ex_old = new_states[new_id].transitions.exceptions.clone();
            new_states[new_id].transitions.exceptions.clear();
            for (ch, _) in ex_old {
                let cls = part[*rep_state.transitions.exceptions.get(&ch).unwrap()];
                new_states[new_id].transitions.exceptions.insert(ch, *pid_to_new.get(&cls).unwrap());
            }
        }

        states.0 = new_states;
        Self::normalize_edges_inplace(states);
        let start_pid = part[body.start_state];
        body.start_state = *pid_to_new.get(&start_pid).unwrap();

        true
    }

    pub fn prune_unreachable(states: &mut DWAStates, body: &mut DWABody) -> bool {
        if states.0.is_empty() {
            return false;
        }
        let n = states.0.len();

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
                if !live[v] { live[v] = true; q_live.push_back(v); }
            }
        }

        // 2. Remove transitions to non-live states.
        let mut changed = false;
        for i in 0..n {
            let st = &mut states[i];
            if let Some(d) = st.transitions.default {
                if !live[d] {
                    st.transitions.default = None;
                    st.trans_weight_default = None;
                    changed = true;
                }
            }
            let before = st.transitions.exceptions.len();
            st.transitions.exceptions.retain(|_, tgt| live[*tgt]);
            if st.transitions.exceptions.len() != before {
                changed = true;
                st.trans_weights_exceptions.retain(|ch, _| st.transitions.exceptions.contains_key(ch));
            }
        }

        // 3. Forward reachability from start_state to find actually reachable states.
        let mut visited = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        if body.start_state < n {
            visited[body.start_state] = true;
            q.push_back(body.start_state);
        }
        while let Some(u) = q.pop_front() {
            for (_lbl, v, _w) in states[u].iter_edges() {
                if v < n && !visited[v] { visited[v] = true; q.push_back(v); }
            }
        }

        if visited.iter().all(|&b| b) && !changed {
            return false;
        }

        // 4. Remap kept states.
        let mut map = vec![usize::MAX; n];
        let mut next_id = 0usize;
        for i in 0..n { if visited[i] { map[i] = next_id; next_id += 1; } }

        if next_id == n && !changed { return false; }

        let mut new_states: Vec<DWAState> = Vec::with_capacity(next_id);
        for old in 0..n {
            if !visited[old] { continue; }
            let mut st = states[old].clone();
            if let Some(d) = st.transitions.default { st.transitions.default = Some(map[d]); }
            let ex = st.transitions.exceptions.clone();
            st.transitions.exceptions.clear();
            for (ch, tgt) in ex { st.transitions.exceptions.insert(ch, map[tgt]); }
            new_states.push(st);
        }
        states.0 = new_states;
        if next_id > 0 {
            body.start_state = map[body.start_state];
        } else if n > 0 {
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
        if pass() {
            *changed_any = true;
        }
    }

    pub fn simplify(&mut self) -> bool {
        let now = Instant::now();
        let initial_n = self.states.len();
        let max_passes = 5;
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
            if let Some(p) = &pb {
                p.inc(1);
            }
            changed = false;
            Self::run_pass(&pb, "collapse SCCs", &mut changed, || self.collapse_all_weight_epsilon_sccs());
            Self::run_pass(&pb, "prune unreachable", &mut changed, || self.prune_unreachable());
            Self::run_pass(&pb, "prune dead ends", &mut changed, || self.prune_dead_ends());
        }
        if let Some(p) = &pb {
            p.finish_with_message(format!("Simplified to {} states", self.states.len()));
        }
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
            for (_, (v, _)) in &st.transitions {
                if *v < n && !reachable[*v] { reachable[*v] = true; q.push_back(*v); }
            }
            if let Some((v, _)) = &st.default {
                if *v < n && !reachable[*v] { reachable[*v] = true; q.push_back(*v); }
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
            st.transitions.values_mut().for_each(|(v, _)| *v = remap[*v]);
            if let Some((v, _)) = &mut st.default {
                *v = remap[*v];
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

        // Relax edge weights using future weights.
        // This is semantically correct because any path from a state `v` will have its weight
        // intersected with `fut[v]`. So, for an edge `u -> v` with weight `w`, we can relax `w`
        // to `w | !fut[v]` without changing the language, because
        // `(w | !fut[v]) & fut[v] == w & fut[v]`.
        // This relaxation can turn some weights into `Weight::all()`, enabling more
        // `collapse_all_weight_epsilon_sccs` in subsequent passes.
        let not_fut: Vec<Weight> = fut.iter().map(|w| !w).collect();
        let mut changed_weights = false;
        for i in 0..n {
            let st = &mut self.states[i];
            for (v, w) in &mut st.epsilons {
                let new_w = &*w | &not_fut[*v];
                if new_w != *w {
                    *w = new_w;
                    changed_weights = true;
                }
            }
            for (_, (v, w)) in &mut st.transitions {
                let new_w = &*w | &not_fut[*v];
                if new_w != *w {
                    *w = new_w;
                    changed_weights = true;
                }
            }
            if let Some((v, w)) = &mut st.default {
                let new_w = &*w | &not_fut[*v];
                if new_w != *w {
                    *w = new_w;
                    changed_weights = true;
                }
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

            st.transitions.retain(|_, (v, _)| live[*v]);
            st.transitions.values_mut().for_each(|(v, _)| *v = remap[*v]);

            if let Some((v, _)) = &st.default {
                if !live[*v] {
                    st.default = None;
                } else {
                    st.default.as_mut().unwrap().0 = remap[*v];
                }
            }
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
                    if let Some(acc) = &mut final_weight {
                        *acc |= fw;
                    } else {
                        final_weight = Some(fw.clone());
                    }
                }
            }
            new_state.final_weight = final_weight;

            for &sid in comp {
                for (&lbl, &(to, ref w)) in &self.states[sid].transitions {
                    let to_comp = comp_of[to];
                    Self::add_transition_to_state(&mut new_state, lbl, to_comp, w.clone());
                }
                if let Some((to, w)) = &self.states[sid].default {
                    if new_state.default.is_none() {
                        new_state.default = Some((comp_of[*to], w.clone()));
                    }
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

        for st in &mut new_states_vec {
            st.epsilons.iter_mut().for_each(|(v, _)| *v = comp_of[*v]);
            st.transitions.values_mut().for_each(|(v, _)| *v = comp_of[*v]);
            if let Some((v, _)) = &mut st.default {
                *v = comp_of[*v];
            }
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
        if let Some((old_to, old_w)) = state.transitions.get_mut(&on) {
            assert_eq!(*old_to, to, "NWA restricted: cannot merge states with same-label transitions to different components");
            *old_w |= &w;
        } else {
            state.transitions.insert(on, (to, w));
        }
    }
}
