// src/precompute4/weighted_automata/simplification.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{StateID, Weight, STOCHASTIC_DEBUG};
use super::dwa::{DWABody, DWAState, DWAStates, DWA};
use super::nwa::{NWAState, NWAStates, NWA};
use crate::precompute4::test_weighted_automata;
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::r#macro::is_debug_level_enabled;
use indicatif::{ProgressBar, ProgressStyle};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque, hash_map::DefaultHasher};
use std::time::Instant;

/// For very large DWAs, we skip heavy fixpoint/minimization passes to guarantee fast simplification.
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
        if is_debug_level_enabled(3) {
            eprintln!("DWA::simplify_components ({} states -> {} states) took: {:?}", initial_len, states.len(), now.elapsed());
        }
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
        if is_debug_level_enabled(3) {
            eprintln!("DWA simplify pass '{}' took {:?} (changed: {})", pass_name, elapsed, changed);
        }

        if let Some(before) = before_dwa {
            if changed {
                let after_dwa = DWA { states: states.clone(), body: body.clone() };
                IN_SIMPLIFY_CHECK.with(|c| c.set(true));
                if is_debug_level_enabled(1) {
                    eprintln!("Stochastic testing DWA after pass: {}", pass_name);
                }
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

        if let Some(p) = &pb { p.set_message("initial prune"); }
        let _ = Self::run_pass_with_test(states, body, "initial prune_unreachable", |s, b| {
            Self::prune_unreachable(s, b)
        });
        let _ = Self::run_pass_with_test(states, body, "initial prune_dead_ends", |s, _b| {
            Self::prune_dead_ends(s)
        });

        let mut changed_any = true;
        let mut passes = 0usize;
        while changed_any && passes < max_passes {
            passes += 1;
            if let Some(p) = &pb { p.inc(1); }
            changed_any = false;

            if let Some(p) = &pb { p.set_message("normalize"); }
            changed_any |= Self::run_pass_with_test(states, body, "normalize_edges_inplace", |s, _b| {
                Self::normalize_edges_inplace(s)
            });

            if let Some(p) = &pb { p.set_message("constrain/absorb"); }
            changed_any |= Self::run_pass_with_test(states, body, "constrain_and_absorb_sinks_fixpoint", |s, b| {
                Self::constrain_and_absorb_sinks_fixpoint(s, b)
            });

            if let Some(p) = &pb { p.set_message("relax local future"); }
            changed_any |=
                Self::run_pass_with_test(states, body, "relax_weights_by_local_future", |s, _b| {
                    Self::relax_weights_by_local_future(s)
                });

            if !large {
                if let Some(p) = &pb { p.set_message("minimize"); }
                changed_any |=
                    Self::run_pass_with_test(states, body, "minimize_partition_refinement", |s, b| {
                        Self::minimize_partition_refinement(s, b)
                    });
            }

            if let Some(p) = &pb { p.set_message("prune dead"); }
            changed_any |= Self::run_pass_with_test(states, body, "prune_dead_ends", |s, _b| {
                Self::prune_dead_ends(s)
            });

            if let Some(p) = &pb { p.set_message("prune unreachable"); }
            changed_any |= Self::run_pass_with_test(states, body, "prune_unreachable", |s, b| {
                Self::prune_unreachable(s, b)
            });
        }

        if let Some(p) = &pb {
            p.finish_with_message(format!("Simplified to {} states", states.len()));
        }
    }

    pub fn constrain_and_absorb_sinks_fixpoint(states: &mut DWAStates, body: &mut DWABody) -> bool {
        let mut changed_overall = false;
        for _ in 0..5 {
            let n = states.len();
            if n == 0 { return changed_overall; }

            let mut reachable_weights = vec![Weight::zeros(); n];
            let mut worklist = VecDeque::new();

            if body.start_state >= n { return changed_overall; }
            reachable_weights[body.start_state] = Weight::all();
            worklist.push_back(body.start_state);

            while let Some(u) = worklist.pop_front() {
                let u_rw = reachable_weights[u].clone();
                if u_rw.is_empty() { continue; }
                let u_state = &states[u];
                let gated_u_rw = if let Some(sw) = &u_state.state_weight { &u_rw & sw } else { u_rw };
                for (_, v, edge_w) in u_state.iter_edges() {
                    if v >= n { continue; }
                    let new_v_rw = &gated_u_rw & edge_w;
                    if !new_v_rw.is_subset_of(&reachable_weights[v]) {
                        reachable_weights[v] |= &new_v_rw;
                        worklist.push_back(v);
                    }
                }
            }

            let mut changed_this_iteration = false;
            for i in 0..n {
                let old_fw = states[i].final_weight.clone();
                if let Some(fw) = states[i].final_weight.as_mut() {
                    *fw &= &reachable_weights[i];
                    if fw.is_empty() { states[i].final_weight = None; }
                }
                if states[i].final_weight != old_fw { changed_this_iteration = true; }
            }

            let mut preds: Vec<Vec<(StateID, i16)>> = vec![Vec::new(); n];
            for p in 0..n {
                for (ch, d, _w) in states[p].iter_edges() {
                    if d < n { preds[d].push((p, ch)); }
                }
            }

            for v in 0..n {
                if !states[v].transitions.is_empty() { continue; }
                let mut effective_weight = Weight::all();
                if let Some(fw) = &states[v].final_weight { effective_weight &= fw; }
                if let Some(sw) = &states[v].state_weight { effective_weight &= sw; }
                if effective_weight.is_all_fast() || preds[v].is_empty() { continue; }

                for &(p, ch) in &preds[v] {
                    if let Some(w) = states[p].trans_weights.get_mut(&ch) {
                        let old_w = w.clone();
                        *w &= &effective_weight;
                        if *w != old_w { changed_this_iteration = true; }
                    }
                }
            }

            changed_overall |= changed_this_iteration;
            if !changed_this_iteration { break; }
        }
        changed_overall
    }

    pub fn relax_weights_by_local_future(states: &mut DWAStates) -> bool {
        let n = states.len();
        if n == 0 { return false; }
        let mut upper: Vec<Weight> = Vec::with_capacity(n);
        for i in 0..n {
            let mut u = states[i].final_weight.clone().unwrap_or_else(Weight::zeros);
            for w in states[i].trans_weights.values() { u |= w; }
            if let Some(sw) = states[i].state_weight.as_ref() { u &= sw; }
            upper.push(u);
        }
        let not_upper: Vec<Weight> = upper.iter().map(|w| !w).collect();
        let mut any_weight_changed = false;
        for i in 0..n {
            let st = &mut states[i];
            let keys: Vec<i16> = st.transitions.keys().copied().collect();
            for ch in keys {
                if let Some(&v) = st.transitions.get(&ch) {
                    if v < n {
                        if let Some(w) = st.trans_weights.get_mut(&ch) {
                            let new_w = &*w | &not_upper[v];
                            if new_w != *w { *w = new_w; any_weight_changed = true; }
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
            let before_w = st.trans_weights.len();
            st.trans_weights.retain(|ch, _| st.transitions.contains_key(ch));
            if st.trans_weights.len() != before_w { changed = true; }
        }
        changed
    }

    pub fn minimize_partition_refinement(states: &mut DWAStates, body: &mut DWABody) -> bool {
        let n = states.0.len();
        if n <= 1 { return false; }

        let mut part: Vec<usize> = vec![0; n];
        let mut canon0: HashMap<(Option<Weight>, Option<Weight>), usize> = HashMap::new();
        for i in 0..n {
            let key = (states[i].state_weight.clone(), states[i].final_weight.clone());
            let next_id = canon0.len();
            part[i] = *canon0.entry(key).or_insert(next_id);
        }

        let mut changed = true;
        while changed {
            changed = false;
            let mut next_part: Vec<usize> = vec![0; n];
            let mut sig2pid: HashMap<(Option<Weight>, Option<Weight>, BTreeMap<i16, (usize, Weight)>), usize> = HashMap::new();

            for i in 0..n {
                let st = &states[i];
                let trans_sig: BTreeMap<_, _> = st.transitions.iter().map(|(ch, &tgt)| (*ch, (part[tgt], st.trans_weights.get(ch).cloned().unwrap_or_else(Weight::all)))).collect();
                let sig = (st.state_weight.clone(), st.final_weight.clone(), trans_sig);
                let next_pid = sig2pid.len();
                next_part[i] = *sig2pid.entry(sig).or_insert(next_pid);
            }
            if next_part != part { part = next_part; changed = true; }
        }

        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in part.iter().enumerate() { groups.entry(*p).or_default().push(i); }
        if groups.len() == n { return false; }

        let mut pid_to_new: HashMap<usize, usize> = HashMap::new();
        let mut new_states: Vec<DWAState> = vec![DWAState::default(); groups.len()];
        for (pid, _) in &groups { pid_to_new.insert(*pid, pid_to_new.len()); }

        for (pid, members) in &groups {
            let rep = members[0];
            let new_id = *pid_to_new.get(pid).unwrap();
            new_states[new_id].state_weight = states[rep].state_weight.clone();
            new_states[new_id].final_weight = states[rep].final_weight.clone();
            for (ch, tgt) in &states[rep].transitions {
                let cls = part[*tgt];
                new_states[new_id].transitions.insert(*ch, *pid_to_new.get(&cls).unwrap());
                if let Some(w) = states[rep].trans_weights.get(ch) {
                    new_states[new_id].trans_weights.insert(*ch, w.clone());
                }
            }
        }

        states.0 = new_states;
        let start_pid = part[body.start_state];
        body.start_state = *pid_to_new.get(&start_pid).unwrap();
        true
    }

    pub fn prune_dead_ends(states: &mut DWAStates) -> bool {
        let n = states.len();
        if n == 0 { return false; }
        let mut live = vec![false; n];
        let mut q_live: VecDeque<usize> = VecDeque::new();
        let mut rev_adj: Vec<Vec<usize>> = vec![vec![]; n];
        for i in 0..n {
            if states[i].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                live[i] = true;
                q_live.push_back(i);
            }
            for (_, v, w) in states[i].iter_edges() {
                if v < n && !w.is_empty() { rev_adj[v].push(i); }
            }
        }
        while let Some(u) = q_live.pop_front() {
            for &v in &rev_adj[u] {
                if !live[v] { live[v] = true; q_live.push_back(v); }
            }
        }

        let mut changed = false;
        for i in 0..n {
            let st = &mut states[i];
            let before = st.transitions.len();
            st.transitions.retain(|_, tgt| *tgt < n && live[*tgt]);
            if st.transitions.len() != before {
                changed = true;
                st.trans_weights.retain(|ch, _| st.transitions.contains_key(ch));
            }
        }
        changed
    }

    pub fn prune_unreachable(states: &mut DWAStates, body: &mut DWABody) -> bool {
        if states.0.is_empty() { return false; }
        let n = states.0.len();
        let mut visited = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        if body.start_state < n {
            visited[body.start_state] = true;
            q.push_back(body.start_state);
        } else {
            if n > 0 {
                states.0.clear();
                body.start_state = states.add_state();
                return true;
            }
            return false;
        }
        while let Some(u) = q.pop_front() {
            for (_, v, _) in states[u].iter_edges() {
                if v < n && !visited[v] { visited[v] = true; q.push_back(v); }
            }
        }

        let num_reachable = visited.iter().filter(|&&b| b).count();
        if num_reachable == n { return false; }

        let mut map = vec![usize::MAX; n];
        let mut new_states: Vec<DWAState> = Vec::with_capacity(num_reachable);
        for i in 0..n {
            if visited[i] {
                map[i] = new_states.len();
                new_states.push(states[i].clone());
            }
        }

        for st in &mut new_states {
            for tgt in st.transitions.values_mut() { *tgt = map[*tgt]; }
        }
        states.0 = new_states;
        if num_reachable > 0 {
            body.start_state = map[body.start_state];
        } else {
            states.0.clear();
            body.start_state = states.add_state();
        }
        true
    }
}

impl NWA {
    pub fn simplify(&mut self) -> bool {
        let initial_n = self.states.len();
        let initial_body = self.body;

        // 1. Simplify NWA structure before determinization
        let mut changed = self.prune_unreachable();
        changed |= self.prune_dead_ends();
        changed |= self.collapse_all_weight_epsilon_sccs();
        changed |= self.dedup_epsilon_edges();
        changed |= self.dedup_labeled_edges();

        // 2. Determinize to DWA
        let mut dwa = self.determinize();

        // 3. Simplify the resulting DWA
        dwa.simplify();

        // 4. Convert back to NWA
        *self = NWA::from_dwa(&dwa);

        self.states.len() != initial_n || self.body != initial_body || changed
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
            if changed { self.states.0.clear(); self.body.start_state = self.states.add_state(); }
            return changed;
        }

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];
            for (v, _) in &st.epsilons { if *v < n && !reachable[*v] { reachable[*v] = true; q.push_back(*v); } }
            for (_, targets) in &st.transitions { for (v, _) in targets { if *v < n && !reachable[*v] { reachable[*v] = true; q.push_back(*v); } } }
        }

        let num_reachable = reachable.iter().filter(|&&b| b).count();
        if num_reachable == n { return false; }

        let mut remap = vec![usize::MAX; n];
        let mut new_states_vec = Vec::with_capacity(num_reachable);
        for i in 0..n { if reachable[i] { remap[i] = new_states_vec.len(); new_states_vec.push(self.states[i].clone()); } }

        for st in &mut new_states_vec {
            st.epsilons.iter_mut().for_each(|(v, _)| *v = remap[*v]);
            st.transitions.values_mut().for_each(|targets| for (v, _) in targets { *v = remap[*v]; });
        }

        self.states.0 = new_states_vec;
        self.body.start_state = remap[self.body.start_state];
        true
    }

    fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut live = vec![false; n];
        let mut q = VecDeque::new();
        let mut rev_adj: Vec<Vec<NWAStateID>> = vec![vec![]; n];

        for p in 0..n {
            let st = &self.states[p];
            for &(t, ref w) in &st.epsilons { if t < n && !w.is_empty() { rev_adj[t].push(p); } }
            for (_, targets) in &st.transitions { for &(t, ref w) in targets { if t < n && !w.is_empty() { rev_adj[t].push(p); } } }
        }

        for s in 0..n { if self.states[s].final_weight.as_ref().map_or(false, |w| !w.is_empty()) { if !live[s] { live[s] = true; q.push_back(s); } } }

        while let Some(v) = q.pop_front() { for &p in &rev_adj[v] { if !live[p] { live[p] = true; q.push_back(p); } } }

        if self.body.start_state >= n || !live[self.body.start_state] {
            let changed = n > 0;
            if changed { self.states.0.clear(); self.body.start_state = self.states.add_state(); }
            return changed;
        }

        let num_live = live.iter().filter(|&&b| b).count();
        if num_live == n { return false; }

        let mut remap = vec![usize::MAX; n];
        let mut new_states_vec = Vec::with_capacity(num_live);
        for i in 0..n { if live[i] { remap[i] = new_states_vec.len(); new_states_vec.push(self.states[i].clone()); } }

        for st in &mut new_states_vec {
            st.epsilons.retain(|(v, _)| live[*v]);
            st.epsilons.iter_mut().for_each(|(v, _)| *v = remap[*v]);
            st.transitions.values_mut().for_each(|targets| { targets.retain(|(v, _)| live[*v]); targets.iter_mut().for_each(|(v, _)| *v = remap[*v]); });
            st.transitions.retain(|_, targets| !targets.is_empty());
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

        for i in 0..n { if indices[i] == usize::MAX { self.strongconnect(i, &mut index, &mut indices, &mut lowlink, &mut on_stack, &mut stack, &mut comp_of, &mut comps); } }
        if comps.len() == n { return false; }

        let mut new_states_vec = Vec::with_capacity(comps.len());
        for (cid, comp) in comps.iter().enumerate() {
            let mut new_state = NWAState::default();
            let mut final_weight: Option<Weight> = None;
            for &sid in comp { if let Some(fw) = &self.states[sid].final_weight { if let Some(acc) = &mut final_weight { *acc |= fw; } else { final_weight = Some(fw.clone()); } } }
            new_state.final_weight = final_weight;

            for &sid in comp {
                for (&lbl, targets) in &self.states[sid].transitions { for &(to, ref w) in targets { Self::add_transition_to_state(&mut new_state, lbl, comp_of[to], w.clone()); } }
                for &(to, ref w) in &self.states[sid].epsilons { if cid != comp_of[to] || !w.is_all_fast() { new_state.epsilons.push((comp_of[to], w.clone())); } }
            }
            new_states_vec.push(new_state);
        }

        self.states.0 = new_states_vec;
        self.body.start_state = comp_of[self.body.start_state];
        true
    }

    fn strongconnect(&self, v: NWAStateID, index: &mut usize, indices: &mut [usize], lowlink: &mut [usize], on_stack: &mut [bool], stack: &mut Vec<NWAStateID>, comp_of: &mut [usize], comps: &mut Vec<Vec<NWAStateID>>) {
        indices[v] = *index; lowlink[v] = *index; *index += 1; stack.push(v); on_stack[v] = true;
        for (w, weight) in &self.states[v].epsilons {
            if weight.is_all_fast() {
                if indices[*w] == usize::MAX { self.strongconnect(*w, index, indices, lowlink, on_stack, stack, comp_of, comps); lowlink[v] = lowlink[v].min(lowlink[*w]); }
                else if on_stack[*w] { lowlink[v] = lowlink[v].min(indices[*w]); }
            }
        }
        if lowlink[v] == indices[v] {
            let mut comp = Vec::new();
            loop { let w = stack.pop().unwrap(); on_stack[w] = false; comp_of[w] = comps.len(); comp.push(w); if w == v { break; } }
            comps.push(comp);
        }
    }

    fn add_transition_to_state(state: &mut NWAState, on: i16, to: NWAStateID, w: Weight) {
        let targets = state.transitions.entry(on).or_default();
        if let Some((_, existing_w)) = targets.iter_mut().find(|(t, _)| *t == to) { *existing_w |= &w; }
        else { targets.push((to, w)); }
    }

    fn dedup_labeled_edges(&mut self) -> bool {
        let mut changed = false;
        for st in &mut self.states.0 {
            for targets in st.transitions.values_mut() {
                if targets.len() <= 1 { continue; }
                let old_len = targets.len();
                let mut acc: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
                for (to, w) in targets.iter() { *acc.entry(*to).or_insert_with(Weight::zeros) |= w; }
                *targets = acc.into_iter().collect();
                if targets.len() != old_len { changed = true; }
            }
        }
        changed
    }

    fn dedup_epsilon_edges(&mut self) -> bool {
        let mut changed = false;
        for st in &mut self.states.0 {
            if st.epsilons.len() <= 1 { continue; }
            let old_len = st.epsilons.len();
            let mut acc: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
            for (to, w) in &st.epsilons { *acc.entry(*to).or_insert_with(Weight::zeros) |= w; }
            st.epsilons = acc.into_iter().collect();
            if st.epsilons.len() != old_len { changed = true; }
        }
        changed
    }
}