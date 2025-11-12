// src/precompute4/weighted_automata/determinization.rs

use super::common::{StateID, Weight};
use super::dwa::DWA;
use super::nwa::NWA;
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::r#macro::is_debug_level_enabled;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

type WeightedSubset = BTreeMap<NWAStateID, Weight>;

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();
        let mut nwa = self.clone();
        nwa.simplify();

        if is_debug_level_enabled(5) {
            eprintln!("NWA after simplify:\n{}", nwa);
        }
        if nwa.states.0.is_empty() {
            return DWA::new();
        }

        let mut determinizer = Determinizer::new(&nwa);
        let dwa = determinizer.run();

        if is_debug_level_enabled(5) {
            eprintln!("NWA::determinize_to_dwa result DWA stats:\n{}", dwa.stats());
            eprintln!("NWA::determinize_to_dwa took: {:?}", now.elapsed());
        }
        dwa
    }
}

struct Determinizer<'a> {
    nwa: &'a NWA,
    dwa: DWA,
    work_queue: VecDeque<StateID>,
    dwa_state_map: BTreeMap<WeightedSubset, StateID>,
    weighted_subsets: Vec<WeightedSubset>,
    eps_closure_cache: HashMap<NWAStateID, Arc<WeightedSubset>>,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA) -> Self {
        Self {
            nwa,
            dwa: DWA::new(),
            work_queue: VecDeque::new(),
            dwa_state_map: BTreeMap::new(),
            weighted_subsets: Vec::new(),
            eps_closure_cache: HashMap::new(),
        }
    }

    fn run(&mut self) -> DWA {
        let start_subset = self.compute_epsilon_closure(self.nwa.body.start_state);
        if start_subset.is_empty() {
            return DWA::new();
        }

        let start_dwa_id = self.dwa.body.start_state;
        self.dwa_state_map.insert(start_subset.as_ref().clone(), start_dwa_id);
        self.weighted_subsets.push(start_subset.as_ref().clone());
        self.work_queue.push_back(start_dwa_id);

        let pb = Self::progress_bar(0, "Discovering states");

        while let Some(dwa_id) = self.work_queue.pop_front() {
            if let Some(p) = &pb { p.inc(1); }

            let p_prime = self.weighted_subsets[dwa_id].clone();

            let final_weight = self.calculate_final_weight(&p_prime);
            if !final_weight.is_empty() {
                self.dwa.set_final_weight(dwa_id, final_weight).unwrap();
            }

            let (default_target, exception_targets) = self.collect_outgoing_transitions(&p_prime);

            if let Some(target_subset) = default_target {
                let w_prime = target_subset.values().fold(Weight::zeros(), |mut acc, w| { acc |= w; acc });
                if !w_prime.is_empty() {
                    let target_dwa_id = self.get_or_create_dwa_state(&target_subset);
                    self.dwa.set_default_transition(dwa_id, target_dwa_id, w_prime).unwrap();
                }
            }

            for (label, target_subset) in exception_targets {
                let w_prime = target_subset.values().fold(Weight::zeros(), |mut acc, w| { acc |= w; acc });
                if !w_prime.is_empty() {
                    let target_dwa_id = self.get_or_create_dwa_state(&target_subset);
                    self.dwa.add_transition(dwa_id, label, target_dwa_id, w_prime).unwrap();
                }
            }

            if let Some(p) = &pb { p.set_length(self.dwa.states.len() as u64); }
        }

        if let Some(p) = pb { p.finish_with_message(format!("Discovered {} DWA states", self.dwa.states.len())); }

        std::mem::take(&mut self.dwa)
    }

    fn get_or_create_dwa_state(&mut self, subset: &WeightedSubset) -> StateID {
        if let Some(&id) = self.dwa_state_map.get(subset) {
            return id;
        }
        let new_id = self.dwa.add_state();
        self.dwa_state_map.insert(subset.clone(), new_id);
        self.weighted_subsets.push(subset.clone());
        self.work_queue.push_back(new_id);
        new_id
    }

    fn compute_epsilon_closure(&mut self, start_state: NWAStateID) -> Arc<WeightedSubset> {
        if let Some(cached) = self.eps_closure_cache.get(&start_state) {
            return cached.clone();
        }
        let mut closure = WeightedSubset::new();
        let mut q: VecDeque<(NWAStateID, Weight)> = VecDeque::new();
        closure.insert(start_state, Weight::all());
        q.push_back((start_state, Weight::all()));
        while let Some((u, w_u)) = q.pop_front() {
            for (v, w_uv) in &self.nwa.states[u].epsilons {
                let new_weight = &w_u & w_uv;
                if new_weight.is_empty() { continue; }
                let current_v_weight = closure.entry(*v).or_default();
                if !new_weight.is_subset_of(current_v_weight) {
                    *current_v_weight |= &new_weight;
                    q.push_back((*v, new_weight.clone()));
                }
            }
        }
        let result = Arc::new(closure);
        self.eps_closure_cache.insert(start_state, result.clone());
        result
    }

    // --- REIMPLEMENTED FUNCTION ---
    fn collect_outgoing_transitions(&mut self, p_prime: &WeightedSubset) -> (Option<WeightedSubset>, BTreeMap<i16, WeightedSubset>) {
        let mut transitions_map: BTreeMap<i16, WeightedSubset> = BTreeMap::new();
        let mut relevant_labels = BTreeSet::new();

        // 1. Collect all labels that need special handling.
        for (&nwa_state, _) in p_prime {
            let state = &self.nwa.states[nwa_state];
            relevant_labels.extend(state.transitions.keys());
            for default in &state.default {
                relevant_labels.extend(&default.exceptions);
            }
        }

        // 2. For each special label, compute its precise target subset.
        for &label in &relevant_labels {
            let target_subset = transitions_map.entry(label).or_default();
            for (&nwa_state, residual_weight) in p_prime {
                self.compute_target_for_state(target_subset, nwa_state, label, residual_weight);
            }
        }

        // 3. Compute the target for a generic default character.
        let generic_char = relevant_labels.iter().max().map_or(0, |x| x.wrapping_add(1));
        let mut default_target = WeightedSubset::new();
        for (&nwa_state, residual_weight) in p_prime {
            self.compute_target_for_state(&mut default_target, nwa_state, generic_char, residual_weight);
        }

        // 4. Prune the explicit transitions that are identical to the new default.
        let mut exception_targets = transitions_map;
        exception_targets.retain(|_, subset| *subset != default_target);

        let final_default_target = if default_target.is_empty() { None } else { Some(default_target) };
        (final_default_target, exception_targets)
    }

    /// Helper: For a single NWA state and a label, compute its contribution to a target subset.
    fn compute_target_for_state(&mut self, target_subset: &mut WeightedSubset, nwa_state: NWAStateID, label: i16, residual_weight: &Weight) {
        let state = &self.nwa.states[nwa_state];

        // Explicit transitions have priority.
        if let Some(targets) = state.transitions.get(&label) {
            for (target_state, trans_weight) in targets {
                self.propagate_weight(target_subset, *target_state, residual_weight, trans_weight);
            }
            return; // Found an explicit transition, so we don't check defaults.
        }

        // If no explicit transition, check defaults.
        for default in &state.default {
            if !default.exceptions.contains(&label) {
                self.propagate_weight(target_subset, default.target, residual_weight, &default.weight);
            }
        }
    }

    /// Helper to propagate a weight through an NWA transition and its target's ε-closure.
    fn propagate_weight(&mut self, dest_subset: &mut WeightedSubset, target_state: NWAStateID, residual_weight: &Weight, trans_weight: &Weight) {
        let propagated_weight = residual_weight & trans_weight;
        if propagated_weight.is_empty() { return; }

        let target_closure = self.compute_epsilon_closure(target_state);
        for (closure_state, closure_weight) in target_closure.iter() {
            let final_weight = &propagated_weight & closure_weight;
            if !final_weight.is_empty() {
                *dest_subset.entry(*closure_state).or_default() |= &final_weight;
            }
        }
    }

    fn calculate_final_weight(&self, p_prime: &WeightedSubset) -> Weight {
        let mut final_weight = Weight::zeros();
        for (&nwa_state, residual_weight) in p_prime {
            if let Some(nwa_final_weight) = &self.nwa.states[nwa_state].final_weight {
                final_weight |= &(residual_weight & nwa_final_weight);
            }
        }
        final_weight
    }

    fn progress_bar(len: u64, label: &str) -> Option<ProgressBar> {
        if !PROGRESS_BAR_ENABLED {
            return None;
        }
        let style = ProgressStyle::default_bar()
            .template(&format!("{{spinner:.green}} [Determinize: {{elapsed_precise}}] [{{wide_bar:.cyan/blue}}] {{pos}}/{{len}} ({})", label))
            .unwrap();
        let pb = if len > 0 { ProgressBar::new(len) } else { ProgressBar::new_spinner() };
        Some(pb.with_style(style))
    }
}