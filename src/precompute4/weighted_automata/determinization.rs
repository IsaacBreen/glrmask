// src/precompute4/weighted_automata/determinization.rs

use super::common::{StateID, Weight};
use super::dwa::DWA;
use super::nwa::NWA;
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::r#macro::is_debug_level_enabled;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

/// A weighted subset of NWA states, which defines a single DWA state.
/// The BTreeMap provides a canonical representation (sorted by NWAStateID).
type WeightedSubset = BTreeMap<NWAStateID, Weight>;

// Public API: determinize an NWA into a DWA.
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

/// Implements Mohri's weighted determinization algorithm.
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
            dwa: DWA::new(), // Creates a DWA with one state at index 0
            work_queue: VecDeque::new(),
            dwa_state_map: BTreeMap::new(),
            weighted_subsets: Vec::new(),
            eps_closure_cache: HashMap::new(),
        }
    }

    /// Executes the determinization algorithm.
    fn run(&mut self) -> DWA {
        // 1. The initial DWA state is the epsilon-closure of the NWA's start state.
        let start_subset = self.compute_epsilon_closure(self.nwa.body.start_state);
        if start_subset.is_empty() {
            // The NWA accepts no strings. The default DWA correctly accepts nothing.
            return DWA::new();
        }

        // --- FIX START ---
        // Manually set up the start state (ID 0), which already exists in the DWA.
        let start_dwa_id = self.dwa.body.start_state; // This is guaranteed to be 0
        self.dwa_state_map.insert(start_subset.as_ref().clone(), start_dwa_id);
        self.weighted_subsets.push(start_subset.as_ref().clone());
        self.work_queue.push_back(start_dwa_id);
        // --- FIX END ---

        let pb = Self::progress_bar(0, "Discovering states");

        // 2. Process states from the queue until no new states are created.
        while let Some(dwa_id) = self.work_queue.pop_front() {
            if let Some(p) = &pb { p.inc(1); }

            // This line is now safe because `weighted_subsets` is kept in sync with DWA state IDs.
            let p_prime = self.weighted_subsets[dwa_id].clone();

            // 3. For the current DWA state, compute its final weight.
            let final_weight = self.calculate_final_weight(&p_prime);
            if !final_weight.is_empty() {
                self.dwa.set_final_weight(dwa_id, final_weight).unwrap();
            }

            // 4. Compute all possible outgoing transitions.
            let outgoing_transitions = self.collect_outgoing_transitions(&p_prime);

            // 5. For each transition, create the target DWA state and add the transition.
            for (label, q_prime) in outgoing_transitions {
                let w_prime = q_prime.values().fold(Weight::zeros(), |mut acc, w| {
                    acc |= w;
                    acc
                });

                if w_prime.is_empty() { continue; }

                let target_dwa_id = self.get_or_create_dwa_state(&q_prime);
                self.dwa.add_transition(dwa_id, label, target_dwa_id, w_prime).unwrap();
            }
             if let Some(p) = &pb { p.set_length(self.dwa.states.len() as u64); }
        }

        if let Some(p) = pb { p.finish_with_message(format!("Discovered {} DWA states", self.dwa.states.len())); }

        std::mem::take(&mut self.dwa)
    }

    /// For a given weighted subset, finds or creates a corresponding DWA state.
    fn get_or_create_dwa_state(&mut self, subset: &WeightedSubset) -> StateID {
        if let Some(&id) = self.dwa_state_map.get(subset) {
            return id;
        }

        // This is for a *new* state, so we add one to the DWA.
        let new_id = self.dwa.add_state();
        self.dwa_state_map.insert(subset.clone(), new_id);

        // The new subset corresponds to the new_id. This keeps the vectors in sync.
        self.weighted_subsets.push(subset.clone());
        self.work_queue.push_back(new_id);
        new_id
    }

    /// Computes the epsilon-closure from a given NWA state.
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
                    q.push_back((*v, new_weight.clone())); // Clone here
                }
            }
        }

        let result = Arc::new(closure);
        self.eps_closure_cache.insert(start_state, result.clone());
        result
    }

    /// For a given DWA state (p_prime), find all outgoing labeled transitions.
    fn collect_outgoing_transitions(&mut self, p_prime: &WeightedSubset) -> BTreeMap<i16, WeightedSubset> {
        let mut transitions = BTreeMap::<i16, WeightedSubset>::new();

        for (&nwa_state, residual_weight) in p_prime {
            let state = &self.nwa.states[nwa_state];

            for (label, targets) in &state.transitions {
                for (target_state, trans_weight) in targets {
                    let propagated_weight = residual_weight & trans_weight;
                    if propagated_weight.is_empty() { continue; }

                    let target_closure = self.compute_epsilon_closure(*target_state);
                    let next_subset = transitions.entry(*label).or_default();

                    for (closure_state, closure_weight) in target_closure.iter() {
                        let final_weight = &propagated_weight & closure_weight;
                        if !final_weight.is_empty() {
                            *next_subset.entry(*closure_state).or_default() |= &final_weight;
                        }
                    }
                }
            }
        }
        transitions
    }

    /// Calculates the final weight of a DWA state.
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