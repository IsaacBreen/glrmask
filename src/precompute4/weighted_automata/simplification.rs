#![allow(dead_code)]

use super::common::{BENCHMARK_DEBUG, Label, NWAStateID, StateID, Weight};
use super::dwa::{DWAState, DWAStates, DWA};
use super::nwa::{NWAState, NWAStates, NWA};
use rayon::prelude::*;
use rustfst::algorithms::{minimize, minimize_with_config, MinimizeConfig};
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, VecDeque, HashSet, HashMap};
use std::sync::Arc;

pub const MAX_OPTIMIZE_ITERATIONS: usize = 1000;

#[derive(Clone, Debug)]
struct Partition {
    class_of: Vec<usize>,
    num_classes: usize,
}

impl Partition {
    fn new(num_states: usize) -> Self {
        Self {
            class_of: vec![0; num_states],
            num_classes: if num_states == 0 { 0 } else { 1 },
        }
    }
    fn num_classes(&self) -> usize { self.num_classes }
}

// ---------------- DWA minimization ----------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaTransitionSig {
    label: Label,
    dest_class: usize,
    weight: Weight,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaStateSignature {
    final_weight: Option<Weight>,
    outgoing: Vec<DwaTransitionSig>,
}

impl DwaStateSignature {
    fn from_state(state_id: StateID, states: &DWAStates, classes: &[usize]) -> Self {
        let st = &states[state_id];

        // For a DWA, there is at most one transition per (state, label).
        // This means we never have to aggregate multiple transitions with the
        // same (label, dest_class): each label contributes at most one
        // (label, dest_class, weight) triple to the signature.
        let mut outgoing = Vec::with_capacity(st.transitions.len());
        for (&label, &dest) in &st.transitions {
            let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
            if w.is_empty() {
                continue;
            }
            let dest_class = classes[dest];
            outgoing.push(DwaTransitionSig {
                label,
                dest_class,
                weight: w,
            });
        }
        // Iteration over the BTreeMap `transitions` yields labels in a
        // canonical sorted order, so `outgoing` is canonical without any
        // additional sorting.
        DwaStateSignature {
            final_weight: st.final_weight.clone(),
            outgoing,
        }
    }
}

/// DWA minimization using partition refinement.
fn minimize_dwa_partition(states: &DWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition { class_of: vec![], num_classes: 0 };
    }

    // OPTIMIZATION: Initialize partition based on final weights to reduce iterations
    // States with the same final weight start in the same class
    let mut initial_class_map: FxHashMap<Option<Weight>, usize> = FxHashMap::default();
    let mut class_of = Vec::with_capacity(n);
    let mut num_classes = 0;
    
    for s in 0..n {
        let fw = states[s].final_weight.clone();
        let c = *initial_class_map.entry(fw).or_insert_with(|| {
            let id = num_classes;
            num_classes += 1;
            id
        });
        class_of.push(c);
    }
    
    let mut partition = Partition { class_of, num_classes };
    let mut iter_count = 0;
    
    loop {
        iter_count += 1;
        
        // Pre-size HashMap based on expected number of classes (previous + some growth)
        let expected_classes = partition.num_classes.max(n / 4);
        let mut sig_to_class: FxHashMap<DwaStateSignature, usize> = 
            FxHashMap::with_capacity_and_hasher(expected_classes, Default::default());
        let mut new_classes = vec![0; n];
        let mut next_class = 0;

        for s in 0..n {
            let sig = DwaStateSignature::from_state(s, states, &partition.class_of);
            let entry = sig_to_class.entry(sig).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
            new_classes[s] = *entry;
        }

        if new_classes == partition.class_of {
            partition.num_classes = next_class;
            crate::debug!(7, "Minimize converged after {} iterations with {} classes", iter_count, next_class);
            return partition;
        }

        partition.class_of = new_classes;
        partition.num_classes = next_class;
    }
}

#[derive(Clone, Debug, Default)]
struct DwaStateBuilder {
    final_weight: Option<Weight>,
    trans: BTreeMap<Label, (StateID, Weight)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DwaPass {
    PruneUnreachable,
    PruneDeadEnds,
    PushWeights,
    PushWeightsToInitial,
    PushWeightsRustfst,  // Use rustfst's push_weights algorithm
    FactorUniformOutgoing,  // Factor out uniform outgoing weights to enable merging
    Minimize,
}

impl DWA {
    pub fn simplify(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        if BENCHMARK_DEBUG {
            let initial_states = self.states.len();
            let mut internal = self.clone();
            let internal_start = std::time::Instant::now();
            internal.simplify_internal();
            let internal_time = internal_start.elapsed();
            let internal_states = internal.states.len();

            let mut rustfst = self.clone();
            let rustfst_start = std::time::Instant::now();
            rustfst.simplify_with_rustfst();
            let rustfst_time = rustfst_start.elapsed();
            let rustfst_states = rustfst.states.len();

            if internal_time + rustfst_time > std::time::Duration::from_secs(1) {
                let state_cmp = match internal_states.cmp(&rustfst_states) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };
                let time_cmp = match internal_time.cmp(&rustfst_time) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };

                crate::debug!(6, "[DWA Simplify({})] Internal: t={:.2?}, s={} | RustFST: t={:.2?}, s={}. [s: {}, t: {}]", initial_states, internal_time, internal_states, rustfst_time, rustfst_states, state_cmp, time_cmp);
            }

            *self = internal;
        } else {
            self.simplify_internal();
        }
    }

    /// Performs linear-time optimizations only (Pruning, Weight Pushing).
    /// Skips the expensive O(N log N) or O(N^2) state minimization.
    /// Useful for template generation where we just want a clean graph quickly.
    pub fn simplify_lightweight(&mut self) {
        // PruneDeadEnds, PushWeights, PruneUnreachable
        let ordering = &[
            DwaPass::PruneDeadEnds,
            DwaPass::PushWeights,
            DwaPass::PruneUnreachable,
        ];

        for _ in 0..10 {
            let mut changed_in_iteration = false;
            for &pass in ordering {
                let pass_changed = match pass {
                    DwaPass::PruneUnreachable => self.prune_unreachable(),
                    DwaPass::PruneDeadEnds => self.prune_dead_ends(),
                    DwaPass::PushWeights => self.push_weights_into_transitions_and_finals(),
                    DwaPass::PushWeightsToInitial => self.push_weights_to_initial(),
                    DwaPass::PushWeightsRustfst => self.push_weights_with_rustfst(),
                    DwaPass::FactorUniformOutgoing => self.factor_uniform_outgoing_weights(),
                    DwaPass::Minimize => unreachable!(),
                };
                changed_in_iteration |= pass_changed;
            }
            if !changed_in_iteration {
                break;
            }
        }
    }

    /// Performs a single pass of all optimization passes including minimize.
    /// Unlike simplify(), this does NOT iterate until fixpoint - it runs each pass once.
    /// Useful for terminal DWAs where we want minimize but don't need full convergence.
    pub fn simplify_single_pass(&mut self) {
        // Order: clean graph, minimize, push weights, factor uniform, minimize again, clean
        self.prune_dead_ends();
        self.prune_unreachable();
        self.minimize_states();
        self.push_weights_into_transitions_and_finals();
        self.factor_uniform_outgoing_weights();  // Factor out uniform weights to enable more merging
        self.minimize_states();  // Second minimize after factoring
        self.prune_dead_ends();
        self.prune_unreachable();
    }

    pub fn minimize_with_rustfst(&mut self) {
        let mut fst = self.to_rustfst();
        minimize(&mut fst).unwrap();
        *self = DWA::from_rustfst(&fst);
    }

    pub fn simplify_with_rustfst(&mut self) -> bool {
        let min_config = MinimizeConfig::default();
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, min_config).unwrap();
        *self = DWA::from_rustfst(&fst);
        true
    }
    
    /// Push weights toward initial state using rustfst's push algorithm.
    /// This normalizes the FST so each state's outgoing weights "sum" to one.
    pub fn push_weights_with_rustfst(&mut self) -> bool {
        use rustfst::algorithms::{push_weights, ReweightType};
        let initial_states = self.states.len();
        let mut fst = self.to_rustfst();
        // Push toward initial - each non-initial state will have outgoing weights that "sum" to 1
        if let Err(e) = push_weights(&mut fst, ReweightType::ReweightToInitial) {
            crate::debug!(1, "Warning: rustfst push_weights failed: {:?}", e);
            return false;
        }
        *self = DWA::from_rustfst(&fst);
        self.states.len() != initial_states
    }

    pub fn simplify_internal(&mut self) -> bool {
        let initial_num_states = self.states.len();
        if initial_num_states > 1000 {
            crate::debug!(6, "[DWA::simplify] Starting simplification. Initial stats: {}", self.stats());
        }
        
        // OPTIMIZATION: For small DWAs (< 1000 states), use a faster single-pass approach
        // instead of the iterative history-based algorithm. Template DWAs are typically small
        // and benefit from this optimization (saves ~200ms for 78 templates).
        if initial_num_states < 1000 {
            let mut changed = false;
            let prune1 = self.prune_dead_ends();
            let min1 = self.minimize_states();
            let push1 = self.push_weights_into_transitions_and_finals();
            let push2 = self.push_weights_to_initial();
            let factor1 = self.factor_uniform_outgoing_weights();  // Factor out uniform weights
            let prune2 = self.prune_unreachable();
            changed = prune1 || min1 || push1 || push2 || factor1 || prune2;
            
            // Second pass ONLY if pruning/pushing/factoring changed something (not just minimize)
            // After minimize, the DWA is already minimal so re-minimizing won't help
            // unless structure was changed by prune/push/factor
            if prune1 || push1 || push2 || factor1 || prune2 {
                self.prune_dead_ends();
                self.minimize_states();
                self.prune_unreachable();
            }
            return changed;
        }
        
        let mut total_changed = false;
        let ordering = &[
            DwaPass::PruneDeadEnds,
            DwaPass::Minimize,
            DwaPass::PushWeights,
            DwaPass::PushWeightsToInitial,
            DwaPass::FactorUniformOutgoing,  // Factor out uniform weights to enable more merging
            DwaPass::PruneUnreachable,
        ];
        
        // History of which passes changed things in the last 2 iterations.
        // Start empty - first iteration will run all passes unconditionally.
        let mut history: Vec<HashSet<DwaPass>> = vec![];
        
        // Track whether minimize has been run and found no improvement since last structure change
        let mut minimize_fully_explored = false;
        
        let mut force_all_passes = true;  // Force all passes on first iteration
        let mut converged = false;

        for iter_num in 0..MAX_OPTIMIZE_ITERATIONS {
            let mut current_changing_passes = HashSet::new();
            let mut changed_in_iteration = false;
            
            for &pass in ordering {
                // Special handling for Minimize: skip if we've already determined it can't help
                // This prevents the expensive partition computation when we know it won't reduce states
                if pass == DwaPass::Minimize && minimize_fully_explored {
                    continue;
                }
                
                // Skip if:
                // 1. Not forcing all passes
                // 2. Pass didn't change anything in the last 2 iterations
                // 3. Nothing has changed yet in this iteration (cascade effect)
                let recent_activity = history.iter().any(|s| s.contains(&pass));
                if !force_all_passes && !recent_activity && !changed_in_iteration {
                    continue;
                }
                
                let pass_changed = match pass {
                    DwaPass::PruneUnreachable => self.prune_unreachable(),
                    DwaPass::PruneDeadEnds => self.prune_dead_ends(),
                    DwaPass::PushWeights => self.push_weights_into_transitions_and_finals(),
                    DwaPass::PushWeightsToInitial => self.push_weights_to_initial(),
                    DwaPass::PushWeightsRustfst => self.push_weights_with_rustfst(),
                    DwaPass::FactorUniformOutgoing => self.factor_uniform_outgoing_weights(),
                    DwaPass::Minimize => {
                        let changed = self.minimize_states();
                        if !changed {
                            // Minimize found no improvement - mark as fully explored
                            // It will only be tried again if a structure-modifying pass changes something
                            minimize_fully_explored = true;
                        }
                        changed
                    },
                };
                
                if pass_changed {
                    current_changing_passes.insert(pass);
                    // If a non-minimize pass changed something, minimize might have new opportunities
                    if pass != DwaPass::Minimize {
                        minimize_fully_explored = false;
                    }
                }
                changed_in_iteration |= pass_changed;
            }
            
            history.push(current_changing_passes);
            if history.len() > 2 {
                history.remove(0);
            }

            total_changed |= changed_in_iteration;
            if !changed_in_iteration {
                if force_all_passes {
                    converged = true;
                    break;
                }
                force_all_passes = true;
            } else {
                force_all_passes = false;
            }
        }

        if !converged {
            let last_changes = history.last().map(|s| s.iter().copied().collect::<Vec<_>>()).unwrap_or_default();
            crate::debug!(4, "DWA simplification did not converge after {} iterations. Still changing: {:?}", MAX_OPTIMIZE_ITERATIONS, last_changes);
        }

        if initial_num_states > 1000 {
            crate::debug!(6, "[DWA::simplify] Simplification finished. Total changed: {}. Final stats: {}", total_changed, self.stats());
        }
        total_changed
    }

    pub fn push_weights_into_transitions_and_finals(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }
        let start = self.body.start_state;
        if start >= n {
            return false;
        }

        let mut changed = false;
        let mut preds: Vec<Vec<(StateID, Label)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for (&label, &v) in &st.transitions {
                if v < n {
                    preds[v].push((u, label));
                }
            }
        }

        for v in 0..n {
            if v == start {
                continue;
            }
            if let Some(sw) = self.states[v].state_weight.take() {
                if sw.is_empty() {
                    changed = true;
                    for (u, label) in &preds[v] {
                        if let Some(w) = self.states[*u].trans_weights.get_mut(label) {
                            *w &= &sw;
                        }
                    }
                } else if sw != Weight::all() {
                    changed = true;
                    for (u, label) in &preds[v] {
                        if let Some(w) = self.states[*u].trans_weights.get_mut(label) {
                            *w &= &sw;
                        }
                    }
                } else {
                    changed = true;
                }
            }
        }

        if let Some(sw0) = self.states[start].state_weight.take() {
            if !sw0.is_empty() && sw0 != Weight::all() {
                changed = true;
                for st in &mut self.states.0 {
                    if let Some(ref mut fw) = st.final_weight {
                        *fw &= &sw0;
                    }
                }
            } else if sw0.is_empty() {
                changed = true;
                for st in &mut self.states.0 {
                    if let Some(ref mut fw) = st.final_weight {
                        *fw &= &sw0;
                    }
                }
            } else {
                changed = true;
            }
        }

        changed
    }

    pub fn push_weights_to_initial(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }

        // 1. Compute backward distance (accumulated weight to final)
        let mut d = vec![Weight::zeros(); n];
        let mut q = VecDeque::new();
        let mut in_queue = vec![false; n];

        // Initialize with final weights
        for i in 0..n {
            if let Some(fw) = &self.states[i].final_weight {
                if !fw.is_empty() {
                    d[i] = fw.clone();
                    q.push_back(i);
                    in_queue[i] = true;
                }
            }
        }

        // Build reverse graph for propagation
        let mut preds: Vec<Vec<(StateID, Label, Weight)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for (&label, &v) in &st.transitions {
                if v < n {
                    let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
                    preds[v].push((u, label, w));
                }
            }
        }

        while let Some(v) = q.pop_front() {
            in_queue[v] = false;
            let d_v = d[v].clone();
            if d_v.is_empty() { continue; }

            for (u, label, w) in &preds[v] {
                // d[u] += w * d[v]
                let new_d = w & &d_v;
                if !new_d.is_subset_of(&d[*u]) {
                    d[*u] |= &new_d;
                    if !in_queue[*u] {
                        q.push_back(*u);
                        in_queue[*u] = true;
                    }
                }
            }
        }

        // 2. Reweight
        let mut changed = false;
        let start_node = self.body.start_state;
        for (u, st) in self.states.0.iter_mut().enumerate() {
            let d_u = &d[u];
            let inv_d_u = if u == start_node { Weight::zeros() } else { d_u.complement() };

            // Transitions
            for (&label, &v) in &st.transitions {
                if v < n {
                    let d_v = &d[v];
                    if let Some(w) = st.trans_weights.get_mut(&label) {
                        let new_w = (&*w & d_v) | &inv_d_u;
                        if *w != new_w {
                            *w = new_w;
                            changed = true;
                        }
                    }
                }
            }
            
            // Final weights
            if let Some(fw) = &mut st.final_weight {
                let new_fw = &*fw | &inv_d_u;
                if *fw != new_fw {
                    *fw = new_fw;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Factor out uniform outgoing weights and push them to incoming edges.
    /// 
    /// If state v has the SAME weight W on ALL outgoing transitions:
    /// 1. Multiply W into all incoming transitions to v
    /// 2. Set all outgoing trans_weights from v to identity (Weight::all())
    /// 
    /// This enables states that only differ in their uniform outgoing factor
    /// to become equivalent and merge via minimization.
    pub fn factor_uniform_outgoing_weights(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }
        let start = self.body.start_state;
        
        crate::debug!(1, "factor_uniform_outgoing_weights: {} states, start={}", n, start);

        // Build predecessor map: preds[v] = [(u, label)] where u -> v on label
        let mut preds: Vec<Vec<(StateID, Label)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for (&label, &v) in &st.transitions {
                if v < n {
                    preds[v].push((u, label));
                }
            }
        }

        let mut changed = false;

        // Process each state
        let mut checked_count = 0;
        let mut identity_count = 0;
        let mut not_all_equal_count = 0;
        for v in 0..n {
            // Skip start state - factoring from it would be complicated
            if v == start {
                continue;
            }

            let st = &self.states[v];
            if st.transitions.is_empty() {
                continue;
            }
            checked_count += 1;

            // Check if all outgoing trans_weights are EXACTLY equal
            let mut iter = st.transitions.keys();
            let first_label = *iter.next().unwrap();
            let first_weight = st.trans_weights.get(&first_label).cloned().unwrap_or_else(Weight::all);
            
            // If first weight is already identity, nothing to factor out
            if first_weight == Weight::all() {
                identity_count += 1;
                continue;
            }

            let mut all_equal = true;
            for &label in iter {
                let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
                if w != first_weight {
                    all_equal = false;
                    break;
                }
            }

            if !all_equal {
                not_all_equal_count += 1;
                continue;
            }

            // All outgoing weights are equal to first_weight
            // Factor it out: push first_weight to incoming edges
            for (u, label) in &preds[v] {
                if let Some(w) = self.states[*u].trans_weights.get_mut(label) {
                    *w &= &first_weight;  // times = intersection
                    changed = true;
                }
            }

            // Collect labels first to avoid borrow conflict
            let labels: Vec<_> = self.states[v].transitions.keys().cloned().collect();
            // Set all outgoing trans_weights to identity
            for label in labels {
                if let Some(w) = self.states[v].trans_weights.get_mut(&label) {
                    *w = Weight::all();
                }
            }
            crate::debug!(1, "  Factored state {}: uniform weight {:?}", v, first_weight);
            changed = true;
        }

        if n >= 100 {
            crate::debug!(1, "factor_uniform_outgoing_weights stats: checked={}, identity={}, not_all_equal={}", 
                checked_count, identity_count, not_all_equal_count);
        }
        crate::debug!(1, "factor_uniform_outgoing_weights: changed={}", changed);
        changed
    }

    pub fn minimize_states(&mut self) -> bool {
        let n = self.states.len();
        if n <= 1 {
            return false;
        }
        // Skip minimize for very small automata (optimization)
        if n < 3 {
            return false;
        }
        
        // Quick check: count distinct final weights. If all states have distinct
        // final weights, no merging is possible. This is a common case.
        let mut fw_count = 0;
        let mut seen_final_weights: rustc_hash::FxHashSet<Option<Weight>> = rustc_hash::FxHashSet::default();
        for s in 0..n {
            if seen_final_weights.insert(self.states[s].final_weight.clone()) {
                fw_count += 1;
            }
        }
        // If we have as many distinct final weight classes as states, minimize won't help
        // (partition refinement starts from final weights, so at best we get fw_count classes)
        if fw_count == n {
            return false;
        }
        
        let partition = minimize_dwa_partition(&self.states);
        if partition.num_classes() >= n {
            return false;
        }
        self.rebuild_from_partition(partition);
        true
    }

    fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        if n == 0 {
            return;
        }
        let mut class_to_new: FxHashMap<usize, StateID> = FxHashMap::default();
        let mut builders: Vec<DwaStateBuilder> = Vec::new();

        for s in 0..n {
            let c = partition.class_of[s];
            class_to_new.entry(c).or_insert_with(|| {
                let id = builders.len();
                builders.push(DwaStateBuilder::default());
                id
            });
        }

        for old_s in 0..n {
            let c = partition.class_of[old_s];
            let new_id = class_to_new[&c];
            let builder = &mut builders[new_id];
            let st = &self.states[old_s];

            debug_assert!(st.state_weight.is_none());

            if let Some(ref fw) = st.final_weight {
                if !fw.is_empty() {
                    match &mut builder.final_weight {
                        Some(existing) => *existing |= fw,
                        None => builder.final_weight = Some(fw.clone()),
                    }
                }
            }

            for (&label, &dest) in &st.transitions {
                let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
                if w.is_empty() {
                    continue;
                }
                let dest_class = partition.class_of[dest];
                let dest_new = class_to_new[&dest_class];
                use std::collections::btree_map::Entry;
                match builder.trans.entry(label) {
                    Entry::Vacant(e) => {
                        e.insert((dest_new, w));
                    }
                    Entry::Occupied(mut e) => {
                        let (existing_dest, existing_w) = e.get_mut();
                        debug_assert_eq!(
                            *existing_dest, dest_new,
                            "Determinism violated while rebuilding DWA: multiple destinations for label {} in class {}",
                            label, c
                        );
                        *existing_w |= &w;
                    }
                }
            }
        }

        let mut new_states = DWAStates::default();
        for _ in 0..builders.len() {
            new_states.add_state();
        }

        for (new_id, builder) in builders.into_iter().enumerate() {
            let st = &mut new_states[new_id];
            st.state_weight = None;
            st.final_weight = builder.final_weight;
            st.transitions.clear();
            st.trans_weights.clear();
            for (label, (dest_new, weight)) in builder.trans {
                st.transitions.insert(label, dest_new);
                st.trans_weights.insert(label, weight);
            }
        }

        let start_class = partition.class_of[self.body.start_state];
        let new_start = class_to_new[&start_class];
        self.states = new_states;
        self.body.start_state = new_start;
    }

    pub fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }
        if self.body.start_state >= n {
            let changed = n > 0;
            if changed {
                self.states = DWAStates::default();
                let start = self.states.add_state();
                self.body.start_state = start;
            }
            return changed;
        }

        let mut visited = vec![false; n];
        let mut q: VecDeque<StateID> = VecDeque::new();
        visited[self.body.start_state] = true;
        q.push_back(self.body.start_state);

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];
            for &v in st.transitions.values() {
                if v < n && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }

        if visited.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = DWAStates::default();

        for i in 0..n {
            if visited[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            let mut new_transitions: BTreeMap<Label, StateID> = BTreeMap::new();
            let mut new_trans_weights: BTreeMap<Label, Weight> = BTreeMap::new();
            for (&label, &old_dest) in &st.transitions {
                if old_dest < n && visited[old_dest] {
                    let new_dest = map[old_dest];
                    new_transitions.insert(label, new_dest);
                    if let Some(w) = st.trans_weights.get(&label) {
                        new_trans_weights.insert(label, w.clone());
                    }
                }
            }
            st.transitions = new_transitions;
            st.trans_weights = new_trans_weights;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }

    pub fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        let mut live = vec![false; n];
        let mut q: VecDeque<StateID> = VecDeque::new();
        let mut rev: Vec<Vec<StateID>> = vec![vec![]; n];

        for u in 0..n {
            let st = &self.states[u];
            for (&label, &v) in &st.transitions {
                if v >= n {
                    continue;
                }
                let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
                if !w.is_empty() {
                    rev[v].push(u);
                }
            }
        }

        for s in 0..n {
            if self.states[s].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                live[s] = true;
                q.push_back(s);
            }
        }

        while let Some(v) = q.pop_front() {
            for &u in &rev[v] {
                if !live[u] {
                    live[u] = true;
                    q.push_back(u);
                }
            }
        }

        if self.body.start_state >= n || !live[self.body.start_state] {
            if n == 1 && self.states[0] == DWAState::default() {
                return false;
            }

            let changed = n > 0;
            if changed {
                self.states = DWAStates::default();
                let start = self.states.add_state();
                self.body.start_state = start;
            }
            return changed;
        }

        if live.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = DWAStates::default();

        for i in 0..n {
            if live[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            let mut new_transitions: BTreeMap<Label, StateID> = BTreeMap::new();
            let mut new_trans_weights: BTreeMap<Label, Weight> = BTreeMap::new();
            for (&label, &old_dest) in &st.transitions {
                if old_dest < n && live[old_dest] {
                    let new_dest = map[old_dest];
                    new_transitions.insert(label, new_dest);
                    if let Some(w) = st.trans_weights.get(&label) {
                        new_trans_weights.insert(label, w.clone());
                    }
                }
            }
            st.transitions = new_transitions;
            st.trans_weights = new_trans_weights;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }
}

// ---------------- NWA minimization ----------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ArcLabel {
    Eps,
    Label(Label),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaTransitionSig {
    label: ArcLabel,
    dest_class: usize,
    weight: Weight,
}

impl NwaTransitionSig {
    /// Key used to sort transitions into a canonical order.
    ///
    /// Only the label and destination class matter for the language
    /// semantics; the weight is later aggregated (unioned) for runs
    /// sharing the same (label, dest_class). Therefore the sort key
    /// ignores the weight.
    fn sort_key(&self) -> (u8, Label, usize) {
        let label_tag = match self.label {
            ArcLabel::Eps => 0,
            ArcLabel::Label(_) => 1,
        };
        let label_val = match self.label {
            ArcLabel::Eps => 0,
            ArcLabel::Label(v) => v,
        };
        (label_tag, label_val, self.dest_class)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaStateSignature {
    final_weight: Option<Weight>,
    outgoing: Vec<NwaTransitionSig>,
}

impl NwaStateSignature {
    fn from_state(state_id: NWAStateID, states: &NWAStates, classes: &[usize]) -> Self {
        let st = &states[state_id];

        // Fast aggregation:
        //  1. Materialize one NwaTransitionSig per outgoing transition (ε and labeled),
        //     annotated with the current destination class.
        //  2. Sort by (label, dest_class).
        //  3. Linearly merge runs with the same (label, dest_class) by OR-ing their weights.
        //
        // The result is exactly the same as aggregating with a BTreeMap keyed by
        // (label, dest_class) and then iterating that map in key order.

        // Estimate the number of outgoing transitions to reserve capacity.
        let mut num_out = st.epsilons.len();
        for targets in st.transitions.values() {
            num_out += targets.len();
        }
        let mut tmp: Vec<NwaTransitionSig> = Vec::with_capacity(num_out);

        // Epsilon transitions
        for &(dest, ref w) in &st.epsilons {
            if w.is_empty() {
                continue;
            }
            tmp.push(NwaTransitionSig {
                label: ArcLabel::Eps,
                dest_class: classes[dest],
                weight: w.clone(),
            });
        }

        // Labeled transitions
        for (&lbl, targets) in &st.transitions {
            let label = ArcLabel::Label(lbl);
            for &(dest, ref w) in targets {
                if w.is_empty() {
                    continue;
                }
                tmp.push(NwaTransitionSig {
                    label,
                    dest_class: classes[dest],
                    weight: w.clone(),
                });
            }
        }

        if tmp.is_empty() {
            return NwaStateSignature {
                final_weight: st.final_weight.clone(),
                outgoing: Vec::new(),
            };
        }

        // Sort by (label, dest_class) to make equal-keys contiguous.
        tmp.sort_by_key(|sig| sig.sort_key());

        // Compress runs with the same (label, dest_class) by OR-ing the weights.
        let mut outgoing: Vec<NwaTransitionSig> = Vec::new();
        let mut iter = tmp.into_iter();
        if let Some(mut cur) = iter.next() {
            for sig in iter {
                if cur.label == sig.label && cur.dest_class == sig.dest_class {
                    cur.weight |= &sig.weight;
                } else {
                    if !cur.weight.is_empty() {
                        outgoing.push(cur);
                    }
                    cur = sig;
                }
            }
            if !cur.weight.is_empty() {
                outgoing.push(cur);
            }
        }

        NwaStateSignature {
            final_weight: st.final_weight.clone(),
            outgoing,
        }
    }
}

fn minimize_nwa_partition(states: &NWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition { class_of: vec![], num_classes: 0 };
    }

    let mut partition = Partition::new(n);
    loop {
        let mut sig_to_class: FxHashMap<NwaStateSignature, usize> = FxHashMap::default();
        let mut new_classes = vec![0; n];
        let mut next_class = 0;

        for s in 0..n {
            let sig = NwaStateSignature::from_state(s, states, &partition.class_of);
            let entry = sig_to_class.entry(sig).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
            new_classes[s] = *entry;
        }

        if new_classes == partition.class_of {
            partition.num_classes = next_class;
            return partition;
        }

        partition.class_of = new_classes;
        partition.num_classes = next_class;
    }
}

#[derive(Clone, Debug, Default)]
struct NwaStateBuilder {
    final_weight: Option<Weight>,
    eps: BTreeMap<NWAStateID, Weight>,
    trans: BTreeMap<Label, BTreeMap<NWAStateID, Weight>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NwaPass {
    PruneUnreachable,
    PruneDeadEnds,
    PushFinalWeights,
    PushWeightsToInitial,
    CompressTransitions,
    Minimize,
}

impl NWA {
    pub fn simplify(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        if BENCHMARK_DEBUG {
            let initial_states = self.states.len();
            let mut internal = self.clone();
            let internal_start = std::time::Instant::now();
            internal.simplify_internal();
            let internal_time = internal_start.elapsed();
            let internal_states = internal.states.len();

            let mut rustfst = self.clone();
            let rustfst_start = std::time::Instant::now();
            rustfst.simplify_with_rustfst();
            let rustfst_time = rustfst_start.elapsed();
            let rustfst_states = rustfst.states.len();

            if internal_time + rustfst_time > std::time::Duration::from_secs(1) {
                let state_cmp = match internal_states.cmp(&rustfst_states) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };
                let time_cmp = match internal_time.cmp(&rustfst_time) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };

                crate::debug!(6, "[NWA Simplify({})] Internal: t={:.2?}, s={} | RustFST: t={:.2?}, s={}. [s: {}, t: {}]", initial_states, internal_time, internal_states, rustfst_time, rustfst_states, state_cmp, time_cmp);
            }

            *self = internal;
        } else {
            self.simplify_internal();
        }
    }

    pub fn minimize_with_rustfst(&mut self) {
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, MinimizeConfig::default().with_allow_nondet(true)).unwrap();
        *self = NWA::from_rustfst(&fst);
    }

    pub fn simplify_with_rustfst(&mut self) -> bool {
        let min_config = MinimizeConfig::default().with_allow_nondet(true);
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, min_config).unwrap();
        *self = NWA::from_rustfst(&fst);
        true
    }

    pub fn simplify_internal(&mut self) -> bool {
        crate::debug!(6, "[NWA::simplify] Starting simplification. Initial stats: {}", self.stats());
        let mut total_changed = false;
        // PruneUnreachable, CompressTransitions, PushFinalWeights, PruneDeadEnds, Minimize
        let ordering = &[
            NwaPass::PruneUnreachable,
            NwaPass::CompressTransitions,
            NwaPass::PushFinalWeights,
            NwaPass::PushFinalWeights,
            NwaPass::PushWeightsToInitial,
            NwaPass::PruneDeadEnds,
            NwaPass::Minimize,
        ];

        // History of which passes changed things in the last 2 iterations.
        // Initialized with all passes so we run everything at least twice before skipping.
        let all_passes: HashSet<NwaPass> = ordering.iter().copied().collect();
        let mut history: Vec<HashSet<NwaPass>> = vec![all_passes.clone(), all_passes];

        let mut force_all_passes = false;
        let mut converged = false;

        for iter_num in 0..MAX_OPTIMIZE_ITERATIONS {
            let mut current_changing_passes = HashSet::new();
            let mut changed_in_iteration = false;
            for &pass in ordering {
                let recent_activity = history.iter().any(|s| s.contains(&pass));
                if !force_all_passes && !recent_activity && !changed_in_iteration {
                    continue;
                }

                let pass_changed = match pass {
                    NwaPass::PruneUnreachable => self.prune_unreachable(),
                    NwaPass::PruneDeadEnds => self.prune_dead_ends(),
                    NwaPass::PushFinalWeights => self.push_final_weights_along_epsilons(),
                    NwaPass::PushWeightsToInitial => self.push_weights_to_initial(),
                    NwaPass::CompressTransitions => self.compress_transitions(),
                    NwaPass::Minimize => self.minimize_states(),
                };
                if pass_changed {
                    current_changing_passes.insert(pass);
                }
                changed_in_iteration |= pass_changed;
            }

            history.push(current_changing_passes);
            if history.len() > 2 {
                history.remove(0);
            }

            total_changed |= changed_in_iteration;
            if !changed_in_iteration {
                if force_all_passes {
                    converged = true;
                    break;
                }
                force_all_passes = true;
            } else {
                force_all_passes = false;
            }
        }

        if !converged {
            let last_changes = history.last().map(|s| s.iter().copied().collect::<Vec<_>>()).unwrap_or_default();
            crate::debug!(4, "NWA simplification did not converge after {} iterations. Still changing: {:?}", MAX_OPTIMIZE_ITERATIONS, last_changes);
        }

        crate::debug!(6, "[NWA::simplify] Simplification finished. Total changed: {}. Final stats: {}", total_changed, self.stats());
        total_changed
    }

    pub fn minimize_states(&mut self) -> bool {
        crate::debug!(7, "[NWA] Minimizing states...");
        let n = self.states.len();
        if n <= 1 {
            return false;
        }
        let partition = minimize_nwa_partition(&self.states);
        if partition.num_classes() >= n {
            return false;
        }
        self.rebuild_from_partition(partition);
        true
    }

    /// Canonicalize NWA transitions by merging parallel transitions:
    ///  - For each state and epsilon edge, merge multiple (to, w) by unioning weights per `to`.
    ///  - For each state, label, and destination, merge multiple (label, to, w) by unioning weights.
    /// Transitions with empty weight are removed.
    pub fn compress_transitions(&mut self) -> bool {
        crate::debug!(7, "[NWA] Compressing transitions...");
        let mut changed = false;

        for st in &mut self.states.0 {
            // Compress epsilons: (to, w) -> union of weights per `to`.
            if !st.epsilons.is_empty() {
                let mut eps_map: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
                for &(to, ref w) in &st.epsilons {
                    if w.is_empty() {
                        continue;
                    }
                    eps_map.entry(to).and_modify(|acc| *acc |= w).or_insert(w.clone());
                }
                if eps_map.len() != st.epsilons.len() {
                    changed = true;
                }
                st.epsilons = eps_map.into_iter().filter(|(_, w)| !w.is_empty()).collect();
            }

            // Compress labeled transitions: per (label, to) aggregate weights by union.
            if !st.transitions.is_empty() {
                let mut new_transitions: BTreeMap<Label, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
                for (&lbl, targets) in &st.transitions {
                    let mut per_dest: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
                    for &(to, ref w) in targets {
                        if w.is_empty() {
                            continue;
                        }
                        per_dest.entry(to).and_modify(|acc| *acc |= w).or_insert(w.clone());
                    }
                    if per_dest.len() != targets.len() {
                        changed = true;
                    }
                    let merged: Vec<(NWAStateID, Weight)> =
                        per_dest.into_iter().filter(|(_, w)| !w.is_empty()).collect();
                    if !merged.is_empty() {
                        new_transitions.insert(lbl, merged);
                    }
                }
                if new_transitions.len() != st.transitions.len() {
                    changed = true;
                }
                st.transitions = new_transitions;
            }
        }

        changed
    }

    pub fn push_final_weights_along_epsilons(&mut self) -> bool {
        crate::debug!(7, "[NWA] Pushing final weights along epsilons...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        // Build reverse epsilon adjacency: for each target v, collect (u, w_uv) with u --w_uv--> v.
        let mut rev_eps: Vec<Vec<(NWAStateID, Weight)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for &(v, ref w) in &st.epsilons {
                if v < n && !w.is_empty() {
                    rev_eps[v].push((u, w.clone()));
                }
            }
        }

        // Initialise final_weights with the existing final weights.
        let mut final_weights: Vec<Weight> = Vec::with_capacity(n);
        let mut queue: VecDeque<NWAStateID> = VecDeque::new();
        for i in 0..n {
            let w = self.states.0[i].final_weight.clone().unwrap_or_else(Weight::zeros);
            if !w.is_empty() {
                queue.push_back(i);
            }
            final_weights.push(w);
        }

        let mut changed = false;

        // Worklist fixpoint for F(q) = f0(q) ∨ ⋁_{q --w--> r} (w ∧ F(r)).
        while let Some(v) = queue.pop_front() {
            let w_v = final_weights[v].clone();
            if w_v.is_empty() {
                continue;
            }

            for &(u, ref w_uv) in &rev_eps[v] {
                let candidate = &w_v & w_uv;
                if candidate.is_empty() {
                    continue;
                }
                let new_w = &final_weights[u] | &candidate;
                if new_w != final_weights[u] {
                    final_weights[u] = new_w;
                    queue.push_back(u);
                }
            }
        }

        for i in 0..n {
            let new_w = &final_weights[i];
            let new_final = if new_w.is_empty() { None } else { Some(new_w.clone()) };
            if self.states.0[i].final_weight != new_final {
                self.states.0[i].final_weight = new_final;
                changed = true;
            }
        }

        changed
    }

    pub fn push_weights_to_initial(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }

        // 1. Compute backward distance (accumulated weight to final)
        let mut d = vec![Weight::zeros(); n];
        let mut q = VecDeque::new();
        let mut in_queue = vec![false; n];

        // Initialize with final weights
        for i in 0..n {
            if let Some(fw) = &self.states[i].final_weight {
                if !fw.is_empty() {
                    d[i] = fw.clone();
                    q.push_back(i);
                    in_queue[i] = true;
                }
            }
        }

        // Build reverse graph
        // For NWA, we have transitions and epsilons.
        let mut preds: Vec<Vec<(NWAStateID, Option<Label>, Weight)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            // Epsilons
            for &(v, ref w) in &st.epsilons {
                if v < n {
                    preds[v].push((u, None, w.clone()));
                }
            }
            // Labeled
            for (&lbl, targets) in &st.transitions {
                for &(v, ref w) in targets {
                    if v < n {
                        preds[v].push((u, Some(lbl), w.clone()));
                    }
                }
            }
        }

        while let Some(v) = q.pop_front() {
            in_queue[v] = false;
            let d_v = d[v].clone();
            if d_v.is_empty() { continue; }

            for (u, _, w) in &preds[v] {
                // d[u] += w * d[v]
                let new_d = w & &d_v;
                if !new_d.is_subset_of(&d[*u]) {
                    d[*u] |= &new_d;
                    if !in_queue[*u] {
                        q.push_back(*u);
                        in_queue[*u] = true;
                    }
                }
            }
        }

        // 2. Reweight
        let mut changed = false;
        // For NWA, check if u is in start_states
        let starts: HashSet<NWAStateID> = self.body.start_states.iter().cloned().collect();

        for (u, st) in self.states.0.iter_mut().enumerate() {
            let d_u = &d[u];
            let inv_d_u = if starts.contains(&u) { Weight::zeros() } else { d_u.complement() };

            // Epsilons
            for (v, w) in &mut st.epsilons {
                if *v < n {
                    let d_v = &d[*v];
                    // w' = (w & d[v]) | !d[u]
                    let new_w = (&*w & d_v) | &inv_d_u;
                    if *w != new_w {
                        *w = new_w;
                        changed = true;
                    }
                }
            }
            // Labeled
            for targets in st.transitions.values_mut() {
                for (v, w) in targets {
                    if *v < n {
                        let d_v = &d[*v];
                        let new_w = (&*w & d_v) | &inv_d_u;
                        if *w != new_w {
                            *w = new_w;
                            changed = true;
                        }
                    }
                }
            }
            // Final weights
            if let Some(fw) = &mut st.final_weight {
                let new_fw = &*fw | &inv_d_u;
                if *fw != new_fw {
                    *fw = new_fw;
                    changed = true;
                }
            }
        }
        changed
    }

    fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        let mut class_to_new: FxHashMap<usize, NWAStateID> = FxHashMap::default();
        let mut builders: Vec<NwaStateBuilder> = Vec::new();

        for s in 0..n {
            let c = partition.class_of[s];
            class_to_new.entry(c).or_insert_with(|| {
                let id = builders.len();
                builders.push(NwaStateBuilder::default());
                id
            });
        }

        for old_s in 0..n {
            let c = partition.class_of[old_s];
            let new_id = class_to_new[&c];
            let builder = &mut builders[new_id];
            let st = &self.states[old_s];

            if let Some(ref fw) = st.final_weight {
                if !fw.is_empty() {
                    match &mut builder.final_weight {
                        Some(existing) => *existing |= fw,
                        None => builder.final_weight = Some(fw.clone()),
                    }
                }
            }

            for &(dest, ref w) in &st.epsilons {
                if w.is_empty() {
                    continue;
                }
                let dest_class = partition.class_of[dest];
                let new_dest = class_to_new[&dest_class];
                let entry = builder.eps.entry(new_dest).or_insert_with(Weight::zeros);
                *entry |= w;
            }

            for (&lbl, targets) in &st.transitions {
                for &(dest, ref w) in targets {
                    if w.is_empty() {
                        continue;
                    }
                    let dest_class = partition.class_of[dest];
                    let new_dest = class_to_new[&dest_class];
                    let per_label = builder.trans.entry(lbl).or_insert_with(BTreeMap::new);
                    let entry = per_label.entry(new_dest).or_insert_with(Weight::zeros);
                    *entry |= w;
                }
            }
        }

        let mut new_states = NWAStates::default();
        for _ in 0..builders.len() {
            new_states.add_state();
        }

        for (new_id, builder) in builders.into_iter().enumerate() {
            let st = &mut new_states[new_id];
            st.final_weight = builder.final_weight;
            st.epsilons.clear();
            for (dest_new, w) in builder.eps {
                if !w.is_empty() {
                    st.epsilons.push((dest_new, w));
                }
            }
            st.transitions.clear();
            for (lbl, dests_map) in builder.trans {
                let mut dests_vec: Vec<(NWAStateID, Weight)> = Vec::new();
                for (dest_new, w) in dests_map {
                    if !w.is_empty() {
                        dests_vec.push((dest_new, w));
                    }
                }
                if !dests_vec.is_empty() {
                    st.transitions.insert(lbl, dests_vec);
                }
            }
        }

        let mut new_start_states = Vec::new();
        for &old_start in &self.body.start_states {
            let start_class = partition.class_of[old_start];
            let new_start = class_to_new[&start_class];
            if !new_start_states.contains(&new_start) {
                new_start_states.push(new_start);
            }
        }
        self.states = new_states;
        self.body.start_states = new_start_states;
    }

    pub fn prune_unreachable(&mut self) -> bool {
        crate::debug!(7, "[NWA] Pruning unreachable states...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        if self.body.start_states.is_empty() {
            let changed = n > 0;
            if changed {
                self.states = NWAStates::default();
                self.body.start_states.clear();
            }
            return changed;
        }

        let mut reachable = vec![false; n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();

        for &start in &self.body.start_states {
            if start < n && !reachable[start] {
                reachable[start] = true;
                q.push_back(start);
            }
        }

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];

            for &(v, ref w) in &st.epsilons {
                if v < n && !reachable[v] && !w.is_empty() {
                    reachable[v] = true;
                    q.push_back(v);
                }
            }

            for (_, targets) in &st.transitions {
                for &(v, ref w) in targets {
                    if v < n && !reachable[v] && !w.is_empty() {
                        reachable[v] = true;
                        q.push_back(v);
                    }
                }
            }
        }

        if reachable.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = NWAStates::default();

        for i in 0..n {
            if reachable[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            st.epsilons.retain(|(v, w)| *v < n && !w.is_empty());
            for (v, _) in &mut st.epsilons {
                *v = map[*v];
            }

            let mut new_transitions: BTreeMap<Label, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
            for (&lbl, targets) in &st.transitions {
                let mut new_targets = Vec::new();
                for &(v, ref w) in targets {
                    if v < n && !w.is_empty() && reachable[v] {
                        new_targets.push((map[v], w.clone()));
                    }
                }
                if !new_targets.is_empty() {
                    new_transitions.insert(lbl, new_targets);
                }
            }
            st.transitions = new_transitions;
        }

        let mut new_start_states = Vec::new();
        for &s in &self.body.start_states {
            if s < n && reachable[s] {
                new_start_states.push(map[s]);
            }
        }
        self.body.start_states = new_start_states;
        self.states = new_states;
        true
    }

    pub fn prune_dead_ends(&mut self) -> bool {
        crate::debug!(7, "[NWA] Pruning dead ends...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        let mut live = vec![false; n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        let mut rev: Vec<Vec<NWAStateID>> = vec![vec![]; n];

        for p in 0..n {
            let st = &self.states[p];
            for &(t, ref w) in &st.epsilons {
                if t < n && !w.is_empty() {
                    rev[t].push(p);
                }
            }
            for (_, targets) in &st.transitions {
                for &(t, ref w) in targets {
                    if t < n && !w.is_empty() {
                        rev[t].push(p);
                    }
                }
            }
        }

        for s in 0..n {
            if self.states[s].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                if !live[s] {
                    live[s] = true;
                    q.push_back(s);
                }
            }
        }

        while let Some(v) = q.pop_front() {
            for &p in &rev[v] {
                if !live[p] {
                    live[p] = true;
                    q.push_back(p);
                }
            }
        }

        if live.iter().all(|&b| b) {
            return false;
        }
        
        // Check if start states survive
        let any_start_live = self.body.start_states.iter().any(|&s| s < n && live[s]);
        if !any_start_live {
             // If all start states are dead, the NWA accepts empty language.
             // Collapse to empty.
             if n == 0 { return false; }
             self.states = NWAStates::default();
             self.body.start_states.clear();
             return true;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = NWAStates::default();

        for i in 0..n {
            if live[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            st.epsilons.retain(|(v, w)| *v < n && !w.is_empty() && live[*v]);
            for (v, _) in &mut st.epsilons {
                *v = map[*v];
            }

            let mut new_transitions: BTreeMap<Label, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
            for (&lbl, targets) in &st.transitions {
                let mut new_targets = Vec::new();
                for &(v, ref w) in targets {
                    if v < n && !w.is_empty() && live[v] {
                        new_targets.push((map[v], w.clone()));
                    }
                }
                if !new_targets.is_empty() {
                    new_transitions.insert(lbl, new_targets);
                }
            }
            st.transitions = new_transitions;
        }

        let mut new_start_states = Vec::new();
        for &s in &self.body.start_states {
            if s < n && live[s] {
                new_start_states.push(map[s]);
            }
        }
        self.body.start_states = new_start_states;
        self.states = new_states;
        true
    }
}
