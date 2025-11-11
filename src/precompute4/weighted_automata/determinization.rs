use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::r#macro::is_debug_level_enabled;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::Hash;
use std::time::Instant;

// --- Determinization Algorithm ---

impl NWA {
    /// Determinizes the NWA into a DWA.
    /// The process involves several steps:
    /// 1.  Epsilon-closure computation for all states.
    /// 2.  Grouping NWA states by behavior into "macro signatures".
    /// 3.  Subset construction on these macro signatures to build DWA states ("composition nodes").
    /// 4.  State merging heuristics to keep the resulting DWA smaller.
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();
        let mut nwa = self.clone();
        nwa.simplify(); // Assumes simplify() exists on NWA.

        if is_debug_level_enabled(5) {
            eprintln!("NWA after simplify:\n{}", nwa);
        }

        if nwa.states.0.is_empty() {
            return DWA::new();
        }

        let mut determinizer = Determinizer::new(&nwa);
        let result = determinizer.run();

        if is_debug_level_enabled(5) {
            eprintln!("NWA::determinize_to_dwa result DWA stats:\n{}", result.stats());
            eprintln!("NWA::determinize_to_dwa took: {:?}", now.elapsed());
        }

        result
    }
}

// --- Helper Structs for Determinization ---

/// Interns sequences of (NWAStateID, Weight) pairs to avoid duplication.
struct StepPool {
    raw: Vec<Vec<(NWAStateID, Weight)>>,
    map: HashMap<u64, Vec<usize>>,
}

/// A compiled step represents transitions to a set of macro signatures with associated weights.
#[derive(Clone)]
struct CompiledStep {
    by_sig: Vec<(usize, Weight)>,
}

/// Signature for a default transition within a macro state.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct DefSig {
    step_id: usize,
    exceptions: BTreeSet<i16>,
}

/// A macro signature captures the behavior of an NWA state after epsilon closure.
#[derive(Clone)]
struct MacroSig {
    final_weight: Option<Weight>,
    default_transitions: Vec<DefSig>,
    exception_transitions: BTreeMap<i16, Vec<usize>>,
}

/// A key used for interning MacroSigs.
#[derive(Clone, Hash, Eq, PartialEq)]
struct MacroSigKey {
    final_weight_fp: u64,
    default_transitions: Vec<(usize, Vec<i16>)>,
    exception_transitions: Vec<(i16, Vec<usize>)>,
}

/// A `DWAStateBuilder` represents a state in the DWA being constructed.
/// It's a composition of NWA macro signatures, each with a "gate" weight.
struct DWAStateBuilder {
    final_weight: Option<Weight>,
    default_target_idx: Option<usize>,
    default_mask: Option<Weight>,
    exception_targets: BTreeMap<i16, usize>,
    exception_masks: BTreeMap<i16, Weight>,
    /// Map from macro signature ID to its gate weight in this composition.
    gates: HashMap<usize, Weight>,
    /// Union of all weights of incoming transitions to this node. Used for state merging heuristics.
    incoming_weight_union: Weight,
}

// --- Main Determinizer ---

struct Determinizer<'a> {
    nwa: &'a NWA,
    future_weights: Vec<Weight>,
    eps_cache: Vec<Vec<(NWAStateID, Weight)>>,
    step_pool: StepPool,
    signatures: Vec<MacroSig>,
    state_to_sig_id: Vec<usize>,
    compiled_steps: Vec<CompiledStep>,
    nodes: Vec<DWAStateBuilder>,
    work_queue: VecDeque<usize>,
    in_queue: Vec<bool>,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA) -> Self {
        let n = nwa.states.len();
        Self {
            nwa,
            future_weights: Vec::new(),
            eps_cache: vec![Vec::new(); n],
            step_pool: StepPool::new(),
            signatures: Vec::with_capacity(n),
            state_to_sig_id: vec![0; n],
            compiled_steps: Vec::new(),
            nodes: Vec::new(),
            work_queue: VecDeque::new(),
            in_queue: Vec::new(),
        }
    }

    /// Executes the determinization algorithm.
    fn run(&mut self) -> DWA {
        self.compute_future_weights();
        self.precompute_eps_closures();
        self.build_macro_signatures();
        self.compile_steps();
        self.discover_composition_nodes();
        self.build_dwa()
    }

    // --- Major Steps of Determinization ---

    /// Step 1: Compute future weights for all NWA states.
    /// A future weight of a state is the union of weights of all paths from that state to a final state.
    fn compute_future_weights(&mut self) {
        let n = self.nwa.states.len();
        let mut fut = vec![Weight::zeros(); n];
        let mut rev_adj: Vec<Vec<(NWAStateID, &Weight)>> = vec![vec![]; n];

        // Build reverse adjacency list for weight propagation.
        for p in 0..n {
            for &(t, ref w) in &self.nwa.states[p].epsilons {
                if t < n { rev_adj[t].push((p, w)); }
            }
            for (_, targets) in &self.nwa.states[p].transitions {
                for (t, w) in targets {
                    if *t < n { rev_adj[*t].push((p, w)); }
                }
            }
            for def in &self.nwa.states[p].default {
                if def.target < n { rev_adj[def.target].push((p, &def.weight)); }
            }
        }

        // Initialize work queue with final states.
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        for s in 0..n {
            if let Some(fw) = &self.nwa.states[s].final_weight {
                if !fw.is_empty() {
                    fut[s] = fw.clone();
                    q.push_back(s);
                }
            }
        }

        // Propagate weights backwards until a fixed point is reached.
        while let Some(v) = q.pop_front() {
            let fv = fut[v].clone();
            if fv.is_empty() { continue; }
            for &(p, w_pv) in &rev_adj[v] {
                let propagated_weight = &fv & w_pv;
                if !propagated_weight.is_empty() && !propagated_weight.is_subset_of(&fut[p]) {
                    fut[p] |= &propagated_weight;
                    q.push_back(p);
                }
            }
        }
        self.future_weights = fut;
    }

    /// Step 2: Precompute epsilon closures for all NWA states, masked by future weights.
    fn precompute_eps_closures(&mut self) {
        let n = self.nwa.states.len();
        let pb = Self::progress_bar(n as u64, "ε-closures");

        let mut scratch_w = vec![Weight::zeros(); n];
        let mut q = VecDeque::new();
        let mut touched = Vec::new();
        for s in 0..n {
            self.eps_cache[s] = self.compute_eps_closure_for_state(s, &mut scratch_w, &mut q, &mut touched);
            if let Some(p) = &pb { p.inc(1); }
        }
        if let Some(p) = pb { p.finish_with_message("ε-closures done"); }
    }

    /// Step 3: Build macro signatures for each NWA state.
    fn build_macro_signatures(&mut self) {
        let n = self.nwa.states.len();
        let pb = Self::progress_bar(n as u64, "Macro signatures");
        let mut sig_interner: HashMap<MacroSigKey, usize> = HashMap::new();

        for s in 0..n {
            let (sig, key) = self.build_one_macro_sig(s);
            let sig_id = *sig_interner.entry(key).or_insert_with(|| {
                let id = self.signatures.len();
                self.signatures.push(sig);
                id
            });
            self.state_to_sig_id[s] = sig_id;
            if let Some(p) = &pb { p.inc(1); }
        }
        if let Some(p) = pb { p.finish_with_message("Macro signatures done"); }
    }

    /// Step 4: Compile interned steps into transitions between macro signatures.
    fn compile_steps(&mut self) {
        let num_steps = self.step_pool.raw.len();
        let pb = Self::progress_bar(num_steps as u64, "Compile steps");

        self.compiled_steps = Vec::with_capacity(num_steps);
        for pairs in &self.step_pool.raw {
            let mut acc: HashMap<usize, Weight> = HashMap::new();
            for (t, w) in pairs.iter() {
                *acc.entry(self.state_to_sig_id[*t]).or_default() |= w;
            }
            let mut by_sig: Vec<(usize, Weight)> = acc.into_iter().collect();
            by_sig.sort_by_key(|(sid, _)| *sid);
            self.compiled_steps.push(CompiledStep { by_sig });
            if let Some(p) = &pb { p.inc(1); }
        }
        if let Some(p) = pb { p.finish_with_message("Compile steps done"); }
    }

    /// Step 5: Discover DWA states (DWAStateBuilders) via subset construction.
    fn discover_composition_nodes(&mut self) {
        let pb = Self::progress_bar(0, "Discovering states");

        let mut initial_gates: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in &self.eps_cache[self.nwa.body.start_state] {
            *initial_gates.entry(self.state_to_sig_id[*t]).or_default() |= w;
        }

        self.add_dwa_state_builder(initial_gates, Weight::all());

        while let Some(idx) = self.work_queue.pop_front() {
            self.in_queue[idx] = false;
            if let Some(p) = &pb { p.inc(1); }

            self.process_dwa_state_builder(idx);

            if let Some(p) = &pb { p.set_length(self.nodes.len() as u64); }
        }
        if let Some(p) = pb { p.finish_with_message(format!("Discovered {} DWA states", self.nodes.len())); }
    }

    /// Step 6: Build the final DWA from the discovered DWAStateBuilders.
    fn build_dwa(&self) -> DWA {
        let mut dwa = DWA::new();
        if self.nodes.is_empty() { return dwa; }
        dwa.states.0.resize(self.nodes.len(), Default::default());
        dwa.body.start_state = 0;

        for (i, node) in self.nodes.iter().enumerate() {
            dwa.states[i].final_weight = node.final_weight.clone();
            if let (Some(target), Some(mask)) = (node.default_target_idx, &node.default_mask) {
                if !mask.is_empty() {
                    dwa.set_default_transition(i, target, mask.clone()).unwrap();
                }
            }
            for (lbl, &target) in &node.exception_targets {
                let mask = node.exception_masks.get(lbl).cloned().unwrap_or_else(Weight::zeros);
                if !mask.is_empty() {
                    dwa.add_transition(i, *lbl, target, mask).unwrap();
                }
            }
        }
        dwa
    }

    // --- Helper Methods ---

    /// Computes the epsilon closure from a single state `s`.
    fn compute_eps_closure_for_state(
        &self, s: NWAStateID,
        scratch_w: &mut [Weight], q: &mut VecDeque<NWAStateID>, touched: &mut Vec<NWAStateID>,
    ) -> Vec<(NWAStateID, Weight)> {
        if s >= self.nwa.states.len() || self.future_weights[s].is_empty() {
            return Vec::new();
        }

        scratch_w[s] = self.future_weights[s].clone();
        touched.push(s);
        q.push_back(s);

        while let Some(u) = q.pop_front() {
            let base_weight = scratch_w[u].clone();
            if base_weight.is_empty() { continue; }

            for &(v, ref w_eps) in &self.nwa.states[u].epsilons {
                if v >= self.nwa.states.len() { continue; }
                let mut prop_weight = &base_weight & w_eps;
                if prop_weight.is_empty() { continue; }
                prop_weight &= &self.future_weights[v];
                if prop_weight.is_empty() { continue; }

                if !prop_weight.is_subset_of(&scratch_w[v]) {
                    if scratch_w[v].is_empty() { touched.push(v); }
                    scratch_w[v] |= &prop_weight;
                    q.push_back(v);
                }
            }
        }

        let mut out: Vec<(NWAStateID, Weight)> = Vec::with_capacity(touched.len());
        for &i in touched.iter() {
            out.push((i, scratch_w[i].clone()));
            scratch_w[i] = Weight::zeros();
        }
        touched.clear();
        out.sort_by_key(|(sid, _)| *sid);
        out
    }

    /// Constructs a `MacroSig` and its `MacroSigKey` for a single NWA state.
    fn build_one_macro_sig(&mut self, s: NWAStateID) -> (MacroSig, MacroSigKey) {
        // Final weight is the union of weights of epsilon paths to final states.
        let final_acc = self.eps_cache[s].iter().fold(Weight::zeros(), |mut acc, (t, w)| {
            if let Some(fw) = &self.nwa.states[*t].final_weight { acc |= &(w & fw); }
            acc
        });
        let final_weight = if final_acc.is_empty() { None } else { Some(final_acc) };

        // Collect default transitions, applying epsilon closures.
        let mut default_transitions: Vec<DefSig> = Vec::new();
        for d in &self.nwa.states[s].default {
            if d.target >= self.nwa.states.len() { continue; }
            let pairs = Self::apply_weight_to_pairs(&self.eps_cache[d.target], &d.weight);
            if !pairs.is_empty() {
                default_transitions.push(DefSig {
                    step_id: self.step_pool.intern(pairs),
                    exceptions: d.exceptions.clone(),
                });
            }
        }

        // Collect exception transitions, applying epsilon closures.
        let mut exception_transitions: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
        for (lbl, targets) in &self.nwa.states[s].transitions {
            let mut step_exs: Vec<usize> = Vec::new();
            for (to, w) in targets {
                if *to >= self.nwa.states.len() { continue; }
                let pairs = Self::apply_weight_to_pairs(&self.eps_cache[*to], w);
                if !pairs.is_empty() { step_exs.push(self.step_pool.intern(pairs)); }
            }
            if !step_exs.is_empty() {
                // Optimization: if exception transitions are identical to default transitions,
                // they are redundant and can be omitted.
                step_exs.sort_unstable();
                let mut sorted_def_ids: Vec<_> = default_transitions.iter().map(|d| d.step_id).collect();
                sorted_def_ids.sort_unstable();
                if step_exs != sorted_def_ids {
                    exception_transitions.insert(*lbl, step_exs);
                }
            }
        }

        let key = Self::create_macro_sig_key(&final_weight, &default_transitions, &exception_transitions);
        let sig = MacroSig { final_weight, default_transitions, exception_transitions };
        (sig, key)
    }

    /// Computes transitions and final weight for a single DWA state builder.
    fn process_dwa_state_builder(&mut self, idx: usize) {
        let node_gates = self.nodes[idx].gates.clone();
        let target_maps = self.compute_target_maps_for_gates(&node_gates);

        let mut resolved_transitions = BTreeMap::new();
        for (label, map) in target_maps {
            let total_weight = map.values().fold(Weight::zeros(), |mut a, b| { a |= b; a });
            if total_weight.is_empty() { continue; }

            let target_idx = self.find_or_create_target_node(&map);
            if self.propagate_weights_to_node(target_idx, &map) {
                self.enqueue_node(target_idx);
            }
            resolved_transitions.insert(label, (target_idx, total_weight));
        }

        let node = &mut self.nodes[idx];
        if let Some((target_idx, mask)) = resolved_transitions.remove(&None) {
            node.default_target_idx = Some(target_idx);
            node.default_mask = Some(mask);
        }
        for (label, (target_idx, mask)) in resolved_transitions {
            if let Some(lbl) = label {
                node.exception_targets.insert(lbl, target_idx);
                node.exception_masks.insert(lbl, mask);
            }
        }
        node.final_weight = Some(node_gates.iter().fold(Weight::zeros(), |mut acc, (sig_id, gate)| {
            if let Some(fw) = &self.signatures[*sig_id].final_weight { acc |= &(gate & fw); }
            acc
        }));
    }

    /// Computes transition maps for a set of gates, grouped by character label.
    fn compute_target_maps_for_gates(&self, node_gates: &HashMap<usize, Weight>) -> BTreeMap<Option<i16>, HashMap<usize, Weight>> {
        // This function computes the outgoing transitions from a DWA state (represented by `node_gates`).
        // The result is a map from label (None for default) to a "target map".
        // A target map is a set of (macro_signature, weight) pairs that define the target DWA state.

        // 1. Aggregate transitions from all macro signatures in the current DWA state.
        let mut all_default_steps: HashMap<usize, Weight> = HashMap::new();
        let mut all_exception_steps: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
        // `overridden_defaults`: for a given label, which default steps are overridden by an explicit exception transition.
        let mut overridden_defaults: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
        // `excepted_defaults`: for a given label, which default steps are explicitly excepted via `def.exceptions`.
        let mut excepted_defaults: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();

        for (&sig_id, gate) in node_gates {
            if gate.is_empty() { continue; }
            let sig = &self.signatures[sig_id];
            for def in &sig.default_transitions {
                *all_default_steps.entry(def.step_id).or_default() |= gate;
                for &lbl in &def.exceptions {
                    *excepted_defaults.entry(lbl).or_default().entry(def.step_id).or_default() |= gate;
                }
            }
            for (lbl, ex_steps) in &sig.exception_transitions {
                for &ex_step in ex_steps {
                    *all_exception_steps.entry(*lbl).or_default().entry(ex_step).or_default() |= gate;
                }
                // If a signature has an exception on `lbl`, all its default transitions are considered overridden for that label.
                for def in &sig.default_transitions {
                    *overridden_defaults.entry(*lbl).or_default().entry(def.step_id).or_default() |= gate;
                }
            }
        }

        let mut target_maps = BTreeMap::new();

        // 2. Compute the target map for the default transition. This is the union of all default steps.
        let mut default_target_map = HashMap::new();
        for (step_id, gate) in &all_default_steps {
            self.accumulate_step_targets(&mut default_target_map, *step_id, gate);
        }
        if !default_target_map.is_empty() {
            target_maps.insert(None, default_target_map.clone());
        }

        // 3. Compute target maps for all exception labels.
        let all_labels: BTreeSet<i16> = all_exception_steps.keys().copied()
            .chain(excepted_defaults.keys().copied()).collect();

        for lbl in all_labels {
            let mut map_for_lbl = HashMap::new();
            // Part A: Add default transitions that are not overridden or excepted on this label.
            for (step_id, total_gate) in &all_default_steps {
                let mut effective_gate = total_gate.clone();
                // Subtract parts of the gate where the default is overridden by an explicit exception.
                if let Some(g) = overridden_defaults.get(&lbl).and_then(|m| m.get(step_id)) {
                    effective_gate -= g;
                }
                // Subtract parts of the gate where the default is explicitly excepted.
                if let Some(g) = excepted_defaults.get(&lbl).and_then(|m| m.get(step_id)) {
                    effective_gate -= g;
                }
                if !effective_gate.is_empty() {
                    self.accumulate_step_targets(&mut map_for_lbl, *step_id, &effective_gate);
                }
            }
            // Part B: Add in the explicit exception transitions for this label.
            if let Some(ex_steps) = all_exception_steps.get(&lbl) {
                for (step_id, gate) in ex_steps {
                    self.accumulate_step_targets(&mut map_for_lbl, *step_id, gate);
                }
            }

            // An exception is only needed if the resulting transition is different from the default.
            if !map_for_lbl.is_empty() && map_for_lbl != default_target_map {
                target_maps.insert(Some(lbl), map_for_lbl);
            }
        }
        target_maps
    }

    /// Finds an existing DWA state builder to merge with, or creates a new one.
    fn find_or_create_target_node(&mut self, map: &HashMap<usize, Weight>) -> usize {
        let incoming_weight = map.values().fold(Weight::zeros(), |mut a, b| { a |= b; a });

        // Heuristic for finding the best existing state to merge into.
        // The cost function prioritizes merges that add fewer new macro signatures to a state,
        // and secondarily prefers merging into smaller states.
        let calculate_merge_cost = |cand_node: &DWAStateBuilder| -> (usize, usize) {
            let spec_increase = map.keys().filter(|k| !cand_node.gates.contains_key(k)).count();
            (spec_increase, cand_node.gates.len())
        };

        let best_cand_idx = self.nodes.iter().enumerate()
            .filter(|(_, cand_node)| self.are_nodes_mergeable(cand_node, map, &incoming_weight))
            .min_by_key(|&(_, cand_node)| calculate_merge_cost(cand_node))
            .map(|(idx, _)| idx);

        if let Some(merge_idx) = best_cand_idx {
            self.nodes[merge_idx].incoming_weight_union |= &incoming_weight;
            return merge_idx;
        }

        self.add_dwa_state_builder(HashMap::new(), incoming_weight)
    }

    /// Checks if a new set of gates can be merged into an existing candidate node.
    fn are_nodes_mergeable(&self, cand_node: &DWAStateBuilder, new_gates: &HashMap<usize, Weight>, new_incoming_weight: &Weight) -> bool {
        // Two sets of incoming transitions can be merged into the same target state if their behavior is consistent.
        // If their incoming weights are disjoint, they can always be merged.
        let intersect = &cand_node.incoming_weight_union & new_incoming_weight;
        if intersect.is_empty() { return true; }

        // If their weights overlap, we must check that their behavior on the overlapping part is identical.
        // Behavior is defined by the transitions they produce.
        let filter_map_by_weight = |gates: &HashMap<usize, Weight>, w: &Weight| -> HashMap<usize, Weight> {
            gates.iter().map(|(s, g)| (*s, g & w)).filter(|(_, g)| !g.is_empty()).collect()
        };

        let cand_gates_intersect = filter_map_by_weight(&cand_node.gates, &intersect);
        let new_gates_intersect = filter_map_by_weight(new_gates, &intersect);

        if cand_gates_intersect.is_empty() && new_gates_intersect.is_empty() { return true; }

        let cand_transitions = self.compute_target_maps_for_gates(&cand_gates_intersect);
        let new_transitions = self.compute_target_maps_for_gates(&new_gates_intersect);

        cand_transitions == new_transitions
    }

    fn add_dwa_state_builder(&mut self, gates: HashMap<usize, Weight>, incoming_weight_union: Weight) -> usize {
        let new_idx = self.nodes.len();
        self.nodes.push(DWAStateBuilder {
            final_weight: None, default_target_idx: None, default_mask: None,
            exception_targets: BTreeMap::new(), exception_masks: BTreeMap::new(),
            gates, incoming_weight_union,
        });
        self.enqueue_node(new_idx);
        new_idx
    }

    fn enqueue_node(&mut self, idx: usize) {
        if idx >= self.in_queue.len() { self.in_queue.resize(idx + 1, false); }
        if !self.in_queue[idx] {
            self.in_queue[idx] = true;
            self.work_queue.push_back(idx);
        }
    }

    fn propagate_weights_to_node(&mut self, node_idx: usize, weights: &HashMap<usize, Weight>) -> bool {
        let mut any_change = false;
        for (sig_id, weight) in weights {
            let entry = self.nodes[node_idx].gates.entry(*sig_id).or_default();
            if !weight.is_subset_of(entry) {
                *entry |= weight;
                any_change = true;
            }
        }
        any_change
    }

    // --- Static Helpers ---

    fn create_macro_sig_key(
        final_weight: &Option<Weight>,
        default_transitions: &[DefSig],
        exception_transitions: &BTreeMap<i16, Vec<usize>>,
    ) -> MacroSigKey {
        let mut sorted_def_key: Vec<_> = default_transitions.iter()
            .map(|d| (d.step_id, d.exceptions.iter().copied().collect::<Vec<_>>()))
            .collect();
        sorted_def_key.sort_unstable();

        MacroSigKey {
            final_weight_fp: final_weight.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO),
            default_transitions: sorted_def_key,
            exception_transitions: exception_transitions.iter().map(|(k, v)| (*k, v.clone())).collect(),
        }
    }

    fn apply_weight_to_pairs(base: &[(NWAStateID, Weight)], w: &Weight) -> Vec<(NWAStateID, Weight)> {
        if w.is_all_fast() { return base.to_vec(); }
        base.iter().map(|(sid, wt)| (*sid, wt & w)).filter(|(_, x)| !x.is_empty()).collect()
    }

    fn accumulate_step_targets(&self, dst: &mut HashMap<usize, Weight>, step_id: usize, gate: &Weight) {
        for (sid, w) in &self.compiled_steps[step_id].by_sig {
            let x = w & gate;
            if !x.is_empty() { *dst.entry(*sid).or_default() |= &x; }
        }
    }

    fn progress_bar(len: u64, stage: &str) -> Option<ProgressBar> {
        if PROGRESS_BAR_ENABLED {
            let pb = ProgressBar::new(len).with_style(
                ProgressStyle::default_bar()
                    .template(&format!("{{spinner:.green}} [Determinize: {{elapsed_precise}}] [{{wide_bar:.cyan/blue}}] {{pos}}/{{len}} ({})", stage))
                    .unwrap(),
            );
            Some(pb)
        } else {
            None
        }
    }
}

impl StepPool {
    fn new() -> Self { Self { raw: Vec::new(), map: HashMap::new() } }

    fn fingerprint(pairs: &[(NWAStateID, Weight)]) -> u64 {
        pairs.iter().fold(FP_ZERO, |fp, (sid, w)| mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2)))
    }

    fn intern(&mut self, mut pairs: Vec<(NWAStateID, Weight)>) -> usize {
        pairs.retain(|(_, w)| !w.is_empty());
        let fp = Self::fingerprint(&pairs);
        if let Some(cands) = self.map.get(&fp) {
            for &id in cands { if self.raw[id] == pairs { return id; } }
        }
        let id = self.raw.len();
        self.raw.push(pairs);
        self.map.entry(fp).or_default().push(id);
        id
    }
}
