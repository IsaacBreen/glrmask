use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::{NWADefaultTransition, NWAStateID};
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::r#macro::is_debug_level_enabled;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::Hash;
use std::time::Instant;

// --- Helper Structs for Determinization ---

struct StepPool {
    raw: Vec<Vec<(NWAStateID, Weight)>>,
    map: HashMap<u64, Vec<usize>>,
}

#[derive(Clone)]
struct CompiledStep {
    by_sig: Vec<(usize, Weight)>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct DefSig {
    step_id: usize,
    exceptions: BTreeSet<i16>,
}

#[derive(Clone)]
struct MacroSig {
    final_w: Option<Weight>,
    def: Vec<DefSig>,
    ex: BTreeMap<i16, Vec<usize>>,
}

#[derive(Clone, Hash, Eq, PartialEq)]
struct MacroSigKey {
    final_fp: u64,
    def: Vec<(usize, Vec<i16>)>,
    ex: Vec<(i16, Vec<usize>)>,
}

struct CompositionNode {
    final_weight: Option<Weight>,
    default_target_idx: Option<usize>,
    default_mask: Option<Weight>,
    exception_targets: BTreeMap<i16, usize>,
    exception_masks: BTreeMap<i16, Weight>,
    gates: HashMap<usize, Weight>,
    incoming_weight_union: Weight,
}

// --- Free Helper Functions ---

fn apply_weight_to_pairs(base: &[(NWAStateID, Weight)], w: &Weight) -> Vec<(NWAStateID, Weight)> {
    if w.is_all_fast() {
        return base.to_vec();
    }
    base.iter().map(|(sid, wt)| (*sid, wt & w)).filter(|(_, x)| !x.is_empty()).collect()
}

fn accumulate(dst: &mut HashMap<usize, Weight>, compiled: &[(usize, Weight)], gate: &Weight) {
    for (sid, w) in compiled.iter() {
        let x = w & gate;
        if !x.is_empty() {
            *dst.entry(*sid).or_default() |= &x;
        }
    }
}

fn eps_closure_masked_vec_one(
    s: NWAStateID,
    states: &NWAStates,
    fut: &[Weight],
    scratch_w: &mut [Weight],
    q: &mut VecDeque<NWAStateID>,
    touched: &mut Vec<NWAStateID>,
) -> Vec<(NWAStateID, Weight)> {
    if s >= states.len() || fut[s].is_empty() {
        return Vec::new();
    }

    scratch_w[s] = fut[s].clone();
    touched.push(s);
    q.push_back(s);

    while let Some(u) = q.pop_front() {
        let base = scratch_w[u].clone();
        if base.is_empty() {
            continue;
        }
        for &(v, ref w_eps) in &states[u].epsilons {
            if v >= states.len() {
                continue;
            }
            let mut prop = &base & w_eps;
            if prop.is_empty() {
                continue;
            }
            prop &= &fut[v];
            if prop.is_empty() {
                continue;
            }
            let old = &scratch_w[v];
            if (&prop | old) != *old {
                if old.is_empty() {
                    touched.push(v);
                }
                scratch_w[v] |= &prop;
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

// --- Determinizer Struct and Implementation ---

struct Determinizer<'a> {
    nwa: &'a NWA,
    fut: Vec<Weight>,
    eps_cache: Vec<Vec<(NWAStateID, Weight)>>,
    step_pool: StepPool,
    sigs: Vec<MacroSig>,
    state_to_sig_id: Vec<usize>,
    compiled_steps: Vec<CompiledStep>,
    nodes: Vec<CompositionNode>,
    work: VecDeque<usize>,
    in_queue: Vec<bool>,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA) -> Self {
        let n = nwa.states.len();
        Self {
            nwa,
            fut: Vec::new(),
            eps_cache: vec![Vec::new(); n],
            step_pool: StepPool::new(),
            sigs: Vec::with_capacity(n),
            state_to_sig_id: vec![0; n],
            compiled_steps: Vec::new(),
            nodes: Vec::new(),
            work: VecDeque::new(),
            in_queue: Vec::new(),
        }
    }

    fn run(&mut self) -> DWA {
        if self.nwa.states.0.is_empty() {
            return DWA::new();
        }

        self.fut = self.nwa.compute_future_weights();
        self.precompute_eps_closures();
        self.build_macro_signatures();
        self.compile_steps();
        self.discover_composition_nodes();
        self.build_dwa()
    }

    fn precompute_eps_closures(&mut self) {
        let n = self.nwa.states.len();
        let pb = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(n as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (ε-closures)")
                    .unwrap(),
            ))
        } else { None };

        let mut scratch_w = vec![Weight::zeros(); n];
        let mut q = VecDeque::new();
        let mut touched = Vec::new();
        for s in 0..n {
            self.eps_cache[s] = eps_closure_masked_vec_one(s, &self.nwa.states, &self.fut, &mut scratch_w, &mut q, &mut touched);
            if let Some(p) = &pb { p.inc(1); }
        }
        if let Some(p) = pb { p.finish_with_message("ε-closures done"); }
    }

    fn build_macro_signatures(&mut self) {
        let n = self.nwa.states.len();
        let pb = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(n as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Macro signatures)")
                    .unwrap(),
            ))
        } else { None };
        
        let mut sig_intern: HashMap<MacroSigKey, usize> = HashMap::new();

        for s in 0..n {
            let final_acc = self.eps_cache[s].iter().fold(Weight::zeros(), |mut acc, (t, w)| {
                if let Some(fw) = &self.nwa.states[*t].final_weight { acc |= &(w & fw); }
                acc
            });
            let final_w = if final_acc.is_empty() { None } else { Some(final_acc) };

            let mut def: Vec<DefSig> = Vec::new();
            for d in &self.nwa.states[s].default {
                if d.target >= n { continue; }
                let pairs = apply_weight_to_pairs(&self.eps_cache[d.target], &d.weight);
                if !pairs.is_empty() {
                    def.push(DefSig { step_id: self.step_pool.intern(pairs), exceptions: d.exceptions.clone() });
                }
            }

            let mut ex: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
            for (lbl, targets) in &self.nwa.states[s].transitions {
                let mut step_exs: Vec<usize> = Vec::new();
                for (to, w) in targets {
                    if *to >= n { continue; }
                    let pairs = apply_weight_to_pairs(&self.eps_cache[*to], w);
                    if !pairs.is_empty() { step_exs.push(self.step_pool.intern(pairs)); }
                }
                if !step_exs.is_empty() {
                    step_exs.sort_unstable();
                    let mut sorted_def_ids: Vec<_> = def.iter().map(|d| d.step_id).collect();
                    sorted_def_ids.sort_unstable();
                    if step_exs != sorted_def_ids { ex.insert(*lbl, step_exs); }
                }
            }

            let mut sorted_def_key: Vec<_> = def.iter().map(|d| (d.step_id, d.exceptions.iter().copied().collect::<Vec<_>>())).collect();
            sorted_def_key.sort_unstable();
            let key = MacroSigKey {
                final_fp: final_w.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO),
                def: sorted_def_key,
                ex: ex.iter().map(|(k, v)| (*k, v.clone())).collect(),
            };

            let sig_id = *sig_intern.entry(key).or_insert_with(|| {
                let id = self.sigs.len();
                self.sigs.push(MacroSig { final_w, def, ex });
                id
            });
            self.state_to_sig_id[s] = sig_id;
            if let Some(p) = &pb { p.inc(1); }
        }
        if let Some(p) = pb { p.finish_with_message("Macro signatures done"); }
    }

    fn compile_steps(&mut self) {
        let num_steps = self.step_pool.raw.len();
        let pb = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(num_steps as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Compile steps)")
                    .unwrap(),
            ))
        } else { None };

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

    fn discover_composition_nodes(&mut self) {
        let pb = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(0).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Discovering states)")
                    .unwrap(),
            ))
        } else { None };

        let mut init_map: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in &self.eps_cache[self.nwa.body.start_state] {
            *init_map.entry(self.state_to_sig_id[*t]).or_default() |= w;
        }

        let start_idx = 0;
        self.nodes.push(CompositionNode {
            final_weight: None, default_target_idx: None, default_mask: None,
            exception_targets: BTreeMap::new(), exception_masks: BTreeMap::new(),
            gates: init_map, incoming_weight_union: Weight::all(),
        });
        self.in_queue = vec![false; 1];
        self.in_queue[start_idx] = true;
        self.work.push_back(start_idx);

        while let Some(idx) = self.work.pop_front() {
            self.in_queue[idx] = false;
            if let Some(p) = &pb { p.inc(1); }

            let node_gates = self.nodes[idx].gates.clone();
            let target_maps = self.compute_target_maps_for_gates(&node_gates);

            let mut resolved_transitions = BTreeMap::new();
            for (label, map) in target_maps {
                let total_weight = map.values().fold(Weight::zeros(), |mut a, b| { a |= b; a });
                if total_weight.is_empty() {
                    if label.is_some() { resolved_transitions.insert(label, (idx, Weight::zeros())); }
                    continue;
                }

                let target_idx = self.find_or_create_target_node(&map);
                let mut any_change = false;
                for (sig_id, weight) in &map {
                    let entry = self.nodes[target_idx].gates.entry(*sig_id).or_default();
                    let new_w = &*entry | weight;
                    if new_w != *entry { *entry = new_w; any_change = true; }
                }
                if any_change {
                    if target_idx >= self.in_queue.len() { self.in_queue.resize(target_idx + 1, false); }
                    if !self.in_queue[target_idx] {
                        self.in_queue[target_idx] = true;
                        self.work.push_back(target_idx);
                    }
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
                if let Some(fw) = &self.sigs[*sig_id].final_w { acc |= &(gate & fw); }
                acc
            }));
            if let Some(p) = &pb { p.set_length(self.nodes.len() as u64); }
        }
        if let Some(p) = pb { p.finish_with_message(format!("Discovered {} compositions", self.nodes.len())); }
    }

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
                dwa.add_transition(i, *lbl, target, mask).unwrap();
            }
        }
        dwa
    }

    fn compute_target_maps_for_gates(&self, node_gates: &HashMap<usize, Weight>) -> BTreeMap<Option<i16>, HashMap<usize, Weight>> {
        let mut def_groups = HashMap::new();
        let mut ex_groups_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
        let mut def_exers_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
        let mut def_exceptions_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();

        for (sig_id, gate) in node_gates {
            if gate.is_empty() { continue; }
            for def in &self.sigs[*sig_id].def {
                *def_groups.entry(def.step_id).or_default() |= gate;
                for &lbl in &def.exceptions {
                    *def_exceptions_by_label.entry(lbl).or_default().entry(def.step_id).or_default() |= gate;
                }
            }
            for (lbl, ex_steps) in &self.sigs[*sig_id].ex {
                for ex_step in ex_steps {
                    *ex_groups_by_label.entry(*lbl).or_default().entry(*ex_step).or_default() |= gate;
                }
                for def in &self.sigs[*sig_id].def {
                    *def_exers_by_label.entry(*lbl).or_default().entry(def.step_id).or_default() |= gate;
                }
            }
        }

        let mut target_maps = BTreeMap::new();
        let mut def_target_map = HashMap::new();
        for (def_step, g) in &def_groups {
            accumulate(&mut def_target_map, &self.compiled_steps[*def_step].by_sig, g);
        }
        if !def_target_map.is_empty() {
            target_maps.insert(None, def_target_map);
        }

        let mut labels_to_consider: BTreeSet<i16> = BTreeSet::new();
        labels_to_consider.extend(ex_groups_by_label.keys().copied());
        labels_to_consider.extend(def_exceptions_by_label.keys().copied());

        for lbl in labels_to_consider {
            let mut map = HashMap::new();
            for (def_step, total_g) in &def_groups {
                let mut g_nonex = total_g.clone();
                if let Some(g) = def_exers_by_label.get(&lbl).and_then(|de| de.get(def_step)) { g_nonex -= g; }
                if let Some(g) = def_exceptions_by_label.get(&lbl).and_then(|dx| dx.get(def_step)) { g_nonex -= g; }
                if !g_nonex.is_empty() {
                    accumulate(&mut map, &self.compiled_steps[*def_step].by_sig, &g_nonex);
                }
            }
            if let Some(ex_groups) = ex_groups_by_label.get(&lbl) {
                for (ex_step, g_ex) in ex_groups {
                    accumulate(&mut map, &self.compiled_steps[*ex_step].by_sig, g_ex);
                }
            }
            target_maps.insert(Some(lbl), map);
        }
        target_maps
    }

    fn are_nodes_mergeable(&self, cand_node: &CompositionNode, new_gates: &HashMap<usize, Weight>, new_incoming_weight: &Weight) -> bool {
        let intersect = &cand_node.incoming_weight_union & new_incoming_weight;
        if intersect.is_empty() { return true; }

        let cand_gates_intersect: HashMap<_, _> = cand_node.gates.iter().map(|(s, w)| (*s, w & &intersect)).filter(|(_, w)| !w.is_empty()).collect();
        let new_gates_intersect: HashMap<_, _> = new_gates.iter().map(|(s, w)| (*s, w & &intersect)).filter(|(_, w)| !w.is_empty()).collect();

        if cand_gates_intersect.is_empty() && new_gates_intersect.is_empty() { return true; }

        let cand_transitions = self.compute_target_maps_for_gates(&cand_gates_intersect);
        let new_transitions = self.compute_target_maps_for_gates(&new_gates_intersect);

        cand_transitions == new_transitions
    }

    fn find_or_create_target_node(&mut self, map: &HashMap<usize, Weight>) -> usize {
        let incoming_weight = map.values().fold(Weight::zeros(), |mut a, b| { a |= b; a });

        let calculate_merge_cost = |cand_node: &CompositionNode| -> (usize, usize) {
            let mut spec_increase = 0;
            for sig_id in map.keys() {
                if !cand_node.gates.contains_key(sig_id) { spec_increase += 1; }
            }
            (spec_increase, cand_node.gates.len())
        };

        let best_cand_idx = self.nodes.iter().enumerate()
            .filter(|(_, cand_node)| self.are_nodes_mergeable(cand_node, map, &incoming_weight))
            .min_by_key(|(idx, cand_node)| (calculate_merge_cost(cand_node), *idx))
            .map(|(idx, _)| idx);

        if let Some(merge_idx) = best_cand_idx {
            self.nodes[merge_idx].incoming_weight_union |= &incoming_weight;
            return merge_idx;
        }

        let new_idx = self.nodes.len();
        self.nodes.push(CompositionNode {
            final_weight: None, default_target_idx: None, default_mask: None,
            exception_targets: BTreeMap::new(), exception_masks: BTreeMap::new(),
            gates: HashMap::new(), incoming_weight_union: incoming_weight,
        });
        new_idx
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

// --- NWA Public API ---

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();
        let mut nwa = self.clone();
        nwa.simplify();

        if is_debug_level_enabled(5) {
            eprintln!("NWA after simplify:\n{}", nwa);
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
