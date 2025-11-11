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

type Composition = BTreeMap<usize, Weight>;
type Behavior = BTreeMap<Option<i16>, Composition>;

fn apply_weight_to_pairs(base: &[(NWAStateID, Weight)], w: &Weight) -> Vec<(NWAStateID, Weight)> {
    if w.is_all_fast() {
        return base.to_vec();
    }
    base.iter()
        .map(|(sid, wt)| (*sid, wt & w))
        .filter(|(_, x)| !x.is_empty())
        .collect()
}

struct StepPool {
    raw: Vec<Vec<(NWAStateID, Weight)>>,
    map: HashMap<u64, Vec<usize>>,
}

impl StepPool {
    fn new() -> Self {
        Self { raw: Vec::new(), map: HashMap::new() }
    }
    fn fingerprint(pairs: &[(NWAStateID, Weight)]) -> u64 {
        pairs
            .iter()
            .fold(FP_ZERO, |fp, (sid, w)| mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2)))
    }
    fn intern(&mut self, mut pairs: Vec<(NWAStateID, Weight)>) -> usize {
        pairs.retain(|(_, w)| !w.is_empty());
        let fp = Self::fingerprint(&pairs);
        if let Some(cands) = self.map.get(&fp) {
            for &id in cands {
                if self.raw[id] == pairs {
                    return id;
                }
            }
        }
        let id = self.raw.len();
        self.raw.push(pairs);
        self.map.entry(fp).or_default().push(id);
        id
    }
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
    gates: Composition,
    incoming_weight_union: Weight,
}

/// The main struct for the determinization process. It holds all intermediate data structures
/// and the memoization cache for the expensive behavior computation.
struct Determinizer<'a> {
    nwa: &'a NWA,
    sigs: Vec<MacroSig>,
    compiled_steps: Vec<CompiledStep>,
    nodes: Vec<CompositionNode>,
    behavior_cache: HashMap<Composition, Behavior>,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA, sigs: Vec<MacroSig>, compiled_steps: Vec<CompiledStep>) -> Self {
        Self {
            nwa,
            sigs,
            compiled_steps,
            nodes: Vec::new(),
            behavior_cache: HashMap::new(),
        }
    }

    fn accumulate(dst: &mut Composition, compiled: &[(usize, Weight)], gate: &Weight) {
        for (sid, w) in compiled.iter() {
            let x = w & gate;
            if !x.is_empty() {
                *dst.entry(*sid).or_default() |= &x;
            }
        }
    }

    /// Computes the behavior (outgoing transitions) for a given composition of NWA states.
    /// This is the most performance-critical function, so its results are cached.
    fn compute_behavior(&mut self, gates: &Composition) -> Behavior {
        if let Some(cached) = self.behavior_cache.get(gates) {
            return cached.clone();
        }

        let mut def_groups: HashMap<usize, Weight> = HashMap::new();
        let mut ex_groups_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();
        let mut def_ex_sets_by_label: BTreeMap<i16, HashMap<usize, Weight>> = BTreeMap::new();

        for (&sig_id, gate) in gates {
            if gate.is_empty() {
                continue;
            }
            let sig = &self.sigs[sig_id];
            for def in &sig.def {
                *def_groups.entry(def.step_id).or_default() |= gate;
                for &lbl in &def.exceptions {
                    *def_ex_sets_by_label.entry(lbl).or_default().entry(def.step_id).or_default() |= gate;
                }
            }
            for (&lbl, ex_steps) in &sig.ex {
                for &ex_step in ex_steps {
                    *ex_groups_by_label.entry(lbl).or_default().entry(ex_step).or_default() |= gate;
                }
            }
        }

        let mut behavior: Behavior = Behavior::new();
        let mut def_target_map = Composition::new();
        for (def_step, g) in &def_groups {
            Self::accumulate(&mut def_target_map, &self.compiled_steps[*def_step].by_sig, g);
        }

        let mut labels_to_consider: BTreeSet<i16> = BTreeSet::new();
        labels_to_consider.extend(ex_groups_by_label.keys().copied());
        labels_to_consider.extend(def_ex_sets_by_label.keys().copied());

        for lbl in labels_to_consider {
            let mut map = Composition::new();
            let ex_groups = ex_groups_by_label.get(&lbl);
            let def_ex_set = def_ex_sets_by_label.get(&lbl);

            for (def_step, total_g) in &def_groups {
                let mut g_nonex = total_g.clone();
                if let Some(ex_gates) = ex_groups.and_then(|eg| eg.get(def_step)) {
                    g_nonex -= ex_gates;
                }
                if let Some(def_ex_gates) = def_ex_set.and_then(|de| de.get(def_step)) {
                    g_nonex -= def_ex_gates;
                }
                if !g_nonex.is_empty() {
                    Self::accumulate(&mut map, &self.compiled_steps[*def_step].by_sig, &g_nonex);
                }
            }
            if let Some(ex_groups) = ex_groups {
                for (ex_step, g_ex) in ex_groups {
                    Self::accumulate(&mut map, &self.compiled_steps[*ex_step].by_sig, g_ex);
                }
            }
            if map != def_target_map {
                behavior.insert(Some(lbl), map);
            }
        }

        if !def_target_map.is_empty() {
            behavior.insert(None, def_target_map);
        }

        self.behavior_cache.insert(gates.clone(), behavior.clone());
        behavior
    }

    /// Finds the best existing node to merge a new state composition into, or creates a new node.
    /// This contains the core merging heuristic.
    fn find_or_create_target_node(&mut self, map: Composition) -> usize {
        let incoming_transition_weight = map.values().fold(Weight::zeros(), |mut a, b| {
            a |= b;
            a
        });

        let best_cand_idx = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, cand_node)| {
                let intersect = &cand_node.incoming_weight_union & &incoming_transition_weight;
                if intersect.is_empty() {
                    return Some((idx, (0, cand_node.gates.len()))); // Disjoint is always ok, cost is low.
                }

                let cand_gates_intersect: Composition = cand_node
                    .gates
                    .iter()
                    .map(|(sid, w)| (*sid, w & &intersect))
                    .filter(|(_, w)| !w.is_empty())
                    .collect();

                let new_gates_intersect: Composition = map
                    .iter()
                    .map(|(sid, w)| (*sid, w & &intersect))
                    .filter(|(_, w)| !w.is_empty())
                    .collect();

                if self.compute_behavior(&cand_gates_intersect) == self.compute_behavior(&new_gates_intersect) {
                    let mut spec_increase = 0;
                    for sig_id in map.keys() {
                        if !cand_node.gates.contains_key(sig_id) {
                            spec_increase += 1;
                        }
                    }
                    Some((idx, (spec_increase, cand_node.gates.len())))
                } else {
                    None
                }
            })
            .min_by_key(|&(_idx, cost)| cost)
            .map(|(idx, _)| idx);

        if let Some(merge_idx) = best_cand_idx {
            self.nodes[merge_idx].incoming_weight_union |= &incoming_transition_weight;
            for (sig_id, weight) in map {
                *self.nodes[merge_idx].gates.entry(sig_id).or_default() |= &weight;
            }
            return merge_idx;
        }

        let new_idx = self.nodes.len();
        self.nodes.push(CompositionNode {
            final_weight: None,
            default_target_idx: None,
            default_mask: None,
            exception_targets: BTreeMap::new(),
            exception_masks: BTreeMap::new(),
            gates: map,
            incoming_weight_union: incoming_transition_weight,
        });
        new_idx
    }
}

/// Faster ε-closure from a single source with masked propagation.
fn eps_closure_fast(
    s: NWAStateID,
    states: &NWAStates,
    fut: &[Weight],
    scratch_w: &mut [Weight],
    q: &mut VecDeque<NWAStateID>,
    touched: &mut Vec<NWAStateID>,
) -> Vec<(NWAStateID, Weight)> {
    let n = states.len();
    if s >= n || fut[s].is_empty() {
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
            if v >= n {
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

            let old_v_w = &scratch_w[v];
            if !prop.is_subset_of(old_v_w) {
                let was_empty = old_v_w.is_empty();
                scratch_w[v] |= &prop;
                if was_empty {
                    touched.push(v);
                }
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

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();

        let mut nwa = self.clone();
        nwa.simplify();

        if is_debug_level_enabled(5) {
            eprintln!("NWA after simplify:\n{}", nwa);
        }
        let result = nwa.det_fixpoint();
        if is_debug_level_enabled(5) {
            eprintln!("NWA::determinize_to_dwa result DWA stats:\n{}", result.stats());
        }
        eprintln!("NWA::determinize_to_dwa took: {:?}", now.elapsed());

        result
    }

    fn det_fixpoint(&self) -> DWA {
        let fut = self.compute_future_weights();
        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }

        let pb_eps = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(n as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (ε-closures)")
                    .unwrap(),
            ))
        } else {
            None
        };
        let mut eps_cache: Vec<Vec<(NWAStateID, Weight)>> = vec![Vec::new(); n];
        let mut scratch_w: Vec<Weight> = vec![Weight::zeros(); n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        let mut touched: Vec<NWAStateID> = Vec::new();
        for s in 0..n {
            eps_cache[s] = eps_closure_fast(s, &self.states, &fut, &mut scratch_w, &mut q, &mut touched);
            if let Some(p) = &pb_eps {
                p.inc(1);
            }
        }
        if let Some(p) = pb_eps {
            p.finish_with_message("ε-closures done");
        }

        let mut step_pool = StepPool::new();
        let mut sigs: Vec<MacroSig> = Vec::with_capacity(n);
        let mut state_to_sig_id: Vec<usize> = vec![0; n];
        let mut sig_intern: HashMap<MacroSigKey, usize> = HashMap::new();
        let pb_sigs = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(n as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Macro signatures)")
                    .unwrap(),
            ))
        } else {
            None
        };
        for s in 0..n {
            let final_acc = eps_cache[s].iter().fold(Weight::zeros(), |mut acc, (t, w)| {
                if let Some(fw) = &self.states[*t].final_weight {
                    acc |= &(w & fw);
                }
                acc
            });
            let final_acc = if final_acc.is_empty() { None } else { Some(final_acc) };

            let mut def_steps: Vec<DefSig> = Vec::new();
            for default in &self.states[s].default {
                let pairs_def = apply_weight_to_pairs(&eps_cache[default.target], &default.weight);
                if !pairs_def.is_empty() {
                    def_steps.push(DefSig {
                        step_id: step_pool.intern(pairs_def),
                        exceptions: default.exceptions.clone(),
                    });
                }
            }

            let mut ex: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
            for (lbl, targets) in self.states[s].transitions.iter() {
                let mut step_exs: Vec<usize> = Vec::new();
                for (to, wlbl) in targets {
                    let pairs_ex = apply_weight_to_pairs(&eps_cache[*to], wlbl);
                    if !pairs_ex.is_empty() {
                        step_exs.push(step_pool.intern(pairs_ex));
                    }
                }
                if !step_exs.is_empty() {
                    ex.insert(*lbl, step_exs);
                }
            }

            let mut sorted_def_steps_key: Vec<(usize, Vec<i16>)> = def_steps
                .iter()
                .map(|d| (d.step_id, d.exceptions.iter().copied().collect()))
                .collect();
            sorted_def_steps_key.sort_unstable();
            let key = MacroSigKey {
                final_fp: final_acc.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO),
                def: sorted_def_steps_key,
                ex: ex.iter().map(|(k, v)| (*k, v.clone())).collect(),
            };
            let sig_id = *sig_intern.entry(key).or_insert_with(|| {
                let id = sigs.len();
                sigs.push(MacroSig { final_w: final_acc, def: def_steps, ex });
                id
            });
            state_to_sig_id[s] = sig_id;
            if let Some(p) = &pb_sigs {
                p.inc(1);
            }
        }
        if let Some(p) = pb_sigs {
            p.finish_with_message("Macro signatures done");
        }

        let pb_compile = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(step_pool.raw.len() as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Compile steps)")
                    .unwrap(),
            ))
        } else {
            None
        };
        let mut compiled_steps: Vec<CompiledStep> = Vec::with_capacity(step_pool.raw.len());
        for pairs in &step_pool.raw {
            let mut acc: BTreeMap<usize, Weight> = BTreeMap::new();
            for (t, w) in pairs.iter() {
                *acc.entry(state_to_sig_id[*t]).or_default() |= w;
            }
            let mut by_sig: Vec<(usize, Weight)> = acc.into_iter().collect();
            by_sig.sort_by_key(|(sid, _)| *sid);
            compiled_steps.push(CompiledStep { by_sig });
            if let Some(p) = &pb_compile {
                p.inc(1);
            }
        }
        if let Some(p) = pb_compile {
            p.finish_with_message("Compile steps done");
        }

        let mut det = Determinizer::new(self, sigs, compiled_steps);
        let mut work: VecDeque<usize> = VecDeque::new();

        let pb_discover = if PROGRESS_BAR_ENABLED {
            Some(
                ProgressBar::new(0).with_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Discovering states)")
                        .unwrap(),
                ),
            )
        } else {
            None
        };

        let mut init_map: Composition = BTreeMap::new();
        for (t, w) in eps_cache[self.body.start_state].iter() {
            *init_map.entry(state_to_sig_id[*t]).or_default() |= w;
        }

        let start_idx = det.nodes.len();
        det.nodes.push(CompositionNode {
            final_weight: None,
            default_target_idx: None,
            default_mask: None,
            exception_targets: BTreeMap::new(),
            exception_masks: BTreeMap::new(),
            gates: init_map,
            incoming_weight_union: Weight::all(),
        });
        let mut in_queue = vec![false; 1];
        in_queue[start_idx] = true;
        work.push_back(start_idx);

        while let Some(idx) = work.pop_front() {
            in_queue[idx] = false;
            if let Some(p) = &pb_discover {
                p.inc(1);
            }
            let node_gates = det.nodes[idx].gates.clone();
            let behavior = det.compute_behavior(&node_gates);

            let mut resolved_transitions: BTreeMap<Option<i16>, (usize, Weight)> = BTreeMap::new();
            for (label, map) in behavior {
                let total_weight = map.values().fold(Weight::zeros(), |mut a, b| {
                    a |= b;
                    a
                });
                if total_weight.is_empty() {
                    if label.is_some() {
                        resolved_transitions.insert(label, (idx, Weight::zeros()));
                    }
                    continue;
                }

                let target_idx = det.find_or_create_target_node(map);

                if target_idx >= in_queue.len() {
                    in_queue.resize(target_idx + 1, false);
                }
                if !in_queue[target_idx] {
                    in_queue[target_idx] = true;
                    work.push_back(target_idx);
                }
                resolved_transitions.insert(label, (target_idx, total_weight));
            }

            let node = &mut det.nodes[idx];
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
            node.final_weight = Into::into(node_gates.iter().fold(Weight::zeros(), |mut acc, (sig_id, gate)| {
                if let Some(fw) = &det.sigs[*sig_id].final_w {
                    acc |= &(gate & fw);
                }
                acc
            }));

            if let Some(p) = &pb_discover {
                p.set_length(det.nodes.len() as u64);
            }
        }
        if let Some(p) = pb_discover {
            p.finish_with_message(format!("Discovered {} compositions", det.nodes.len()));
        }

        let mut dwa = DWA::new();
        if det.nodes.is_empty() {
            return dwa;
        }
        dwa.states.0.resize(det.nodes.len(), Default::default());
        dwa.body.start_state = start_idx;

        for (i, node) in det.nodes.iter().enumerate() {
            dwa.states[i].final_weight = node.final_weight.clone();
            if let (Some(target_idx), Some(mask)) = (node.default_target_idx, &node.default_mask) {
                if !mask.is_empty() {
                    dwa.set_default_transition(i, target_idx, mask.clone()).unwrap();
                }
            }
            for (lbl, &target_idx) in &node.exception_targets {
                let mask = node.exception_masks.get(lbl).cloned().unwrap_or_else(Weight::zeros);
                dwa.add_transition(i, *lbl, target_idx, mask).unwrap();
            }
        }
        dwa
    }
}
