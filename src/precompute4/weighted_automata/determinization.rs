use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::NWA;
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::r#macro::is_debug_level_enabled;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::Instant;

type Label = i16;

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

        let mut det = Determinizer::new(&nwa);
        let dwa = det.run();

        if is_debug_level_enabled(5) {
            eprintln!("NWA::determinize_to_dwa result DWA stats:\n{}", dwa.stats());
            eprintln!("NWA::determinize_to_dwa took: {:?}", now.elapsed());
        }
        dwa
    }
}

// Interned step = a sequence of (NWA state, weight) pairs.
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
    exceptions: BTreeSet<Label>,
}

#[derive(Clone)]
struct MacroSig {
    final_weight: Option<Weight>,
    default_transitions: Vec<DefSig>,
    exception_transitions: BTreeMap<Label, Vec<usize>>,
}

#[derive(Clone, Hash, Eq, PartialEq)]
struct MacroSigKey {
    final_weight_fp: u64,
    default_transitions: Vec<(usize, Vec<Label>)>,
    exception_transitions: Vec<(Label, Vec<usize>)>,
}

// Outgoing summary restricted to a weight slice.
// Maps: None => default, Some(label) => exceptions.
type OutMap = HashMap<usize, Weight>;
#[derive(Clone, Default)]
struct OutSummary {
    def: OutMap,
    ex: BTreeMap<Label, OutMap>,
}
impl OutSummary {
    fn is_empty(&self) -> bool {
        self.def.is_empty() && self.ex.values().all(|m| m.is_empty())
    }
    fn union_assign(&mut self, other: &OutSummary) {
        for (k, w) in &other.def {
            let e = self.def.entry(*k).or_default();
            *e |= w;
        }
        for (lbl, m) in &other.ex {
            let dst = self.ex.entry(*lbl).or_default();
            for (k, w) in m {
                let e = dst.entry(*k).or_default();
                *e |= w;
            }
        }
    }
}
impl PartialEq for OutSummary {
    fn eq(&self, other: &Self) -> bool {
        if self.def.len() != other.def.len() || self.ex.len() != other.ex.len() {
            return false;
        }
        if self.def != other.def {
            return false;
        }
        if self.ex.len() != other.ex.len() {
            return false;
        }
        for (lbl, m) in &self.ex {
            if let Some(m2) = other.ex.get(lbl) {
                if m != m2 {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}
impl Eq for OutSummary {}

// A DWA state builder: a composition of macro signatures reachable under gate weights.
struct DWAStateBuilder {
    final_weight: Option<Weight>,
    default_target_idx: Option<usize>,
    default_mask: Option<Weight>,
    exception_targets: BTreeMap<Label, usize>,
    exception_masks: BTreeMap<Label, Weight>,
    gates: HashMap<usize, Weight>, // macro_sig_id -> gate mask
    incoming_weight_union: Weight, // union of incoming masks (for merge heuristic)
    macro_keys: BTreeSet<usize>,   // keys(gates)
    atom_covered: BTreeSet<usize>, // indices of atoms covered in the AtomIndex
    out_cache_by_atom: HashMap<usize, OutSummary>, // per-atom cache of outgoing summary
}

// Atom index over disjoint atoms of the weight domain.
// Every mask is expressed as union of some atoms after refinement.
struct AtomIndex {
    atoms: Vec<Weight>,
    nodes_in_atom: Vec<BTreeSet<usize>>, // atoms[i] -> set of node indices
}
impl AtomIndex {
    fn new() -> Self {
        Self { atoms: Vec::new(), nodes_in_atom: Vec::new() }
    }

    // Refine atoms so that 'w' is a union of atoms; return indices of atoms contained in 'w'.
    fn refine_and_covering_atoms(&mut self, w: &Weight) -> Vec<usize> {
        if w.is_empty() {
            return Vec::new();
        }
        // If empty universe: just add w as the first atom.
        if self.atoms.is_empty() {
            self.atoms.push(w.clone());
            self.nodes_in_atom.push(BTreeSet::new());
            return vec![0];
        }

        let mut covering = Vec::new();
        let mut pending_new_atoms: Vec<(usize, Weight)> = Vec::new(); // split results: (copy_from_atom_idx, new_atom_mask)
        let mut w_remainder = w.clone();

        for i in 0..self.atoms.len() {
            let ai = &self.atoms[i];
            let i1 = ai & w; // overlap
            if i1.is_empty() {
                continue;
            }
            covering.push(i);
            // Split atom if needed: ai becomes i1, create i2 = ai - w
            let i2 = ai - &w;
            if !i2.is_empty() {
                // ai becomes i1
                self.atoms[i] = i1;
                // new atom spawned, copy node set later
                pending_new_atoms.push((i, i2));
            }
            // Remove covered portion from w_remainder (optional, but keeps new residual small)
            w_remainder -= &self.atoms[i];
        }

        // Add residual part (disjoint from all existing atoms)
        if !w_remainder.is_empty() {
            self.atoms.push(w_remainder);
            self.nodes_in_atom.push(BTreeSet::new());
            covering.push(self.atoms.len() - 1);
        }

        // Materialize splits; copy node sets for new atoms.
        for (src_idx, mask) in pending_new_atoms {
            self.atoms.push(mask);
            let mut copy = self.nodes_in_atom[src_idx].clone();
            self.nodes_in_atom.push(std::mem::take(&mut copy)); // move copy, leave src unchanged
            covering.push(self.atoms.len() - 1);
        }

        covering.sort_unstable();
        covering.dedup();
        covering
    }

    fn register_node_on_atoms(&mut self, node_idx: usize, atom_indices: &[usize]) {
        for &ai in atom_indices {
            self.nodes_in_atom[ai].insert(node_idx);
        }
    }

    // All nodes that have any overlap with w (exact superset, no false negatives).
    fn query_nodes_overlapping(&mut self, w: &Weight) -> BTreeSet<usize> {
        let atoms = self.refine_and_covering_atoms(w);
        let mut out: BTreeSet<usize> = BTreeSet::new();
        for ai in atoms {
            out.extend(self.nodes_in_atom[ai].iter().copied());
        }
        out
    }
}

struct Determinizer<'a> {
    nwa: &'a NWA,
    future_weights: Vec<Weight>,
    eps_cache: Vec<Vec<(NWAStateID, Weight)>>,
    step_pool: StepPool,
    signatures: Vec<MacroSig>,
    state_to_sig_id: Vec<usize>,
    compiled_steps: Vec<CompiledStep>,

    // Composition nodes
    nodes: Vec<DWAStateBuilder>,
    work_queue: VecDeque<usize>,
    in_queue: Vec<bool>,

    // New indices for faster merging
    atom_index: AtomIndex,                            // weight-space index
    size_index: BTreeMap<usize, BTreeSet<usize>>,     // gates.len() -> node IDs
    pb: Option<ProgressBar>,
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
            atom_index: AtomIndex::new(),
            size_index: BTreeMap::new(),
            pb: None,
        }
    }

    fn run(&mut self) -> DWA {
        self.compute_future_weights();
        self.precompute_eps_closures();
        self.build_macro_signatures();
        self.compile_steps();
        self.discover_composition_nodes();
        self.build_dwa()
    }

    // Step 1: backward propagate final weights through all edges to get "future weights".
    fn compute_future_weights(&mut self) {
        let n = self.nwa.states.len();
        let mut fut = vec![Weight::zeros(); n];
        let mut rev: Vec<Vec<(NWAStateID, &Weight)>> = vec![vec![]; n];

        for p in 0..n {
            for &(t, ref w) in &self.nwa.states[p].epsilons {
                if t < n {
                    rev[t].push((p, w));
                }
            }
            for (_, targets) in &self.nwa.states[p].transitions {
                for (t, w) in targets {
                    if *t < n {
                        rev[*t].push((p, w));
                    }
                }
            }
            for def in &self.nwa.states[p].default {
                if def.target < n {
                    rev[def.target].push((p, &def.weight));
                }
            }
        }

        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        for s in 0..n {
            if let Some(fw) = &self.nwa.states[s].final_weight {
                if !fw.is_empty() {
                    fut[s] = fw.clone();
                    q.push_back(s);
                }
            }
        }

        while let Some(v) = q.pop_front() {
            let fv = fut[v].clone();
            if fv.is_empty() {
                continue;
            }
            for &(p, w_pv) in &rev[v] {
                let add = &fv & w_pv;
                if !add.is_empty() && !add.is_subset_of(&fut[p]) {
                    fut[p] |= &add;
                    q.push_back(p);
                }
            }
        }
        self.future_weights = fut;
    }

    // Step 2: ε-closure per state masked by future weights.
    fn precompute_eps_closures(&mut self) {
        let n = self.nwa.states.len();
        self.init_progress_bar(n as u64, "ε-closures");

        let mut scratch = vec![Weight::zeros(); n];
        let mut q = VecDeque::new();
        let mut touched = Vec::new();

        for s in 0..n {
            self.eps_cache[s] = self.eps_closure(s, &mut scratch, &mut q, &mut touched);
            if let Some(p) = &self.pb {
                p.inc(1);
            }
        }
        if let Some(p) = &self.pb {
            p.finish_with_message("ε-closures done");
        }
    }

    fn eps_closure(
        &self,
        s: NWAStateID,
        scratch: &mut [Weight],
        q: &mut VecDeque<NWAStateID>,
        touched: &mut Vec<NWAStateID>,
    ) -> Vec<(NWAStateID, Weight)> {
        if s >= self.nwa.states.len() || self.future_weights[s].is_empty() {
            return Vec::new();
        }
        scratch[s] = self.future_weights[s].clone();
        touched.push(s);
        q.push_back(s);

        while let Some(u) = q.pop_front() {
            let base = scratch[u].clone();
            if base.is_empty() { continue; }
            for &(v, ref w) in &self.nwa.states[u].epsilons {
                if v >= self.nwa.states.len() { continue; }
                let mut prop = &base & w;
                if prop.is_empty() { continue; }
                prop &= &self.future_weights[v];
                if prop.is_empty() { continue; }
                if !prop.is_subset_of(&scratch[v]) {
                    if scratch[v].is_empty() { touched.push(v); }
                    scratch[v] |= &prop;
                    q.push_back(v);
                }
            }
        }

        let mut out = Vec::with_capacity(touched.len());
        for &i in touched.iter() {
            out.push((i, scratch[i].clone()));
            scratch[i] = Weight::zeros();
        }
        touched.clear();
        out.sort_by_key(|(sid, _)| *sid);
        out
    }

    // Step 3: build macro signatures (behavior after ε-closure).
    fn build_macro_signatures(&mut self) {
        let n = self.nwa.states.len();
        self.init_progress_bar(n as u64, "Macro signatures");
        let mut interner: HashMap<MacroSigKey, usize> = HashMap::new();

        for s in 0..n {
            let (sig, key) = self.build_one_macro_sig(s);
            let sig_id = *interner.entry(key).or_insert_with(|| {
                let id = self.signatures.len();
                self.signatures.push(sig);
                id
            });
            self.state_to_sig_id[s] = sig_id;
            if let Some(p) = &self.pb { p.inc(1); }
        }
        if let Some(p) = &self.pb { p.finish_with_message("Macro signatures done"); }
    }

    fn build_one_macro_sig(&mut self, s: NWAStateID) -> (MacroSig, MacroSigKey) {
        let final_acc = self
            .eps_cache[s]
            .iter()
            .fold(Weight::zeros(), |mut acc, (t, w)| {
                if let Some(fw) = &self.nwa.states[*t].final_weight {
                    acc |= &(w & fw);
                }
                acc
            });
        let final_weight = if final_acc.is_empty() { None } else { Some(final_acc) };

        // Defaults.
        let mut defaults: Vec<DefSig> = Vec::new();
        for d in &self.nwa.states[s].default {
            if d.target >= self.nwa.states.len() { continue; }
            let pairs = Self::apply_weight_to_pairs(&self.eps_cache[d.target], &d.weight);
            if !pairs.is_empty() {
                defaults.push(DefSig {
                    step_id: self.step_pool.intern(pairs),
                    exceptions: d.exceptions.clone(),
                });
            }
        }

        // Exceptions.
        let mut exceptions: BTreeMap<Label, Vec<usize>> = BTreeMap::new();
        for (lbl, targets) in &self.nwa.states[s].transitions {
            let mut ids: Vec<usize> = Vec::new();
            for (to, w) in targets {
                if *to >= self.nwa.states.len() { continue; }
                let pairs = Self::apply_weight_to_pairs(&self.eps_cache[*to], w);
                if !pairs.is_empty() { ids.push(self.step_pool.intern(pairs)); }
            }
            if !ids.is_empty() {
                ids.sort_unstable();
                let mut def_ids: Vec<_> = defaults.iter().map(|d| d.step_id).collect();
                def_ids.sort_unstable();
                if ids != def_ids {
                    exceptions.insert(*lbl, ids);
                }
            }
        }

        let key = Self::make_sig_key(&final_weight, &defaults, &exceptions);
        (MacroSig { final_weight, default_transitions: defaults, exception_transitions: exceptions }, key)
    }

    // Step 4: compile steps to macro-signature space.
    fn compile_steps(&mut self) {
        let m = self.step_pool.raw.len();
        self.init_progress_bar(m as u64, "Compile steps");
        self.compiled_steps = Vec::with_capacity(m);

        for pairs in &self.step_pool.raw {
            let mut acc: HashMap<usize, Weight> = HashMap::new();
            for (t, w) in pairs.iter() {
                *acc.entry(self.state_to_sig_id[*t]).or_default() |= w;
            }
            let mut by_sig: Vec<(usize, Weight)> = acc.into_iter().collect();
            by_sig.sort_by_key(|(sid, _)| *sid);
            self.compiled_steps.push(CompiledStep { by_sig });
            if let Some(p) = &self.pb { p.inc(1); }
        }
        if let Some(p) = &self.pb { p.finish_with_message("Compile steps done"); }
    }

    // Step 5: subset construction over macro signatures (with merge heuristics).
    fn discover_composition_nodes(&mut self) {
        self.init_progress_bar(0, "Discovering states");

        let mut init: HashMap<usize, Weight> = HashMap::new();
        for (t, w) in &self.eps_cache[self.nwa.body.start_state] {
            *init.entry(self.state_to_sig_id[*t]).or_default() |= w;
        }
        self.add_dwa_state_builder(init, Weight::all());

        while let Some(idx) = self.work_queue.pop_front() {
            self.in_queue[idx] = false;
            if let Some(p) = &self.pb { p.inc(1); }
            self.process_state(idx);
            if let Some(p) = &self.pb { p.set_length(self.nodes.len() as u64); }
        }
        if let Some(p) = &self.pb {
            p.finish_with_message(format!("Discovered {} DWA states", self.nodes.len()));
        }
    }

    fn process_state(&mut self, idx: usize) {
        if let Some(p) = self.pb.as_ref() {
            p.set_message(format!("state {}: compute_target_maps", idx));
        }
        let gates = self.nodes[idx].gates.clone();
        let target_maps = self.compute_target_maps(&gates);

        let mut resolved = BTreeMap::new();
        let num_target_maps = target_maps.len();
        for (i, (label, map)) in target_maps.into_iter().enumerate() {
            if let Some(p) = self.pb.as_ref() {
                p.set_message(format!("state {}: find_or_create_target_node {}/{}", idx, i + 1, num_target_maps));
            }
            let mask = map.values().fold(Weight::zeros(), |mut a, b| { a |= b; a });
            if mask.is_empty() {
                if label.is_some() { resolved.insert(label, (idx, Weight::zeros())); }
                continue;
            }
            let target_idx = self.find_or_create_target_node(&map);
            if self.propagate_weights_to_node(target_idx, &map) {
                self.enqueue_node(target_idx);
            }
            resolved.insert(label, (target_idx, mask));
        }

        if let Some(p) = self.pb.as_ref() {
            p.set_message(format!("state {}: resolving", idx));
        }
        let node = &mut self.nodes[idx];
        if let Some((t, m)) = resolved.get(&None).cloned() {
            node.default_target_idx = Some(t);
            node.default_mask = Some(m);
        }
        for (lbl, (t, m)) in resolved.into_iter().filter_map(|(k, v)| k.map(|l| (l, v))) {
            node.exception_targets.insert(lbl, t);
            node.exception_masks.insert(lbl, m);
        }

        if let Some(p) = self.pb.as_ref() {
            p.set_message(format!("state {}: final_weight", idx));
        }
        node.final_weight = Some(gates.iter().fold(Weight::zeros(), |mut acc, (sig_id, gate)| {
            if let Some(fw) = &self.signatures[*sig_id].final_weight {
                acc |= &(gate & fw);
            }
            acc
        }));
        if let Some(p) = self.pb.as_ref() {
            p.set_message("");
        }
    }

    // Compute outgoing transitions for a set of gates, grouped by label (None = default).
    fn compute_target_maps(&self, gates: &HashMap<usize, Weight>) -> BTreeMap<Option<Label>, HashMap<usize, Weight>> {
        let mut all_defaults: HashMap<usize, Weight> = HashMap::new();
        let mut all_exceptions: BTreeMap<Label, HashMap<usize, Weight>> = BTreeMap::new();
        let mut overridden: BTreeMap<Label, HashMap<usize, Weight>> = BTreeMap::new();
        let mut excepted: BTreeMap<Label, HashMap<usize, Weight>> = BTreeMap::new();

        for (&sig_id, gate) in gates {
            if gate.is_empty() { continue; }
            let sig = &self.signatures[sig_id];

            for def in &sig.default_transitions {
                *all_defaults.entry(def.step_id).or_default() |= gate;
                for &lbl in &def.exceptions {
                    *excepted.entry(lbl).or_default().entry(def.step_id).or_default() |= gate;
                }
            }

            for (lbl, ex_steps) in &sig.exception_transitions {
                for &ex_step in ex_steps {
                    *all_exceptions.entry(*lbl).or_default().entry(ex_step).or_default() |= gate;
                }
                for def in &sig.default_transitions {
                    *overridden.entry(*lbl).or_default().entry(def.step_id).or_default() |= gate;
                }
            }
        }

        let mut target_maps = BTreeMap::new();

        // Default map = union of all default steps.
        let mut def_map = HashMap::new();
        for (step_id, gate) in &all_defaults {
            self.accumulate_step_targets(&mut def_map, *step_id, gate);
        }
        if !def_map.is_empty() {
            target_maps.insert(None, def_map.clone());
        }

        // Exception maps per label: defaults minus overrides/excepts + explicit exceptions.
        let labels: BTreeSet<Label> = all_exceptions.keys().copied().chain(excepted.keys().copied()).collect();
        for lbl in labels {
            let mut map = HashMap::new();

            for (step_id, total_gate) in &all_defaults {
                let mut g = total_gate.clone();
                if let Some(x) = overridden.get(&lbl).and_then(|m| m.get(step_id)) {
                    g -= x;
                }
                if let Some(x) = excepted.get(&lbl).and_then(|m| m.get(step_id)) {
                    g -= x;
                }
                if !g.is_empty() {
                    self.accumulate_step_targets(&mut map, *step_id, &g);
                }
            }

            if let Some(ex_steps) = all_exceptions.get(&lbl) {
                for (step_id, g) in ex_steps {
                    self.accumulate_step_targets(&mut map, *step_id, g);
                }
            }

            if map != def_map {
                target_maps.insert(Some(lbl), map);
            }
        }

        target_maps
    }

    // Fast target-node selection with atom index + per-atom caching.
    fn find_or_create_target_node(&mut self, map: &HashMap<usize, Weight>) -> usize {
        // Compute incoming mask once
        let incoming = map.values().fold(Weight::zeros(), |mut a, b| { a |= b; a });

        // Overlap candidates via AtomIndex (exact, sublinear in #nodes).
        let overlap_candidates = self.atom_index.query_nodes_overlapping(&incoming);

        // Precompute per-atom OutSummary for "map" across atoms of incoming once.
        let overlap_atoms = self.atom_index.refine_and_covering_atoms(&incoming);
        let mut out_for_map_by_atom: HashMap<usize, OutSummary> = HashMap::new();
        for &ai in &overlap_atoms {
            let atom_mask = &self.atom_index.atoms[ai];
            let mut filtered: HashMap<usize, Weight> = HashMap::new();
            for (k, g) in map {
                let x = g & atom_mask;
                if !x.is_empty() { filtered.insert(*k, x); }
            }
            out_for_map_by_atom.insert(ai, self.compute_summary_for_gates(&filtered));
        }

        // Search a best overlapped candidate (prefer these to keep structure tight).
        let mut best: Option<(usize, (usize, usize))> = None; // (idx, cost=(inc, gates_len))
        for idx in overlap_candidates {
            // Intersect atoms for candidate and incoming
            let inter_atoms: Vec<usize> = overlap_atoms.iter().copied()
                .filter(|a| self.nodes[idx].atom_covered.contains(a)).collect();
            if inter_atoms.is_empty() {
                // Shouldn't happen given overlap selection, but guard anyway.
                continue;
            }

            // Build OutSummary for candidate on inter_atoms via per-atom cache.
            let mut cand_out = OutSummary::default();
            for &ai in &inter_atoms {
                let s = self.node_out_for_atom(idx, ai).clone();
                cand_out.union_assign(&s);
            }
            if cand_out.is_empty() {
                // If the candidate has no outgoing behavior on overlap (degenerate), consider unequal only if map has.
                let mut map_out = OutSummary::default();
                for &ai in &inter_atoms {
                    if let Some(m) = out_for_map_by_atom.get(&ai) {
                        map_out.union_assign(m);
                    }
                }
                if !map_out.is_empty() { continue; }
            } else {
                // Build map's summary across the same inter atoms.
                let mut map_out = OutSummary::default();
                for &ai in &inter_atoms {
                    if let Some(m) = out_for_map_by_atom.get(&ai) {
                        map_out.union_assign(m);
                    }
                }
                if cand_out != map_out {
                    continue; // not mergeable on overlap
                }
            }

            // Mergeable: compute cost (inc = number of new macro keys to add).
            let inc = map.keys().filter(|k| !self.nodes[idx].macro_keys.contains(k)).count();
            let cost = (inc, self.nodes[idx].gates.len());
            if best.as_ref().map_or(true, |(_, c)| cost < *c) {
                best = Some((idx, cost));
            }
        }

        if let Some((idx, _)) = best {
            // Reuse overlapped candidate
            self.register_incoming_and_maybe_update_cache(idx, &incoming, map);
            return idx;
        }

        // No overlapped mergeable: choose a disjoint node from the smallest gates bucket.
        // Iterate through size buckets to find the first available disjoint node.
        for (_, ids) in &self.size_index {
            for &idx in ids {
                if (&self.nodes[idx].incoming_weight_union & &incoming).is_empty() {
                    // Found a disjoint node to reuse.
                    self.register_incoming_and_maybe_update_cache(idx, &incoming, map);
                    return idx;
                }
            }
        }

        // Else create a fresh node.
        self.add_dwa_state_builder(HashMap::new(), incoming)
    }

    // Per-atom outgoing summary for an existing node (cached).
    fn node_out_for_atom(&mut self, idx: usize, atom_idx: usize) -> &OutSummary {
        if !self.nodes[idx].out_cache_by_atom.contains_key(&atom_idx) {
            let atom_mask = self.atom_index.atoms[atom_idx].clone();
            let gates = self.nodes[idx].gates.clone();
            // Filter gates by atom
            let mut filtered: HashMap<usize, Weight> = HashMap::new();
            for (k, g) in &gates {
                let x = g & &atom_mask;
                if !x.is_empty() { filtered.insert(*k, x); }
            }
            let summary = self.compute_summary_for_gates(&filtered);
            self.nodes[idx].out_cache_by_atom.insert(atom_idx, summary);
        }
        self.nodes[idx].out_cache_by_atom.get(&atom_idx).unwrap()
    }

    // OutSummary for any gates set (helper).
    fn compute_summary_for_gates(&self, gates: &HashMap<usize, Weight>) -> OutSummary {
        let maps = self.compute_target_maps(gates);
        let mut out = OutSummary::default();
        for (k, m) in maps {
            match k {
                None => out.def = m,
                Some(lbl) => { out.ex.insert(lbl, m); }
            }
        }
        out
    }

    // Merge heuristic: avoid recomputing target maps for entire gates repeatedly by caching per-atom;
    // then register the chosen node's growth into indices.
    fn register_incoming_and_maybe_update_cache(&mut self, idx: usize, incoming: &Weight, map: &HashMap<usize, Weight>) {
        // Before modifying node, check whether macro keys grow to update size_index and invalidate per-atom cache.
        let old_size = self.nodes[idx].macro_keys.len();
        let mut added_macro_key = false;
        for k in map.keys() {
            if self.nodes[idx].macro_keys.insert(*k) {
                added_macro_key = true;
            }
        }
        if added_macro_key {
            if let Some(set) = self.size_index.get_mut(&old_size) {
                set.remove(&idx);
                if set.is_empty() { self.size_index.remove(&old_size); }
            }
        }

        // Update gates
        if self.propagate_weights_to_node(idx, map) {
            self.enqueue_node(idx);
        }

        // Grow bucket if size changed
        let new_size = self.nodes[idx].macro_keys.len();
        if added_macro_key {
            self.size_index.entry(new_size).or_default().insert(idx);
            // Invalidate per-atom cache since gates changed
            self.nodes[idx].out_cache_by_atom.clear();
        }

        // Update incoming coverage and AtomIndex
        let prior = self.nodes[idx].incoming_weight_union.clone();
        let delta = incoming - &prior;
        if !delta.is_empty() {
            self.nodes[idx].incoming_weight_union |= incoming;
            // Register coverage for new atoms
            let atoms = self.atom_index.refine_and_covering_atoms(&delta);
            self.atom_index.register_node_on_atoms(idx, &atoms);
            for a in atoms { self.nodes[idx].atom_covered.insert(a); }
        }
    }

    // Mergeability (original) retained for reference and for correctness in target computation equality via per-atom caches.
    // The per-atom approach ensures equality over the exact overlap region.

    fn add_dwa_state_builder(&mut self, gates: HashMap<usize, Weight>, incoming_weight_union: Weight) -> usize {
        let idx = self.nodes.len();
        let macro_keys: BTreeSet<usize> = gates.keys().copied().collect();
        let mut node = DWAStateBuilder {
            final_weight: None,
            default_target_idx: None,
            default_mask: None,
            exception_targets: BTreeMap::new(),
            exception_masks: BTreeMap::new(),
            gates,
            incoming_weight_union: incoming_weight_union.clone(),
            macro_keys,
            atom_covered: BTreeSet::new(),
            out_cache_by_atom: HashMap::new(),
        };

        // Index into AtomIndex if incoming is non-empty.
        if !incoming_weight_union.is_empty() {
            let atoms = self.atom_index.refine_and_covering_atoms(&incoming_weight_union);
            self.atom_index.register_node_on_atoms(idx, &atoms);
            for a in atoms { node.atom_covered.insert(a); }
        }

        // Size bucket
        self.size_index.entry(node.macro_keys.len()).or_default().insert(idx);

        self.nodes.push(node);
        self.enqueue_node(idx);
        idx
    }

    fn enqueue_node(&mut self, idx: usize) {
        if idx >= self.in_queue.len() { self.in_queue.resize(idx + 1, false); }
        if !self.in_queue[idx] {
            self.in_queue[idx] = true;
            self.work_queue.push_back(idx);
        }
    }

    fn propagate_weights_to_node(&mut self, node_idx: usize, weights: &HashMap<usize, Weight>) -> bool {
        let mut changed = false;
        for (sig_id, w) in weights {
            let entry = self.nodes[node_idx].gates.entry(*sig_id).or_default();
            if !w.is_subset_of(entry) {
                *entry |= w;
                changed = true;
            }
        }
        changed
    }

    // Final assembly.
    fn build_dwa(&self) -> DWA {
        let mut dwa = DWA::new();
        if self.nodes.is_empty() {
            return dwa;
        }
        dwa.states.0.resize(self.nodes.len(), Default::default());
        dwa.body.start_state = 0;

        for (i, n) in self.nodes.iter().enumerate() {
            dwa.states[i].final_weight = n.final_weight.clone();

            if let (Some(t), Some(m)) = (n.default_target_idx, &n.default_mask) {
                if !m.is_empty() {
                    dwa.set_default_transition(i, t, m.clone()).unwrap();
                }
            }
            for (lbl, &t) in &n.exception_targets {
                let m = n.exception_masks.get(lbl).cloned().unwrap_or_else(Weight::zeros);
                dwa.add_transition(i, *lbl, t, m).unwrap();
            }
        }
        dwa
    }

    // Helpers

    fn make_sig_key(
        final_weight: &Option<Weight>,
        defs: &[DefSig],
        exs: &BTreeMap<Label, Vec<usize>>,
    ) -> MacroSigKey {
        let mut defs_key: Vec<_> = defs
            .iter()
            .map(|d| (d.step_id, d.exceptions.iter().copied().collect::<Vec<_>>()))
            .collect();
        defs_key.sort_unstable();

        MacroSigKey {
            final_weight_fp: final_weight.as_ref().map(|w| w.fp).unwrap_or(FP_ZERO),
            default_transitions: defs_key,
            exception_transitions: exs.iter().map(|(k, v)| (*k, v.clone())).collect(),
        }
    }

    fn apply_weight_to_pairs(base: &[(NWAStateID, Weight)], w: &Weight) -> Vec<(NWAStateID, Weight)> {
        if w.is_all_fast() {
            return base.to_vec();
        }
        base.iter()
            .map(|(sid, wt)| (*sid, wt & w))
            .filter(|(_, x)| !x.is_empty())
            .collect()
    }

    fn accumulate_step_targets(&self, dst: &mut HashMap<usize, Weight>, step_id: usize, gate: &Weight) {
        for (sid, w) in &self.compiled_steps[step_id].by_sig {
            let x = w & gate;
            if !x.is_empty() {
                *dst.entry(*sid).or_default() |= &x;
            }
        }
    }

    fn init_progress_bar(&mut self, len: u64, label: &str) {
        if !PROGRESS_BAR_ENABLED {
            self.pb = None;
            return;
        }
        let style = ProgressStyle::default_bar()
            .template(&format!("{{spinner:.green}} [Determinize: {{elapsed_precise}}] [{{wide_bar:.cyan/blue}}] {{pos}}/{{len}} ({}) {{msg}}", label))
            .unwrap();
        self.pb = Some(ProgressBar::new(len).with_style(style));
    }
}

impl StepPool {
    fn new() -> Self {
        Self { raw: Vec::new(), map: HashMap::new() }
    }

    fn fingerprint(pairs: &[(NWAStateID, Weight)]) -> u64 {
        pairs.iter().fold(FP_ZERO, |fp, (sid, w)| {
            mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2))
        })
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
