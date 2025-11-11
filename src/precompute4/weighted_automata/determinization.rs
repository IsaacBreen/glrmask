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
    by_sig: Vec<(usize, Weight)>, // macro_signature_id -> weight
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
    exception_transitions: BTreeMap<Label, Vec<usize>>, // label -> step_ids
}

#[derive(Clone, Hash, Eq, PartialEq)]
struct MacroSigKey {
    final_weight_fp: u64,
    default_transitions: Vec<(usize, Vec<Label>)>,
    exception_transitions: Vec<(Label, Vec<usize>)>,
}

struct Determinizer<'a> {
    nwa: &'a NWA,
    // Precomputations
    future_weights: Vec<Weight>,
    eps_cache: Vec<Vec<(NWAStateID, Weight)>>, // ε-closure masked by future weights
    step_pool: StepPool,
    signatures: Vec<MacroSig>,      // per NWA state -> macro signature
    state_to_sig_id: Vec<usize>,    // NWA state -> macro signature id
    compiled_steps: Vec<CompiledStep>, // interned step -> macro-signature space

    // Canonical subset-construction: unique-table of determinized states
    // Key: fingerprint -> collision bucket of canonical gate maps
    // Canonical gate map is a Vec<(macro_sig_id, Weight)> sorted by macro_sig_id
    state_cache: HashMap<u64, Vec<(Vec<(usize, Weight)>, usize)>>,

    // Discovered states (in order of id)
    nodes: Vec<NodeData>,
}

#[derive(Clone, Default)]
struct NodeData {
    // Canonical "gate map" defining this determinized state
    gates: HashMap<usize, Weight>, // macro_sig_id -> weight

    // Outgoing edges
    default_edge: Option<(usize, Weight)>,                 // (to_id, mask)
    exception_edges: BTreeMap<Label, (usize, Weight)>,     // label -> (to_id, mask)

    // Final weight
    final_weight: Option<Weight>,
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
            state_cache: HashMap::new(),
            nodes: Vec::new(),
        }
    }

    fn run(&mut self) -> DWA {
        self.compute_future_weights();
        self.precompute_eps_closures();
        self.build_macro_signatures();
        self.compile_steps();
        self.discover_states_bfs();
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
        let pb = Self::progress_bar(n as u64, "ε-closures");

        let mut scratch = vec![Weight::zeros(); n];
        let mut q = VecDeque::new();
        let mut touched = Vec::new();

        for s in 0..n {
            self.eps_cache[s] = self.eps_closure(s, &mut scratch, &mut q, &mut touched);
            if let Some(p) = &pb {
                p.inc(1);
            }
        }
        if let Some(p) = pb {
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
            if base.is_empty() {
                continue;
            }
            for &(v, ref w) in &self.nwa.states[u].epsilons {
                if v >= self.nwa.states.len() {
                    continue;
                }
                let mut prop = &base & w;
                if prop.is_empty() {
                    continue;
                }
                prop &= &self.future_weights[v];
                if prop.is_empty() {
                    continue;
                }
                if !prop.is_subset_of(&scratch[v]) {
                    if scratch[v].is_empty() {
                        touched.push(v);
                    }
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
        let pb = Self::progress_bar(n as u64, "Macro signatures");
        let mut interner: HashMap<MacroSigKey, usize> = HashMap::new();

        for s in 0..n {
            let (sig, key) = self.build_one_macro_sig(s);
            let sig_id = *interner.entry(key).or_insert_with(|| {
                let id = self.signatures.len();
                self.signatures.push(sig);
                id
            });
            self.state_to_sig_id[s] = sig_id;
            if let Some(p) = &pb {
                p.inc(1);
            }
        }
        if let Some(p) = pb {
            p.finish_with_message("Macro signatures done");
        }
    }

    fn build_one_macro_sig(&mut self, s: NWAStateID) -> (MacroSig, MacroSigKey) {
        // Final weight = union over ε-closure to final states.
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
            if d.target >= self.nwa.states.len() {
                continue;
            }
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
                if *to >= self.nwa.states.len() {
                    continue;
                }
                let pairs = Self::apply_weight_to_pairs(&self.eps_cache[*to], w);
                if !pairs.is_empty() {
                    ids.push(self.step_pool.intern(pairs));
                }
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
        let pb = Self::progress_bar(m as u64, "Compile steps");
        self.compiled_steps = Vec::with_capacity(m);

        for pairs in &self.step_pool.raw {
            let mut acc: HashMap<usize, Weight> = HashMap::new(); // macro_sig_id -> weight
            for (t, w) in pairs.iter() {
                *acc.entry(self.state_to_sig_id[*t]).or_default() |= w;
            }
            let mut by_sig: Vec<(usize, Weight)> = acc.into_iter().collect();
            by_sig.sort_by_key(|(sid, _)| *sid);
            self.compiled_steps.push(CompiledStep { by_sig });
            if let Some(p) = &pb {
                p.inc(1);
            }
        }
        if let Some(p) = pb {
            p.finish_with_message("Compile steps done");
        }
    }

    // Step 5: canonical subset construction via BFS with gate-map interning.
    fn discover_states_bfs(&mut self) {
        // Initial gate map from ε-closure of NWA start.
        let mut init: HashMap<usize, Weight> = HashMap::new(); // macro_sig_id -> weight
        for (t, w) in &self.eps_cache[self.nwa.body.start_state] {
            *init.entry(self.state_to_sig_id[*t]).or_default() |= w;
        }

        let pb = Self::progress_bar(0, "Discovering states (subset-construction)");

        let start_id = self.get_or_create_state(init);
        let mut q = VecDeque::new();
        q.push_back(start_id);

        let mut processed = Vec::<bool>::new();
        processed.resize(self.nodes.len(), false);

        while let Some(idx) = q.pop_front() {
            if idx >= processed.len() {
                processed.resize(idx + 1, false);
            }
            if processed[idx] {
                continue;
            }

            if let Some(p) = &pb {
                p.inc(1);
                p.set_length(self.nodes.len() as u64);
            }

            // Compute outgoing transitions from current gate map.
            let target_maps = self.compute_target_maps(&self.nodes[idx].gates);

            // Resolve default
            if let Some(def_map) = target_maps.get(&None) {
                let def_mask = Self::union_mask(def_map);
                if !def_mask.is_empty() {
                    let to_id = self.get_or_create_state(def_map.clone());
                    self.nodes[idx].default_edge = Some((to_id, def_mask.clone()));
                    if !processed.get(to_id).copied().unwrap_or(false) {
                        q.push_back(to_id);
                    }
                }
            }

            // Resolve exceptions
            for (lbl, ex_map) in target_maps.iter().filter_map(|(k, v)| k.map(|l| (l, v))) {
                let ex_mask = Self::union_mask(ex_map);
                if ex_mask.is_empty() {
                    continue;
                }
                let to_id = self.get_or_create_state(ex_map.clone());
                self.nodes[idx].exception_edges.insert(lbl, (to_id, ex_mask.clone()));
                if !processed.get(to_id).copied().unwrap_or(false) {
                    q.push_back(to_id);
                }
            }

            // Compute final weight for this node
            let mut final_w = Weight::zeros();
            for (sig_id, gate) in &self.nodes[idx].gates {
                if let Some(fw) = &self.signatures[*sig_id].final_weight {
                    final_w |= &(gate & fw);
                }
            }
            if !final_w.is_empty() {
                self.nodes[idx].final_weight = Some(final_w);
            }

            processed[idx] = true;
        }

        if let Some(p) = pb {
            p.finish_with_message(format!("Discovered {} DWA states", self.nodes.len()));
        }
    }

    // Compute outgoing transitions for a set of gates, grouped by label (None = default).
    fn compute_target_maps(&self, gates: &HashMap<usize, Weight>) -> BTreeMap<Option<Label>, HashMap<usize, Weight>> {
        let mut all_defaults: HashMap<usize, Weight> = HashMap::new();                 // step_id -> gate union
        let mut all_exceptions: BTreeMap<Label, HashMap<usize, Weight>> = BTreeMap::new(); // label -> step_id -> gate union
        let mut overridden: BTreeMap<Label, HashMap<usize, Weight>> = BTreeMap::new(); // label -> step_id -> gate union
        let mut excepted: BTreeMap<Label, HashMap<usize, Weight>> = BTreeMap::new();   // label -> step_id -> gate union

        for (&sig_id, gate) in gates {
            if gate.is_empty() {
                continue;
            }
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
        let mut def_map: HashMap<usize, Weight> = HashMap::new(); // macro_sig_id -> weight
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

            if let Some((t, m)) = &n.default_edge {
                if !m.is_empty() {
                    dwa.set_default_transition(i, *t, m.clone()).unwrap();
                }
            }
            for (lbl, (t, m)) in &n.exception_edges {
                if !m.is_empty() {
                    dwa.add_transition(i, *lbl, *t, m.clone()).unwrap();
                }
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

    fn union_mask(map: &HashMap<usize, Weight>) -> Weight {
        map.values().fold(Weight::zeros(), |mut a, b| {
            a |= b;
            a
        })
    }

    // Unique-table for determinized states (gate maps).
    // Returns existing id or creates a new one, interning the canonical gate map.
    fn get_or_create_state(&mut self, gates: HashMap<usize, Weight>) -> usize {
        let pairs = Self::canonicalize_gates(&gates);
        let fp = Self::fingerprint_pairs(&pairs);

        if let Some(bucket) = self.state_cache.get(&fp) {
            for (existing_pairs, id) in bucket {
                if Self::pairs_equal(existing_pairs, &pairs) {
                    return *id;
                }
            }
        }

        let id = self.nodes.len();
        let node = NodeData {
            gates,
            default_edge: None,
            exception_edges: BTreeMap::new(),
            final_weight: None,
        };
        self.nodes.push(node);

        self.state_cache.entry(fp).or_default().push((pairs, id));
        id
    }

    fn canonicalize_gates(gates: &HashMap<usize, Weight>) -> Vec<(usize, Weight)> {
        let mut v: Vec<(usize, Weight)> = gates
            .iter()
            .filter_map(|(k, w)| if w.is_empty() { None } else { Some((*k, w.clone())) })
            .collect();
        v.sort_by_key(|(k, _)| *k);
        v
    }

    fn fingerprint_pairs(pairs: &[(usize, Weight)]) -> u64 {
        pairs.iter().fold(FP_ZERO, |fp, (sid, w)| {
            mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2))
        })
    }

    fn pairs_equal(a: &[(usize, Weight)], b: &[(usize, Weight)]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for ((sa, wa), (sb, wb)) in a.iter().zip(b.iter()) {
            if sa != sb || wa != wb {
                return false;
            }
        }
        true
    }

    fn progress_bar(len: u64, label: &str) -> Option<ProgressBar> {
        if !PROGRESS_BAR_ENABLED {
            return None;
        }
        let style = ProgressStyle::default_bar()
            .template(&format!("{{spinner:.green}} [Determinize: {{elapsed_precise}}] [{{wide_bar:.cyan/blue}}] {{pos}}/{{len}} ({})", label))
            .unwrap();
        Some(ProgressBar::new(len).with_style(style))
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
