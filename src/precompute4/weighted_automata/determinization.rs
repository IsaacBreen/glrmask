use super::bitset::{mix3, FP_K1, FP_K2, FP_ZERO};
use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWA, NWADefaultTransition};
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::r#macro::is_debug_level_enabled;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::Instant;

type Label = i16;

// Public determinization entry point.
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

// A default step: closure(target) masked by default weight; carries the default's exception set.
#[derive(Clone)]
struct DefaultStep {
    exceptions: BTreeSet<Label>,
    pairs: Vec<(NWAStateID, Weight)>,
}

// Per-NWA-state precomputation: labeled steps and default steps (all closure-applied).
#[derive(Clone, Default)]
struct StatePrecomp {
    // For label l: union over all edges (s -l,w-> t) of (closure(t) & w)
    labeled: BTreeMap<Label, Vec<(NWAStateID, Weight)>>,
    // For defaults: for each default d, closure(d.target) & d.weight with its exception set.
    defaults: Vec<DefaultStep>,
    // All labels that may differ from default: labeled keys ∪ all default exception labels
    labels_of_interest: BTreeSet<Label>,
}

// A simple interner for determinized states represented as canonical vectors (sid, mask).
struct NodeInterner {
    nodes: Vec<Vec<(NWAStateID, Weight)>>,
    map: HashMap<u64, Vec<usize>>,
}
impl NodeInterner {
    fn new() -> Self {
        Self { nodes: Vec::new(), map: HashMap::new() }
    }
    fn fingerprint(pairs: &[(NWAStateID, Weight)]) -> u64 {
        pairs.iter().fold(FP_ZERO, |fp, (sid, w)| {
            mix3(fp, (*sid as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2))
        })
    }
    // Intern canonical vector; return (id, is_new)
    fn intern(&mut self, pairs: Vec<(NWAStateID, Weight)>) -> (usize, bool) {
        let fp = Self::fingerprint(&pairs);
        if let Some(ids) = self.map.get(&fp) {
            for &i in ids {
                if self.nodes[i] == pairs {
                    return (i, false);
                }
            }
        }
        let id = self.nodes.len();
        self.nodes.push(pairs);
        self.map.entry(fp).or_default().push(id);
        (id, true)
    }
    fn get(&self, id: usize) -> &[(NWAStateID, Weight)] {
        &self.nodes[id]
    }
    fn len(&self) -> usize { self.nodes.len() }
}

// Main determinizer (concise, cache-friendly).
struct Determinizer<'a> {
    nwa: &'a NWA,

    // Backward reachability weights (weights that can reach some final).
    future: Vec<Weight>,
    // ε-closure(s): vector of (state, mask) pairs masked by 'future'.
    closures: Vec<Vec<(NWAStateID, Weight)>>,
    // For each s: union over closure(s) of mask & final_weight, used for finalization.
    final_via_closure: Vec<Weight>,

    // Per-state precomputation of labeled/default steps (applied to closures).
    pre: Vec<StatePrecomp>,

    interner: NodeInterner,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA) -> Self {
        let n = nwa.states.len();
        Self {
            nwa,
            future: vec![Weight::zeros(); n],
            closures: vec![Vec::new(); n],
            final_via_closure: vec![Weight::zeros(); n],
            pre: vec![StatePrecomp::default(); n],
            interner: NodeInterner::new(),
        }
    }

    fn run(&mut self) -> DWA {
        self.compute_future_weights();
        self.compute_eps_closures();
        self.precompute_steps();

        // Start determinized state = ε-closure(start)
        let start_pairs = self.closures[self.nwa.body.start_state].clone();
        let (start_id, _) = self.interner.intern(start_pairs);
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        dwa.states.0.resize(self.interner.len(), Default::default());
        dwa.body.start_state = start_id;

        let pb = Self::progress_bar(0, "Determinize (BFS)");

        // BFS over determinized states
        let mut q: VecDeque<usize> = VecDeque::new();
        let mut seen: Vec<bool> = vec![false; 1.max(self.interner.len())];
        q.push_back(start_id);
        seen.resize(self.interner.len(), false);
        seen[start_id] = true;

        while let Some(u) = q.pop_front() {
            if let Some(p) = &pb {
                p.inc(1);
                p.set_length(self.interner.len() as u64);
            }
            // Ensure DWA arena is big enough
            if dwa.states.len() < self.interner.len() {
                dwa.states.0.resize(self.interner.len(), Default::default());
            }

            // Content of this determinized node
            let content = self.interner.get(u).to_vec();

            // 1) Final weight
            let mut fin = Weight::zeros();
            for (sid, gate) in &content {
                let f = &self.final_via_closure[*sid];
                if !f.is_empty() {
                    fin |= &(gate & f);
                }
            }
            if !fin.is_empty() {
                dwa.set_final_weight(u, fin).unwrap();
            }

            // 2) Default target (content + mask) aggregated over defaults ignoring exceptions.
            let (def_vec, def_mask) = self.build_default_target(&content);
            let mut def_target_id = None;
            if !def_vec.is_empty() && !def_mask.is_empty() {
                let (tid, is_new) = self.interner.intern(def_vec.clone());
                def_target_id = Some(tid);
                if is_new {
                    if self.interner.len() > dwa.states.len() {
                        dwa.states.0.resize(self.interner.len(), Default::default());
                    }
                    if tid >= seen.len() { seen.resize(tid + 1, false); }
                    if !seen[tid] {
                        seen[tid] = true;
                        q.push_back(tid);
                    }
                }
                // Set default transition; it may be overridden by exceptions next.
                dwa.set_default_transition(u, tid, def_mask.clone()).unwrap();
            }

            // 3) Exception labels: labels where behavior differs from default.
            let labels = self.labels_of_interest_for(&content);
            for lbl in labels {
                let (lbl_vec, lbl_mask) = self.build_label_target(&content, lbl);
                if lbl_vec.is_empty() || lbl_mask.is_empty() {
                    continue;
                }
                // If label's target content equals default content, it's not an exception.
                if def_target_id.is_some() {
                    if lbl_vec == def_vec {
                        continue;
                    }
                } else {
                    // If no default exists, any non-empty labeled behavior is necessarily an exception.
                }

                let (tid, is_new) = self.interner.intern(lbl_vec);
                if is_new {
                    if self.interner.len() > dwa.states.len() {
                        dwa.states.0.resize(self.interner.len(), Default::default());
                    }
                    if tid >= seen.len() { seen.resize(tid + 1, false); }
                    if !seen[tid] {
                        seen[tid] = true;
                        q.push_back(tid);
                    }
                }
                dwa.add_transition(u, lbl, tid, lbl_mask).unwrap();
            }
        }

        if let Some(p) = pb {
            p.finish_with_message(format!("Discovered {} DWA states", self.interner.len()));
        }
        dwa
    }

    // -------------------- Precomputation --------------------

    fn compute_future_weights(&mut self) {
        let n = self.nwa.states.len();
        let mut rev: Vec<Vec<(NWAStateID, Weight)>> = vec![Vec::new(); n];

        for p in 0..n {
            // ε
            for &(t, ref w) in &self.nwa.states.0[p].epsilons {
                if t < n { rev[t].push((p, w.clone())); }
            }
            // labeled
            for (_, targets) in &self.nwa.states.0[p].transitions {
                for (t, w) in targets {
                    if *t < n { rev[*t].push((p, w.clone())); }
                }
            }
            // defaults (ignore exception sets for "reachability" – if a default exists, some label uses it)
            for d in &self.nwa.states.0[p].default {
                if d.target < n {
                    rev[d.target].push((p, d.weight.clone()));
                }
            }
        }

        let mut fut = vec![Weight::zeros(); n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        for s in 0..n {
            if let Some(fw) = &self.nwa.states.0[s].final_weight {
                if !fw.is_empty() {
                    fut[s] = fw.clone();
                    q.push_back(s);
                }
            }
        }

        while let Some(v) = q.pop_front() {
            let fv = fut[v].clone();
            if fv.is_empty() { continue; }
            for (p, w) in &rev[v] {
                let add = &fv & w;
                if !add.is_empty() && !add.is_subset_of(&fut[*p]) {
                    fut[*p] |= &add;
                    q.push_back(*p);
                }
            }
        }
        self.future = fut;
    }

    fn compute_eps_closures(&mut self) {
        let n = self.nwa.states.len();
        let pb = Self::progress_bar(n as u64, "ε-closures");

        let mut scratch = vec![Weight::zeros(); n];
        let mut q = VecDeque::new();
        let mut touched: Vec<NWAStateID> = Vec::new();

        for s in 0..n {
            let pairs = self.eps_closure(s, &mut scratch, &mut q, &mut touched);
            // final via closure(s)
            let mut fw = Weight::zeros();
            for (t, w) in &pairs {
                if let Some(f) = &self.nwa.states.0[*t].final_weight {
                    fw |= &(w & f);
                }
            }
            self.closures[s] = pairs;
            self.final_via_closure[s] = fw;
            if let Some(p) = &pb { p.inc(1); }
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
        let n = self.nwa.states.len();
        if s >= n || self.future[s].is_empty() {
            return Vec::new();
        }

        scratch[s] = self.future[s].clone();
        touched.push(s);
        q.push_back(s);

        while let Some(u) = q.pop_front() {
            let base = scratch[u].clone();
            if base.is_empty() { continue; }
            for &(v, ref w) in &self.nwa.states.0[u].epsilons {
                if v >= n { continue; }
                let mut prop = &base & w;
                if prop.is_empty() { continue; }
                prop &= &self.future[v];
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
            if !scratch[i].is_empty() {
                out.push((i, scratch[i].clone()));
                scratch[i] = Weight::zeros();
            }
        }
        touched.clear();
        out.sort_by_key(|(sid, _)| *sid);
        out
    }

    fn precompute_steps(&mut self) {
        let n = self.nwa.states.len();
        let pb = Self::progress_bar(n as u64, "Precompute steps");

        for s in 0..n {
            // Defaults
            let mut defaults: Vec<DefaultStep> = Vec::new();
            for d in &self.nwa.states.0[s].default {
                if d.target >= n { continue; }
                let pairs = Self::apply_weight_to_pairs(&self.closures[d.target], &d.weight);
                if !pairs.is_empty() {
                    defaults.push(DefaultStep { exceptions: d.exceptions.clone(), pairs });
                }
            }

            // Labeled
            let mut labeled: BTreeMap<Label, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
            for (lbl, targets) in &self.nwa.states.0[s].transitions {
                let mut acc: HashMap<NWAStateID, Weight> = HashMap::new();
                for (to, w) in targets {
                    if *to >= n { continue; }
                    for (t, wt) in Self::apply_weight_to_pairs(&self.closures[*to], w) {
                        *acc.entry(t).or_default() |= &wt;
                    }
                }
                let mut v: Vec<(NWAStateID, Weight)> = acc.into_iter().filter(|(_, w)| !w.is_empty()).collect();
                v.sort_by_key(|(t, _)| *t);
                if !v.is_empty() {
                    labeled.insert(*lbl, v);
                }
            }

            // Labels of interest: union of labeled keys and default exception labels
            let mut labels_of_interest: BTreeSet<Label> = labeled.keys().copied().collect();
            for d in &defaults {
                labels_of_interest.extend(d.exceptions.iter().copied());
            }

            self.pre[s] = StatePrecomp { labeled, defaults, labels_of_interest };
            if let Some(p) = &pb { p.inc(1); }
        }

        if let Some(p) = pb {
            p.finish_with_message("Precompute steps done");
        }
    }

    // -------------------- Builders for transitions --------------------

    fn labels_of_interest_for(&self, content: &[(NWAStateID, Weight)]) -> BTreeSet<Label> {
        let mut out = BTreeSet::new();
        for (sid, gate) in content {
            if gate.is_empty() { continue; }
            out.extend(self.pre[*sid].labels_of_interest.iter().copied());
        }
        out
    }

    // Default target content and its mask for a determinized state
    fn build_default_target(&self, content: &[(NWAStateID, Weight)]) -> (Vec<(NWAStateID, Weight)>, Weight) {
        let mut acc: HashMap<NWAStateID, Weight> = HashMap::new();
        let mut mask = Weight::zeros();

        for (sid, gate) in content {
            if gate.is_empty() { continue; }
            for d in &self.pre[*sid].defaults {
                for (t, w) in &d.pairs {
                    let add = gate & w;
                    if !add.is_empty() {
                        *acc.entry(*t).or_default() |= &add;
                        mask |= &add;
                    }
                }
            }
        }

        let mut v: Vec<(NWAStateID, Weight)> = acc.into_iter().filter(|(_, w)| !w.is_empty()).collect();
        v.sort_by_key(|(t, _)| *t);
        (v, mask)
    }

    // Label target content and its mask for a determinized state on label 'lbl'.
    // Semantics: use labeled edges if present at a state; otherwise fallback to defaults (if 'lbl' not an exception).
    fn build_label_target(&self, content: &[(NWAStateID, Weight)], lbl: Label) -> (Vec<(NWAStateID, Weight)>, Weight) {
        let mut acc: HashMap<NWAStateID, Weight> = HashMap::new();
        let mut mask = Weight::zeros();

        for (sid, gate) in content {
            if gate.is_empty() { continue; }
            let sp = &self.pre[*sid];

            let mut used_labeled = false;
            if let Some(v) = sp.labeled.get(&lbl) {
                used_labeled = true;
                for (t, w) in v {
                    let add = gate & w;
                    if !add.is_empty() {
                        *acc.entry(*t).or_default() |= &add;
                        mask |= &add;
                    }
                }
            }
            if !used_labeled {
                // fallback via defaults unless label is in exceptions
                for d in &sp.defaults {
                    if d.exceptions.contains(&lbl) { continue; }
                    for (t, w) in &d.pairs {
                        let add = gate & w;
                        if !add.is_empty() {
                            *acc.entry(*t).or_default() |= &add;
                            mask |= &add;
                        }
                    }
                }
            }
        }

        let mut v: Vec<(NWAStateID, Weight)> = acc.into_iter().filter(|(_, w)| !w.is_empty()).collect();
        v.sort_by_key(|(t, _)| *t);
        (v, mask)
    }

    // -------------------- Utilities --------------------

    fn apply_weight_to_pairs(base: &[(NWAStateID, Weight)], w: &Weight) -> Vec<(NWAStateID, Weight)> {
        if w.is_all_fast() {
            return base.to_vec();
        }
        let mut out = Vec::with_capacity(base.len());
        for (sid, bw) in base {
            let x = bw & w;
            if !x.is_empty() {
                out.push((*sid, x));
            }
        }
        out
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
