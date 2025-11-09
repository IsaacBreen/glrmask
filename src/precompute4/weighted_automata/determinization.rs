#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::SimpleBitset as Weight;
use super::dwa::{DWABody, DWAStates, DWA};
use super::nwa::{NWA, NWAStates};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use range_set_blaze::RangeSetBlaze;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::RangeInclusive;
use std::time::Instant;

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now_total = Instant::now();
        debug_log(3, || format!("Starting determinization for NWA with {} states", self.states.len()));

        let fut = self.compute_future_weights();

        // 1. Atomization
        let atoms = WeightPartition::from_nwa(self);
        if atoms.intervals.is_empty() {
            let mut dwa = DWA::new();
            dwa.states.0.clear();
            dwa.body.start_state = 0;
            return dwa;
        }

        // 2. Alphabet
        let sigma = Alphabet::from_nwa_with_future(self, &fut);

        // 3. Component DFAs
        let comp_dfas: Vec<DetDFA> = atoms
            .intervals
            .iter()
            .map(|atom| {
                let nfa = PerAtomNFA::from_nwa(&self.states, self.body.start_state, &sigma, atom, &fut);
                let mut dfa = nfa.determinize(&sigma);
                dfa.minimize(&sigma);
                dfa
            })
            .collect();

        let mut determinizer = Determinizer::new(atoms, sigma, comp_dfas);
        let dwa = determinizer.run();

        debug_log(3, || format!("NWA::determinize_to_dwa total time: {:?}", now_total.elapsed()));
        dwa
    }
}

type GlobalStateID = (usize, usize); // (template_idx, state_idx)
type Support = BTreeSet<GlobalStateID>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Signature {
    final_weight: Weight,
    transitions: BTreeMap<usize, (usize, Weight)>, // sym_idx -> (dwa_state_id, weight)
}

struct Determinizer {
    atoms: WeightPartition,
    sigma: Alphabet,
    templates: Vec<TemplateDFA>,
    atom_to_template: Vec<(usize, Vec<usize>)>, // (template_idx, role_for_each_orig_symbol)
    dispatch_tables: Vec<Vec<Vec<Weight>>>,     // [template_idx][role_sym_idx][orig_sym_idx]
    // Construction state
    signature_map: HashMap<Signature, usize>,
    memo: HashMap<Support, usize>,
    dwa_states: DWAStates,
}

impl Determinizer {
    fn new(atoms: WeightPartition, sigma: Alphabet, comp_dfas: Vec<DetDFA>) -> Self {
        // Template Extraction
        let mut templates = Vec::new();
        let mut template_map = HashMap::new();
        let mut atom_to_template = Vec::with_capacity(comp_dfas.len());

        for dfa in &comp_dfas {
            let (canon, perm) = canonicalize(dfa, &sigma);
            let template_idx = *template_map.entry(canon).or_insert_with(|| {
                let idx = templates.len();
                templates.push(TemplateDFA::from_det_dfa(dfa, &sigma));
                idx
            });
            atom_to_template.push((template_idx, perm));
        }

        // Precomputation of dispatch tables
        let mut dispatch_tables = vec![vec![vec![Weight::zeros(); sigma.size()]; sigma.size()]; templates.len()];
        for atom_idx in 0..comp_dfas.len() {
            let (template_idx, ref perm) = atom_to_template[atom_idx];
            for orig_sym_idx in 0..sigma.size() {
                let role_sym_idx = perm[orig_sym_idx];
                dispatch_tables[template_idx][role_sym_idx][orig_sym_idx].add(atom_idx);
            }
        }

        Self {
            atoms,
            sigma,
            templates,
            atom_to_template,
            dispatch_tables,
            signature_map: HashMap::new(),
            memo: HashMap::new(),
            dwa_states: DWAStates::default(),
        }
    }

    fn run(&mut self) -> DWA {
        let start_support: Support = (0..self.templates.len()).map(|t_idx| (t_idx, self.templates[t_idx].start)).collect();
        let start_id = self.resolve_support(start_support);

        let mut id_to_signature: Vec<Option<Signature>> = vec![None; self.dwa_states.len()];
        for (sig, id) in &self.signature_map {
            id_to_signature[*id] = Some(sig.clone());
        }

        for (id, sig_opt) in id_to_signature.into_iter().enumerate() {
            if let Some(sig) = sig_opt {
                if !sig.final_weight.is_empty() {
                    self.dwa_states[id].final_weight = Some(sig.final_weight);
                }
                let mut transitions = BTreeMap::new();
                let mut default_target = None;
                let mut default_weight = None;

                for (sym_idx, (target_id, weight)) in sig.transitions {
                    if self.sigma.is_other(sym_idx) {
                        default_target = Some(target_id);
                        default_weight = Some(weight);
                    } else {
                        let label = self.sigma.labels[sym_idx];
                        self.dwa_states[id].transitions.exceptions.insert(label, target_id);
                        self.dwa_states[id].trans_weights_exceptions.insert(label, weight);
                    }
                }
                self.dwa_states[id].transitions.default = default_target;
                self.dwa_states[id].trans_weight_default = default_weight;
            }
        }

        DWA { states: self.dwa_states.clone(), body: DWABody { start_state: start_id } }
    }

    fn resolve_support(&mut self, s: Support) -> usize {
        if let Some(&id) = self.memo.get(&s) {
            return id;
        }

        // Placeholder to break recursion
        let placeholder_id = self.dwa_states.add_state();
        self.memo.insert(s.clone(), placeholder_id);

        let final_weight = self.compute_final_weight(&s);
        let mut transitions = BTreeMap::new();
        for sym_idx in 0..self.sigma.size() {
            let (s_next, w) = self.compute_transition(&s, sym_idx);
            let s_next_id = self.resolve_support(s_next);
            if !w.is_empty() {
                transitions.insert(sym_idx, (s_next_id, w));
            }
        }
        let signature = Signature { final_weight, transitions };

        if let Some(&existing_id) = self.signature_map.get(&signature) {
            // Found equivalent state, reuse it.
            self.memo.insert(s, existing_id);
            // The placeholder is no longer needed. We don't remove it to keep indices stable,
            // it will be an unreachable state. A later simplification pass could remove it.
            return existing_id;
        }

        // This is a new, unique state.
        self.signature_map.insert(signature, placeholder_id);
        placeholder_id
    }

    fn compute_final_weight(&self, s: &Support) -> Weight {
        let mut w = Weight::zeros();
        for &(template_idx, state_idx) in s {
            if self.templates[template_idx].finals[state_idx] {
                for (atom_idx, (atom_t_idx, _)) in self.atom_to_template.iter().enumerate() {
                    if *atom_t_idx == template_idx {
                        w |= &self.atoms.atoms[atom_idx];
                    }
                }
            }
        }
        w
    }

    fn compute_transition(&self, s: &Support, sym_idx: usize) -> (Support, Weight) {
        let mut next_s = Support::new();
        let mut w = Weight::zeros();

        for &(template_idx, state_idx) in s {
            let template = &self.templates[template_idx];
            for role_sym_idx in 0..self.sigma.size() {
                let dispatch_w = &self.dispatch_tables[template_idx][role_sym_idx][sym_idx];
                if !dispatch_w.is_empty() {
                    let next_state_idx = template.trans[state_idx][role_sym_idx];
                    next_s.insert((template_idx, next_state_idx));
                    if template.sink.map_or(true, |sink| next_state_idx != sink) {
                        w |= dispatch_w;
                    }
                }
            }
        }
        (next_s, w)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CanonicalDFA {
    n_states: usize,
    finals: Vec<bool>,
    trans: Vec<Vec<usize>>,
}

fn canonicalize(dfa: &DetDFA, sigma: &Alphabet) -> (CanonicalDFA, Vec<usize>) {
    // 1. Canonical symbol ordering
    let mut sym_cols = BTreeMap::new();
    for sym_idx in 0..sigma.size() {
        let col: Vec<usize> = (0..dfa.n_states).map(|s| dfa.trans[s][sym_idx]).collect();
        sym_cols.entry(col).or_insert_with(Vec::new).push(sym_idx);
    }
    let mut canon_to_orig_sym = Vec::new();
    for (_, mut sym_indices) in sym_cols {
        sym_indices.sort();
        canon_to_orig_sym.extend(sym_indices);
    }
    let mut orig_to_canon_sym = vec![0; sigma.size()];
    for (canon_idx, &orig_idx) in canon_to_orig_sym.iter().enumerate() {
        orig_to_canon_sym[orig_idx] = canon_idx;
    }

    // 2. Canonical state ordering
    let mut canon_to_orig_state = vec![0; dfa.n_states];
    let mut orig_to_canon_state = vec![usize::MAX; dfa.n_states];
    let mut next_canon_id = 0;
    let mut q = VecDeque::new();

    if dfa.n_states > 0 {
        orig_to_canon_state[dfa.start] = next_canon_id;
        canon_to_orig_state[next_canon_id] = dfa.start;
        next_canon_id += 1;
        q.push_back(dfa.start);
    }

    while let Some(orig_s) = q.pop_front() {
        for &orig_sym in &canon_to_orig_sym {
            let orig_t = dfa.trans[orig_s][orig_sym];
            if orig_to_canon_state[orig_t] == usize::MAX {
                orig_to_canon_state[orig_t] = next_canon_id;
                canon_to_orig_state[next_canon_id] = orig_t;
                next_canon_id += 1;
                q.push_back(orig_t);
            }
        }
    }

    // Fill in unreachable states
    for i in 0..dfa.n_states {
        if orig_to_canon_state[i] == usize::MAX {
            orig_to_canon_state[i] = next_canon_id;
            canon_to_orig_state[next_canon_id] = i;
            next_canon_id += 1;
        }
    }

    // 3. Build canonical DFA
    let mut canon_finals = vec![false; dfa.n_states];
    let mut canon_trans = vec![vec![0; sigma.size()]; dfa.n_states];
    for canon_s in 0..dfa.n_states {
        let orig_s = canon_to_orig_state[canon_s];
        canon_finals[canon_s] = dfa.finals[orig_s];
        for canon_sym in 0..sigma.size() {
            let orig_sym = canon_to_orig_sym[canon_sym];
            let orig_t = dfa.trans[orig_s][orig_sym];
            canon_trans[canon_s][canon_sym] = orig_to_canon_state[orig_t];
        }
    }

    (
        CanonicalDFA { n_states: dfa.n_states, finals: canon_finals, trans: canon_trans },
        orig_to_canon_sym,
    )
}

#[derive(Debug)]
struct TemplateDFA {
    n_states: usize,
    start: usize,
    finals: Vec<bool>,
    trans: Vec<Vec<usize>>,
    sink: Option<usize>,
}

impl TemplateDFA {
    fn from_det_dfa(dfa: &DetDFA, sigma: &Alphabet) -> Self {
        Self {
            n_states: dfa.n_states,
            start: dfa.start,
            finals: dfa.finals.clone(),
            trans: dfa.trans.clone(),
            sink: dfa.find_sink_index(sigma),
        }
    }
}

/* ------------------------------
   Utilities and support structs (from original file, required by the new implementation)
   ------------------------------ */

fn debug_log(level: usize, msg: impl FnOnce() -> String) {
    crate::debug!(level, "{}", msg());
}

#[derive(Clone, Debug)]
struct Alphabet {
    labels: Vec<i16>,
    other_index: usize,
}
impl Alphabet {
    fn from_nwa_with_future(nwa: &NWA, fut: &[Weight]) -> Self {
        let mut set = BTreeSet::new();
        for st in &nwa.states.0 {
            for &lbl in st.transitions.keys() {
                set.insert(lbl);
            }
            for def in &st.default {
                for &lbl in &def.exceptions {
                    set.insert(lbl);
                }
            }
        }
        let labels: Vec<i16> = set.into_iter().collect();
        let other_index = labels.len();
        Alphabet { labels, other_index }
    }
    #[inline]
    fn size(&self) -> usize {
        self.labels.len() + 1
    }
    #[inline]
    fn is_other(&self, sym: usize) -> bool {
        sym == self.other_index
    }
}

#[derive(Clone, Debug)]
struct WeightPartition {
    intervals: Vec<RangeInclusive<usize>>,
    atoms: Vec<Weight>,
}
impl WeightPartition {
    fn from_nwa(nwa: &NWA) -> Self {
        let mut breaks = BTreeSet::new();
        let mut feed_weight = |w: &Weight| {
            for r in w.rsb.ranges() {
                breaks.insert(*r.start());
                if *r.end() < usize::MAX {
                    breaks.insert(r.end() + 1);
                }
            }
        };
        for st in &nwa.states.0 {
            if let Some(w) = &st.final_weight {
                feed_weight(w);
            }
            for (_, w) in &st.epsilons {
                feed_weight(w);
            }
            for def in &st.default {
                feed_weight(&def.weight);
            }
            for (_, targets) in &st.transitions {
                for (_, w) in targets {
                    feed_weight(w);
                }
            }
        }
        let breaks: Vec<usize> = breaks.into_iter().collect();
        if breaks.is_empty() {
            return WeightPartition { intervals: vec![], atoms: vec![] };
        }
        let mut intervals = Vec::new();
        for i in 0..breaks.len() - 1 {
            intervals.push(breaks[i]..=breaks[i + 1] - 1);
        }
        if *breaks.last().unwrap() < usize::MAX {
            intervals.push(*breaks.last().unwrap()..=usize::MAX);
        }
        let atoms = intervals.iter().map(|r| std::iter::once(r.clone()).collect()).collect();
        WeightPartition { intervals, atoms }
    }
}

#[derive(Clone, Debug)]
struct PerAtomNFA {
    n: usize,
    start: usize,
    finals: Vec<bool>,
    ex_by_state: Vec<BTreeMap<i16, Vec<usize>>>,
    def_by_state: Vec<Vec<usize>>,
    eps_by_state: Vec<Vec<usize>>,
}
impl PerAtomNFA {
    fn from_nwa(states: &NWAStates, start: usize, sigma: &Alphabet, atom: &RangeInclusive<usize>, fut: &[Weight]) -> Self {
        let n = states.len();
        let atom_w: Weight = std::iter::once(atom.clone()).collect();
        let mut live = vec![false; n];
        for s in 0..n {
            if !(&fut[s] & &atom_w).is_empty() {
                live[s] = true;
            }
        }
        if start >= n || !live[start] {
            return PerAtomNFA { n: 1, start: 0, finals: vec![false], ex_by_state: vec![BTreeMap::new()], def_by_state: vec![vec![]], eps_by_state: vec![vec![]] };
        }
        let mut q = VecDeque::new();
        q.push_back(start);
        let mut visited = vec![false; n];
        visited[start] = true;
        let mut reachable_live = vec![start];
        while let Some(u) = q.pop_front() {
            for (v, w) in &states[u].epsilons {
                if *v < n && live[*v] && !(&atom_w & w).is_empty() && !visited[*v] {
                    visited[*v] = true;
                    q.push_back(*v);
                    reachable_live.push(*v);
                }
            }
            for (_, targets) in &states[u].transitions {
                for (v, w) in targets {
                    if *v < n && live[*v] && !(&atom_w & w).is_empty() && !visited[*v] {
                        visited[*v] = true;
                        q.push_back(*v);
                        reachable_live.push(*v);
                    }
                }
            }
            for def in &states[u].default {
                let v = def.target;
                if v < n && live[v] && !(&atom_w & &def.weight).is_empty() && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                    reachable_live.push(v);
                }
            }
        }
        reachable_live.sort();
        let mut remap = vec![usize::MAX; n];
        for (i, &s) in reachable_live.iter().enumerate() {
            remap[s] = i;
        }
        let m = reachable_live.len();
        let mut nfa = PerAtomNFA { n: m, start: remap[start], finals: vec![false; m], ex_by_state: vec![BTreeMap::new(); m], def_by_state: vec![vec![]; m], eps_by_state: vec![vec![]; m] };
        for (i, &s) in reachable_live.iter().enumerate() {
            if let Some(w) = &states[s].final_weight {
                if !(&atom_w & w).is_empty() {
                    nfa.finals[i] = true;
                }
            }
            for (v, w) in &states[s].epsilons {
                if remap[*v] != usize::MAX && !(&atom_w & w).is_empty() {
                    nfa.eps_by_state[i].push(remap[*v]);
                }
            }
            for (&lbl, targets) in &states[s].transitions {
                for (v, w) in targets {
                    if remap[*v] != usize::MAX && !(&atom_w & w).is_empty() {
                        nfa.ex_by_state[i].entry(lbl).or_default().push(remap[*v]);
                    }
                }
            }
            for def in &states[s].default {
                if remap[def.target] != usize::MAX && !(&atom_w & &def.weight).is_empty() {
                    let has_ex = sigma.labels.iter().any(|l| def.exceptions.contains(l) && nfa.ex_by_state[i].contains_key(l));
                    if !has_ex {
                        nfa.def_by_state[i].push(remap[def.target]);
                    }
                }
            }
        }
        nfa
    }
    fn eps_closure_per_state(&self) -> Vec<Vec<usize>> {
        let mut closures = vec![vec![]; self.n];
        for i in 0..self.n {
            let mut q = VecDeque::new();
            q.push_back(i);
            let mut visited = vec![false; self.n];
            visited[i] = true;
            let mut closure = vec![i];
            while let Some(u) = q.pop_front() {
                for &v in &self.eps_by_state[u] {
                    if !visited[v] {
                        visited[v] = true;
                        q.push_back(v);
                        closure.push(v);
                    }
                }
            }
            closure.sort();
            closures[i] = closure;
        }
        closures
    }
    fn eps_closure_set(&self, set: &[usize], per_state: &[Vec<usize>]) -> Vec<usize> {
        let mut result = BTreeSet::new();
        for &s in set {
            for &v in &per_state[s] {
                result.insert(v);
            }
        }
        result.into_iter().collect()
    }
    fn determinize(&self, sigma: &Alphabet) -> DetDFA {
        let eps_closures = self.eps_closure_per_state();
        let mut dfa_states = Vec::new();
        let mut dfa_map = HashMap::new();
        let mut q = VecDeque::new();
        let start_set = self.eps_closure_set(&[self.start], &eps_closures);
        dfa_map.insert(start_set.clone(), 0);
        dfa_states.push(start_set);
        q.push_back(0);
        let mut trans = Vec::new();
        let mut finals = Vec::new();
        while let Some(u_idx) = q.pop_front() {
            let u_set = dfa_states[u_idx].clone();
            let is_final = u_set.iter().any(|&s| self.finals[s]);
            finals.push(is_final);
            let mut u_trans = vec![0; sigma.size()];
            for sym_idx in 0..sigma.size() {
                let mut next_raw = BTreeSet::new();
                if sigma.is_other(sym_idx) {
                    for &s in &u_set {
                        for &t in &self.def_by_state[s] {
                            next_raw.insert(t);
                        }
                    }
                } else {
                    let lbl = sigma.labels[sym_idx];
                    for &s in &u_set {
                        if let Some(targets) = self.ex_by_state[s].get(&lbl) {
                            for &t in targets {
                                next_raw.insert(t);
                            }
                        } else {
                            for &t in &self.def_by_state[s] {
                                next_raw.insert(t);
                            }
                        }
                    }
                }
                let next_set = self.eps_closure_set(&next_raw.into_iter().collect::<Vec<_>>(), &eps_closures);
                let v_idx = *dfa_map.entry(next_set.clone()).or_insert_with(|| {
                    let new_idx = dfa_states.len();
                    dfa_states.push(next_set);
                    q.push_back(new_idx);
                    new_idx
                });
                u_trans[sym_idx] = v_idx;
            }
            trans.push(u_trans);
        }
        let n_states = dfa_states.len();
        if n_states == 0 {
            return DetDFA { n_states: 1, start: 0, finals: vec![false], trans: vec![vec![0; sigma.size()]] };
        }
        DetDFA { n_states, start: 0, finals, trans }
    }
}

#[derive(Clone, Debug)]
struct DetDFA {
    n_states: usize,
    start: usize,
    finals: Vec<bool>,
    trans: Vec<Vec<usize>>,
}
impl DetDFA {
    fn minimize(&mut self, sigma: &Alphabet) {
        if self.n_states == 0 {
            return;
        }
        let mut part = vec![if self.finals[s] { 0 } else { 1 } for s in 0..self.n_states];
        let mut num_parts = 2;
        loop {
            let mut changed = false;
            for sym_idx in 0..sigma.size() {
                let mut inv_trans = vec![vec![]; num_parts];
                for s in 0..self.n_states {
                    inv_trans[part[self.trans[s][sym_idx]]].push(s);
                }
                for i in 0..num_parts {
                    if inv_trans[i].len() < self.n_states {
                        let mut splits = HashMap::new();
                        for &s in &inv_trans[i] {
                            splits.entry(part[s]).or_insert_with(Vec::new).push(s);
                        }
                        for (_, split_group) in splits {
                            if split_group.len() < inv_trans[i].len() {
                                changed = true;
                                for &s in &split_group {
                                    part[s] = num_parts;
                                }
                                num_parts += 1;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut remap = vec![usize::MAX; num_parts];
        let mut new_n = 0;
        let new_start = {
            let p = part[self.start];
            if remap[p] == usize::MAX {
                remap[p] = new_n;
                new_n += 1;
            }
            remap[p]
        };
        let mut new_finals = vec![false; num_parts];
        let mut new_trans = vec![vec![0; sigma.size()]; num_parts];
        for s in 0..self.n_states {
            let p = part[s];
            if remap[p] == usize::MAX {
                remap[p] = new_n;
                new_n += 1;
            }
            let new_s = remap[p];
            new_finals[new_s] = self.finals[s];
            for sym_idx in 0..sigma.size() {
                let t = self.trans[s][sym_idx];
                let pt = part[t];
                if remap[pt] == usize::MAX {
                    remap[pt] = new_n;
                    new_n += 1;
                }
                new_trans[new_s][sym_idx] = remap[pt];
            }
        }
        new_finals.truncate(new_n);
        new_trans.truncate(new_n);
        self.n_states = new_n;
        self.start = new_start;
        self.finals = new_finals;
        self.trans = new_trans;
    }
    fn find_sink_index(&self, sigma: &Alphabet) -> Option<usize> {
        'outer: for s in 0..self.n_states {
            if self.finals[s] {
                continue;
            }
            for sym in 0..sigma.size() {
                if self.trans[s][sym] != s {
                    continue 'outer;
                }
            }
            return Some(s);
        }
        None
    }
}