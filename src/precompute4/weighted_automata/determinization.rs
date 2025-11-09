#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

/*
A new determinization algorithm that avoids the exponential blow-up of the
"per-atom product DFA" by explicitly leveraging DWA's ability to place
weights on edges and final states.

High-level idea (control-then-weights):
- Build a single "control DFA" from the NWA by ignoring weights. This DFA
  determinizes the structural nondeterminism induced by labeled, default,
  and epsilon transitions (epsilon considered in the subset construction).
  The alphabet is Sigma' = all labels mentioned anywhere in the NWA plus
  the OTHER symbol (which triggers defaults).
- Then, for each control DFA state U and symbol a:
    Edge weight(U --a--> V) = the union (over all paths that read exactly
    one symbol a with epsilons before/after) of the intersection of all
    edge weights along that one-letter+epsilon path. Intuitively, it is
    the set of atoms that can traverse U -a-> V in the original NWA with
    some epsilon-pre/post path.
  For final weights at a control DFA state U:
    Final(U) = the union (over epsilon paths from U to any NWA final state f)
               of the intersection of epsilon weights and the final weight at f.

These two annotations (edge weights and final weights) are sufficient for
a deterministic weighted automaton (DWA) to evaluate any word w to the same
weight set as the NWA: at each step, the DWA applies the edge's weight (which
absorbs epsilon weights too), and at the end, it intersects the accumulated
weight with final_weight at the reached control state.

Crucially, this construction's state-space depends only on the structural
nondeterminism of the NWA (ignoring weights) and the size of Sigma', and
not on the number of atoms K in the weight partition, avoiding the product
explosion that arises when K is large.
*/

use super::bitset::SimpleBitset as Weight;
use super::dwa::{DWAState, DWAStates, DWA, DWABody};
use super::nwa::{NWA, NWAStates};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

fn debug_log(level: usize, msg: impl FnOnce() -> String) {
    crate::debug!(level, "{}", msg());
}

/// Alphabet = all labels that appear as exceptions, plus a special OTHER symbol.
/// OTHER means: take defaults (subject to exceptions when we are using concrete labels).
#[derive(Clone, Debug)]
struct Alphabet {
    labels: Vec<i16>, // sorted unique exception labels
    other_index: usize,
}
impl Alphabet {
    fn from_nwa(nwa: &NWA) -> Self {
        let mut set = BTreeSet::new();
        for st in &nwa.states.0 {
            for (&lbl, _) in &st.transitions {
                set.insert(lbl);
            }
            for def in &st.default {
                for &lbl in &def.exceptions {
                    set.insert(lbl);
                }
            }
        }
        let labels: Vec<i16> = set.into_iter().collect();
        let other_index = labels.len(); // last slot is OTHER
        Alphabet { labels, other_index }
    }
    #[inline]
    fn size(&self) -> usize {
        self.labels.len() + 1
    }
    #[inline]
    fn index_of_label(&self, l: i16) -> Option<usize> {
        self.labels.binary_search(&l).ok()
    }
    #[inline]
    fn is_other(&self, sym: usize) -> bool {
        sym == self.other_index
    }
    #[inline]
    fn label_at(&self, sym: usize) -> Option<i16> {
        if sym < self.labels.len() {
            Some(self.labels[sym])
        } else {
            None
        }
    }
}

/* ------------------------------
   Control DFA (weights ignored)
   ------------------------------ */

#[derive(Clone, Debug)]
struct ControlDFA {
    n_states: usize,
    start: usize,
    /// transitions[state][symbol] -> next_state
    trans: Vec<Vec<usize>>,
    /// For inspection/weight computation: the underlying NWA-state set for each DFA state
    subsets: Vec<Vec<usize>>,
}

impl ControlDFA {
    fn from_nwa(nwa: &NWA, sigma: &Alphabet) -> Self {
        let n = nwa.states.len();
        // Precompute epsilon closures per single state (weights ignored)
        let per_eps = eps_closure_per_state_ignoring_weights(&nwa.states);

        // Helper to compute closure for a set
        let eps_closure_set = |base: &[usize]| -> Vec<usize> {
            let mut mark = vec![false; n];
            let mut out = Vec::new();
            for &s in base {
                if s >= n {
                    continue;
                }
                for &u in &per_eps[s] {
                    if !mark[u] {
                        mark[u] = true;
                        out.push(u);
                    }
                }
            }
            out.sort_unstable();
            out
        };

        // Start subset = eps-closure({start})
        let start_subset = eps_closure_set(&[nwa.body.start_state]);

        // Subset construction over Sigma' (labels plus OTHER)
        let mut map: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut subsets: Vec<Vec<usize>> = Vec::new();
        let mut trans: Vec<Vec<usize>> = Vec::new();

        let mut intern = |subset: Vec<usize>,
                          subsets: &mut Vec<Vec<usize>>,
                          trans: &mut Vec<Vec<usize>>,
                          map: &mut HashMap<Vec<usize>, usize>| {
            if let Some(&id) = map.get(&subset) {
                return id;
            }
            let id = subsets.len();
            subsets.push(subset.clone());
            trans.push(vec![0usize; sigma.size()]);
            map.insert(subset, id);
            id
        };

        let start = intern(start_subset, &mut subsets, &mut trans, &mut map);

        let pb = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new_spinner();
            p.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} [Determinize/Control: {elapsed_precise}] States: {pos}")
                    .unwrap(),
            );
            p.set_position(subsets.len() as u64);
            Some(p)
        } else {
            None
        };

        let mut q = VecDeque::new();
        q.push_back(start);

        while let Some(u) = q.pop_front() {
            let subset = subsets[u].clone();

            for sym in 0..sigma.size() {
                let mut next_raw: Vec<usize> = Vec::new();
                match sigma.label_at(sym) {
                    Some(lbl) => {
                        // Label 'lbl': from each s in subset, take explicit lbl transitions and defaults that allow lbl
                        for &s in &subset {
                            // explicit
                            if let Some(targets) = nwa.states[s].transitions.get(&lbl) {
                                for (to, _) in targets {
                                    if *to < n {
                                        next_raw.push(*to);
                                    }
                                }
                            }
                            // defaults that allow lbl
                            for def in &nwa.states[s].default {
                                if !def.exceptions.contains(&lbl) {
                                    if def.target < n {
                                        next_raw.push(def.target);
                                    }
                                }
                            }
                        }
                    }
                    None => {
                        // OTHER: take all defaults from subset
                        for &s in &subset {
                            for def in &nwa.states[s].default {
                                if def.target < n {
                                    next_raw.push(def.target);
                                }
                            }
                        }
                    }
                }
                next_raw.sort_unstable();
                next_raw.dedup();
                let next_subset = eps_closure_set(&next_raw);
                let v = intern(next_subset, &mut subsets, &mut trans, &mut map);
                trans[u][sym] = v;
                if v == subsets.len() - 1 && Some(&(v as u64)) != pb.as_ref().map(|p| p.position()).as_ref() {
                    if let Some(p) = &pb {
                        p.set_position(subsets.len() as u64);
                    }
                    q.push_back(v);
                }
            }
        }

        if let Some(p) = pb {
            p.finish_with_message(format!("Control DFA built with {} states", subsets.len()));
        }

        ControlDFA {
            n_states: subsets.len(),
            start,
            trans,
            subsets,
        }
    }
}

/* ------------------------------
   Epsilon utilities (weights/no)
   ------------------------------ */

/// Epsilon closure per state (ignoring weights)
fn eps_closure_per_state_ignoring_weights(states: &NWAStates) -> Vec<Vec<usize>> {
    let n = states.len();
    let mut out = vec![Vec::<usize>::new(); n];
    for s in 0..n {
        let mut visited = vec![false; n];
        let mut stack = vec![s];
        visited[s] = true;
        let mut closure = Vec::new();
        while let Some(u) = stack.pop() {
            closure.push(u);
            for &(v, _) in &states[u].epsilons {
                if v < n && !visited[v] {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }
        closure.sort_unstable();
        out[s] = closure;
    }
    out
}

/// From a set of sources, compute for each state the union-of-weights of epsilon paths
/// from any source to that state. Start weight at each source is ALL.
fn eps_forward_masks_from_sources(states: &NWAStates, sources: &[usize]) -> Vec<Weight> {
    let n = states.len();
    let mut mask: Vec<Weight> = vec![Weight::zeros(); n];
    let mut inq = vec![false; n];
    let mut q = VecDeque::new();

    for &s in sources {
        if s >= n {
            continue;
        }
        // Add ALL at the source itself
        if (&Weight::all() & &mask[s]) != Weight::all() {
            mask[s] |= &Weight::all();
            if !inq[s] {
                inq[s] = true;
                q.push_back(s);
            }
        }
    }

    while let Some(u) = q.pop_front() {
        inq[u] = false;
        let mu = mask[u].clone();
        if mu.is_empty() {
            continue;
        }
        for &(v, ref w) in &states[u].epsilons {
            if v >= n {
                continue;
            }
            let m2 = &mu & w;
            if m2.is_empty() {
                continue;
            }
            if (&m2 & &mask[v]) != m2 {
                mask[v] |= &m2;
                if !inq[v] {
                    inq[v] = true;
                    q.push_back(v);
                }
            }
        }
    }

    mask
}

/// Given initial per-node seeds (weights located at nodes), propagate along epsilon edges.
/// Returns the fixpoint vector of weights at each node.
///
/// seeds[u] contributes to v via each epsilon u -> v with AND by edge weight; multiple paths
/// OR their weights together. Standard monotone propagation over the semiring (∨, ∧).
fn eps_propagate_from_seeds(states: &NWAStates, seeds: &[Weight]) -> Vec<Weight> {
    let n = states.len();
    let mut mask: Vec<Weight> = seeds.to_vec();
    let mut inq = vec![false; n];
    let mut q = VecDeque::new();

    for u in 0..n {
        if !mask[u].is_empty() {
            inq[u] = true;
            q.push_back(u);
        }
    }

    while let Some(u) = q.pop_front() {
        inq[u] = false;
        let mu = mask[u].clone();
        if mu.is_empty() {
            continue;
        }
        for &(v, ref w) in &states[u].epsilons {
            if v >= n {
                continue;
            }
            let m2 = &mu & w;
            if m2.is_empty() {
                continue;
            }
            if (&m2 & &mask[v]) != m2 {
                mask[v] |= &m2;
                if !inq[v] {
                    inq[v] = true;
                    q.push_back(v);
                }
            }
        }
    }

    mask
}

/* ------------------------------
   Symbol propagation with weights
   ------------------------------ */

/// Given pre-masks (weights at nodes after epsilon-closure from U), compute the one-symbol
/// propagation followed by epsilon-closure. Returns a vector 'out' of weights per node after
/// reading symbol 'sym' (Some(lbl) or None for OTHER) starting from U.
fn one_symbol_propagate(states: &NWAStates, pre_masks: &[Weight], sym: Option<i16>) -> Vec<Weight> {
    let n = states.len();
    let mut seeds: Vec<Weight> = vec![Weight::zeros(); n];

    // For each node 's' with pre_masks[s] ≠ ∅, accumulate labeled/default contributions.
    for s in 0..n {
        let ms = &pre_masks[s];
        if ms.is_empty() {
            continue;
        }
        match sym {
            Some(lbl) => {
                // explicit on lbl
                if let Some(targets) = states[s].transitions.get(&lbl) {
                    for (to, w) in targets {
                        if *to < n {
                            let add = ms & w;
                            if add.is_empty() {
                                continue;
                            }
                            if (&add & &seeds[*to]) != add {
                                seeds[*to] |= &add;
                            }
                        }
                    }
                }
                // defaults that allow lbl
                for def in &states[s].default {
                    if !def.exceptions.contains(&lbl) {
                        let to = def.target;
                        if to < n {
                            let add = ms & &def.weight;
                            if add.is_empty() {
                                continue;
                            }
                            if (&add & &seeds[to]) != add {
                                seeds[to] |= &add;
                            }
                        }
                    }
                }
            }
            None => {
                // OTHER: take all defaults
                for def in &states[s].default {
                    let to = def.target;
                    if to < n {
                        let add = ms & &def.weight;
                        if add.is_empty() {
                            continue;
                        }
                        if (&add & &seeds[to]) != add {
                            seeds[to] |= &add;
                        }
                    }
                }
            }
        }
    }

    // Epsilon-closure propagation after consuming the symbol
    eps_propagate_from_seeds(states, &seeds)
}

/* ------------------------------
   NWA -> DWA determinization
   ------------------------------ */

impl NWA {
    /// Determinize the NWA into a DWA using the "control-then-weights" construction.
    ///
    /// - Build the control DFA (ignoring weights) over Sigma' (labels + OTHER).
    /// - For each control state U and symbol a, compute the edge weight W(U,a) by
    ///   existential a-step reachability with epsilon pre/post (weights ANDed along each path).
    /// - For each control state U, compute final weight Final(U) via epsilon reachability to finals.
    /// - Emit a DWA with one state per control DFA state, start mapped directly; exceptions for
    ///   each explicit label, and a single default edge for OTHER.
    pub fn determinize_to_dwa(&self) -> DWA {
        debug_log(3, || format!("Starting efficient determinization for NWA with {} states", self.states.len()));

        let sigma = Alphabet::from_nwa(self);
        debug_log(4, || format!("Alphabet size: {} ({} labels + OTHER)", sigma.size(), sigma.labels.len()));

        // 1) Control DFA (weights ignored)
        let ctrl = ControlDFA::from_nwa(self, &sigma);

        // 2) Prepare DWA with the same number of states; direct start mapping.
        let mut dwa_states = DWAStates::default();
        for _ in 0..ctrl.n_states {
            dwa_states.add_state();
        }
        let mut dwa = DWA { states: dwa_states, body: DWABody { start_state: ctrl.start } };

        // Convenience alias
        let n_ctrl = ctrl.n_states;

        // Precompute epsilon-forward masks for each control state once
        let pb_pre = if PROGRESS_BAR_ENABLED {
            Some(
                ProgressBar::new(n_ctrl as u64).with_style(
                    ProgressStyle::default_bar()
                        .template(
                            "{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Precompute epsilon masks)",
                        )
                        .unwrap(),
                ),
            )
        } else {
            None
        };

        let mut pre_masks: Vec<Vec<Weight>> = Vec::with_capacity(n_ctrl);
        for sid in 0..n_ctrl {
            let sources = &ctrl.subsets[sid];
            let masks = eps_forward_masks_from_sources(&self.states, sources);
            pre_masks.push(masks);
            if let Some(p) = &pb_pre {
                p.inc(1);
            }
        }
        if let Some(p) = pb_pre {
            p.finish_with_message("Epsilon masks ready");
        }

        // 3) Final weights per control state
        let pb_final = if PROGRESS_BAR_ENABLED {
            Some(
                ProgressBar::new(n_ctrl as u64).with_style(
                    ProgressStyle::default_bar()
                        .template(
                            "{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Compute final weights)",
                        )
                        .unwrap(),
                ),
            )
        } else {
            None
        };

        for sid in 0..n_ctrl {
            let masks = &pre_masks[sid];
            let mut w_final = Weight::zeros();
            for (u, m) in masks.iter().enumerate() {
                if m.is_empty() {
                    continue;
                }
                if let Some(fw) = &self.states[u].final_weight {
                    let add = m & fw;
                    if !add.is_empty() {
                        w_final |= &add;
                    }
                }
            }
            if !w_final.is_empty() {
                let _ = dwa.set_final_weight(sid, w_final);
            }
            if let Some(p) = &pb_final {
                p.inc(1);
            }
        }
        if let Some(p) = pb_final {
            p.finish_with_message("Final weights computed");
        }

        // 4) Edges (default OTHER and labeled exceptions)
        let total_edges = (n_ctrl as u64) * (sigma.size() as u64);
        let pb_edges = if PROGRESS_BAR_ENABLED {
            Some(
                ProgressBar::new(total_edges).with_style(
                    ProgressStyle::default_bar()
                        .template(
                            "{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Attach DWA edges)",
                        )
                        .unwrap(),
                ),
            )
        } else {
            None
        };

        for sid in 0..n_ctrl {
            // For each symbol, compute edge weight by one-symbol propagation + epsilon
            for sym in 0..sigma.size() {
                let lbl_opt = sigma.label_at(sym);
                let post_masks = one_symbol_propagate(&self.states, &pre_masks[sid], lbl_opt);

                let dst = ctrl.trans[sid][sym];
                let dst_subset = &ctrl.subsets[dst];

                // Union weights only over the destination control subset
                let mut w_edge = Weight::zeros();
                for &u in dst_subset {
                    let add = &post_masks[u];
                    if !add.is_empty() {
                        w_edge |= add;
                    }
                }

                if sigma.is_other(sym) {
                    let _ = dwa.set_default_transition(sid, dst, w_edge);
                } else {
                    let lbl = lbl_opt.unwrap();
                    let _ = dwa.add_transition(sid, lbl, dst, w_edge);
                }

                if let Some(p) = &pb_edges {
                    p.inc(1);
                }
            }
        }
        if let Some(p) = pb_edges {
            p.finish_with_message("DWA edges attached");
        }

        debug_log(3, || format!("Efficient NWA::determinize_to_dwa: produced {} states", dwa.states.len()));
        dwa
    }
}
