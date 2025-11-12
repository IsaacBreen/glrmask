// src/precompute4/weighted_automata/determinization.rs
//
// Determinization of NWA -> DWA specialized to the Weight bitset semiring,
// with an aggressively minimized construction following Mohri's insights:
//   - General case: weighted-subset determinization with ε-closure integration,
//     default/exception compression of labels, and compact state-entry/final weights.
//   - Special optimized case (fast-path): if the NWA is a union of singleton-loop
//     components reachable by ε from the start, whose per-component base weights
//     are pairwise disjoint (the "hypercube" class), then we compile directly a
//     1-state DWA. In that case, the entire "history" is tracked in the accumulating
//     weight, not in the state space, avoiding 2^N blowup.
//
// Correctness sketch for the fast-path (bitset semiring):
//   Let components C = {q}. Each q is a single state with self-loops only and no ε-out,
//   and has a base weight B_q (start ε weight ∧ final weight at q). For each symbol a,
//   let W_a = ⋁_{q allows a} (B_q ∧ w(q,a)) where w(q,a) is the (union of) loop weights at q.
//   Then the 1-state DWA uses per-letter self-loops on a with weight W_a, and final weight
//   F = ⋁_q B_q. For any input word x = a1...ak, the DWA assigns
//       (W_a1 ∧ W_a2 ∧ ... ∧ W_ak) ∧ F
//   The bitset semiring (∧ distributes over ⋁) and pairwise disjointness of the B_q guarantees
//   this equals
//       ⋁_q ( B_q ∧ w(q,a1) ∧ ... ∧ w(q,ak) )
//   which is exactly the NWA semantics for this class (each q "reads" the sequence in-place).
//
// Implementation notes about state identity canonicalization:
//   - Prior versions keyed determinized states by String built from Weight's Display.
//     If Weight's Display is non-canonical (e.g., unsorted iteration), equal states got
//     distinct keys, causing non-termination on cyclic inputs.
//   - We fix this by using Weight::canonical_string(), which is deterministic,
//     when building closure keys.

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use super::common::{NWAStateID, Weight};
use super::dwa::{DWA, DWABuildError};
use super::nwa::{NWA, NWAStates};

// Weighted subset and ε-closure maps
type WeightedSubset = BTreeMap<NWAStateID, Weight>;
type ClosureMap = BTreeMap<NWAStateID, Weight>;

// Helper ops on Weight
fn weight_union(mut a: Weight, b: &Weight) -> Weight {
    a | b.clone()
}
fn weight_union_in_place(dst: &mut Weight, src: &Weight) {
    *dst = dst.clone() | src.clone();
}
fn weight_intersection(a: &Weight, b: &Weight) -> Weight {
    a & b
}
fn is_zero(w: &Weight) -> bool {
    w.is_empty()
}

// Canonical key for Weight (deterministic)
fn weight_key(w: &Weight) -> String {
    // Using w.to_string() is incorrect if the underlying implementation (e.g.,
    // iterating over a hash map) is not deterministic. A non-canonical key
    // prevents the determinizer from recognizing equivalent states, leading to
    // state explosion and non-termination for cyclic NWAs.
    let mut items: Vec<_> = w.iter().collect();
    items.sort_unstable();
    format!("{:?}", items)
}

// Create a canonical key for a weighted subset to identify states in a HashMap.
fn subset_key(sub: &WeightedSubset) -> String {
    let mut s = String::new();
    for (sid, w) in sub {
        s.push_str(&format!("{}#{};", sid, weight_key(w)));
    }
    s
}

// Create a canonical key for a closure map used to identify determinized states.
fn closure_key(cl: &ClosureMap) -> String {
    let mut s = String::new();
    for (sid, w) in cl {
        s.push_str(&format!("{}#{};", sid, weight_key(w)));
    }
    s
}

// Epsilon-closure from a weighted subset:
// Input: initial weighted subset 'seed' mapping NWA states to input weights (bitsets).
// Output: closure map mapping every reachable state via ε-paths to the union of all
// weights of those ε-paths (seed_weight ∧ ε-edges ∧ ...).
fn epsilon_closure(nwa_states: &NWAStates, seed: &WeightedSubset) -> ClosureMap {
    let mut closure: ClosureMap = ClosureMap::new();
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();

    for (sid, w) in seed {
        if !is_zero(w) {
            let prev = closure.get(sid).cloned().unwrap_or_else(Weight::zeros);
            let neww = weight_union(prev.clone(), w);
            if neww != prev {
                closure.insert(*sid, neww.clone());
                queue.push_back(*sid);
            }
        }
    }

    while let Some(u) = queue.pop_front() {
        let uw = closure.get(&u).cloned().unwrap_or_else(Weight::zeros);
        if is_zero(&uw) {
            continue;
        }
        for (v, w_eps) in &nwa_states[u].epsilons {
            let cand = weight_intersection(&uw, w_eps);
            if is_zero(&cand) {
                continue;
            }
            let prev = closure.get(v).cloned().unwrap_or_else(Weight::zeros);
            let merged = weight_union(prev.clone(), &cand);
            if merged != prev {
                closure.insert(*v, merged.clone());
                queue.push_back(*v);
            }
        }
    }

    closure
}

// Compute S_total: explicit labels and default exceptions in this ε-closure
fn collect_exception_labels(nwa_states: &NWAStates, closure: &ClosureMap) -> BTreeSet<i16> {
    let mut labels: BTreeSet<i16> = BTreeSet::new();
    for (sid, cw) in closure {
        if is_zero(cw) {
            continue;
        }
        for (lbl, _) in nwa_states[*sid].transitions.iter() {
            labels.insert(*lbl);
        }
        for def in &nwa_states[*sid].default {
            for ex in &def.exceptions {
                labels.insert(*ex);
            }
        }
    }
    labels
}

// Next subset for "others" labels not in S_total: use all defaults
fn next_subset_for_others(nwa_states: &NWAStates, closure: &ClosureMap) -> WeightedSubset {
    let mut next: WeightedSubset = WeightedSubset::new();
    for (sid, cw) in closure {
        if is_zero(cw) {
            continue;
        }
        let st = &nwa_states[*sid];
        for def in &st.default {
            let cand = weight_intersection(cw, &def.weight);
            if is_zero(&cand) {
                continue;
            }
            next.entry(def.target)
                .and_modify(|w| weight_union_in_place(w, &cand))
                .or_insert(cand);
        }
    }
    next.retain(|_, w| !is_zero(w));
    next
}

// Next subset for a specific label ch
fn next_subset_for_label(
    nwa_states: &NWAStates,
    closure: &ClosureMap,
    ch: i16,
) -> WeightedSubset {
    let mut next: WeightedSubset = WeightedSubset::new();
    for (sid, cw) in closure {
        if is_zero(cw) {
            continue;
        }
        let st = &nwa_states[*sid];
        if let Some(targets) = st.transitions.get(&ch) {
            for (to, w_edge) in targets {
                let cand = weight_intersection(cw, w_edge);
                if is_zero(&cand) {
                    continue;
                }
                next.entry(*to)
                    .and_modify(|w| weight_union_in_place(w, &cand))
                    .or_insert(cand);
            }
        } else {
            for def in &st.default {
                if !def.exceptions.contains(&ch) {
                    let cand = weight_intersection(cw, &def.weight);
                    if is_zero(&cand) {
                        continue;
                    }
                    next.entry(def.target)
                        .and_modify(|w| weight_union_in_place(w, &cand))
                        .or_insert(cand);
                }
            }
        }
    }
    next.retain(|_, w| !is_zero(w));
    next
}

// Union over values (for edge/default weights)
fn union_over_values(map: &WeightedSubset) -> Weight {
    let mut acc = Weight::zeros();
    for w in map.values() {
        acc = acc | w.clone();
    }
    acc
}

// Compute DWA state-entry and final weights from ε-closure
fn compute_state_and_final_weights(
    nwa: &NWA,
    closure: &ClosureMap,
) -> (Option<Weight>, Option<Weight>) {
    // State-entry weights are not part of Mohri's determinization algorithm for
    // standard weighted automata. Their inclusion was conceptually incorrect and
    // could interfere with the DWA's semantics, even if idempotent operations
    // masked immediate issues. The correct approach is to encode all necessary
    // information in the DWA's state transitions and final weights.
    let entry_opt = None;
    let mut finalw = Weight::zeros();
    for (sid, cw) in closure {
        if let Some(fw) = &nwa.states[*sid].final_weight {
            let cand = cw & fw;
            if !is_zero(&cand) {
                finalw = finalw | cand;
            }
        }
    }
    let final_opt = if finalw.is_empty() { None } else { Some(finalw) };
    (entry_opt, final_opt)
}

// Determinizer

struct Determinizer<'a> {
    nwa: &'a NWA,
    seen: HashMap<String, usize>,  // closure key -> DWA state id (canonical, deterministic)
    subsets: Vec<WeightedSubset>,  // for each DWA id, its (raw) weighted subset
    closures: Vec<ClosureMap>,     // for each DWA id, its ε-closure
    queue: VecDeque<usize>,        // worklist of DWA ids
    dwa: DWA,
    mp: Option<MultiProgress>,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA, mp: Option<MultiProgress>) -> Self {
        let mut dwa = DWA::new();
        // We'll manage our own state indexing; clear the auto-created state.
        dwa.states.0.clear();
        dwa.body.start_state = 0; // placeholder; will be reset after first register_state
        Determinizer {
            nwa,
            seen: HashMap::new(),
            subsets: Vec::new(),
            closures: Vec::new(),
            queue: VecDeque::new(),
            dwa,
            mp,
        }
    }

    fn register_state(&mut self, subset: WeightedSubset) -> usize {
        // 1) Canonicalize incoming seed by dropping empty weights.
        let mut subset_clean: WeightedSubset = WeightedSubset::new();
        for (sid, w) in subset {
            if !is_zero(&w) { subset_clean.insert(sid, w); }
        }
        // 2) Compute ε-closure; determinized state identity depends on closure
        let closure = epsilon_closure(&self.nwa.states, &subset_clean);
        let key = closure_key(&closure);
        if let Some(&id) = self.seen.get(&key) { return id; }

        // 3) Create brand-new DWA state for this closure
        let id = self.dwa.add_state();

        // 4) Install state-entry and final weights from closure
        // State-entry weights are not used in the standard determinization algorithm.
        let (_entry_opt, final_opt) = compute_state_and_final_weights(self.nwa, &closure);
        if let Some(w) = final_opt { let _ = self.dwa.set_final_weight(id, w); }

        self.seen.insert(key, id);
        self.subsets.push(subset_clean);
        self.closures.push(closure);
        self.queue.push_back(id);
        id
    }

    fn expand_state(&mut self, sid: usize) {
        let closure = &self.closures[sid];
        if closure.is_empty() {
            return;
        }

        let labels_pb = if let Some(mp) = &self.mp {
            let pb = mp.add(ProgressBar::new(0));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("  {spinner:.green} [{elapsed_precise}] Labels: [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb.set_message(format!("State {}", sid));
            Some(pb)
        } else {
            None
        };

        // exception labels are all explicit labels and default exceptions visible in closure
        let exception_labels = collect_exception_labels(&self.nwa.states, closure);

        if let Some(pb) = &labels_pb {
            pb.set_length(exception_labels.len() as u64);
        }

        // default "others"
        let others_subset = next_subset_for_others(&self.nwa.states, closure);
        let others_weight = union_over_values(&others_subset);

        let mut exception_data: BTreeMap<i16, (WeightedSubset, Weight)> = BTreeMap::new();
        for ch in &exception_labels {
            let sub_ch = next_subset_for_label(&self.nwa.states, closure, *ch);
            let w_ch = union_over_values(&sub_ch);
            if !sub_ch.is_empty() && !is_zero(&w_ch) {
                exception_data.insert(*ch, (sub_ch, w_ch));
            }
            if let Some(pb) = &labels_pb {
                pb.inc(1);
            }
        }

        if let Some(pb) = &labels_pb {
            pb.set_message(format!("State {}: installing transitions", sid));
            pb.set_length(exception_data.len() as u64);
            pb.set_position(0);
        }

        // install default if non-empty
        let mut default_target_id: Option<usize> = None;
        if !others_subset.is_empty() && !is_zero(&others_weight) {
            let to_id = self.register_state(others_subset);
            default_target_id = Some(to_id);
            let _ = self
                .dwa
                .set_default_transition(sid, to_id, others_weight.clone())
                .map_err(|_e: DWABuildError| ())
                .ok();
        }

        for (ch, (sub_ch, w_ch)) in exception_data {
            let to_ch_id = self.register_state(sub_ch);
            let need_exception = match default_target_id {
                None => true,
                Some(def_id) => def_id != to_ch_id || w_ch != others_weight,
            };
            if need_exception {
                let _ = self
                    .dwa
                    .add_transition(sid, ch, to_ch_id, w_ch.clone())
                    .map_err(|_e: DWABuildError| ())
                    .ok();
            }
            if let Some(pb) = &labels_pb {
                pb.inc(1);
            }
        }

        if let Some(pb) = labels_pb {
            pb.finish_and_clear();
        }
    }
}

// Fast-path: try to build a 1-state DWA for a union of singleton-loop components.
fn try_build_singleton_loop_union(nwa: &NWA) -> Option<DWA> {
    if nwa.states.0.is_empty() || nwa.body.start_state >= nwa.states.len() {
        return None;
    }

    let start = nwa.body.start_state;
    if !nwa.states[start].transitions.is_empty() {
        return None;
    }
    if !nwa.states[start].default.is_empty() {
        return None;
    }

    let mut seed: WeightedSubset = WeightedSubset::new();
    seed.insert(start, Weight::all());
    let start_closure = epsilon_closure(&nwa.states, &seed);

    let mut comps: Vec<(NWAStateID, Weight)> = Vec::new();
    for (sid, cw) in start_closure.iter() {
        if *sid == start || is_zero(cw) {
            continue;
        }
        let st = &nwa.states[*sid];

        // singleton: no ε-out, no defaults, all labeled loops to itself
        if !st.epsilons.is_empty() {
            return None;
        }
        if !st.default.is_empty() {
            return None;
        }
        for (_lbl, vec_targets) in st.transitions.iter() {
            for (to, _w) in vec_targets {
                if *to != *sid {
                    return None;
                }
            }
        }

        if let Some(fw) = &st.final_weight {
            let base = cw & fw;
            if !base.is_empty() {
                comps.push((*sid, base));
            }
        }
    }

    if comps.is_empty() {
        return None;
    }

    // Pairwise disjointness of base weights
    for i in 0..comps.len() {
        for j in (i + 1)..comps.len() {
            if !(comps[i].1.clone() & comps[j].1.clone()).is_empty() {
                return None;
            }
        }
    }

    // Build per-label weights W[l] = ⋁_q ( base_q ∧ ⋁_{q -l-> q} w_edge )
    let mut label_to_weight: BTreeMap<i16, Weight> = BTreeMap::new();
    for (sid, base) in &comps {
        let st = &nwa.states[*sid];
        for (lbl, vec_targets) in st.transitions.iter() {
            let mut w_union = Weight::zeros();
            for (_to, w) in vec_targets {
                w_union = w_union | w.clone();
            }
            if !w_union.is_empty() {
                let contrib = base.clone() & w_union;
                if !contrib.is_empty() {
                    let prev = label_to_weight.get(lbl).cloned().unwrap_or_else(Weight::zeros);
                    label_to_weight.insert(*lbl, prev | contrib);
                }
            }
        }
    }

    // Single DWA state with final F = ⋁ base_q and loops with weights W[l].
    let mut final_union = Weight::zeros();
    for (_sid, base) in &comps {
        final_union = final_union | base.clone();
    }

    let mut dwa = DWA::new();
    let s0 = dwa.body.start_state;
    if !final_union.is_empty() {
        let _ = dwa.set_final_weight(s0, final_union);
    }
    for (lbl, w) in label_to_weight {
        if !w.is_empty() {
            let _ = dwa.add_transition(s0, lbl, s0, w);
        }
    }

    Some(dwa)
}

impl NWA {
    /// Determinize the subgraph reachable from `self.body.start_state` into a DWA.
    pub fn determinize_to_dwa(&self) -> DWA {
        // Fast-path: avoid blow-up for "singleton-loop union" cases
        if let Some(dwa) = try_build_singleton_loop_union(self) {
            return dwa;
        }

        eprintln!(
            "[DEBUG] Determinization: Using general-purpose subset construction (fast-path not taken)."
        );
        if self.states.0.is_empty() || self.body.start_state >= self.states.len() {
            return DWA::new();
        }

        let mp = MultiProgress::new();
        let main_pb = mp.add(ProgressBar::new(1));
        main_pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] States: [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        main_pb.set_message("Determinizing NWA");

        let mut det = Determinizer::new(self, Some(mp));

        // Start subset: { start_state -> 1 (Weight::all) }.
        let mut start_subset: WeightedSubset = WeightedSubset::new();
        start_subset.insert(self.body.start_state, Weight::all());
        let start_id = det.register_state(start_subset);
        det.dwa.body.start_state = start_id;

        let mut processed_count = 0;
        while let Some(sid) = det.queue.pop_front() {
            let total_states = det.seen.len();
            main_pb.set_length(total_states as u64);
            main_pb.set_position(processed_count as u64);
            main_pb.set_message(format!("Expanding state {}/{}", processed_count + 1, total_states));

            det.expand_state(sid);
            processed_count += 1;
        }
        main_pb.finish_with_message("Determinization complete");

        det.dwa
    }
}
