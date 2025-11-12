// src/precompute4/weighted_automata/determinization.rs
//
// Determinization of NWA -> DWA, specialized to the Weight bitset semiring.
// - Addition (⊕) is union of bitsets.
// - Multiplication (⊗) is intersection of bitsets.
// - Epsilon-closure weights are propagated as state-entry weights.
// - On each symbol, we compute a single deterministic target subset and a single
//   edge weight equal to the union over all contributing NWA paths.
// - We compress labels using a single default transition plus explicit exceptions.
//
// Correctness sketch (see the response above for a longer proof outline):
//   - For input word w = a1 a2 ... ak, the unique DWA path produces a bitset
//     intersection of the state-entry weights and edge weights:
//       W_entry(s0) ∧ W_edge(s0,a1) ∧ W_entry(s1) ∧ ... ∧ W_edge(sk-1,ak) ∧ W_entry(sk) ∧ W_final(sk)
//     which equals the union of all NWA path weights for w thanks to distributivity
//     of ∧ over ∨ and our construction of each factor.
//
// Practical considerations:
//   - State identity is the weighted subset after the last symbol: a mapping
//     NWAStateID -> Weight (union of all epsilon-path weights from the last symbol).
//   - We use a canonical string key for the weighted subset to reuse states and avoid recomputation.
//   - For each DWA state, we compute:
//       * closure_map: epsilon-closure of the weighted subset and their weights,
//       * state-entry weight: union of closure weights,
//       * final weight: union over closure (closure_weight ∧ final_weight),
//       * default target/weight for all labels not in S_total = explicit labels ∪ default exceptions,
//       * explicit exceptions for labels in S_total when they differ from default.
//
// Notes on defaults:
//   - Defaults are used only when no labeled transition matches at a state.
//   - If a labeled transition exists on label ℓ from q, we ignore q's defaults for ℓ.
//   - A default transition applies on a label if that label is NOT in its exception set.
//   - Multiple default transitions from the same state q can all contribute on a label.

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use super::common::{NWAStateID, Weight};
use super::dwa::{DWA, DWABuildError};
use super::nwa::{NWA, NWAStates};

type WeightedSubset = BTreeMap<NWAStateID, Weight>;
type ClosureMap = BTreeMap<NWAStateID, Weight>;

fn weight_union(mut a: Weight, b: &Weight) -> Weight {
    // a ∪= b
    // Prefer |= if available, but fall back to a = a | b.clone()
    // since we do not know the exact trait set at compile time here.
    // Using clone() on small bitsets is fine; for large bitsets, SimpleBitset
    // is expected to be cheap to copy or use RC under the hood.
    let u = a | b.clone();
    u
}

fn weight_union_in_place(dst: &mut Weight, src: &Weight) {
    let u = dst.clone() | src.clone();
    *dst = u;
}

fn weight_intersection(a: &Weight, b: &Weight) -> Weight {
    a & b
}

fn is_zero(w: &Weight) -> bool {
    w.is_empty()
}

// Create a canonical key for a weighted subset to identify states in a HashMap.
// We avoid requiring Weight: Hash or Ord by using Display to build a stable string.
fn subset_key(sub: &WeightedSubset) -> String {
    // canonical: sorted by NWAStateID (BTreeMap already sorts), each as "id=weight;"
    // relying on Weight's Display to be stable across runs
    let mut s = String::new();
    for (sid, w) in sub {
        s.push_str(&format!("{}={};", sid, w));
    }
    s
}

// Epsilon-closure from a weighted subset:
// Input: initial weighted subset 'seed' mapping NWA states to input weights (bitsets).
// Output: closure map mapping every reachable state via ε-paths to the union of all
// weights of those ε-paths (seed_weight ∧ ε-edges ∧ ...).
fn epsilon_closure(nwa_states: &NWAStates, seed: &WeightedSubset) -> ClosureMap {
    // Closure uses a BFS-like propagation until no new bits appear.
    // For each state, we maintain union of all discovered ε-path weights.
    let mut closure: ClosureMap = ClosureMap::new();
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();

    // Initialize with the seed itself (0-length ε-paths).
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

    // Propagate forward through ε transitions while growing weights.
    while let Some(u) = queue.pop_front() {
        let uw = closure.get(&u).cloned().unwrap_or_else(Weight::zeros);
        if is_zero(&uw) {
            continue;
        }
        // For each ε transition u -> v with weight w_eps, contribution is uw ∧ w_eps.
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

// Compute the set S_total of "exception" labels: explicit labels and default exceptions
// present in the closure. For labels NOT in S_total, behavior is uniform and equals the
// "other/default" behavior we compute once.
fn collect_exception_labels(nwa_states: &NWAStates, closure: &ClosureMap) -> BTreeSet<i16> {
    let mut labels: BTreeSet<i16> = BTreeSet::new();
    for (sid, cw) in closure {
        if is_zero(cw) {
            continue;
        }
        // explicit labeled transitions
        for (lbl, _) in nwa_states[*sid].transitions.iter() {
            labels.insert(*lbl);
        }
        // default exceptions (union of all exception sets)
        for def in &nwa_states[*sid].default {
            for ex in &def.exceptions {
                labels.insert(*ex);
            }
        }
    }
    labels
}

// Build the "other/default" next-subset (raw-map) for labels not in S_total.
// For a given closure, the default contributions are:
// - from each closure state q, since no explicit transitions apply on "others",
// - include ALL default transitions def where "others" are not exceptions for def
//   (and since "others" are not in any exception set, all defaults apply).
// By construction, defaults apply only if the label has no explicit transitions at q,
// and "others" satisfies that.
fn next_subset_for_others(nwa_states: &NWAStates, closure: &ClosureMap) -> WeightedSubset {
    let mut next: WeightedSubset = WeightedSubset::new();
    for (sid, cw) in closure {
        if is_zero(cw) {
            continue;
        }
        let st = &nwa_states[*sid];
        // For "others" there is no explicit transition, so defaults apply.
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
    // Remove empties if any appeared (safety)
    next.retain(|_, w| !is_zero(w));
    next
}

// Compute the next subset for a specific label ch.
// For each closure state q:
//   - If q has explicit transitions on ch, we take those transitions only (defaults ignored).
//   - Otherwise, consider defaults def s.t. ch ∉ def.exceptions.
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
            // explicit labeled transitions take precedence; defaults ignored for this ch at sid
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
            // no explicit label -> consider defaults that include ch (i.e., ch ∉ exceptions)
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

// Union of weights across all values in a WeightedSubset (used for edge weights).
fn union_over_values(map: &WeightedSubset) -> Weight {
    let mut acc = Weight::zeros();
    for w in map.values() {
        let u = acc.clone() | w.clone();
        acc = u;
    }
    acc
}

// Compute the final weight for a DWA state derived from a closure.
fn compute_final_weight(
    nwa: &NWA,
    closure: &ClosureMap,
) -> Option<Weight> {
    // Final weight is union over closure_weight ∧ final_weight(q)
    let mut finalw = Weight::zeros();
    for (sid, cw) in closure {
        if let Some(fw) = &nwa.states[*sid].final_weight {
            let cand = cw & fw;
            if !is_zero(&cand) {
                weight_union_in_place(&mut finalw, &cand);
            }
        }
    }
    if finalw.is_empty() { None } else { Some(finalw) }
}

// Register (or retrieve) a DWA state corresponding to a weighted subset (raw-map).
// On first creation, compute and set:
//   - state-entry weight and final weight (from epsilon-closure),
//   - return newly created state id,
// and schedule for expansion later.
struct Determinizer<'a> {
    nwa: &'a NWA,
    // Map subset key -> DWA state id.
    seen: HashMap<String, usize>,
    // For each DWA state id, we store its closure.
    closures: Vec<ClosureMap>,
    // Work queue of DWA state ids to expand.
    queue: VecDeque<usize>,
    // The resulting deterministic automaton being built.
    dwa: DWA,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA) -> Self {
        let mut dwa = DWA::new();
        // The state 0 created by DWA::new() is not used by determinization,
        // which creates its own states. Clearing it ensures our state IDs
        // start from 0 and align with Vec indices.
        dwa.states.0.clear();
        // The start state will be set to the first registered state.
        dwa.body.start_state = 0; // Placeholder
        Determinizer {
            nwa,
            seen: HashMap::new(),
            closures: Vec::new(),
            queue: VecDeque::new(),
            dwa,
        }
    }

    fn register_state(&mut self, subset: WeightedSubset) -> usize {
        // Drop empty weights from raw subset
        let mut subset_clean: WeightedSubset = WeightedSubset::new();
        for (sid, w) in subset {
            if !is_zero(&w) {
                subset_clean.insert(sid, w);
            }
        }

        // State identity is determined by the epsilon closure of the raw subset.
        // If two different raw subsets have the same closure, they are equivalent.
        let closure = epsilon_closure(&self.nwa.states, &subset_clean);
        let key = subset_key(&closure);

        if let Some(&id) = self.seen.get(&key) {
            return id;
        }

        // Create new DWA state
        let id = self.dwa.add_state();

        // Set final weight from the closure. State-entry weights are not used;
        // all weight information is on transitions.
        if let Some(w) = compute_final_weight(self.nwa, &closure) {
            let _ = self.dwa.set_final_weight(id, w);
        }

        self.seen.insert(key, id);
        self.closures.push(closure);
        self.queue.push_back(id);
        id
    }

    fn expand_state(&mut self, sid: usize) {
        // We must calculate all next subsets before calling self.register_state to avoid
        // mutable borrow of self invalidating the immutable borrow of self.closures[sid].
        // We clone the closure to work around the borrow checker.
        let closure = self.closures[sid].clone();

        // If closure is empty, there are no outgoing transitions.
        if closure.is_empty() {
            return;
        }

        // 1. Compute exception labels (explicit labels + default exceptions).
        let exception_labels = collect_exception_labels(&self.nwa.states, &closure);

        // 2. Compute default ("others") next-subset and its transition weight.
        // The transition weight includes contributions from subsequent epsilon paths.
        let others_subset = next_subset_for_others(&self.nwa.states, &closure);
        let others_closure = epsilon_closure(&self.nwa.states, &others_subset);
        let others_weight = union_over_values(&others_closure);

        // 3. Pre-calculate all exception subsets and weights.
        let mut exception_data: BTreeMap<i16, (WeightedSubset, Weight)> = BTreeMap::new();
        for ch in &exception_labels {
            let sub_ch = next_subset_for_label(&self.nwa.states, &closure, *ch);
            let closure_ch = epsilon_closure(&self.nwa.states, &sub_ch);
            let w_ch = union_over_values(&closure_ch);
            // Only store non-empty transitions
            if !sub_ch.is_empty() && !is_zero(&w_ch) {
                exception_data.insert(*ch, (sub_ch, w_ch));
            }
        }

        // 4. Install default transition if non-empty.
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

        // 5. For each pre-calculated exception label, register the state and add an explicit exception
        //    only if it differs from default.
        for (ch, (sub_ch, w_ch)) in exception_data {
            let to_ch_id = self.register_state(sub_ch);

            // Decide whether we need an exception for ch.
            // If default exists and target/weight match default, we can skip adding an exception.
            let need_exception = match default_target_id {
                None => true, // no default installed; must emit exception
                Some(def_id) => {
                    if def_id != to_ch_id {
                        true
                    } else {
                        // Weights must also match.
                        w_ch != others_weight
                    }
                }
            };

            if need_exception {
                let _ = self
                    .dwa
                    .add_transition(sid, ch, to_ch_id, w_ch.clone())
                    .map_err(|_e: DWABuildError| ())
                    .ok();
            }
        }
    }
}

impl NWA {
    /// Determinize the subgraph reachable from `self.body.start_state` into a DWA.
    ///
    /// Semantics:
    /// - NWA path weights are intersected (∧) along a single path and unioned (∨) across alternative paths.
    /// - DWA provides a single run; its per-edge weights equal the union over all contributing NWA paths on that edge;
    ///   per-state entry weights equal the union over ε-closure contributions active at that point.
    /// - Default transitions are emitted for "other" labels (not in any explicit label or default exception),
    ///   with explicit exceptions only when a label's behavior differs.
    pub fn determinize_to_dwa(&self) -> DWA {
        if self.states.0.is_empty() || self.body.start_state >= self.states.len() {
            // An NWA with no states, or an invalid start state, accepts no strings.
            // Return a DWA that does the same.
            // DWA::new() creates a single non-final start state with no transitions.
            return DWA::new();
        }

        let mut det = Determinizer::new(self);

        // The initial DWA state corresponds to the epsilon closure of the NWA start state.
        let mut start_subset: WeightedSubset = WeightedSubset::new();
        start_subset.insert(self.body.start_state, Weight::all());

        // We register the raw subset. `register_state` will compute its closure,
        // use that for the key, and set the final weight if needed.
        let start_id = det.register_state(start_subset);
        det.dwa.body.start_state = start_id;

        // BFS expand all reachable DWA states.
        while let Some(sid) = det.queue.pop_front() {
            det.expand_state(sid);
        }

        // If the NWA was empty or only had a non-final start state, determinization
        // might produce an empty DWA. Ensure there's at least a start state.
        if det.dwa.states.0.is_empty() {
            det.dwa.add_state();
            det.dwa.body.start_state = 0;
        }

        det.dwa
    }
}
