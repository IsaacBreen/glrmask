#![allow(dead_code)]

use super::common::{NWAStateID, StateID, Weight};
use super::dwa::{DWAState, DWAStates, DWA};
use super::nwa::{NWAState, NWAStates, NWA};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// A partition of states into equivalence classes.
/// `class_of[s]` is the index of the equivalence class containing state `s`.
#[derive(Clone, Debug)]
struct Partition {
    class_of: Vec<usize>,
    num_classes: usize,
}

impl Partition {
    fn new(num_states: usize) -> Self {
        Self {
            class_of: vec![0; num_states],
            num_classes: if num_states == 0 { 0 } else { 1 },
        }
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Helper: hash any Hash + Eq type into a u64.
/// We use this in some ordering keys to make ordering deterministic but cheap.
fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

// ============================================================================
// DWA MINIMIZATION (DETERMINISTIC WEIGHTED AUTOMATA)
// ============================================================================
//
// High-level design (analogy with rustfst weighted acceptor minimization):
//
// 1. "Connect": trim unreachable and dead-end states.
// 2. Push weights:
//
//    - For every non-start state v with a state_weight W_v, push W_v onto all
//      incoming transitions and clear v.state_weight. This is semantics-
//      preserving because state weights are always intersected when entering
//      v; moving that intersection to the incoming arcs preserves the number
//      of times W_v is intersected along any path.
//
//    - For the start state s0 with state_weight W_0, push W_0 into *all final
//      weights* reachable from s0, and clear s0.state_weight. For any
//      accepting path from s0, the path weight was:
//          W_0 & (∧_edges w_e) & final_weight(f)
//      After pushing: state_weight is None, final_weight'(f) = final_weight(f) & W_0,
//      so the new path weight is:
//          (∧_edges w_e) & final_weight(f) & W_0
//      which is identical by associativity/commutativity of intersection.
//
//    After this step, *all* state_weight fields are None; all gating has been
//    pushed onto transitions or final weights. This is an exact preservation
//    of semantics, not an approximation.
//
// 3. Cyclic minimization via partition refinement:
//
//    - We look for an equivalence relation ~ on states capturing equality of
//      "right languages" (i.e. same function word -> weight when starting
//      from that state).
//
//    - We define a sequence of partitions P_0, P_1, ... where each refinement
//      refines states by their *signatures* w.r.t the current partition.
//
//    - A DWA state signature consists of:
//        * its final_weight;
//        * for each outgoing label `a`, a triple:
//              (label = a, dest_class = class_of[target], weight = w).
//
//      Determinism ensures that from a given state and label there is at most
//      one destination; we capture that exactly.
//
//    - Two states with identical signatures under partition P_k must be in the
//      same class in P_{k+1}. We iterate until P_{k+1} == P_k. This is the
//      standard partition-refinement scheme analogous to Hopcroft’s DFA
//      minimization, generalized to bitset weights.
//
// 4. Quotient construction:
//
//    - Given the final stable partition P, we build a quotient DWA where
//      each equivalence class becomes a single state.
//
//    - For each new state (class C), we *union* the contributions of all old
//      states in C:
//        * state_weight is already None (after pushing), so no contribution;
//        * final_weight is the union (bitset OR) of final weights of members;
//        * for transitions: for each label a and destination class D, we union
//          all weights w from any member of C that had such a transition.
//          Since the semantics of a DWA is: for a given label, there is at
//          most one successor, and the path weight is the intersection of
//          all weights along the path, merging transitions via union over
//          alternative predecessors preserves the language of the *start*
//          state.
//
//    - Because all states in the same class had the same right-language,
//      the language of the quotient start state equals that of the original
//      start state. Thus, the overall recognized function word -> Weight is
//      preserved exactly.
//
// 5. Trim again.
//
// Note: our Weight = SimpleBitset semiring has ⊕ = union and ⊗ = intersection.
// It is idempotent and commutative, but lacks multiplicative inverses, so we
// cannot implement OpenFst-style potential-based weight pushing. We carefully
// restrict ourselves to transformations that are provably semantics-preserving.

// ---------------------------------------------------------------------------
// DWA state signatures
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaTransitionSig {
    label: i16,
    dest_class: usize,
    weight: Weight,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaStateSignature {
    final_weight: Option<Weight>,
    outgoing: Vec<DwaTransitionSig>,
}

impl DwaStateSignature {
    fn from_state(state_id: StateID, states: &DWAStates, classes: &[usize]) -> Self {
        let st = &states[state_id];

        // Since DWA is deterministic: transitions is BTreeMap<i16, StateID>,
        // and there is at most one destination per label.
        // We still normalize by sorting labels to get deterministic order.
        let mut outgoing = Vec::with_capacity(st.transitions.len());
        for (&label, &dest) in &st.transitions {
            let weight = st
                .trans_weights
                .get(&label)
                .cloned()
                .unwrap_or_else(Weight::all);
            let dest_class = classes[dest];
            outgoing.push(DwaTransitionSig {
                label,
                dest_class,
                weight,
            });
        }

        // Already deterministic ordering, but sort anyway in case of future changes.
        outgoing.sort_by(|a, b| {
            a.label
                .cmp(&b.label)
                .then_with(|| a.dest_class.cmp(&b.dest_class))
                .then_with(|| {
                    let ha = hash_value(&a.weight);
                    let hb = hash_value(&b.weight);
                    ha.cmp(&hb)
                })
        });

        DwaStateSignature {
            final_weight: st.final_weight.clone(),
            outgoing,
        }
    }
}

/// Compute the stable partition of DWA states under the forward-weighted
/// bisimulation defined by DwaStateSignature.
fn minimize_dwa_partition(states: &DWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition {
            class_of: vec![],
            num_classes: 0,
        };
    }

    let mut partition = Partition::new(n);

    loop {
        let mut sig_to_class: HashMap<DwaStateSignature, usize> = HashMap::new();
        let mut new_classes = vec![0; n];
        let mut next_class = 0;

        for s in 0..n {
            let sig = DwaStateSignature::from_state(s, states, &partition.class_of);
            let entry = sig_to_class.entry(sig).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
            new_classes[s] = *entry;
        }

        if new_classes == partition.class_of {
            partition.num_classes = next_class;
            return partition;
        }

        partition.class_of = new_classes;
        partition.num_classes = next_class;
    }
}

// ---------------------------------------------------------------------------
// DWA quotient construction and trimming
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct DwaStateBuilder {
    final_weight: Option<Weight>,
    // label -> (dest_new, weight)
    trans: BTreeMap<i16, (StateID, Weight)>,
}

impl DWA {
    /// Public entry point: simplifies the DWA in place.
    ///
    /// This method:
    /// - trims unreachable and dead-end states;
    /// - pushes state weights into transitions and finals (semantics-preserving);
    /// - performs cyclic minimization via partition refinement;
    /// - trims again.
    pub fn simplify(&mut self) {
        // Overall "changed" flag is not used by callers, but we keep it to
        // structure the algorithm and potentially help with future debugging.
        let _ = self.simplify_internal();
    }

    /// Returns true if the DWA changed structurally.
    fn simplify_internal(&mut self) -> bool {
        let mut changed = false;

        changed |= self.prune_unreachable();
        changed |= self.prune_dead_ends();

        // Weight pushing: remove state_weight by pushing it onto transitions
        // (for non-start states) and final weights (for start state).
        changed |= self.push_weights_into_transitions_and_finals();

        // Minimization: merge equivalent states via partition refinement.
        changed |= self.minimize_states();

        // Final cleanup: trim the quotient automaton.
        changed |= self.prune_unreachable();
        changed |= self.prune_dead_ends();

        changed
    }

    /// Weight pushing for DWA:
    ///
    /// 1. For every non-start state v with state_weight W_v:
    ///    - for every incoming transition u --a--> v with weight w(u,a,v),
    ///      replace it by w'(u,a,v) = w(u,a,v) & W_v;
    ///    - set v.state_weight = None.
    ///
    ///    Proof of correctness:
    ///    Consider any accepting path that enters v k times via edges
    ///      e_1, ..., e_k,
    ///    with weights w(e_i). Originally, each entry multiplies the path
    ///    weight by W_v (intersection). After pushing, each edge e_i carries
    ///    an extra factor W_v, so the path weight is intersected by W_v
    ///    exactly k times either way. Since intersection is idempotent and
    ///    associative/commutative, the overall path weight is unchanged.
    ///
    /// 2. For the start state s0 with state_weight W_0 (if any):
    ///    - for every state f with final_weight F_f, set final_weight'(f) =
    ///      F_f & W_0;
    ///    - set state_weight[s0] = None.
    ///
    ///    For any accepting path starting at s0 and ending at f, old weight:
    ///      W_0 & (∧_edges w_e) & F_f
    ///    New weight:
    ///      (∧_edges w_e) & (F_f & W_0)
    ///    which is equal by associativity/commutativity of intersection.
    ///
    /// After this function returns, all state_weight fields are None, and
    /// path weights are exactly preserved.
    fn push_weights_into_transitions_and_finals(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        let start = self.body.start_state;
        if start >= n {
            // Inconsistent automaton; we'll fix this in prune_unreachable.
            return false;
        }

        let mut changed = false;

        // Build predecessor lists: for each target v, a list of (u, label)
        // such that u --label--> v exists.
        let mut preds: Vec<Vec<(StateID, i16)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for (&label, &v) in &st.transitions {
                if v < n {
                    preds[v].push((u, label));
                }
            }
        }

        // 1. Push non-start state weights into incoming transitions.
        for v in 0..n {
            if v == start {
                continue;
            }
            if let Some(sw) = self.states[v].state_weight.take() {
                // If sw is "all", pushing has no effect; we can treat it as no-op.
                if sw.is_empty() {
                    // Weight is empty: all paths entering v become impossible.
                    // Pushing this onto incoming arcs keeps semantics identical.
                    changed = true;
                    for (u, label) in &preds[v] {
                        if let Some(w) = self.states[*u].trans_weights.get_mut(label) {
                            *w &= &sw;
                        }
                    }
                } else if sw != Weight::all() {
                    changed = true;
                    for (u, label) in &preds[v] {
                        if let Some(w) = self.states[*u].trans_weights.get_mut(label) {
                            *w &= &sw;
                        }
                    }
                }
                // In all cases, state_weight[v] is now None.
            }
        }

        // 2. Push start state weight into all final weights.
        if let Some(sw0) = self.states[start].state_weight.take() {
            if !sw0.is_empty() && sw0 != Weight::all() {
                changed = true;
                for st in &mut self.states.0 {
                    if let Some(ref mut fw) = st.final_weight {
                        *fw &= &sw0;
                    }
                }
            } else if sw0.is_empty() {
                // If start state weight is empty, all accepting words
                // get weight 0; intersecting final weights with empty is
                // consistent.
                changed = true;
                for st in &mut self.states.0 {
                    if let Some(ref mut fw) = st.final_weight {
                        *fw &= &sw0;
                    }
                }
            } else {
                // sw0 == all(): no effect, but we still clear state_weight.
                changed = true;
            }
        }

        changed
    }

    /// Minimize the DWA via partition refinement and quotient construction.
    /// Returns true if any states were merged.
    fn minimize_states(&mut self) -> bool {
        let n = self.states.len();
        if n <= 1 {
            return false;
        }

        let partition = minimize_dwa_partition(&self.states);
        if partition.num_classes() >= n {
            // No merges.
            return false;
        }

        self.rebuild_from_partition(partition);
        true
    }

    /// Rebuild the DWA from a stable partition of states.
    ///
    /// For each equivalence class C:
    ///   - final_weight is the union of final weights of all s ∈ C;
    ///   - for each label a and destination class D, the transition weight is
    ///     the union of weights of all transitions s --a--> t with
    ///     s ∈ C and t ∈ D.
    ///
    /// Because all states in a class have identical right-languages, this
    /// quotient DWA preserves the language of the original start state.
    fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Map each class to a new state id (0..num_classes-1).
        let mut class_to_new: HashMap<usize, StateID> = HashMap::new();
        let mut builders: Vec<DwaStateBuilder> = Vec::new();

        for s in 0..n {
            let c = partition.class_of[s];
            class_to_new.entry(c).or_insert_with(|| {
                let id = builders.len();
                builders.push(DwaStateBuilder::default());
                id
            });
        }

        // Aggregate contributions from states in the same class.
        for old_s in 0..n {
            let c = partition.class_of[old_s];
            let new_id = class_to_new[&c];
            let builder = &mut builders[new_id];
            let st = &self.states[old_s];

            // state_weight should already be None after weight pushing.
            debug_assert!(st.state_weight.is_none());

            if let Some(ref fw) = st.final_weight {
                if !fw.is_empty() {
                    match &mut builder.final_weight {
                        Some(existing) => *existing |= fw,
                        None => builder.final_weight = Some(fw.clone()),
                    }
                }
            }

            for (&label, &dest) in &st.transitions {
                let w = st
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                if w.is_empty() {
                    continue;
                }
                let dest_class = partition.class_of[dest];
                let dest_new = class_to_new[&dest_class];

                use std::collections::btree_map::Entry;
                match builder.trans.entry(label) {
                    Entry::Vacant(e) => {
                        e.insert((dest_new, w));
                    }
                    Entry::Occupied(mut e) => {
                        let (existing_dest, existing_w) = e.get_mut();
                        debug_assert_eq!(
                            *existing_dest, dest_new,
                            "Determinism violated while rebuilding DWA: multiple destinations for label {} in class {}",
                            label, c
                        );
                        *existing_w |= &w;
                    }
                }
            }
        }

        // Construct new states.
        let mut new_states = DWAStates::default();
        for _ in 0..builders.len() {
            new_states.add_state();
        }

        for (new_id, builder) in builders.into_iter().enumerate() {
            let st = &mut new_states[new_id];
            st.state_weight = None;
            st.final_weight = builder.final_weight;

            st.transitions.clear();
            st.trans_weights.clear();
            for (label, (dest_new, weight)) in builder.trans {
                st.transitions.insert(label, dest_new);
                st.trans_weights.insert(label, weight);
            }
        }

        // Map start state.
        let start_class = partition.class_of[self.body.start_state];
        let new_start = class_to_new[&start_class];

        self.states = new_states;
        self.body.start_state = new_start;
    }

    /// Remove states that are not reachable from the start state.
    ///
    /// This is the analogue of `connect`'s "accessible" part.
    /// Returns true if any states were removed or the start was fixed.
    fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        if self.body.start_state >= n {
            // Invalid start; reset to a singleton empty DWA.
            let changed = n > 0;
            if changed {
                self.states = DWAStates::default();
                let start = self.states.add_state();
                self.body.start_state = start;
            }
            return changed;
        }

        let mut visited = vec![false; n];
        let mut q: VecDeque<StateID> = VecDeque::new();
        visited[self.body.start_state] = true;
        q.push_back(self.body.start_state);

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];
            for &v in st.transitions.values() {
                if v < n && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }

        if visited.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = DWAStates::default();

        for i in 0..n {
            if visited[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        // Reindex transitions.
        for st in &mut new_states.0 {
            let mut new_transitions: BTreeMap<i16, StateID> = BTreeMap::new();
            let mut new_trans_weights: BTreeMap<i16, Weight> = BTreeMap::new();
            for (&label, &old_dest) in &st.transitions {
                if old_dest < n && visited[old_dest] {
                    let new_dest = map[old_dest];
                    new_transitions.insert(label, new_dest);
                    if let Some(w) = st.trans_weights.get(&label) {
                        new_trans_weights.insert(label, w.clone());
                    }
                }
            }
            st.transitions = new_transitions;
            st.trans_weights = new_trans_weights;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }

    /// Remove states that cannot reach any state with a non-empty final weight.
    ///
    /// This is the analogue of `connect`'s "coaccessible" part.
    /// Returns true if any states were removed or the start was fixed.
    fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        let mut live = vec![false; n];
        let mut q: VecDeque<StateID> = VecDeque::new();
        let mut rev: Vec<Vec<StateID>> = vec![vec![]; n];

        for u in 0..n {
            let st = &self.states[u];
            for (&label, &v) in &st.transitions {
                if v >= n {
                    continue;
                }
                let w = st
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                if !w.is_empty() {
                    rev[v].push(u);
                }
            }
        }

        // Any state with non-empty final weight is live.
        for s in 0..n {
            if self.states[s]
                .final_weight
                .as_ref()
                .map_or(false, |w| !w.is_empty())
            {
                live[s] = true;
                q.push_back(s);
            }
        }

        while let Some(v) = q.pop_front() {
            for &u in &rev[v] {
                if !live[u] {
                    live[u] = true;
                    q.push_back(u);
                }
            }
        }

        // If the start state is not live, the language is empty.
        if self.body.start_state >= n || !live[self.body.start_state] {
            let changed = n > 0;
            if changed {
                self.states = DWAStates::default();
                let start = self.states.add_state();
                self.body.start_state = start;
            }
            return changed;
        }

        if live.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = DWAStates::default();

        for i in 0..n {
            if live[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            let mut new_transitions: BTreeMap<i16, StateID> = BTreeMap::new();
            let mut new_trans_weights: BTreeMap<i16, Weight> = BTreeMap::new();
            for (&label, &old_dest) in &st.transitions {
                if old_dest < n && live[old_dest] {
                    let new_dest = map[old_dest];
                    new_transitions.insert(label, new_dest);
                    if let Some(w) = st.trans_weights.get(&label) {
                        new_trans_weights.insert(label, w.clone());
                    }
                }
            }
            st.transitions = new_transitions;
            st.trans_weights = new_trans_weights;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }
}

// ============================================================================
// NWA MINIMIZATION (NON-DETERMINISTIC WEIGHTED AUTOMATA)
// ============================================================================
//
// High-level design:
//
// 1. Trim unreachable and dead-end states (connect-like).
//
// 2. Partition refinement to compute a weighted forward-bisimulation
//    equivalence:
//    - States are considered equivalent if they have the same "future"
//      behaviour in terms of:
//         * their final_weight; and
//         * epsilon transitions and labeled transitions, grouped by
//           (ArcLabel, weight) and *sets of destination classes*.
//
//    - This is the natural generalization of DFA minimization / unweighted
//      acceptor minimization to the non-deterministic, weighted setting.
//
//    - Crucially, **we do not determinize** the NWA. We work directly on the
//      NWA graph, and the partition refinement uses only local structure
//      relative to the current partition.
//
// 3. Quotient construction:
//    - For each equivalence class C, we build one new state whose:
//         * final_weight is the union (bitset OR) of the final weights of
//           members;
//         * epsilon transitions are constructed by, for each destination
//           class D, unioning all weights from any member in C to any state
//           in D;
//         * labeled transitions similarly.
//
//    - For a given word w, the weight recognized from the start state of the
//      quotient is exactly the union over weights of all accepting paths in
//      the original automaton. This holds because:
//         * Partition refinement ensures that all states in the same class
//           have identical right-languages;
//         * Merging them and unioning their contributions produces exactly
//           that same right-language as a function.
//    - We trim again after quotienting.
//
// NOTE ON WEIGHT PUSHING:
// For NWA we do not perform OpenFst-style weight pushing because the bitset
// semiring lacks multiplicative inverses. General reweighting formulas
// (using "potentials") need such inverses to preserve all path weights. For
// this reason, we restrict ourselves to:
//   - trimming unreachable/dead states;
//   - merging states via weighted forward bisimulation;
//   - merging duplicate transitions (implicitly through the quotient).
//
// These are all algebraically sound and preserve semantics exactly.

// ---------------------------------------------------------------------------
// NWA state signatures
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ArcLabel {
    Eps,
    Label(i16),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaTransitionSig {
    label: ArcLabel,
    weight: Weight,
    // Sorted list of destination classes.
    dest_classes: Vec<usize>,
}

impl NwaTransitionSig {
    fn sort_key(&self) -> (u8, i16, u64, u64) {
        let label_tag = match self.label {
            ArcLabel::Eps => 0,
            ArcLabel::Label(_) => 1,
        };
        let label_val = match self.label {
            ArcLabel::Eps => 0,
            ArcLabel::Label(v) => v,
        };
        let w_hash = hash_value(&self.weight);
        let dest_hash = hash_value(&self.dest_classes);
        (label_tag, label_val, w_hash, dest_hash)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaStateSignature {
    final_weight: Option<Weight>,
    outgoing: Vec<NwaTransitionSig>,
}

impl NwaStateSignature {
    fn from_state(state_id: NWAStateID, states: &NWAStates, classes: &[usize]) -> Self {
        let st = &states[state_id];

        // Map (ArcLabel, weight) -> set of destination classes.
        let mut temp: HashMap<(ArcLabel, Weight), BTreeSet<usize>> = HashMap::new();

        // Epsilon transitions.
        for &(dest, ref w) in &st.epsilons {
            if w.is_empty() {
                continue;
            }
            let key = (ArcLabel::Eps, w.clone());
            temp.entry(key)
                .or_insert_with(BTreeSet::new)
                .insert(classes[dest]);
        }

        // Labeled transitions.
        for (&lbl, targets) in &st.transitions {
            let label = ArcLabel::Label(lbl);
            for &(dest, ref w) in targets {
                if w.is_empty() {
                    continue;
                }
                let key = (label, w.clone());
                temp.entry(key)
                    .or_insert_with(BTreeSet::new)
                    .insert(classes[dest]);
            }
        }

        let mut outgoing = Vec::with_capacity(temp.len());
        for ((label, weight), dest_set) in temp {
            let dest_classes: Vec<usize> = dest_set.into_iter().collect();
            outgoing.push(NwaTransitionSig {
                label,
                weight,
                dest_classes,
            });
        }

        // Canonical ordering of outgoing signatures.
        outgoing.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));

        NwaStateSignature {
            final_weight: st.final_weight.clone(),
            outgoing,
        }
    }
}

/// Compute the stable partition of NWA states under weighted forward
/// bisimulation defined by NwaStateSignature.
fn minimize_nwa_partition(states: &NWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition {
            class_of: vec![],
            num_classes: 0,
        };
    }

    let mut partition = Partition::new(n);

    loop {
        let mut sig_to_class: HashMap<NwaStateSignature, usize> = HashMap::new();
        let mut new_classes = vec![0; n];
        let mut next_class = 0;

        for s in 0..n {
            let sig = NwaStateSignature::from_state(s, states, &partition.class_of);
            let entry = sig_to_class.entry(sig).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
            new_classes[s] = *entry;
        }

        if new_classes == partition.class_of {
            partition.num_classes = next_class;
            return partition;
        }

        partition.class_of = new_classes;
        partition.num_classes = next_class;
    }
}

// ---------------------------------------------------------------------------
// NWA quotient construction and trimming
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct NwaStateBuilder {
    final_weight: Option<Weight>,
    // Epsilon transitions: dest_new -> weight
    eps: BTreeMap<NWAStateID, Weight>,
    // Labeled transitions: label -> (dest_new -> weight)
    trans: BTreeMap<i16, BTreeMap<NWAStateID, Weight>>,
}

impl NWA {
    /// Simplify the NWA in place.
    ///
    /// This method:
    /// - trims unreachable and dead-end states;
    /// - computes a weighted forward-bisimulation partition;
    /// - builds the quotient NWA (no determinization);
    /// - trims again.
    ///
    /// Returns true if the automaton changed structurally.
    pub fn simplify(&mut self) -> bool {
        let mut changed = false;

        changed |= self.prune_unreachable();
        changed |= self.prune_dead_ends();

        let n = self.states.len();
        if n > 1 {
            let partition = minimize_nwa_partition(&self.states);
            if partition.num_classes() < n {
                self.rebuild_from_partition(partition);
                changed = true;
            }
        }

        changed |= self.prune_unreachable();
        changed |= self.prune_dead_ends();

        changed
    }

    fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Map each class to a new state id.
        let mut class_to_new: HashMap<usize, NWAStateID> = HashMap::new();
        let mut builders: Vec<NwaStateBuilder> = Vec::new();

        for s in 0..n {
            let c = partition.class_of[s];
            class_to_new.entry(c).or_insert_with(|| {
                let id = builders.len();
                builders.push(NwaStateBuilder::default());
                id
            });
        }

        // Aggregate information from states in the same class.
        for old_s in 0..n {
            let c = partition.class_of[old_s];
            let new_id = class_to_new[&c];
            let builder = &mut builders[new_id];
            let st = &self.states[old_s];

            if let Some(ref fw) = st.final_weight {
                if !fw.is_empty() {
                    match &mut builder.final_weight {
                        Some(existing) => *existing |= fw,
                        None => builder.final_weight = Some(fw.clone()),
                    }
                }
            }

            // Epsilon transitions
            for &(dest, ref w) in &st.epsilons {
                if w.is_empty() {
                    continue;
                }
                let dest_class = partition.class_of[dest];
                let new_dest = class_to_new[&dest_class];
                let entry = builder.eps.entry(new_dest).or_insert_with(Weight::zeros);
                *entry |= w;
            }

            // Labeled transitions
            for (&lbl, targets) in &st.transitions {
                for &(dest, ref w) in targets {
                    if w.is_empty() {
                        continue;
                    }
                    let dest_class = partition.class_of[dest];
                    let new_dest = class_to_new[&dest_class];
                    let per_label = builder.trans.entry(lbl).or_insert_with(BTreeMap::new);
                    let entry = per_label.entry(new_dest).or_insert_with(Weight::zeros);
                    *entry |= w;
                }
            }
        }

        // Build the new NWAStates.
        let mut new_states = NWAStates::default();
        for _ in 0..builders.len() {
            new_states.add_state();
        }

        for (new_id, builder) in builders.into_iter().enumerate() {
            let st = &mut new_states[new_id];
            st.final_weight = builder.final_weight;

            st.epsilons.clear();
            for (dest_new, w) in builder.eps {
                if !w.is_empty() {
                    st.epsilons.push((dest_new, w));
                }
            }

            st.transitions.clear();
            for (lbl, dests_map) in builder.trans {
                let mut dests_vec: Vec<(NWAStateID, Weight)> = Vec::new();
                for (dest_new, w) in dests_map {
                    if !w.is_empty() {
                        dests_vec.push((dest_new, w));
                    }
                }
                if !dests_vec.is_empty() {
                    st.transitions.insert(lbl, dests_vec);
                }
            }
        }

        let start_class = partition.class_of[self.body.start_state];
        let new_start = class_to_new[&start_class];

        self.states = new_states;
        self.body.start_state = new_start;
    }

    /// Remove states not reachable from the start state.
    ///
    /// Returns true if any states were removed or the start was fixed.
    fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        if self.body.start_state >= n {
            let changed = n > 0;
            if changed {
                self.states = NWAStates::default();
                let start = self.states.add_state();
                self.body.start_state = start;
            }
            return changed;
        }

        let mut reachable = vec![false; n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        reachable[self.body.start_state] = true;
        q.push_back(self.body.start_state);

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];

            for &(v, ref w) in &st.epsilons {
                if v < n && !reachable[v] && !w.is_empty() {
                    reachable[v] = true;
                    q.push_back(v);
                }
            }

            for (_, targets) in &st.transitions {
                for &(v, ref w) in targets {
                    if v < n && !reachable[v] && !w.is_empty() {
                        reachable[v] = true;
                        q.push_back(v);
                    }
                }
            }
        }

        if reachable.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = NWAStates::default();

        for i in 0..n {
            if reachable[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            st.epsilons.retain(|(v, w)| *v < n && !w.is_empty());
            for (v, _) in &mut st.epsilons {
                *v = map[*v];
            }

            let mut new_transitions: BTreeMap<i16, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
            for (&lbl, targets) in &st.transitions {
                let mut new_targets = Vec::new();
                for &(v, ref w) in targets {
                    if v < n && !w.is_empty() && reachable[v] {
                        new_targets.push((map[v], w.clone()));
                    }
                }
                if !new_targets.is_empty() {
                    new_transitions.insert(lbl, new_targets);
                }
            }
            st.transitions = new_transitions;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }

    /// Remove states that cannot reach any state with non-empty final weight.
    ///
    /// Returns true if any states were removed or the start was fixed.
    fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        let mut live = vec![false; n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        let mut rev: Vec<Vec<NWAStateID>> = vec![vec![]; n];

        for p in 0..n {
            let st = &self.states[p];
            for &(t, ref w) in &st.epsilons {
                if t < n && !w.is_empty() {
                    rev[t].push(p);
                }
            }
            for (_, targets) in &st.transitions {
                for &(t, ref w) in targets {
                    if t < n && !w.is_empty() {
                        rev[t].push(p);
                    }
                }
            }
        }

        for s in 0..n {
            if self.states[s]
                .final_weight
                .as_ref()
                .map_or(false, |w| !w.is_empty())
            {
                if !live[s] {
                    live[s] = true;
                    q.push_back(s);
                }
            }
        }

        while let Some(v) = q.pop_front() {
            for &p in &rev[v] {
                if !live[p] {
                    live[p] = true;
                    q.push_back(p);
                }
            }
        }

        // If the start state is not live, the language is empty.
        if self.body.start_state >= n || !live[self.body.start_state] {
            let changed = n > 0;
            if changed {
                self.states = NWAStates::default();
                let start = self.states.add_state();
                self.body.start_state = start;
            }
            return changed;
        }

        if live.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = NWAStates::default();

        for i in 0..n {
            if live[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            st.epsilons
                .retain(|(v, w)| *v < n && !w.is_empty() && live[*v]);
            for (v, _) in &mut st.epsilons {
                *v = map[*v];
            }

            let mut new_transitions: BTreeMap<i16, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
            for (&lbl, targets) in &st.transitions {
                let mut new_targets = Vec::new();
                for &(v, ref w) in targets {
                    if v < n && !w.is_empty() && live[v] {
                        new_targets.push((map[v], w.clone()));
                    }
                }
                if !new_targets.is_empty() {
                    new_transitions.insert(lbl, new_targets);
                }
            }
            st.transitions = new_transitions;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }
}
