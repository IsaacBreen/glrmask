use super::common::{StateID, Weight};
use super::dwa::{DWABody, DWAState, DWAStates, DWA};
use super::nwa::NWA;
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use range_set_blaze::RangeSetBlaze;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::time::Instant;

// --- Data structures for standard (unweighted) automata ---

#[derive(Debug, Default)]
struct UnweightedNFA {
    states: Vec<UnweightedNFAState>,
    start_state: usize,
    alphabet: BTreeSet<i16>,
}

#[derive(Debug, Default, Clone)]
struct UnweightedNFAState {
    transitions: BTreeMap<i16, Vec<usize>>,
    epsilons: Vec<usize>,
    is_final: bool,
    default_transitions: Vec<usize>,
}

#[derive(Debug)]
struct UnweightedDFA {
    states: Vec<UnweightedDFAState>,
    start_state: usize,
    alphabet: BTreeSet<i16>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct UnweightedDFAState {
    transitions: BTreeMap<i16, usize>,
    default_transition: Option<usize>,
    is_final: bool,
}

impl NWA {
    /// # Determinization of Weighted Automata
    ///
    /// This function converts a nondeterministic weighted automaton (NWA) into a deterministic
    /// one (DWA). The resulting DWA is guaranteed to be minimal.
    ///
    /// ## Mathematical Foundation
    ///
    /// A weighted automaton can be viewed as a machine that recognizes a "weighted language", where
    /// each word `w` is associated with a `Weight` (a set of abstract items). The core idea behind
    /// this determinization algorithm is to decompose the problem over the weight space into a set
    /// of independent problems over a partitioned weight space, and then combine the results.
    ///
    /// ### Hypothesis 1: Weight Partitioning
    /// The set of all weights appearing in the NWA can be partitioned into a minimal set of disjoint
    /// "atomic" weights, such that any original weight can be perfectly reconstructed by a union of
    /// some of these atomic weights.
    ///
    /// *Proof*: This is a classic set partitioning problem. Given a collection of sets (weights),
    /// we can find a partition of their union. This is achieved by considering all interval endpoints
    /// from all weights. The segments between sorted endpoints form elementary intervals. All elementary
    /// intervals that are contained in the exact same set of original weights are equivalent and can be
    /// merged to form one atomic weight partition. This process yields the desired minimal partition.
    ///
    /// ### Hypothesis 2: Per-Atom Automata
    /// For each atomic weight `A_i` from the partition, we can construct a standard (unweighted) NFA
    /// that accepts a word `w` if and only if the original NWA accepts `w` with a final weight `W`
    /// such that `W ∩ A_i` is non-empty. This NFA can then be determinized and minimized using
    /// standard algorithms to yield a minimal DFA, `D_i`.
    ///
    /// *Proof*: The construction is straightforward. An edge exists in the unweighted NFA if the
    /// corresponding edge in the NWA has a weight that overlaps with `A_i`. A state is final if its
    /// final weight overlaps with `A_i`. Standard subset construction and DFA minimization algorithms
    /// (like Hopcroft's or partition refinement) can then be applied.
    ///
    /// ### Hypothesis 3: Product Construction and Correctness
    /// A minimal DWA can be constructed as the product of all the minimal DFAs `D_i`. A state in the
    /// DWA corresponds to a tuple of states, one from each `D_i`.
    ///
    /// *   **States**: `S = (s_1, s_2, ..., s_k)` where `s_i` is a state in `D_i`.
    /// *   **Transitions**: `δ(S, c) = (δ_1(s_1, c), ..., δ_k(s_k, c))`. Transition weights are `ALL`.
    /// *   **Final Weights**: The final weight of a DWA state `S` is the union of all atomic weights
    ///     `A_i` for which the corresponding component state `s_i` is a final state in `D_i`.
    ///
    /// *Proof of Correctness*: Let `W(w)` be the weight of a word `w` in the NWA. The product DWA,
    /// after processing `w`, reaches state `S_w = (s_w1, ..., s_wk)`. The final weight it assigns is
    /// `Union { A_i | s_wi is final }`. A state `s_wi` is final iff `w` is accepted by `D_i`, which
    /// by construction means `W(w) ∩ A_i` is non-empty. Since the `A_i` form a partition of the
    /// entire weight space, `Union { A_i | W(w) ∩ A_i ≠ ∅ }` is exactly `W(w)`.
    ///
    /// ### Hypothesis 4: Minimality
    /// The DWA constructed via the lazy product of minimal DFAs is itself minimal.
    ///
    /// *Proof*: Let `S = (s_1, ...)` and `S' = (s'_1, ...)` be two distinct reachable states in the
    /// product DWA. Then `s_j ≠ s'_j` for some `j`. Since `D_j` is minimal, there exists a
    /// distinguishing string `z` for `s_j` and `s'_j`. This `z` will lead one to a final state in `D_j`
    /// and the other to a non-final state. Consequently, the final weight of `δ(S, z)` will contain
    /// `A_j` while that of `δ(S', z)` will not. Thus, their final weights differ, making `z` a
    /// distinguishing string for `S` and `S'`. Since any two states are distinguishable, the automaton is minimal.
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();

        let mut nwa = self.clone();
        nwa.simplify();

        crate::debug!(4, "NWA::determinize_to_dwa stats after simplify:\n{}", nwa.stats());

        // 1. Partition all weights in the NWA into atomic, disjoint sets.
        let partition = partition_weights(&nwa);
        if partition.is_empty() {
            return DWA::new();
        }

        // 2. For each atomic weight, build a minimal unweighted DFA.
        let pb_dfas = if PROGRESS_BAR_ENABLED {
            Some(ProgressBar::new(partition.len() as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinize: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (Building minimal DFAs)")
                    .unwrap(),
            ))
        } else {
            None
        };

        let mut minimal_dfas = Vec::with_capacity(partition.len());
        for p_weight in &partition {
            let unweighted_nfa = build_nfa_for_partition(&nwa, p_weight);
            let dfa = determinize_nfa(unweighted_nfa);
            let minimal_dfa = minimize_dfa(dfa);
            minimal_dfas.push(minimal_dfa);
            if let Some(p) = &pb_dfas {
                p.inc(1);
            }
        }
        if let Some(p) = &pb_dfas {
            p.finish_with_message("Minimal DFAs built");
        }

        // 3. Construct the final DWA as the lazy product of the minimal DFAs.
        let result = build_product_dwa(minimal_dfas, partition);

        crate::debug!(4, "NWA::determinize_to_dwa result DWA stats:\n{}", result.stats());
        crate::debug!(4, "NWA::determinize_to_dwa took: {:?}", now.elapsed());
        result
    }
}

/// Collects all weights from an NWA and partitions them into a minimal set of disjoint "atomic" weights.
fn partition_weights(nwa: &NWA) -> Vec<Weight> {
    let mut all_weights = HashSet::new();
    for state in &nwa.states.0 {
        if let Some(w) = &state.final_weight {
            all_weights.insert(w.clone());
        }
        for (_, targets) in &state.transitions {
            for (_, w) in targets {
                all_weights.insert(w.clone());
            }
        }
        for (_, w) in &state.epsilons {
            all_weights.insert(w.clone());
        }
        for (_, w) in &state.default {
            all_weights.insert(w.clone());
        }
    }

    if all_weights.is_empty() {
        return Vec::new();
    }

    let mut endpoints = BTreeSet::new();
    for weight in &all_weights {
        for range in weight.rsb.ranges() {
            endpoints.insert(*range.start());
            if *range.end() < usize::MAX {
                endpoints.insert(*range.end() + 1);
            }
        }
    }

    let total_union: Weight = all_weights.iter().fold(Weight::zeros(), |acc, w| acc | w);
    if endpoints.is_empty() {
        return if total_union.is_empty() { Vec::new() } else { vec![total_union] };
    }

    let sorted_points: Vec<usize> = endpoints.into_iter().collect();
    let mut elementary_intervals = Vec::new();
    for i in 0..sorted_points.len().saturating_sub(1) {
        let start = sorted_points[i];
        let end = sorted_points[i + 1] - 1;
        if start <= end {
            elementary_intervals.push(start..=end);
        }
    }

    let weights_vec: Vec<_> = all_weights.iter().collect();
    let mut sig_to_intervals: BTreeMap<Vec<bool>, Vec<std::ops::RangeInclusive<usize>>> = BTreeMap::new();

    for interval in elementary_intervals {
        let midpoint = *interval.start();
        let signature: Vec<bool> = weights_vec.iter().map(|w| w.contains(midpoint)).collect();
        if signature.iter().any(|&b| b) {
            sig_to_intervals.entry(signature).or_default().push(interval);
        }
    }

    sig_to_intervals.into_values().map(|intervals| Weight::from_rsb(RangeSetBlaze::from_iter(intervals))).collect()
}

/// Constructs an unweighted NFA for a specific atomic weight partition.
fn build_nfa_for_partition(nwa: &NWA, p_weight: &Weight) -> UnweightedNFA {
    let num_states = nwa.states.len();
    let mut nfa = UnweightedNFA {
        states: vec![UnweightedNFAState::default(); num_states],
        start_state: nwa.body.start_state,
        alphabet: BTreeSet::new(),
    };

    for (i, nwa_state) in nwa.states.0.iter().enumerate() {
        nfa.states[i].is_final = nwa_state.final_weight.as_ref().map_or(false, |fw| !(fw & p_weight).is_empty());

        for (to, w) in &nwa_state.epsilons {
            if !(w & p_weight).is_empty() {
                nfa.states[i].epsilons.push(*to);
            }
        }
        for (label, targets) in &nwa_state.transitions {
            nfa.alphabet.insert(*label);
            for (to, w) in targets {
                if !(w & p_weight).is_empty() {
                    nfa.states[i].transitions.entry(*label).or_default().push(*to);
                }
            }
        }
        for (to, w) in &nwa_state.default {
            if !(w & p_weight).is_empty() {
                nfa.states[i].default_transitions.push(*to);
            }
        }
    }
    nfa
}

/// Computes the epsilon closure of a set of NFA states.
fn epsilon_closure(nfa_states: &BTreeSet<usize>, nfa: &UnweightedNFA) -> BTreeSet<usize> {
    let mut closure = nfa_states.clone();
    let mut worklist: Vec<_> = nfa_states.iter().copied().collect();
    while let Some(u) = worklist.pop() {
        if u < nfa.states.len() {
            for &v in &nfa.states[u].epsilons {
                if closure.insert(v) {
                    worklist.push(v);
                }
            }
        }
    }
    closure
}

/// Converts an unweighted NFA to a complete unweighted DFA using subset construction.
fn determinize_nfa(nfa: UnweightedNFA) -> UnweightedDFA {
    let mut dfa_states = Vec::new();
    let mut dfa_map: HashMap<BTreeSet<usize>, usize> = HashMap::new();
    let mut worklist = VecDeque::new();

    let start_closure = epsilon_closure(&BTreeSet::from([nfa.start_state]), &nfa);
    dfa_map.insert(start_closure.clone(), 0);
    dfa_states.push(UnweightedDFAState::default());
    worklist.push_back(start_closure);

    let mut sink_id = None;
    let get_sink = |sink_id: &mut Option<usize>, dfa_states: &mut Vec<UnweightedDFAState>, alphabet: &BTreeSet<i16>| -> usize {
        *sink_id.get_or_insert_with(|| {
            let new_id = dfa_states.len();
            let mut sink_state = UnweightedDFAState::default();
            for &l in alphabet {
                sink_state.transitions.insert(l, new_id);
            }
            sink_state.default_transition = Some(new_id);
            dfa_states.push(sink_state);
            new_id
        })
    };

    while let Some(current_nfa_states) = worklist.pop_front() {
        let current_dfa_id = dfa_map[&current_nfa_states];
        dfa_states[current_dfa_id].is_final = current_nfa_states.iter().any(|&s| nfa.states.get(s).map_or(false, |st| st.is_final));

        for &label in &nfa.alphabet {
            let mut next_nfa_states_raw = BTreeSet::new();
            for &nfa_state in &current_nfa_states {
                if let Some(targets) = nfa.states.get(nfa_state).and_then(|s| s.transitions.get(&label)) {
                    next_nfa_states_raw.extend(targets);
                }
            }
            let next_nfa_states = epsilon_closure(&next_nfa_states_raw, &nfa);

            if next_nfa_states.is_empty() {
                dfa_states[current_dfa_id].transitions.insert(label, get_sink(&mut sink_id, &mut dfa_states, &nfa.alphabet));
            } else {
                let next_dfa_id = *dfa_map.entry(next_nfa_states.clone()).or_insert_with(|| {
                    let new_id = dfa_states.len();
                    dfa_states.push(UnweightedDFAState::default());
                    worklist.push_back(next_nfa_states);
                    new_id
                });
                dfa_states[current_dfa_id].transitions.insert(label, next_dfa_id);
            }
        }

        let mut next_default_raw = BTreeSet::new();
        for &nfa_state in &current_nfa_states {
            if let Some(state) = nfa.states.get(nfa_state) {
                next_default_raw.extend(&state.default_transitions);
            }
        }
        let next_default_states = epsilon_closure(&next_default_raw, &nfa);
        if next_default_states.is_empty() {
            dfa_states[current_dfa_id].default_transition = Some(get_sink(&mut sink_id, &mut dfa_states, &nfa.alphabet));
        } else {
            let next_dfa_id = *dfa_map.entry(next_default_states.clone()).or_insert_with(|| {
                let new_id = dfa_states.len();
                dfa_states.push(UnweightedDFAState::default());
                worklist.push_back(next_default_states);
                new_id
            });
            dfa_states[current_dfa_id].default_transition = Some(next_dfa_id);
        }
    }

    UnweightedDFA { states: dfa_states, start_state: 0, alphabet: nfa.alphabet }
}

/// Minimizes a DFA using partition refinement.
fn minimize_dfa(dfa: UnweightedDFA) -> UnweightedDFA {
    let n = dfa.states.len();
    if n == 0 {
        return dfa;
    }

    let mut reachable_nodes = BTreeSet::new();
    if dfa.start_state < n {
        let mut q = VecDeque::from([dfa.start_state]);
        reachable_nodes.insert(dfa.start_state);
        while let Some(u) = q.pop_front() {
            for &v in dfa.states[u].transitions.values() {
                if reachable_nodes.insert(v) {
                    q.push_back(v);
                }
            }
            if let Some(v) = dfa.states[u].default_transition {
                if reachable_nodes.insert(v) {
                    q.push_back(v);
                }
            }
        }
    }

    let mut part = vec![0; n];
    let finals_reachable: BTreeSet<_> = reachable_nodes.iter().copied().filter(|&i| dfa.states[i].is_final).collect();
    if !finals_reachable.is_empty() {
        let non_finals_reachable: BTreeSet<_> = reachable_nodes.difference(&finals_reachable).copied().collect();
        for i in finals_reachable {
            part[i] = 0;
        }
        for i in non_finals_reachable {
            part[i] = 1;
        }
    }

    loop {
        let mut changed = false;
        let mut next_partition_id = 0;
        let mut new_part_mapping = vec![0; n];
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &i in &reachable_nodes {
            groups.entry(part[i]).or_default().push(i);
        }

        for group in groups.values() {
            let mut sig_to_subgroup: BTreeMap<_, Vec<usize>> = BTreeMap::new();
            for &i in group {
                let trans_sig: BTreeMap<_, _> = dfa.states[i].transitions.iter().map(|(k, v)| (*k, part[*v])).collect();
                let def_sig = dfa.states[i].default_transition.map(|t| part[t]);
                sig_to_subgroup.entry((trans_sig, def_sig)).or_default().push(i);
            }

            if sig_to_subgroup.len() > 1 {
                changed = true;
            }
            for subgroup in sig_to_subgroup.values() {
                for &i in subgroup {
                    new_part_mapping[i] = next_partition_id;
                }
                next_partition_id += 1;
            }
        }

        if !changed {
            break;
        }
        part = new_part_mapping;
    }

    let num_min_states = part.iter().max().map_or(0, |m| m + 1);
    let mut min_states = vec![UnweightedDFAState::default(); num_min_states];
    for &i in &reachable_nodes {
        let new_id = part[i];
        min_states[new_id].is_final = dfa.states[i].is_final;
        for (&label, &target) in &dfa.states[i].transitions {
            min_states[new_id].transitions.insert(label, part[target]);
        }
        if let Some(target) = dfa.states[i].default_transition {
            min_states[new_id].default_transition = Some(part[target]);
        }
    }

    UnweightedDFA { states: min_states, start_state: part[dfa.start_state], alphabet: dfa.alphabet }
}

/// Constructs the final DWA by taking the lazy product of the minimal DFAs.
fn build_product_dwa(minimal_dfas: Vec<UnweightedDFA>, partition: Vec<Weight>) -> DWA {
    let mut dwa = DWA::new();
    dwa.states.0.clear();

    if minimal_dfas.is_empty() || minimal_dfas.iter().any(|d| d.states.is_empty()) {
        return DWA::new();
    }

    let mut product_state_map: HashMap<Vec<usize>, StateID> = HashMap::new();
    let mut worklist = VecDeque::new();
    let alphabet: BTreeSet<i16> = minimal_dfas.iter().flat_map(|d| d.alphabet.iter().copied()).collect();

    let start_product_state: Vec<usize> = minimal_dfas.iter().map(|d| d.start_state).collect();
    let start_dwa_id = dwa.add_state();
    product_state_map.insert(start_product_state.clone(), start_dwa_id);
    worklist.push_back(start_product_state);
    dwa.body.start_state = start_dwa_id;

    while let Some(current_product_state) = worklist.pop_front() {
        let current_dwa_id = product_state_map[&current_product_state];

        let mut final_w = Weight::zeros();
        for (i, &dfa_state_id) in current_product_state.iter().enumerate() {
            if minimal_dfas[i].states[dfa_state_id].is_final {
                final_w |= &partition[i];
            }
        }
        if !final_w.is_empty() {
            dwa.states[current_dwa_id].final_weight = Some(final_w);
        }

        let next_default_product_state: Vec<usize> = current_product_state
            .iter()
            .enumerate()
            .map(|(i, &s)| minimal_dfas[i].states[s].default_transition.expect("DFA must be complete"))
            .collect();
        let default_target_dwa_id = *product_state_map.entry(next_default_product_state.clone()).or_insert_with(|| {
            let new_id = dwa.add_state();
            worklist.push_back(next_default_product_state);
            new_id
        });
        dwa.set_default_transition(current_dwa_id, default_target_dwa_id, Weight::all()).unwrap();

        for &label in &alphabet {
            let next_product_state: Vec<usize> = current_product_state
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let dfa = &minimal_dfas[i];
                    *dfa.states[s].transitions.get(&label).unwrap_or_else(|| dfa.states[s].default_transition.as_ref().unwrap())
                })
                .collect();

            if next_product_state == next_default_product_state {
                continue;
            }

            let next_dwa_id = *product_state_map.entry(next_product_state.clone()).or_insert_with(|| {
                let new_id = dwa.add_state();
                worklist.push_back(next_product_state);
                new_id
            });
            dwa.add_transition(current_dwa_id, label, next_dwa_id, Weight::all()).unwrap();
        }
    }
    dwa
}
