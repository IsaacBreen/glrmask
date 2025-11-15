// src/precompute4/weighted_automata/simplification.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{StateID, Weight};
use super::dwa::{DWABody, DWAState, DWAStates, DWA};
use super::nwa::NWA;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use crate::precompute4::weighted_automata::NWAStateID;

impl NWA {
    /// Simplifies the NWA by determinizing it to a DWA, minimizing the DWA,
    /// and then converting it back to an NWA. This is a powerful simplification
    /// that often results in a much smaller and more efficient automaton.
    pub fn simplify(&mut self) -> bool {
        let initial_n = self.states.len();
        let initial_body = self.body;

        // 1. Determinize the NWA to a DWA.
        let mut dwa = self.determinize();

        // 2. Simplify the DWA using the minimization pipeline.
        dwa.simplify();

        // 3. Convert the simplified DWA back to an NWA.
        *self = NWA::from_dwa(&dwa);

        // 4. Run final structural cleanups on the resulting NWA.
        let mut changed = self.prune_unreachable();
        changed |= self.prune_dead_ends();

        self.states.len() != initial_n || self.body != initial_body || changed
    }
}

impl DWA {
    /// Simplifies the DWA in-place by running a pipeline of optimization passes
    /// until a fixpoint is reached.
    pub fn simplify(&mut self) {
        // Run a few passes of the simplification pipeline to reach a fixpoint,
        // as some optimizations can enable others.
        for _ in 0..3 {
            let mut changed = false;
            changed |= self.prune_unreachable();
            changed |= self.prune_dead_ends();
            changed |= self.minimize();
            if !changed {
                break;
            }
        }
        // Final cleanup pass.
        self.prune_unreachable();
    }
}

/// Helper struct for the encode/decode process.
struct EncodeTable {
    map: HashMap<(i16, Weight), i16>,
    vec: Vec<(i16, Weight)>,
}

impl EncodeTable {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            vec: Vec::new(),
        }
    }
    fn encode(&mut self, label: i16, weight: Weight) -> i16 {
        let key = (label, weight);
        if let Some(&id) = self.map.get(&key) {
            return id;
        }
        let new_id = self.vec.len() as i16;
        self.vec.push(key.clone());
        self.map.insert(key, new_id);
        new_id
    }
}

/// An unweighted representation of the DWA for minimization.
struct EncodedGraph {
    transitions: Vec<BTreeMap<i16, StateID>>,
    is_final: Vec<bool>,
}

/// Manages state partitions during minimization.
struct Partition {
    classes: Vec<usize>,
    num_classes: usize,
}
impl Partition {
    fn new(n: usize, num_classes: usize) -> Self {
        Self {
            classes: vec![0; n],
            num_classes,
        }
    }
    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

impl DWA {
    /// The core minimization pipeline for a DWA, inspired by rustfst's approach for weighted acceptors.
    fn minimize(&mut self) -> bool {
        let initial_n = self.states.len();
        if initial_n == 0 {
            return false;
        }

        // 1. PREPROCESSING
        // `connect` (prune_unreachable/dead_ends) is handled in the main simplify loop.
        // `push_weights` equivalent: propagate future weight constraints to relax edge weights.
        Self::push_weights_and_relax_edges(&mut self.states);

        // `encode`: Convert the weighted DWA to an unweighted DFA representation.
        let (encoded_graph, encode_table) = Self::encode(&self.states);

        // 2. CORE MINIMIZATION
        // `cyclic_minimize`: Use partition refinement to find equivalent states in the unweighted graph.
        let partition = Self::cyclic_minimize(&encoded_graph);

        if partition.num_classes() == initial_n {
            return false; // No states were merged.
        }

        // 3. POST-PROCESSING
        // `merge_states`: Build a new DWA based on the partitions, with encoded transitions.
        let (new_states, new_start) =
            Self::merge_states_from_partition(self, &partition, &encoded_graph);
        self.states = new_states;
        self.body.start_state = new_start;

        // `decode`: Restore the original labels and weights from the encode table.
        Self::decode(&mut self.states, &encode_table);

        self.states.len() < initial_n
    }

    /// Encodes a weighted DWA into an unweighted graph representation.
    fn encode(states: &DWAStates) -> (EncodedGraph, EncodeTable) {
        let n = states.len();
        let mut table = EncodeTable::new();
        let mut encoded_transitions = vec![BTreeMap::new(); n];
        let mut is_final = vec![false; n];

        for i in 0..n {
            let state = &states[i];
            is_final[i] = state.final_weight.as_ref().map_or(false, |w| !w.is_empty());

            for (label, target, weight) in state.iter_edges() {
                let encoded_label = table.encode(label, weight.clone());
                encoded_transitions[i].insert(encoded_label, target);
            }
        }

        let graph = EncodedGraph {
            transitions: encoded_transitions,
            is_final,
        };
        (graph, table)
    }

    /// Minimizes an unweighted graph using Hopcroft's partition refinement algorithm.
    fn cyclic_minimize(graph: &EncodedGraph) -> Partition {
        let n = graph.transitions.len();
        if n == 0 {
            return Partition::new(0, 0);
        }

        // 1. Initial partition: final vs. non-final states.
        let mut num_classes = 1;
        if graph.is_final.iter().any(|&f| f) {
            num_classes = 2;
        }
        let mut partition = Partition::new(n, num_classes);
        if num_classes == 2 {
            for i in 0..n {
                if graph.is_final[i] {
                    partition.classes[i] = 1;
                }
            }
        } else {
            return partition; // All states are in one class.
        }

        // 2. Build reverse adjacency list for the encoded graph.
        let mut rev_adj: Vec<BTreeMap<i16, Vec<StateID>>> = vec![BTreeMap::new(); n];
        for i in 0..n {
            for (&label, &target) in &graph.transitions[i] {
                rev_adj[target].entry(label).or_default().push(i);
            }
        }

        // 3. Worklist contains partitions to refine.
        let mut worklist: VecDeque<usize> = (0..partition.num_classes).collect();

        while let Some(class_id) = worklist.pop_front() {
            let states_in_class: Vec<_> = (0..n)
                .filter(|&i| partition.classes[i] == class_id)
                .collect();
            if states_in_class.is_empty() {
                continue;
            }

            // Find all incoming transitions to this class, grouped by label.
            let mut incoming_by_label: BTreeMap<i16, BTreeSet<StateID>> = BTreeMap::new();
            for &state in &states_in_class {
                for (label, sources) in &rev_adj[state] {
                    for &source in sources {
                        incoming_by_label.entry(*label).or_default().insert(source);
                    }
                }
            }

            for (_, sources) in incoming_by_label {
                // Group sources by their current partition.
                let mut sources_by_class: BTreeMap<usize, Vec<StateID>> = BTreeMap::new();
                for &source in &sources {
                    sources_by_class
                        .entry(partition.classes[source])
                        .or_default()
                        .push(source);
                }

                for (source_class_id, split_off) in sources_by_class {
                    let total_in_class_count = (0..n)
                        .filter(|&i| partition.classes[i] == source_class_id)
                        .count();
                    if split_off.len() == total_in_class_count {
                        continue; // No split needed.
                    }

                    let new_class_id = partition.num_classes;
                    partition.num_classes += 1;
                    for &state_to_move in &split_off {
                        partition.classes[state_to_move] = new_class_id;
                    }

                    // Update worklist: add the smaller of the two new partitions.
                    if let Some(pos) = worklist.iter().position(|&id| id == source_class_id) {
                        worklist.push_back(new_class_id);
                    } else {
                        if split_off.len() <= total_in_class_count / 2 {
                            worklist.push_back(new_class_id);
                        } else {
                            worklist.push_back(source_class_id);
                        }
                    }
                }
            }
        }
        partition
    }

    /// Builds a new, minimized DWA from the state partitions.
    fn merge_states_from_partition(
        dwa: &DWA,
        partition: &Partition,
        encoded_graph: &EncodedGraph,
    ) -> (DWAStates, StateID) {
        let mut new_states = DWAStates::default();
        let mut class_to_new_id = HashMap::new();
        let mut representatives = BTreeMap::new();

        for i in 0..dwa.states.len() {
            representatives.entry(partition.classes[i]).or_insert(i);
        }

        for (class_id, &rep_id) in &representatives {
            let new_id = new_states.add_state();
            class_to_new_id.insert(*class_id, new_id);
            new_states[new_id].final_weight = dwa.states[rep_id].final_weight.clone();
            new_states[new_id].state_weight = dwa.states[rep_id].state_weight.clone();
        }

        for (class_id, &rep_id) in &representatives {
            let new_id = class_to_new_id[class_id];
            for (&encoded_label, &target) in &encoded_graph.transitions[rep_id] {
                let target_class = partition.classes[target];
                let new_target_id = class_to_new_id[&target_class];
                new_states[new_id]
                    .transitions
                    .insert(encoded_label, new_target_id);
            }
        }

        let start_class = partition.classes[dwa.body.start_state];
        let new_start = class_to_new_id[&start_class];

        (new_states, new_start)
    }

    /// Restores original labels and weights to a minimized DWA.
    fn decode(states: &mut DWAStates, table: &EncodeTable) {
        for state in &mut states.0 {
            let mut new_transitions = BTreeMap::new();
            let mut new_weights = BTreeMap::new();
            for (&encoded_label, &target) in &state.transitions {
                let (original_label, original_weight) = &table.vec[encoded_label as usize];
                new_transitions.insert(*original_label, target);
                let weight_entry = new_weights
                    .entry(*original_label)
                    .or_insert_with(Weight::zeros);
                *weight_entry |= original_weight;
            }
            state.transitions = new_transitions;
            state.trans_weights = new_weights;
        }
    }

    /// Propagates future weight constraints backward to relax edge weights,
    /// making states with similar futures more likely to be identified as equivalent.
    fn push_weights_and_relax_edges(states: &mut DWAStates) -> bool {
        let n = states.len();
        if n == 0 {
            return false;
        }
        let mut upper_bounds: Vec<Weight> = Vec::with_capacity(n);
        for i in 0..n {
            let mut u = states[i].final_weight.clone().unwrap_or_else(Weight::zeros);
            for w in states[i].trans_weights.values() {
                u |= w;
            }
            if let Some(sw) = states[i].state_weight.as_ref() {
                u &= sw;
            }
            upper_bounds.push(u);
        }
        let complements: Vec<Weight> = upper_bounds.iter().map(|w| !w).collect();
        let mut changed = false;
        for i in 0..n {
            let st = &mut states[i];
            let keys: Vec<i16> = st.transitions.keys().copied().collect();
            for ch in keys {
                if let Some(&v) = st.transitions.get(&ch) {
                    if v < n {
                        if let Some(w) = st.trans_weights.get_mut(&ch) {
                            let new_w = &*w | &complements[v];
                            if new_w != *w {
                                *w = new_w;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        changed
    }

    /// Removes states that are not reachable from the start state.
    fn prune_unreachable(&mut self) -> bool {
        if self.states.0.is_empty() {
            return false;
        }
        let n = self.states.0.len();
        let mut visited = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        if self.body.start_state < n {
            visited[self.body.start_state] = true;
            q.push_back(self.body.start_state);
        } else {
            if n > 0 {
                self.states.0.clear();
                self.body.start_state = self.states.add_state();
                return true;
            }
            return false;
        }
        while let Some(u) = q.pop_front() {
            for (_, v, _) in self.states[u].iter_edges() {
                if v < n && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }
        let num_reachable = visited.iter().filter(|&&b| b).count();
        if num_reachable == n {
            return false;
        }
        let mut map = vec![usize::MAX; n];
        let mut new_states: Vec<DWAState> = Vec::with_capacity(num_reachable);
        for i in 0..n {
            if visited[i] {
                map[i] = new_states.len();
                new_states.push(self.states[i].clone());
            }
        }
        for st in &mut new_states {
            for tgt in st.transitions.values_mut() {
                *tgt = map[*tgt];
            }
        }
        self.states.0 = new_states;
        if num_reachable > 0 {
            self.body.start_state = map[self.body.start_state];
        } else {
            self.states.0.clear();
            self.body.start_state = self.states.add_state();
        }
        true
    }

    /// Removes states that cannot reach a final state.
    fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }
        let mut live = vec![false; n];
        let mut q_live: VecDeque<usize> = VecDeque::new();
        let mut rev_adj: Vec<Vec<usize>> = vec![vec![]; n];
        for i in 0..n {
            if self.states[i]
                .final_weight
                .as_ref()
                .map_or(false, |w| !w.is_empty())
            {
                live[i] = true;
                q_live.push_back(i);
            }
            for (_, v, w) in self.states[i].iter_edges() {
                if v < n && !w.is_empty() {
                    rev_adj[v].push(i);
                }
            }
        }
        while let Some(u) = q_live.pop_front() {
            for &v in &rev_adj[u] {
                if !live[v] {
                    live[v] = true;
                    q_live.push_back(v);
                }
            }
        }
        let mut changed = false;
        for i in 0..n {
            let st = &mut self.states[i];
            let before = st.transitions.len();
            st.transitions.retain(|_, tgt| *tgt < n && live[*tgt]);
            if st.transitions.len() != before {
                changed = true;
                st.trans_weights
                    .retain(|ch, _| st.transitions.contains_key(ch));
            }
        }
        changed
    }
}

impl NWA {
    fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut reachable = vec![false; n];
        let mut q = VecDeque::new();

        if self.body.start_state < n {
            reachable[self.body.start_state] = true;
            q.push_back(self.body.start_state);
        } else {
            let changed = n > 0;
            if changed { self.states.0.clear(); self.body.start_state = self.states.add_state(); }
            return changed;
        }

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];
            for (v, _) in &st.epsilons { if *v < n && !reachable[*v] { reachable[*v] = true; q.push_back(*v); } }
            for (_, targets) in &st.transitions { for (v, _) in targets { if *v < n && !reachable[*v] { reachable[*v] = true; q.push_back(*v); } } }
        }

        let num_reachable = reachable.iter().filter(|&&b| b).count();
        if num_reachable == n { return false; }

        let mut remap = vec![usize::MAX; n];
        let mut new_states_vec = Vec::with_capacity(num_reachable);
        for i in 0..n { if reachable[i] { remap[i] = new_states_vec.len(); new_states_vec.push(self.states[i].clone()); } }

        for st in &mut new_states_vec {
            st.epsilons.iter_mut().for_each(|(v, _)| *v = remap[*v]);
            st.transitions.values_mut().for_each(|targets| for (v, _) in targets { *v = remap[*v]; });
        }

        self.states.0 = new_states_vec;
        self.body.start_state = remap[self.body.start_state];
        true
    }

    fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut live = vec![false; n];
        let mut q = VecDeque::new();
        let mut rev_adj: Vec<Vec<NWAStateID>> = vec![vec![]; n];

        for p in 0..n {
            let st = &self.states[p];
            for &(t, ref w) in &st.epsilons { if t < n && !w.is_empty() { rev_adj[t].push(p); } }
            for (_, targets) in &st.transitions { for &(t, ref w) in targets { if t < n && !w.is_empty() { rev_adj[t].push(p); } } }
        }

        for s in 0..n { if self.states[s].final_weight.as_ref().map_or(false, |w| !w.is_empty()) { if !live[s] { live[s] = true; q.push_back(s); } } }

        while let Some(v) = q.pop_front() { for &p in &rev_adj[v] { if !live[p] { live[p] = true; q.push_back(p); } } }

        if self.body.start_state >= n || !live[self.body.start_state] {
            let changed = n > 0;
            if changed { self.states.0.clear(); self.body.start_state = self.states.add_state(); }
            return changed;
        }

        let num_live = live.iter().filter(|&&b| b).count();
        if num_live == n { return false; }

        let mut remap = vec![usize::MAX; n];
        let mut new_states_vec = Vec::with_capacity(num_live);
        for i in 0..n { if live[i] { remap[i] = new_states_vec.len(); new_states_vec.push(self.states[i].clone()); } }

        for st in &mut new_states_vec {
            st.epsilons.retain(|(v, _)| live[*v]);
            st.epsilons.iter_mut().for_each(|(v, _)| *v = remap[*v]);
            st.transitions.values_mut().for_each(|targets| { targets.retain(|(v, _)| live[*v]); targets.iter_mut().for_each(|(v, _)| *v = remap[*v]); });
            st.transitions.retain(|_, targets| !targets.is_empty());
        }

        self.states.0 = new_states_vec;
        self.body.start_state = remap[self.body.start_state];
        true
    }
}