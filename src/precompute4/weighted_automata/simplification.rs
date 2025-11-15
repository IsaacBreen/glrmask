// src/precompute4/weighted_automata/simplification.rs

#![allow(dead_code)]

use super::common::{StateID, Weight};
use super::dwa::{DWABody, DWAState, DWAStates, DWA};
use super::nwa::NWA;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::cmp::Ordering;

// --- Top-Level API ---

impl NWA {
    /// Simplifies the NWA by determinizing it to a DWA, minimizing the DWA using a
    /// high-fidelity port of the rustfst minimization pipeline, and then converting
    /// it back to an NWA.
    pub fn simplify(&mut self) -> bool {
        let initial_n = self.states.len();
        let initial_body = self.body;

        // 1. Determinize the NWA to a DWA.
        let mut dwa = self.determinize();

        // 2. Simplify the DWA using the full minimization pipeline.
        dwa.simplify();

        // 3. Convert the simplified DWA back to an NWA.
        *self = NWA::from_dwa(&dwa);

        // 4. Final structural cleanup.
        let mut changed = self.prune_unreachable();
        changed |= self.prune_dead_ends();

        self.states.len() != initial_n || self.body != initial_body || changed
    }
}

impl DWA {
    /// Simplifies the DWA in-place by running a full minimization pipeline.
    pub fn simplify(&mut self) {
        // The core pipeline: connect, then minimize.
        self.connect();
        self.minimize();
        // Final cleanup after minimization.
        self.connect();
    }
}

// --- Core Minimization Pipeline ---

impl DWA {
    /// The main minimization pipeline, faithfully adapting the rustfst approach for weighted acceptors.
    fn minimize(&mut self) -> bool {
        let initial_n = self.states.len();
        if initial_n == 0 {
            return false;
        }

        // 1. PREPROCESSING
        // `push_weights`: Redistribute weights to normalize the automaton.
        push_weights(&mut self.states, &mut self.body);

        // `encode`: Convert the weighted DWA to an unweighted graph representation.
        let (encoded_graph, encode_table) = encode(&self.states);

        // 2. CORE MINIMIZATION
        // Dispatch to the appropriate algorithm based on graph structure.
        let partition = if is_acyclic_encoded(&encoded_graph) {
            acyclic_minimize::minimize(&encoded_graph)
        } else {
            cyclic_minimize::minimize(&encoded_graph)
        };

        if partition.num_classes() == initial_n {
            return false; // No states were merged.
        }

        // 3. POST-PROCESSING
        // `merge_states` and `decode` are combined into one step.
        let (new_states, new_start) =
            merge_and_decode(self, &partition, &encoded_graph, &encode_table);
        self.states = new_states;
        self.body.start_state = new_start;

        self.states.len() < initial_n
    }

    /// Removes states that are not on a successful path from start to a final state.
    fn connect(&mut self) -> bool {
        let changed = self.prune_unreachable();
        changed | self.prune_dead_ends()
    }
}

// --- Helper Structs ---

/// Helper for the encode/decode process.
struct EncodeTable {
    map: HashMap<(i16, Weight), i16>,
    vec: Vec<(i16, Weight)>,
}

/// An unweighted representation of the DWA for minimization algorithms.
struct EncodedGraph {
    transitions: Vec<BTreeMap<i16, StateID>>,
    is_final: Vec<bool>,
}

/// Manages state partitions during minimization.
#[derive(Debug)]
struct Partition {
    /// `classes[i]` stores the class ID of state `i`.
    classes: Vec<usize>,
    num_classes: usize,
}

// --- Pipeline Stages ---

/// Encodes a weighted DWA into an unweighted graph representation using a super-final state.
fn encode(states: &DWAStates) -> (EncodedGraph, EncodeTable) {
    let n = states.len();
    let super_final_id = n;
    let mut table = EncodeTable::new();
    let mut encoded_transitions = vec![BTreeMap::new(); n + 1];
    let mut is_final = vec![false; n + 1];
    is_final[super_final_id] = true;

    for i in 0..n {
        let state = &states[i];
        // Normal transitions
        for (label, target, weight) in state.iter_edges() {
            let encoded_label = table.encode(label, weight.clone());
            encoded_transitions[i].insert(encoded_label, target);
        }
        // Final weight as transition to super-final state
        if let Some(fw) = &state.final_weight {
            if !fw.is_empty() {
                // Use a reserved label for final weight transitions
                let encoded_label = table.encode(-1, fw.clone());
                encoded_transitions[i].insert(encoded_label, super_final_id);
            }
        }
    }
    let graph = EncodedGraph {
        transitions: encoded_transitions,
        is_final,
    };
    (graph, table)
}

/// Builds a new, minimized DWA from partitions and decodes weights in one pass.
fn merge_and_decode(
    dwa: &DWA,
    partition: &Partition,
    encoded_graph: &EncodedGraph,
    table: &EncodeTable,
) -> (DWAStates, StateID) {
    let n = dwa.states.len();
    let super_final_id = n;
    let super_final_class = partition.classes[super_final_id];

    let mut class_to_new_id = HashMap::new();
    let mut representatives = BTreeMap::new();
    // The partition includes the super-final state, so iterate up to n.
    for i in 0..=n {
        representatives.entry(partition.classes[i]).or_insert(i);
    }

    // Map old classes to temporary new IDs, excluding the super-final class.
    let mut temp_id_counter = 0;
    for (class_id, _) in &representatives {
        if *class_id != super_final_class {
            class_to_new_id.insert(*class_id, temp_id_counter);
            temp_id_counter += 1;
        }
    }

    let mut new_states = DWAStates::default();
    new_states
        .0
        .resize_with(class_to_new_id.len(), DWAState::default);

    for (class_id, &rep_id) in &representatives {
        if *class_id == super_final_class {
            continue;
        }

        let new_id = class_to_new_id[class_id];
        // Don't copy from the placeholder super-final state.
        if rep_id < n {
            new_states[new_id].state_weight = dwa.states[rep_id].state_weight.clone();
        }

        for (&encoded_label, &target) in &encoded_graph.transitions[rep_id] {
            let target_class = partition.classes[target];
            if target_class == super_final_class {
                // This transition represents a final weight.
                let (_label, weight) = &table.vec[encoded_label as usize];
                new_states[new_id].final_weight = Some(weight.clone());
            } else {
                // This is a normal transition.
                let new_target_id = class_to_new_id[&target_class];
                let (original_label, original_weight) = &table.vec[encoded_label as usize];

                new_states[new_id]
                    .transitions
                    .insert(*original_label, new_target_id);
                new_states[new_id]
                    .trans_weights
                    .insert(*original_label, original_weight.clone());
            }
        }
    }

    let start_class = partition.classes[dwa.body.start_state];
    let new_start = class_to_new_id[&start_class];

    (new_states, new_start)
}

/// Pushes weights towards the start state to normalize the automaton for minimization.
fn push_weights(states: &mut DWAStates, _body: &mut DWABody) {
    let distance = shortest_distance::calculate(states, true); // Reversed shortest distance
    reweight::apply(states, &distance);
}

/// Checks if the encoded graph is acyclic.
fn is_acyclic_encoded(graph: &EncodedGraph) -> bool {
    let n = graph.transitions.len();
    let mut visited = vec![0; n]; // 0: unvisited, 1: visiting, 2: visited
    for i in 0..n {
        if visited[i] == 0 {
            if dfs_cycle_check_encoded(i, graph, &mut visited) {
                return false; // Cycle detected
            }
        }
    }
    true
}

fn dfs_cycle_check_encoded(u: StateID, graph: &EncodedGraph, visited: &mut [u8]) -> bool {
    visited[u] = 1; // Mark as visiting
    for (_, &v) in &graph.transitions[u] {
        if v < graph.transitions.len() {
            if visited[v] == 1 {
                return true; // Cycle detected
            }
            if visited[v] == 0 {
                if dfs_cycle_check_encoded(v, graph, visited) {
                    return true;
                }
            }
        }
    }
    visited[u] = 2; // Mark as visited
    false
}

// --- Shortest Distance Module ---
mod shortest_distance {
    use super::*;

    pub fn calculate(states: &DWAStates, reverse: bool) -> Vec<Weight> {
        let n = states.len();
        let mut distance = vec![Weight::zeros(); n];
        let mut worklist = VecDeque::new();

        if reverse {
            let mut rev_adj: Vec<Vec<(StateID, Weight)>> = vec![vec![]; n];
            for u in 0..n {
                for (_label, v, weight) in states[u].iter_edges() {
                    if v < n {
                        rev_adj[v].push((u, weight.clone()));
                    }
                }
            }
            for i in 0..n {
                if let Some(fw) = &states[i].final_weight {
                    distance[i] = fw.clone();
                    worklist.push_back(i);
                }
            }
            while let Some(v) = worklist.pop_front() {
                for (u, weight) in &rev_adj[v] {
                    let new_dist = &distance[v] & weight;
                    if !new_dist.is_subset_of(&distance[*u]) {
                        distance[*u] |= &new_dist;
                        worklist.push_back(*u);
                    }
                }
            }
        } else {
            unimplemented!("Forward shortest distance is not implemented");
        }
        distance
    }
}

// --- Reweight (Weight Pushing) Module ---
mod reweight {
    use super::*;

    pub fn apply(states: &mut DWAStates, potential: &[Weight]) {
        let n = states.len();
        for i in 0..n {
            let state = &mut states[i];
            let inv_potential = !&potential[i];

            // Final weights are not reweighted to preserve equivalence.
            // The original logic `*fw |= &inv_potential` was incorrect.

            // Reweight transitions: w'(s, t) = (w(s, t) & potential(t)) | !potential(s)
            for (label, weight) in &mut state.trans_weights {
                if let Some(&target) = state.transitions.get(label) {
                    if target < n {
                        *weight &= &potential[target];
                        *weight |= &inv_potential;
                    }
                }
            }
        }
    }
}

// --- Acyclic Minimization Module ---
mod acyclic_minimize {
    use super::*;

    pub fn minimize(graph: &EncodedGraph) -> Partition {
        let n = graph.transitions.len();
        if n == 0 {
            return Partition::new(0, 0);
        }

        let heights = compute_heights(graph);
        let mut states_by_height: BTreeMap<usize, Vec<StateID>> = BTreeMap::new();
        for i in 0..n {
            states_by_height.entry(heights[i]).or_default().push(i);
        }

        let mut partition = Partition::new(n, 0);
        let mut num_classes = 0;

        for (_, mut states) in states_by_height {
            if states.is_empty() {
                continue;
            }
            states.sort_by(|&a, &b| state_comparator(a, b, graph, &partition));

            partition.classes[states[0]] = num_classes;
            for i in 1..states.len() {
                if state_comparator(states[i - 1], states[i], graph, &partition) != Ordering::Equal
                {
                    num_classes += 1;
                }
                partition.classes[states[i]] = num_classes;
            }
            num_classes += 1;
        }
        partition.num_classes = num_classes;
        partition
    }

    fn compute_heights(graph: &EncodedGraph) -> Vec<usize> {
        let n = graph.transitions.len();
        let mut heights = vec![0; n];
        let mut rev_adj: Vec<Vec<StateID>> = vec![vec![]; n];
        for i in 0..n {
            for (_, &target) in &graph.transitions[i] {
                rev_adj[target].push(i);
            }
        }
        let mut q = VecDeque::new();
        for i in 0..n {
            if graph.is_final[i] {
                heights[i] = 0;
                q.push_back(i);
            } else {
                heights[i] = usize::MAX;
            }
        }
        while let Some(u) = q.pop_front() {
            for &v in &rev_adj[u] {
                if heights[v] == usize::MAX { // A simple form of Dijkstra/BFS
                    heights[v] = heights[u] + 1;
                    q.push_back(v);
                }
            }
        }
        heights
    }

    fn state_comparator(a: StateID, b: StateID, graph: &EncodedGraph, p: &Partition) -> Ordering {
        if a == b { return Ordering::Equal; }
        graph.is_final[a].cmp(&graph.is_final[b])
            .then_with(|| graph.transitions[a].len().cmp(&graph.transitions[b].len()))
            .then_with(|| {
                let (ta, tb) = (&graph.transitions[a], &graph.transitions[b]);
                for ((la, na), (lb, nb)) in ta.iter().zip(tb.iter()) {
                    let cmp = la.cmp(lb).then_with(|| p.classes[*na].cmp(&p.classes[*nb]));
                    if cmp != Ordering::Equal { return cmp; }
                }
                Ordering::Equal
            })
    }
}

// --- Cyclic Minimization Module ---
mod cyclic_minimize {
    use super::*;
    use std::collections::BTreeSet;

    pub fn minimize(graph: &EncodedGraph) -> Partition {
        let n = graph.transitions.len();
        if n == 0 {
            return Partition::new(0, 0);
        }

        let mut num_classes = if graph.is_final.iter().any(|&f| f) { 2 } else { 1 };
        let mut partition = Partition::new(n, num_classes);
        if num_classes == 2 {
            for i in 0..n {
                if graph.is_final[i] { partition.classes[i] = 1; }
            }
        } else {
            return partition;
        }

        let mut rev_adj: Vec<BTreeMap<i16, Vec<StateID>>> = vec![BTreeMap::new(); n];
        for i in 0..n {
            for (&label, &target) in &graph.transitions[i] {
                rev_adj[target].entry(label).or_default().push(i);
            }
        }

        let mut worklist: VecDeque<usize> = (0..partition.num_classes).collect();

        while let Some(class_id) = worklist.pop_front() {
            let states_in_class: Vec<_> = (0..n).filter(|&i| partition.classes[i] == class_id).collect();
            if states_in_class.is_empty() { continue; }

            let mut incoming_by_label: BTreeMap<i16, BTreeSet<StateID>> = BTreeMap::new();
            for &state in &states_in_class {
                for (label, sources) in &rev_adj[state] {
                    for &source in sources {
                        incoming_by_label.entry(*label).or_default().insert(source);
                    }
                }
            }

            for (_, sources) in incoming_by_label {
                let mut sources_by_class: BTreeMap<usize, Vec<StateID>> = BTreeMap::new();
                for &source in &sources {
                    sources_by_class.entry(partition.classes[source]).or_default().push(source);
                }

                for (source_class_id, split_off) in sources_by_class {
                    let total_in_class_count = (0..n).filter(|&i| partition.classes[i] == source_class_id).count();
                    if split_off.len() == total_in_class_count { continue; }

                    let new_class_id = partition.num_classes;
                    partition.num_classes += 1;
                    for &state_to_move in &split_off {
                        partition.classes[state_to_move] = new_class_id;
                    }

                    if worklist.contains(&source_class_id) {
                        worklist.push_back(new_class_id);
                    } else {
                        if split_off.len() <= (total_in_class_count - split_off.len()) {
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
}

// --- Helper Implementations ---

impl EncodeTable {
    fn new() -> Self {
        Self { map: HashMap::new(), vec: Vec::new() }
    }
    fn encode(&mut self, label: i16, weight: Weight) -> i16 {
        let key = (label, weight);
        if let Some(&id) = self.map.get(&key) { return id; }
        let new_id = self.vec.len() as i16;
        self.vec.push(key.clone());
        self.map.insert(key, new_id);
        new_id
    }
}

impl Partition {
    fn new(n: usize, num_classes: usize) -> Self {
        Self { classes: vec![0; n], num_classes }
    }
    fn num_classes(&self) -> usize { self.num_classes }
}

// --- Pruning and Connectivity ---

impl DWA {
    fn prune_unreachable(&mut self) -> bool {
        if self.states.0.is_empty() { return false; }
        let n = self.states.0.len();
        let mut visited = vec![false; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        if self.body.start_state < n {
            visited[self.body.start_state] = true;
            q.push_back(self.body.start_state);
        } else {
            if n > 0 { self.states.0.clear(); self.body.start_state = self.states.add_state(); return true; }
            return false;
        }
        while let Some(u) = q.pop_front() {
            for (_, v, _) in self.states[u].iter_edges() {
                if v < n && !visited[v] { visited[v] = true; q.push_back(v); }
            }
        }
        let num_reachable = visited.iter().filter(|&&b| b).count();
        if num_reachable == n { return false; }
        let mut map = vec![usize::MAX; n];
        let mut new_states: Vec<DWAState> = Vec::with_capacity(num_reachable);
        for i in 0..n {
            if visited[i] { map[i] = new_states.len(); new_states.push(self.states[i].clone()); }
        }
        for st in &mut new_states {
            for tgt in st.transitions.values_mut() { *tgt = map[*tgt]; }
        }
        self.states.0 = new_states;
        if num_reachable > 0 { self.body.start_state = map[self.body.start_state]; }
        else { self.states.0.clear(); self.body.start_state = self.states.add_state(); }
        true
    }

    fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        let mut live = vec![false; n];
        let mut q_live: VecDeque<usize> = VecDeque::new();
        let mut rev_adj: Vec<Vec<usize>> = vec![vec![]; n];
        for i in 0..n {
            if self.states[i].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                live[i] = true;
                q_live.push_back(i);
            }
            for (_, v, w) in self.states[i].iter_edges() {
                if v < n && !w.is_empty() { rev_adj[v].push(i); }
            }
        }
        while let Some(u) = q_live.pop_front() {
            for &v in &rev_adj[u] {
                if !live[v] { live[v] = true; q_live.push_back(v); }
            }
        }
        let mut changed = false;
        for i in 0..n {
            let st = &mut self.states[i];
            let before = st.transitions.len();
            st.transitions.retain(|_, tgt| *tgt < n && live[*tgt]);
            if st.transitions.len() != before {
                changed = true;
                st.trans_weights.retain(|ch, _| st.transitions.contains_key(ch));
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
        let mut rev_adj: Vec<Vec<StateID>> = vec![vec![]; n];

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