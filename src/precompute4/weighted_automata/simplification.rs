// src/precompute4/weighted_automata/simplification.rs

#![allow(dead_code)]

use super::common::{StateID, Weight};
use super::dwa::{DWAState, DWAStates, DWA};
use super::nwa::NWA;
use std::collections::{BTreeMap, HashMap, VecDeque};

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
    pub fn simplify(&mut self) -> bool {
        let initial_n = self.states.len();
        // The core pipeline: connect, then minimize.
        self.connect();
        let changed = self.minimize();
        // Final cleanup after minimization.
        self.connect();
        changed || self.states.len() < initial_n
    }
}

// --- Core Minimization Pipeline ---

impl DWA {
    /// The main minimization pipeline, adapting the rustfst approach for idempotent weighted acceptors.
    fn minimize(&mut self) -> bool {
        let initial_n = self.states.len();
        if initial_n == 0 {
            return false;
        }

        // 1. PREPROCESSING: Convert the weighted DWA to an unweighted graph representation.
        let (encoded_graph, encode_table) = encode(&self.states);

        // 2. CORE MINIMIZATION: Use Hopcroft's algorithm for all cases.
        let partition = cyclic_minimize::minimize(&encoded_graph);

        // The encoded graph includes a super-final state. If no states were merged,
        // the number of classes will equal the number of original states + 1.
        if partition.num_classes() == initial_n + 1 {
            return false; // No states were merged.
        }

        // 3. POST-PROCESSING: Reconstruct the DWA from the partition.
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

// --- Helper Structs for Minimization ---

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
    let super_final_class = partition.get_class_id(super_final_id);

    let mut class_to_new_id = HashMap::new();
    let mut representatives = BTreeMap::new();
    for i in 0..=n {
        representatives
            .entry(partition.get_class_id(i))
            .or_insert(i);
    }

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

        // Correctly merge state weights by taking the union of all states in the class.
        let mut new_state_weight = None;
        for old_id in partition.iter(*class_id) {
            if old_id < n {
                if let Some(sw) = &dwa.states[old_id].state_weight {
                    if let Some(nsw) = &mut new_state_weight {
                        *nsw |= sw;
                    } else {
                        new_state_weight = Some(sw.clone());
                    }
                }
            }
        }
        new_states[new_id].state_weight = new_state_weight;

        // Reconstruct transitions from the representative state.
        for (&encoded_label, &target) in &encoded_graph.transitions[rep_id] {
            let target_class = partition.get_class_id(target);
            let (original_label, original_weight) = &table.vec[encoded_label as usize];

            if target_class == super_final_class {
                new_states[new_id].final_weight = Some(original_weight.clone());
            } else {
                let new_target_id = class_to_new_id[&target_class];
                new_states[new_id]
                    .transitions
                    .insert(*original_label, new_target_id);
                new_states[new_id]
                    .trans_weights
                    .insert(*original_label, original_weight.clone());
            }
        }
    }

    let start_class = partition.get_class_id(dwa.body.start_state);
    let new_start = class_to_new_id[&start_class];

    (new_states, new_start)
}

// --- Partition Data Structure (Ported from rustfst) ---

#[derive(Debug, Clone)]
struct Element {
    class_id: usize,
    yes: usize,
    next_element: i32,
    prev_element: i32,
}

impl Default for Element {
    fn default() -> Self {
        Self {
            class_id: 0,
            yes: 0,
            next_element: 0,
            prev_element: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct Class {
    size: usize,
    yes_size: usize,
    no_head: i32,
    yes_head: i32,
}

impl Default for Class {
    fn default() -> Self {
        Self {
            size: 0,
            yes_size: 0,
            no_head: -1,
            yes_head: -1,
        }
    }
}

#[derive(Debug, Clone)]
struct Partition {
    elements: Vec<Element>,
    classes: Vec<Class>,
    visited_classes: Vec<usize>,
    yes_counter: usize,
}

struct PartitionIterator<'a> {
    partition: &'a Partition,
    class_id: usize,
    last_element_id: Option<i32>,
}

impl<'a> PartitionIterator<'a> {
    fn new(partition: &'a Partition, class_id: usize) -> Self {
        PartitionIterator {
            partition,
            class_id,
            last_element_id: None,
        }
    }
}

impl<'a> Iterator for PartitionIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let new_element_id = match self.last_element_id {
            None => self.partition.classes[self.class_id].no_head,
            Some(e) => self.partition.elements[e as usize].next_element,
        };
        if new_element_id < 0 {
            None
        } else {
            self.last_element_id = Some(new_element_id);
            Some(new_element_id as usize)
        }
    }
}

impl Partition {
    fn new(num_elements: usize) -> Self {
        Self {
            elements: vec![Element::default(); num_elements],
            classes: Vec::new(),
            visited_classes: Vec::new(),
            yes_counter: 1,
        }
    }

    fn add_class(&mut self) -> usize {
        let num_class = self.classes.len();
        self.classes.push(Class::default());
        num_class
    }

    fn add(&mut self, element_id: usize, class_id: usize) {
        let this_class = &mut self.classes[class_id];
        this_class.size += 1;

        let no_head = this_class.no_head;
        if no_head >= 0 {
            self.elements[no_head as usize].prev_element = element_id as i32;
        }
        this_class.no_head = element_id as i32;

        let this_element = &mut self.elements[element_id];
        this_element.class_id = class_id;
        this_element.yes = 0;
        this_element.next_element = no_head;
        this_element.prev_element = -1;
    }

    fn split_on(&mut self, element_id: usize) {
        let elt_class_id = self.elements[element_id].class_id;
        if self.elements[element_id].yes == self.yes_counter {
            return;
        }

        let elt_prev_elt = self.elements[element_id].prev_element;
        let elt_next_elt = self.elements[element_id].next_element;
        let this_class = &mut self.classes[elt_class_id];

        if elt_prev_elt >= 0 {
            self.elements[elt_prev_elt as usize].next_element = elt_next_elt;
        } else {
            this_class.no_head = elt_next_elt;
        }
        if elt_next_elt >= 0 {
            self.elements[elt_next_elt as usize].prev_element = elt_prev_elt;
        }

        if this_class.yes_head < 0 {
            self.visited_classes.push(elt_class_id);
        } else {
            self.elements[this_class.yes_head as usize].prev_element = element_id as i32;
        }

        self.elements[element_id].yes = self.yes_counter;
        self.elements[element_id].next_element = this_class.yes_head;
        self.elements[element_id].prev_element = -1;
        this_class.yes_head = element_id as i32;
        this_class.yes_size += 1;
    }

    fn finalize_split(&mut self, queue: &mut VecDeque<usize>) {
        let visited_classes = std::mem::take(&mut self.visited_classes);
        for &visited_class in &visited_classes {
            let yes_size = self.classes[visited_class].yes_size;
            let no_size = self.classes[visited_class].size - yes_size;

            if no_size == 0 {
                self.classes[visited_class].no_head = self.classes[visited_class].yes_head;
            } else {
                let new_class_id = self.add_class();
                let (smaller_class, larger_class) = if no_size < yes_size {
                    (new_class_id, visited_class)
                } else {
                    (visited_class, new_class_id)
                };
                if !queue.contains(&larger_class) {
                    queue.push_back(smaller_class);
                }

                if no_size < yes_size {
                    self.classes[new_class_id].no_head = self.classes[visited_class].no_head;
                    self.classes[new_class_id].size = no_size;
                    self.classes[visited_class].no_head = self.classes[visited_class].yes_head;
                    self.classes[visited_class].size = yes_size;
                } else {
                    self.classes[new_class_id].no_head = self.classes[visited_class].yes_head;
                    self.classes[new_class_id].size = yes_size;
                    self.classes[visited_class].size = no_size;
                }

                let mut e = self.classes[new_class_id].no_head;
                while e >= 0 {
                    self.elements[e as usize].class_id = new_class_id;
                    e = self.elements[e as usize].next_element;
                }
            }
            self.classes[visited_class].yes_head = -1;
            self.classes[visited_class].yes_size = 0;
        }
        self.yes_counter += 1;
    }

    fn get_class_id(&self, element_id: usize) -> usize {
        self.elements[element_id].class_id
    }
    fn get_class_size(&self, class_id: usize) -> usize {
        self.classes[class_id].size
    }
    fn num_classes(&self) -> usize {
        self.classes.len()
    }
    fn iter(&self, class_id: usize) -> PartitionIterator {
        PartitionIterator::new(self, class_id)
    }
}

// --- Cyclic Minimization Module (Hopcroft's Algorithm) ---
mod cyclic_minimize {
    use super::*;

    pub fn minimize(graph: &EncodedGraph) -> Partition {
        let n = graph.transitions.len();
        let mut partition = Partition::new(n);
        let mut worklist = VecDeque::new();

        // Initial partition: final vs non-final
        let final_class = partition.add_class();
        let non_final_class = partition.add_class();
        for i in 0..n {
            if graph.is_final[i] {
                partition.add(i, final_class);
            } else {
                partition.add(i, non_final_class);
            }
        }

        if partition.get_class_size(final_class) < partition.get_class_size(non_final_class) {
            worklist.push_back(final_class);
        } else {
            worklist.push_back(non_final_class);
        }

        let mut rev_adj: Vec<BTreeMap<i16, Vec<StateID>>> = vec![BTreeMap::new(); n];
        for i in 0..n {
            for (&label, &target) in &graph.transitions[i] {
                rev_adj[target].entry(label).or_default().push(i);
            }
        }

        while let Some(class_id) = worklist.pop_front() {
            let mut splitters: BTreeMap<i16, Vec<StateID>> = BTreeMap::new();
            for state_id in partition.iter(class_id) {
                for (label, sources) in &rev_adj[state_id] {
                    splitters.entry(*label).or_default().extend(sources);
                }
            }

            for (_, sources) in splitters {
                for source in sources {
                    partition.split_on(source);
                }
                partition.finalize_split(&mut worklist);
            }
        }
        partition
    }
}

// --- Helper Implementations ---

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

// --- Pruning and Connectivity ---

impl DWA {
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
        let mut new_states_vec: Vec<DWAState> = Vec::with_capacity(num_reachable);
        for i in 0..n {
            if visited[i] {
                map[i] = new_states_vec.len();
                new_states_vec.push(self.states[i].clone());
            }
        }
        for st in &mut new_states_vec {
            for tgt in st.transitions.values_mut() {
                *tgt = map[*tgt];
            }
        }
        self.states.0 = new_states_vec;
        if num_reachable > 0 {
            self.body.start_state = map[self.body.start_state];
        } else {
            self.states.0.clear();
            self.body.start_state = self.states.add_state();
        }
        true
    }

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
        if n == 0 {
            return false;
        }
        let mut reachable = vec![false; n];
        let mut q = VecDeque::new();

        if self.body.start_state < n {
            reachable[self.body.start_state] = true;
            q.push_back(self.body.start_state);
        } else {
            let changed = n > 0;
            if changed {
                self.states.0.clear();
                self.body.start_state = self.states.add_state();
            }
            return changed;
        }

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];
            for (v, _) in &st.epsilons {
                if *v < n && !reachable[*v] {
                    reachable[*v] = true;
                    q.push_back(*v);
                }
            }
            for (_, targets) in &st.transitions {
                for (v, _) in targets {
                    if *v < n && !reachable[*v] {
                        reachable[*v] = true;
                        q.push_back(*v);
                    }
                }
            }
        }

        let num_reachable = reachable.iter().filter(|&&b| b).count();
        if num_reachable == n {
            return false;
        }

        let mut remap = vec![usize::MAX; n];
        let mut new_states_vec = Vec::with_capacity(num_reachable);
        for i in 0..n {
            if reachable[i] {
                remap[i] = new_states_vec.len();
                new_states_vec.push(self.states[i].clone());
            }
        }

        for st in &mut new_states_vec {
            st.epsilons.iter_mut().for_each(|(v, _)| *v = remap[*v]);
            st.transitions
                .values_mut()
                .for_each(|targets| for (v, _) in targets { *v = remap[*v]; });
        }

        self.states.0 = new_states_vec;
        self.body.start_state = remap[self.body.start_state];
        true
    }

    fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }
        let mut live = vec![false; n];
        let mut q = VecDeque::new();
        let mut rev_adj: Vec<Vec<StateID>> = vec![vec![]; n];

        for p in 0..n {
            let st = &self.states[p];
            for &(t, ref w) in &st.epsilons {
                if t < n && !w.is_empty() {
                    rev_adj[t].push(p);
                }
            }
            for (_, targets) in &st.transitions {
                for &(t, ref w) in targets {
                    if t < n && !w.is_empty() {
                        rev_adj[t].push(p);
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
            for &p in &rev_adj[v] {
                if !live[p] {
                    live[p] = true;
                    q.push_back(p);
                }
            }
        }

        if self.body.start_state >= n || !live[self.body.start_state] {
            let changed = n > 0;
            if changed {
                self.states.0.clear();
                self.body.start_state = self.states.add_state();
            }
            return changed;
        }

        let num_live = live.iter().filter(|&&b| b).count();
        if num_live == n {
            return false;
        }

        let mut remap = vec![usize::MAX; n];
        let mut new_states_vec = Vec::with_capacity(num_live);
        for i in 0..n {
            if live[i] {
                remap[i] = new_states_vec.len();
                new_states_vec.push(self.states[i].clone());
            }
        }

        for st in &mut new_states_vec {
            st.epsilons.retain(|(v, _)| *v < n && live[*v]);
            st.epsilons.iter_mut().for_each(|(v, _)| *v = remap[*v]);
            st.transitions.values_mut().for_each(|targets| {
                targets.retain(|(v, _)| *v < n && live[*v]);
                targets.iter_mut().for_each(|(v, _)| *v = remap[*v]);
            });
            st.transitions.retain(|_, targets| !targets.is_empty());
        }

        self.states.0 = new_states_vec;
        self.body.start_state = remap[self.body.start_state];
        true
    }
}