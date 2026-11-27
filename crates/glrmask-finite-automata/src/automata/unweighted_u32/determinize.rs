//! Subset-construction determinization: NFA → DFA (unweighted).
//!
//! Implements the classical powerset / subset construction algorithm to convert
//! an `NFA` (with epsilon transitions) into a deterministic `DFA`.
//!
//! The caller is responsible for asserting acyclicity of the input NFA when
//! that invariant is required.

use std::collections::VecDeque;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::dfa::{DFA, Label};
use super::nfa::NFA;

fn subset_is_accepting(nfa: &NFA, subset: &[u32]) -> bool {
    subset
        .iter()
        .any(|&state| nfa.states[state as usize].is_accepting)
}

struct ClosureScratch {
    marks: Vec<u32>,
    epoch: u32,
    stack: Vec<u32>,
    touched: Vec<u32>,
}

impl ClosureScratch {
    fn new(num_states: usize) -> Self {
        Self {
            marks: vec![0; num_states],
            epoch: 0,
            stack: Vec::new(),
            touched: Vec::new(),
        }
    }

    fn next_epoch(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
        self.epoch
    }

    /// Return the sorted epsilon closure of `seeds` in the reusable `touched`
    /// buffer.  Dense epoch marks replace one BTreeSet allocation/insertion
    /// sequence per outgoing label.
    fn epsilon_closure<'a>(&'a mut self, nfa: &NFA, seeds: &[u32]) -> &'a [u32] {
        self.stack.clear();
        self.touched.clear();
        let epoch = self.next_epoch();

        for &seed in seeds {
            let index = seed as usize;
            if index >= self.marks.len() || self.marks[index] == epoch {
                continue;
            }
            self.marks[index] = epoch;
            self.stack.push(seed);
            self.touched.push(seed);
        }

        while let Some(state) = self.stack.pop() {
            for &target in &nfa.states[state as usize].epsilons {
                let index = target as usize;
                if index >= self.marks.len() || self.marks[index] == epoch {
                    continue;
                }
                self.marks[index] = epoch;
                self.stack.push(target);
                self.touched.push(target);
            }
        }
        self.touched.sort_unstable();
        &self.touched
    }
}

/// Singleton subsets need neither a heap key nor a hash lookup. Other
/// subsets share their sorted immutable key between the index and agenda.
enum Subset {
    Singleton(u32),
    Many(Arc<[u32]>),
}

impl Subset {
    fn states(&self) -> &[u32] {
        match self {
            Self::Singleton(state) => std::slice::from_ref(state),
            Self::Many(states) => states,
        }
    }
}

struct Subsets {
    singleton_ids: Vec<u32>,
    many_ids: FxHashMap<Arc<[u32]>, u32>,
    pending: VecDeque<(u32, Subset)>,
}

impl Subsets {
    fn new(n: usize) -> Self {
        Self {
            singleton_ids: vec![u32::MAX; n],
            many_ids: FxHashMap::default(),
            pending: VecDeque::new(),
        }
    }

    fn intern(&mut self, dfa: &mut DFA, states: &[u32]) -> u32 {
        if let [state] = states {
            let slot = &mut self.singleton_ids[*state as usize];
            if *slot != u32::MAX { return *slot; }
            let id = dfa.add_state();
            *slot = id;
            self.pending.push_back((id, Subset::Singleton(*state)));
            return id;
        }
        if let Some(&id) = self.many_ids.get(states) { return id; }
        let id = dfa.add_state();
        let key: Arc<[u32]> = Arc::from(states);
        self.many_ids.insert(Arc::clone(&key), id);
        self.pending.push_back((id, Subset::Many(key)));
        id
    }
}

struct TargetBuckets {
    dense: Option<Vec<Vec<u32>>>,
    sparse: FxHashMap<Label, Vec<u32>>,
    touched: Vec<Label>,
}

impl TargetBuckets {
    fn new(nfa: &NFA) -> Self {
        // Dense grammar-symbol IDs avoid hashing. Sparse or signed parser
        // labels use the same schedule with persistent hash buckets.
        let mut max = 0usize;
        let dense = nfa.states.iter().flat_map(|state| state.transitions.keys()).all(|&label| {
            if let Ok(label) = usize::try_from(label) {
                max = max.max(label);
                label < 4096
            } else {
                false
            }
        });
        Self {
            dense: dense.then(|| vec![Vec::new(); max + 1]),
            sparse: FxHashMap::default(),
            touched: Vec::new(),
        }
    }

    fn bucket(&mut self, label: Label) -> &mut Vec<u32> {
        if let Some(rows) = &mut self.dense {
            &mut rows[label as usize]
        } else {
            self.sparse.entry(label).or_default()
        }
    }

    fn append(&mut self, label: Label, targets: &[u32]) {
        if targets.is_empty() { return; }
        let bucket = self.bucket(label);
        let fresh = bucket.is_empty();
        bucket.extend_from_slice(targets);
        if fresh { self.touched.push(label); }
    }
}

fn connect(
    nfa: &NFA, dfa: &mut DFA, subsets: &mut Subsets,
    closure: &mut ClosureScratch, source: u32, label: Label, targets: &[u32],
) {
    let closed = match targets {
        [target] if nfa.states.get(*target as usize).is_some_and(|s| s.epsilons.is_empty()) => targets,
        _ => closure.epsilon_closure(nfa, targets),
    };
    if !closed.is_empty() {
        let target = subsets.intern(dfa, closed);
        dfa.add_transition(source, label, target);
    }
}

/// Template determinization retains its explicit acyclicity contract.
pub fn determinize(nfa: &NFA) -> DFA {
    assert!(nfa.compute_is_acyclic(), "determinize: input NFA is cyclic");
    determinize_general(nfa)
}

/// Exact breadth-first subset construction, including cyclic epsilon and
/// labelled graphs. The same closure and publication path handles all NFAs.
pub fn determinize_general(nfa: &NFA) -> DFA {
    if nfa.states.is_empty() || nfa.start_states.is_empty() {
        return DFA::new();
    }
    let mut dfa = DFA { states: Vec::new(), start_state: 0 };
    let mut subsets = Subsets::new(nfa.states.len());
    let mut closure = ClosureScratch::new(nfa.states.len());
    let mut buckets = None::<TargetBuckets>;
    let start = closure.epsilon_closure(nfa, &nfa.start_states);
    dfa.start_state = subsets.intern(&mut dfa, start);

    while let Some((source, subset)) = subsets.pending.pop_front() {
        let states = subset.states();
        if subset_is_accepting(nfa, states) {
            dfa.set_accepting(source, true);
        }
        // A singleton already has sorted labels. Directly consume its row,
        // without building label buckets or allocating a subset key.
        if let [state] = states {
            for (&label, targets) in &nfa.states[*state as usize].transitions {
                connect(nfa, &mut dfa, &mut subsets, &mut closure, source, label, targets);
            }
            continue;
        }
        let buckets = buckets.get_or_insert_with(|| TargetBuckets::new(nfa));
        for &state in states {
            for (&label, targets) in &nfa.states[state as usize].transitions {
                buckets.append(label, targets);
            }
        }
        buckets.touched.sort_unstable();
        for index in 0..buckets.touched.len() {
            let label = buckets.touched[index];
            let targets = buckets.bucket(label);
            connect(nfa, &mut dfa, &mut subsets, &mut closure, source, label, targets);
            targets.clear();
        }
        buckets.touched.clear();
    }
    dfa
}
#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

    use super::*;

    fn reference_epsilon_closure(nfa: &NFA, seeds: &[u32]) -> BTreeSet<u32> {
        let mut closed = BTreeSet::new();
        let mut queue: VecDeque<u32> = seeds.iter().copied().collect();
        while let Some(state) = queue.pop_front() {
            if closed.insert(state) {
                for &target in &nfa.states[state as usize].epsilons {
                    if !closed.contains(&target) {
                        queue.push_back(target);
                    }
                }
            }
        }
        closed
    }

    fn reference_determinize(nfa: &NFA) -> DFA {
        if nfa.states.is_empty() || nfa.start_states.is_empty() {
            return DFA::new();
        }
        let mut dfa = DFA {
            states: Vec::new(),
            start_state: 0,
        };
        let mut subset_map: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut worklist: VecDeque<Vec<u32>> = VecDeque::new();
        let start_key = reference_epsilon_closure(nfa, &nfa.start_states)
            .into_iter()
            .collect::<Vec<_>>();
        let start = dfa.add_state();
        dfa.start_state = start;
        subset_map.insert(start_key.clone(), start);
        worklist.push_back(start_key);

        while let Some(subset) = worklist.pop_front() {
            let state_id = subset_map[&subset];
            if subset_is_accepting(nfa, &subset) {
                dfa.set_accepting(state_id, true);
            }
            let mut labels = BTreeMap::<Label, BTreeSet<u32>>::new();
            for &state in &subset {
                for (&label, targets) in &nfa.states[state as usize].transitions {
                    labels.entry(label).or_default().extend(targets.iter().copied());
                }
            }
            for (label, targets) in labels {
                let seeds = targets.into_iter().collect::<Vec<_>>();
                let key = reference_epsilon_closure(nfa, &seeds)
                    .into_iter()
                    .collect::<Vec<_>>();
                if key.is_empty() {
                    continue;
                }
                let target = if let Some(&target) = subset_map.get(&key) {
                    target
                } else {
                    let target = dfa.add_state();
                    subset_map.insert(key.clone(), target);
                    worklist.push_back(key);
                    target
                };
                dfa.add_transition(state_id, label, target);
            }
        }
        dfa
    }

    fn generated_dag(seed: u32) -> NFA {
        let count = 8 + (seed % 17) as usize;
        let mut nfa = NFA::new_empty();
        for _ in 0..count {
            nfa.add_state();
        }
        nfa.start_states = vec![0];
        for source in 0..count {
            if (source as u32 + seed).is_multiple_of(5) {
                nfa.set_accepting(source as u32);
            }
            let remaining = count.saturating_sub(source + 1);
            if remaining == 0 {
                continue;
            }
            for lane in 0..3u32 {
                if (source as u32 + seed + lane).is_multiple_of(2) {
                    let target = source + 1 + ((source as u32 * 7 + seed + lane * 3) as usize % remaining);
                    let label = ((source as i32 * 11 + lane as i32 * 5 + seed as i32) % 13) - 6;
                    nfa.add_transition(source as u32, label, target as u32);
                    if (source as u32 + seed + lane).is_multiple_of(4) {
                        nfa.add_transition(source as u32, label, target as u32);
                    }
                }
            }
            if !(source as u32 * 3 + seed).is_multiple_of(4) {
                let target = source + 1 + ((source as u32 * 5 + seed) as usize % remaining);
                nfa.add_epsilon(source as u32, target as u32);
            }
        }
        nfa
    }

    #[test]
    fn flat_scratch_determinizer_matches_reference_on_generated_dags() {
        for seed in 0..256 {
            let nfa = generated_dag(seed);
            let expected = reference_determinize(&nfa);
            let actual = determinize(&nfa);
            assert_eq!(actual, expected, "determinization mismatch for seed {seed}");
        }
    }
    #[test]
    fn cyclic_epsilon_subsets_match_literal_reference() {
        let mut nfa = NFA::new();
        let a = nfa.add_state();
        let b = nfa.add_state();
        nfa.add_epsilon(0, a);
        nfa.add_epsilon(a, 0);
        nfa.add_transition(a, -3, a);
        nfa.add_transition(0, -3, b);
        nfa.add_transition(b, 9001, 0);
        nfa.set_accepting(b);
        assert_eq!(determinize_general(&nfa), reference_determinize(&nfa));
    }
}
