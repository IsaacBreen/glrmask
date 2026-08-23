//! Subset-construction determinization: NFA → DFA (unweighted).
//!
//! Implements the classical powerset / subset construction algorithm to convert
//! an `NFA` (with epsilon transitions) into a deterministic `DFA`.
//!
//! The caller is responsible for asserting acyclicity of the input NFA when
//! that invariant is required.

use std::collections::VecDeque;

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

fn get_or_create_subset_state(
    dfa: &mut DFA,
    subset_map: &mut FxHashMap<Vec<u32>, u32>,
    worklist: &mut VecDeque<(u32, Vec<u32>)>,
    subset: &[u32],
) -> u32 {
    if let Some(&existing) = subset_map.get(subset) {
        return existing;
    }
    let key = subset.to_vec();
    let new_id = dfa.add_state();
    subset_map.insert(key.clone(), new_id);
    worklist.push_back((new_id, key));
    new_id
}

/// Determinize an acyclic NFA into a DFA using exact subset construction.
///
/// This is the same powerset construction as the reference implementation, but
/// avoids balanced-tree allocation in its two hottest inner loops:
///
/// * outgoing targets are accumulated in persistent label buckets;
/// * epsilon closure uses dense epoch marks plus reusable vectors.
///
/// Subset keys are still sorted `Vec<u32>` values and labels are processed in
/// sorted order, preserving deterministic state-discovery order.
pub fn determinize(nfa: &NFA) -> DFA {
    assert!(nfa.compute_is_acyclic(), "determinize: input NFA is cyclic");

    if nfa.states.is_empty() || nfa.start_states.is_empty() {
        return DFA::new();
    }

    let mut dfa = DFA {
        states: Vec::new(),
        start_state: 0,
    };
    let mut subset_map: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<(u32, Vec<u32>)> = VecDeque::new();
    let mut closure = ClosureScratch::new(nfa.states.len());

    let start_key = closure.epsilon_closure(nfa, &nfa.start_states).to_vec();
    let start_id = dfa.add_state();
    dfa.start_state = start_id;
    subset_map.insert(start_key.clone(), start_id);
    worklist.push_back((start_id, start_key));

    // Keep buckets allocated across DFA states. A bucket is touched exactly
    // when its length changes from zero, so clearing touched buckets is O(labels
    // actually seen), not O(global alphabet size).
    let mut label_targets: FxHashMap<Label, Vec<u32>> = FxHashMap::default();
    let mut touched_labels = Vec::<Label>::new();

    while let Some((dfa_state, subset_key)) = worklist.pop_front() {
        if subset_is_accepting(nfa, &subset_key) {
            dfa.set_accepting(dfa_state, true);
        }

        for &nfa_state in &subset_key {
            for (&label, targets) in &nfa.states[nfa_state as usize].transitions {
                if targets.is_empty() {
                    continue;
                }
                let bucket = label_targets.entry(label).or_default();
                if bucket.is_empty() {
                    touched_labels.push(label);
                }
                bucket.extend_from_slice(targets);
            }
        }
        touched_labels.sort_unstable();

        for &label in &touched_labels {
            let raw_targets = label_targets
                .get(&label)
                .expect("touched determinization label must have a target bucket");
            if raw_targets.is_empty() {
                continue;
            }

            let next_key: &[u32] = if raw_targets.len() == 1
                && nfa.states[raw_targets[0] as usize].epsilons.is_empty()
            {
                raw_targets.as_slice()
            } else {
                closure.epsilon_closure(nfa, raw_targets)
            };
            if next_key.is_empty() {
                continue;
            }
            let next_dfa_state = get_or_create_subset_state(
                &mut dfa,
                &mut subset_map,
                &mut worklist,
                next_key,
            );
            dfa.add_transition(dfa_state, label, next_dfa_state);
        }

        for label in touched_labels.drain(..) {
            label_targets
                .get_mut(&label)
                .expect("touched determinization label must remain allocated")
                .clear();
        }
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
        assert!(nfa.compute_is_acyclic());
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
            if (source as u32 + seed) % 5 == 0 {
                nfa.set_accepting(source as u32);
            }
            let remaining = count.saturating_sub(source + 1);
            if remaining == 0 {
                continue;
            }
            for lane in 0..3u32 {
                if (source as u32 + seed + lane) % 2 == 0 {
                    let target = source + 1 + ((source as u32 * 7 + seed + lane * 3) as usize % remaining);
                    let label = ((source as i32 * 11 + lane as i32 * 5 + seed as i32) % 13) - 6;
                    nfa.add_transition(source as u32, label, target as u32);
                    if (source as u32 + seed + lane) % 4 == 0 {
                        nfa.add_transition(source as u32, label, target as u32);
                    }
                }
            }
            if (source as u32 * 3 + seed) % 4 != 0 {
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
}
