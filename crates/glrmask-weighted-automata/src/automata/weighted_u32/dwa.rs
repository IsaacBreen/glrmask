use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use serde::{Deserialize, Serialize};

use super::nwa::Label;
use crate::ds::weight::{finalize_weight_map, shared_rangeset, Weight};

#[derive(Debug, Clone, Default)]
pub struct DWAState {
    pub transitions: BTreeMap<Label, (u32, Weight)>,
    pub final_weight: Option<Weight>,
}

#[derive(Debug, Clone)]
pub struct DWA {
    states: Vec<DWAState>,
    start_state: u32,
    transition_count_cache: OnceLock<usize>,
    acyclic_cache: OnceLock<bool>,
}

#[derive(Debug, Clone, Copy)]
pub struct DwaStats {
    pub states: usize,
    pub transitions: usize,
    pub transition_pairs: usize,
    pub interned_ranges: usize,
}

// --- Two-level weight-pool serde for DWA ---
// Level 1: Pool unique RangeSetBlaze<u32> (token sets) by Arc pointer
// Level 2: Pool unique Weight (RangeMapBlaze) by Arc pointer, referencing token set indices

/// Serialized token set: Vec of [start, end] range pairs
type EncodedTokenSet = Vec<[u32; 2]>;

/// A single entry in a pooled weight: (tsid_start, tsid_end, token_set_pool_index)
#[derive(Serialize, Deserialize)]
struct WeightPoolEntry {
    all: bool,
    /// Entries: (tsid_range_start, tsid_range_end, token_set_pool_index)
    entries: Vec<(u32, u32, u32)>,
}

#[derive(Serialize, Deserialize)]
struct DWAStateSerde {
    /// transitions: (label, target_state, weight_pool_index)
    transitions: Vec<(Label, u32, u32)>,
    /// final_weight: Some(weight_pool_index) or None
    final_weight: Option<u32>,
}

#[derive(Serialize, Deserialize)]
struct DWASerde {
    /// Pool of unique token sets (level 1)
    token_set_pool: Vec<EncodedTokenSet>,
    /// Pool of unique weights referencing token_set_pool indices (level 2)
    weight_pool: Vec<WeightPoolEntry>,
    states: Vec<DWAStateSerde>,
    start_state: u32,
}

impl Serialize for DWA {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Level 1: Pool unique token sets by Arc pointer
        let mut ts_ptr_to_idx: std::collections::HashMap<usize, u32> =
            std::collections::HashMap::new();
        let mut token_set_pool: Vec<EncodedTokenSet> = Vec::new();

        let mut intern_token_set = |ts: &std::sync::Arc<RangeSetBlaze<u32>>| -> u32 {
            let ptr = std::sync::Arc::as_ptr(ts) as usize;
            *ts_ptr_to_idx.entry(ptr).or_insert_with(|| {
                let idx = token_set_pool.len() as u32;
                token_set_pool.push(
                    ts.ranges()
                        .map(|r| [*r.start(), *r.end()])
                        .collect(),
                );
                idx
            })
        };

        // Level 2: Pool unique weights by Arc pointer
        let mut w_ptr_to_idx: std::collections::HashMap<usize, u32> =
            std::collections::HashMap::new();
        let mut weight_pool: Vec<WeightPoolEntry> = Vec::new();

        let mut intern_weight = |w: &Weight| -> u32 {
            let ptr = w.ptr_key();
            *w_ptr_to_idx.entry(ptr).or_insert_with(|| {
                let idx = weight_pool.len() as u32;
                if w.is_full() {
                    weight_pool.push(WeightPoolEntry {
                        all: true,
                        entries: Vec::new(),
                    });
                } else {
                    let entries = w
                        .raw_range_values()
                        .map(|(range, tokens)| {
                            let ts_idx = intern_token_set(tokens);
                            (*range.start(), *range.end(), ts_idx)
                        })
                        .collect();
                    weight_pool.push(WeightPoolEntry {
                        all: false,
                        entries,
                    });
                }
                idx
            })
        };

        let states: Vec<DWAStateSerde> = self
            .states
            .iter()
            .map(|state| {
                let transitions = state
                    .transitions
                    .iter()
                    .map(|(&label, (target, weight))| (label, *target, intern_weight(weight)))
                    .collect();
                let final_weight = state.final_weight.as_ref().map(|w| intern_weight(w));
                DWAStateSerde {
                    transitions,
                    final_weight,
                }
            })
            .collect();

        let serde_repr = DWASerde {
            token_set_pool,
            weight_pool,
            states,
            start_state: self.start_state,
        };
        serde_repr.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DWA {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let serde_repr = DWASerde::deserialize(deserializer)?;

        // Reconstruct token set pool (shared Arcs)
        let ts_pool: Vec<std::sync::Arc<RangeSetBlaze<u32>>> = serde_repr
            .token_set_pool
            .into_iter()
            .map(|encoded| {
                let rs: RangeSetBlaze<u32> =
                    encoded.into_iter().map(|[s, e]| s..=e).collect();
                shared_rangeset(rs)
            })
            .collect();

        // Reconstruct weight pool
        let w_pool: Vec<Weight> = serde_repr
            .weight_pool
            .into_iter()
            .map(|entry| {
                if entry.all {
                    return Weight::all();
                }
                if entry.entries.is_empty() {
                    return Weight::empty();
                }
                let mut map = RangeMapBlaze::new();
                for (start, end, ts_idx) in entry.entries {
                    let tokens = ts_pool
                        .get(ts_idx as usize)
                        .cloned()
                        .unwrap_or_else(|| std::sync::Arc::new(RangeSetBlaze::new()));
                    map.extend_simple(std::iter::once((start..=end, tokens)));
                }
                finalize_weight_map(map)
            })
            .collect();

        // Reconstruct DWA states
        let states = serde_repr
            .states
            .into_iter()
            .map(|s| {
                let transitions = s
                    .transitions
                    .into_iter()
                    .map(|(label, target, weight_idx)| {
                        let weight = w_pool
                            .get(weight_idx as usize)
                            .cloned()
                            .unwrap_or_else(Weight::empty);
                        (label, (target, weight))
                    })
                    .collect();
                let final_weight = s.final_weight.map(|idx| {
                    w_pool
                        .get(idx as usize)
                        .cloned()
                        .unwrap_or_else(Weight::empty)
                });
                DWAState {
                    transitions,
                    final_weight,
                }
            })
            .collect();

        Ok(DWA {
            states,
            start_state: serde_repr.start_state,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        })
    }
}

impl DWA {
    pub fn new(_num_tsids: u32, _max_token: u32) -> Self {
        Self {
            states: vec![DWAState::default()],
            start_state: 0,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        }
    }

    #[inline]
    fn invalidate_graph_caches(&mut self) {
        let _ = self.transition_count_cache.take();
        let _ = self.acyclic_cache.take();
    }

    #[inline]
    pub fn states(&self) -> &[DWAState] {
        &self.states
    }

    #[inline]
    pub fn states_mut(&mut self) -> &mut Vec<DWAState> {
        self.invalidate_graph_caches();
        &mut self.states
    }

    #[inline]
    pub fn start_state(&self) -> u32 {
        self.start_state
    }

    pub fn from_parts(states: Vec<DWAState>, start_state: u32) -> Self {
        Self {
            states,
            start_state,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        }
    }

    pub fn set_start_state(&mut self, state: u32) {
        self.start_state = state;
    }

    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(DWAState::default());
        id
    }

    pub fn num_states(&self) -> u32 {
        self.states.len() as u32
    }

    pub fn num_transitions(&self) -> usize {
        *self.transition_count_cache.get_or_init(|| {
            self.states
                .iter()
                .map(|state| state.transitions.len())
                .sum()
        })
    }

    pub fn stats(&self) -> DwaStats {
        let mut transition_pairs = 0usize;
        let mut dsts = BTreeSet::new();
        for state in &self.states {
            dsts.clear();
            for (dst, _) in state.transitions.values() {
                dsts.insert(*dst);
            }
            transition_pairs += dsts.len();
        }

        let mut seen_weight_ptrs = BTreeSet::new();
        let mut seen_rangeset_ptrs = BTreeSet::new();
        let mut total_outer_ranges = 0usize;
        let mut total_inner_ranges = 0usize;

        let mut process_weight = |weight: &Weight| {
            let weight_ptr = weight.ptr_key();
            if seen_weight_ptrs.insert(weight_ptr) {
                total_outer_ranges += weight.raw_range_values().count();
            }
            for (_, tokens) in weight.raw_range_values() {
                let token_ptr = std::sync::Arc::as_ptr(tokens) as usize;
                if seen_rangeset_ptrs.insert(token_ptr) {
                    total_inner_ranges += tokens.ranges().count();
                }
            }
        };

        for state in &self.states {
            if let Some(final_weight) = &state.final_weight {
                process_weight(final_weight);
            }
            for (_, weight) in state.transitions.values() {
                process_weight(weight);
            }
        }

        DwaStats {
            states: self.states.len(),
            transitions: self.num_transitions(),
            transition_pairs,
            interned_ranges: total_outer_ranges + total_inner_ranges,
        }
    }

    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.final_weight = Some(weight);
        }
    }

    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
        self.invalidate_graph_caches();
        if let Some(entry) = self.states.get_mut(from as usize) {
            entry.transitions.insert(label, (to, weight));
        }
    }

    pub fn eval_word(&self, word: &[Label]) -> Weight {
        let mut state = self.start_state;
        let mut weight = Weight::all();
        for &label in word {
            let Some((next, edge_weight)) = self.states[state as usize].transitions.get(&label) else {
                return Weight::empty();
            };
            weight = weight.intersection(edge_weight);
            state = *next;
        }
        match self.states.get(state as usize).and_then(|state| state.final_weight.as_ref()) {
            Some(final_weight) => weight.intersection(final_weight),
            None => Weight::empty(),
        }
    }

    /// Clip all weights in the DWA so token sets contain only `0..=max_token`.
    pub fn clip_weights(&mut self, max_token: u32) {
        for state in &mut self.states {
            if let Some(fw) = &mut state.final_weight {
                fw.clip_tokens(max_token);
                if fw.is_empty() {
                    state.final_weight = None;
                }
            }
            for (_, (_, w)) in &mut state.transitions {
                w.clip_tokens(max_token);
            }
        }
    }

    pub fn labels(&self) -> Vec<Label> {
        self.states
            .iter()
            .flat_map(|state| state.transitions.keys().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn is_acyclic(&self) -> bool {
        *self.acyclic_cache.get_or_init(|| self.compute_is_acyclic())
    }

    fn compute_is_acyclic(&self) -> bool {
        let num_states = self.states.len();
        let mut indegree = vec![0u32; num_states];
        for state in &self.states {
            for &(target, _) in state.transitions.values() {
                if let Some(degree) = indegree.get_mut(target as usize) {
                    *degree += 1;
                }
            }
        }
        let mut queue = std::collections::VecDeque::new();
        for (state, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(state);
            }
        }
        let mut visited = 0usize;
        while let Some(state) = queue.pop_front() {
            visited += 1;
            for &(target, _) in self.states[state].transitions.values() {
                let Some(degree) = indegree.get_mut(target as usize) else {
                    continue;
                };
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(target as usize);
                }
            }
        }
        visited == num_states
    }

    /// Convert this DWA to an NWA representation.
    pub fn to_nwa(&self) -> super::nwa::NWA {
        use super::nwa::{NWA, NWAState};
        let mut nwa = NWA::from_parts(
            Vec::with_capacity(self.states.len()),
            vec![self.start_state],
        );
        for state in &self.states {
            let mut nwa_state = NWAState::default();
            nwa_state.final_weight = state.final_weight.clone();
            for (&label, (target, weight)) in &state.transitions {
                nwa_state
                    .transitions
                    .entry(label)
                    .or_default()
                    .push((*target, weight.clone()));
            }
            nwa.states_mut().push(nwa_state);
        }
        nwa
    }
}

fn fmt_dwa_states(
    dwa: &DWA,
    f: &mut std::fmt::Formatter<'_>,
    label_fn: &dyn Fn(Label) -> String,
    weight_fn: &dyn Fn(&Weight) -> String,
) -> std::fmt::Result {
    for (i, state) in dwa.states.iter().enumerate() {
        if state.transitions.is_empty() && state.final_weight.is_none() {
            continue;
        }

        let start_mark = if i as u32 == dwa.start_state { " [START]" } else { "" };
        writeln!(f, "  State {i}{start_mark}")?;

        if let Some(w) = &state.final_weight {
            writeln!(f, "    final: {}", weight_fn(w))?;
        }

        for (label, (tgt, w)) in &state.transitions {
            let lbl = label_fn(*label);
            writeln!(f, "    {lbl} → State {tgt}")?;
            writeln!(f, "      weight: {}", weight_fn(w))?;
        }
    }
    Ok(())
}

impl std::fmt::Display for DWA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA: {} states, start=State {}", self.states.len(), self.start_state)?;
        fmt_dwa_states(self, f, &|l| l.to_string(), &|w| format!("{w}"))
    }
}

impl PartialEq for DWA {
    fn eq(&self, other: &Self) -> bool {
        self.start_state == other.start_state && self.states == other.states
    }
}

impl PartialEq for DWAState {
    fn eq(&self, other: &Self) -> bool {
        self.transitions == other.transitions && self.final_weight == other.final_weight
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn graph_property_caches_invalidate_on_transition_mutation() {
        let mut dwa = DWA::new(1, 1);
        let next = dwa.add_state();
        dwa.add_transition(0, 7, next, Weight::all());

        assert_eq!(dwa.num_transitions(), 1);
        assert!(dwa.is_acyclic());
        assert_eq!(dwa.transition_count_cache.get(), Some(&1));
        assert_eq!(dwa.acyclic_cache.get(), Some(&true));

        dwa.add_transition(next, 8, next, Weight::all());
        assert!(dwa.transition_count_cache.get().is_none());
        assert!(dwa.acyclic_cache.get().is_none());
        assert_eq!(dwa.num_transitions(), 2);
        assert!(!dwa.is_acyclic());
    }

    #[test]
    fn mutable_state_access_and_deserialization_reset_graph_caches() {
        let mut dwa = DWA::new(1, 1);
        let next = dwa.add_state();
        dwa.add_transition(0, 1, next, Weight::all());
        assert_eq!(dwa.num_transitions(), 1);
        assert!(dwa.is_acyclic());

        dwa.states_mut()[next as usize]
            .transitions
            .insert(2, (next, Weight::all()));
        assert!(dwa.transition_count_cache.get().is_none());
        assert!(dwa.acyclic_cache.get().is_none());
        assert_eq!(dwa.num_transitions(), 2);
        assert!(!dwa.is_acyclic());

        let decoded: DWA = bincode::deserialize(&bincode::serialize(&dwa).unwrap()).unwrap();
        assert!(decoded.transition_count_cache.get().is_none());
        assert!(decoded.acyclic_cache.get().is_none());
        assert_eq!(decoded.num_transitions(), 2);
        assert!(!decoded.is_acyclic());
    }
}
