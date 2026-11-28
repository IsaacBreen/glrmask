use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::ds::weight::Weight;

pub type Label = i32;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NWAState {
    pub final_weight: Option<Weight>,
    pub transitions: BTreeMap<Label, Vec<(u32, Weight)>>,
    pub epsilons: Vec<(u32, Weight)>,
}

#[derive(Debug, Clone)]
pub struct NWA {
    states: Vec<NWAState>,
    start_states: Vec<u32>,
    transition_count_cache: OnceLock<usize>,
    acyclic_cache: OnceLock<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NwaBody {
    pub start_states: Vec<u32>,
}

impl NwaBody {
    pub fn union(left: &Self, right: &Self) -> Self {
        let mut start_states = left.start_states.clone();
        start_states.extend(right.start_states.iter().copied());
        Self { start_states }
    }
}

impl NWA {
    fn prune_empty_outgoing(state: &mut NWAState) {
        state.epsilons.retain(|(_, weight)| !weight.is_empty());
        for targets in state.transitions.values_mut() {
            targets.retain(|(_, weight)| !weight.is_empty());
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
    }

    pub fn new(_num_tsids: u32, _max_token: u32) -> Self {
        Self {
            states: Vec::new(),
            start_states: Vec::new(),
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
    pub fn states(&self) -> &[NWAState] {
        &self.states
    }

    #[inline]
    pub fn states_mut(&mut self) -> &mut Vec<NWAState> {
        self.invalidate_graph_caches();
        &mut self.states
    }

    #[inline]
    pub fn start_states(&self) -> &[u32] {
        &self.start_states
    }

    pub fn from_parts(states: Vec<NWAState>, start_states: Vec<u32>) -> Self {
        Self {
            states,
            start_states,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        }
    }

    pub fn into_parts(self) -> (Vec<NWAState>, Vec<u32>) {
        (self.states, self.start_states)
    }

    pub fn set_start_states(&mut self, states: Vec<u32>) {
        self.start_states = states;
    }

    pub fn start_states_mut(&mut self) -> &mut Vec<u32> {
        &mut self.start_states
    }

    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(NWAState::default());
        id
    }

    pub fn num_states(&self) -> u32 {
        self.states.len() as u32
    }

    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.final_weight = Some(weight);
        }
    }

    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
        self.invalidate_graph_caches();
        if let Some(entry) = self.states.get_mut(from as usize) {
            entry.transitions.entry(label).or_default().push((to, weight));
        }
    }

    pub fn add_epsilon(&mut self, from: u32, to: u32, weight: Weight) {
        self.invalidate_graph_caches();
        if let Some(entry) = self.states.get_mut(from as usize) {
            entry.epsilons.push((to, weight));
        }
    }

    /// Subtract final weights from outgoing transitions (epsilon and labeled).
    /// For any state with a final_weight, subtract that weight from all
    /// outgoing edges.  This prevents tokens already accepted at a state
    /// from being further processed through transitions, matching the runtime's
    /// forced-token semantics.
    pub fn subtract_final_weights_from_outgoing(&mut self) {
        self.invalidate_graph_caches();
        for i in 0..self.states.len() {
            let Some(final_weight) = self.states[i].final_weight.clone() else {
                continue;
            };
            if final_weight.is_empty() {
                continue;
            }
            let state = &mut self.states[i];
            for (_, w) in &mut state.epsilons {
                *w = w.difference(&final_weight);
            }
            for targets in state.transitions.values_mut() {
                for (_, w) in targets {
                    *w = w.difference(&final_weight);
                }
            }
            Self::prune_empty_outgoing(state);
        }
    }

    pub fn num_transitions(&self) -> usize {
        *self.transition_count_cache.get_or_init(|| {
            self.states
                .iter()
                .map(|state| {
                    state.epsilons.len()
                        + state.transitions.values().map(Vec::len).sum::<usize>()
                })
                .sum()
        })
    }

    pub fn body(&self) -> NwaBody {
        NwaBody {
            start_states: self.start_states.clone(),
        }
    }

    pub fn append_with_body(&mut self, other: &NWA) -> NwaBody {
        self.append_states(other.states.iter().cloned(), other.start_states.iter().copied())
    }

    /// Append a graph whose ownership has already been transferred.
    pub fn append_owned_with_body(&mut self, other: NWA) -> NwaBody {
        let (states, starts) = other.into_parts();
        self.append_states(states, starts)
    }

    /// Embed a deterministic graph directly in the nondeterministic union.
    /// No temporary NWA or second copy of the state rows is needed.
    pub fn append_dwa_with_body(&mut self, other: &super::dwa::DWA) -> NwaBody {
        self.append_states(other.states().iter().map(|state| NWAState {
            final_weight: state.final_weight.clone(),
            transitions: state.transitions.iter().map(|(&label, (target, weight))| {
                (label, vec![(*target, weight.clone())])
            }).collect(),
            epsilons: Vec::new(),
        }), [other.start_state()])
    }

    fn append_states(
        &mut self, states: impl IntoIterator<Item = NWAState>, starts: impl IntoIterator<Item = u32>,
    ) -> NwaBody {
        self.invalidate_graph_caches();
        let offset = u32::try_from(self.states.len()).expect("NWA state count overflow");
        let states = states.into_iter();
        self.states.reserve(states.size_hint().0);
        for mut state in states {
            assert!(self.states.len() < u32::MAX as usize, "NWA state count overflow");
            for targets in state.transitions.values_mut() {
                for (target, _) in targets {
                    *target = target.checked_add(offset).expect("NWA target overflow");
                }
            }
            for (target, _) in &mut state.epsilons {
                *target = target.checked_add(offset).expect("NWA epsilon target overflow");
            }
            self.states.push(state);
        }
        NwaBody { start_states: starts.into_iter().map(|state| {
            state.checked_add(offset).expect("NWA start state overflow")
        }).collect() }
    }

    pub fn concatenate_in_place(&mut self, left: &NWA, right_body: &NwaBody) -> NwaBody {
        let offset = self.states.len() as u32;
        let left_body = self.append_with_body(left);

        for state_id in offset as usize..(offset as usize + left.states.len()) {
            if let Some(final_weight) = self.states[state_id].final_weight.take()
                && !final_weight.is_empty() {
                    for &right_start in &right_body.start_states {
                        self.add_epsilon(state_id as u32, right_start, final_weight.clone());
                    }
                }
        }

        left_body
    }

    pub fn union_in_place(&mut self, other: &NWA, existing_body: &NwaBody) -> NwaBody {
        let other_body = self.append_with_body(other);
        NwaBody::union(existing_body, &other_body)
    }

    pub fn reverse(&self) -> Self {
        let mut rev = Self {
            states: vec![NWAState::default(); self.states.len()],
            start_states: Vec::new(),
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        };

        let super_start = rev.add_state();
        rev.start_states.push(super_start);

        for (src, state) in self.states.iter().enumerate() {
            for (&label, targets) in &state.transitions {
                for (dst, weight) in targets {
                    rev.add_transition(*dst, label, src as u32, weight.clone());
                }
            }
            for (dst, weight) in &state.epsilons {
                rev.add_epsilon(*dst, src as u32, weight.clone());
            }
            if let Some(final_weight) = &state.final_weight
                && !final_weight.is_empty() {
                    rev.add_epsilon(super_start, src as u32, final_weight.clone());
                }
        }

        for &start in &self.start_states {
            if let Some(state) = rev.states.get_mut(start as usize) {
                let updated = match &state.final_weight {
                    Some(existing) => existing.union(&Weight::all()),
                    None => Weight::all(),
                };
                state.final_weight = Some(updated);
            }
        }

        rev
    }

    pub fn is_acyclic(&self) -> bool {
        *self.acyclic_cache.get_or_init(|| self.compute_is_acyclic())
    }

    fn compute_is_acyclic(&self) -> bool {
        let num_states = self.states.len();
        let mut indegree = vec![0u32; num_states];
        for state in &self.states {
            for &target in state
                .transitions
                .values()
                .flatten()
                .map(|(target, _)| target)
                .chain(state.epsilons.iter().map(|(target, _)| target))
            {
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
        while let Some(state_id) = queue.pop_front() {
            visited += 1;
            let state = &self.states[state_id];
            for &target in state
                .transitions
                .values()
                .flatten()
                .map(|(target, _)| target)
                .chain(state.epsilons.iter().map(|(target, _)| target))
            {
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

}

fn fmt_nwa_states(
    nwa: &NWA,
    f: &mut std::fmt::Formatter<'_>,
    label_fn: &dyn Fn(Label) -> String,
    weight_fn: &dyn Fn(&Weight) -> String,
) -> std::fmt::Result {
    let start_set: BTreeSet<u32> = nwa.start_states.iter().copied().collect();

    for (i, st) in nwa.states.iter().enumerate() {
        if st.transitions.is_empty() && st.epsilons.is_empty() && st.final_weight.is_none() {
            continue;
        }

        let start_mark = if start_set.contains(&(i as u32)) { " [START]" } else { "" };
        writeln!(f, "  State {i}{start_mark}")?;

        if let Some(w) = &st.final_weight {
            writeln!(f, "    final: {}", weight_fn(w))?;
        }

        for (label, targets) in &st.transitions {
            let lbl = label_fn(*label);
            for (tgt, w) in targets {
                writeln!(f, "    {lbl} → State {tgt}")?;
                writeln!(f, "      weight: {}", weight_fn(w))?;
            }
        }

        for (tgt, w) in &st.epsilons {
            writeln!(f, "    ε → State {tgt}")?;
            writeln!(f, "      weight: {}", weight_fn(w))?;
        }
    }
    Ok(())
}

impl std::fmt::Display for NWA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let starts = self.start_states.iter().map(|s| format!("State {s}")).collect::<Vec<_>>().join(", ");
        writeln!(f, "NWA: {} states, start={starts}", self.states.len())?;
        fmt_nwa_states(self, f, &|l| l.to_string(), &|w| format!("{w}"))
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn append_mixed_graphs_rebases_all_references_and_preserves_weights() {
        let weight = Weight::all();
        let mut union = NWA::new(1, 1);
        union.add_state();
        let mut dwa = super::super::dwa::DWA::new(1, 1);
        let first = dwa.add_state();
        let last = dwa.add_state();
        dwa.set_start_state(first);
        dwa.add_transition(first, 13, last, weight.clone());
        dwa.set_final_weight(last, weight.clone());
        let offset = union.num_states();
        let body = union.append_dwa_with_body(&dwa);
        assert_eq!(body.start_states, vec![offset + first]);
        assert_eq!(union.states()[(offset + first) as usize].transitions[&13],
            vec![(offset + last, weight.clone())]);
        assert_eq!(union.states()[(offset + last) as usize].final_weight,
            Some(weight.clone()));

        let mut epsilon = NWA::new(1, 1);
        epsilon.add_state();
        epsilon.add_state();
        epsilon.set_start_states(vec![0, 1]);
        epsilon.add_epsilon(0, 1, weight.clone());
        epsilon.set_final_weight(1, weight.clone());
        let offset = union.num_states();
        let body = union.append_owned_with_body(epsilon);
        assert_eq!(body.start_states, vec![offset, offset + 1]);
        assert_eq!(union.states()[offset as usize].epsilons, vec![(offset + 1, weight.clone())]);
        assert_eq!(union.states()[(offset + 1) as usize].final_weight, Some(weight));
    }

    #[test]
    fn graph_property_caches_invalidate_on_labeled_and_epsilon_mutation() {
        let mut nwa = NWA::new(1, 1);
        let first = nwa.add_state();
        let second = nwa.add_state();
        nwa.set_start_states(vec![first]);
        nwa.add_transition(first, 3, second, Weight::all());

        assert_eq!(nwa.num_transitions(), 1);
        assert!(nwa.is_acyclic());
        assert_eq!(nwa.transition_count_cache.get(), Some(&1));
        assert_eq!(nwa.acyclic_cache.get(), Some(&true));

        nwa.add_epsilon(second, second, Weight::all());
        assert!(nwa.transition_count_cache.get().is_none());
        assert!(nwa.acyclic_cache.get().is_none());
        assert_eq!(nwa.num_transitions(), 2);
        assert!(!nwa.is_acyclic());
    }

    #[test]
    fn mutable_state_access_invalidates_graph_property_caches() {
        let mut nwa = NWA::new(1, 1);
        let state = nwa.add_state();
        assert_eq!(nwa.num_transitions(), 0);
        assert!(nwa.is_acyclic());

        nwa.states_mut()[state as usize]
            .epsilons
            .push((state, Weight::all()));
        assert!(nwa.transition_count_cache.get().is_none());
        assert!(nwa.acyclic_cache.get().is_none());
        assert_eq!(nwa.num_transitions(), 1);
        assert!(!nwa.is_acyclic());
    }
}
