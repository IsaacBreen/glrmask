use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

use super::super::dfa::{DFA as LexerDfa, DEAD};
use super::super::nfa::NFA as LexerNfa;

type TransitionTable = [u32; 256];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProductState {
    left: u32,
    right: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct State {
    pub transitions: Vec<(U8Set, u32)>,
    pub epsilon_transitions: Vec<u32>,
    pub is_end: bool,
}

#[derive(Debug, Clone)]
pub struct Nfa {
    pub states: Vec<State>,
    pub start_state: u32,
    /// Additional selectable entry states. The default remains `start_state`.
    pub additional_start_states: Vec<u32>,
    deterministic: bool,
    minimal: bool,
}

impl Nfa {
    pub fn new(num_states: usize) -> Self {
        Self {
            states: vec![State::default(); num_states.max(1)],
            start_state: 0,
            additional_start_states: Vec::new(),
            deterministic: false,
            minimal: false,
        }
    }

    pub fn with_flags(
        states: Vec<State>,
        start_state: u32,
        deterministic: bool,
        minimal: bool,
    ) -> Self {
        Self {
            states,
            start_state,
            additional_start_states: Vec::new(),
            deterministic,
            minimal,
        }
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn start_states(&self) -> impl Iterator<Item = u32> + '_ {
        std::iter::once(self.start_state).chain(self.additional_start_states.iter().copied())
    }

    pub fn set_start_states(&mut self, states: Vec<u32>) {
        assert!(!states.is_empty(), "lightweight lexer NFA needs a default start state");
        let mut deduplicated = Vec::with_capacity(states.len());
        for state in states {
            assert!(
                (state as usize) < self.states.len(),
                "lightweight lexer NFA start state {state} is out of bounds",
            );
            if !deduplicated.contains(&state) {
                deduplicated.push(state);
            }
        }
        self.start_state = deduplicated[0];
        self.additional_start_states = deduplicated[1..].to_vec();
    }

    pub fn set_default_start_state(&mut self, state: u32) {
        assert!(
            (state as usize) < self.states.len(),
            "lightweight lexer NFA start state {state} is out of bounds",
        );
        let mut starts = Vec::with_capacity(self.additional_start_states.len() + 2);
        starts.push(state);
        starts.extend(self.start_states().filter(|&entry| entry != state));
        self.set_start_states(starts);
    }

    pub fn add_start_state(&mut self, state: u32) {
        assert!(
            (state as usize) < self.states.len(),
            "lightweight lexer NFA start state {state} is out of bounds",
        );
        if state != self.start_state && !self.additional_start_states.contains(&state) {
            self.additional_start_states.push(state);
        }
    }

    pub fn is_deterministic(&self) -> bool {
        self.deterministic
    }

    pub fn is_minimal(&self) -> bool {
        self.minimal
    }

    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(State::default());
        self.deterministic = false;
        self.minimal = false;
        id
    }

    pub fn add_transition(&mut self, from: u32, byte: u8, to: u32) {
        self.add_u8set_transition(from, U8Set::single(byte), to);
    }

    pub fn add_u8set_transition(&mut self, from: u32, set: U8Set, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.transitions.push((set, to));
            self.deterministic = false;
            self.minimal = false;
        }
    }

    pub fn add_epsilon(&mut self, from: u32, to: u32) {
        if let Some(state) = self.states.get_mut(from as usize) {
            state.epsilon_transitions.push(to);
            self.deterministic = false;
            self.minimal = false;
        }
    }

    pub fn set_end(&mut self, state: u32, is_end: bool) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.is_end = is_end;
            self.minimal = false;
        }
    }

    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        debug_assert!(self.deterministic);
        self.states.get(state as usize).and_then(|state| {
            state
                .transitions
                .iter()
                .find_map(|(set, target)| set.contains(byte).then_some(*target))
        })
    }

    pub fn accepting_states(&self) -> impl Iterator<Item = u32> + '_ {
        self.states
            .iter()
            .enumerate()
            .filter(|(_, state)| state.is_end)
            .map(|(state_id, _)| state_id as u32)
    }

    pub fn determinize(&self) -> Self {
        if self.deterministic {
            self.clone()
        } else {
            let lexer_nfa = self.to_lexer_nfa();
            let lexer_dfa = lexer_nfa.to_dfa();
            Self::from_lexer_dfa_impl(&lexer_dfa, false)
        }
    }

    pub fn minimize(&self) -> Self {
        if self.minimal {
            return self.clone();
        }

        let deterministic = self.as_deterministic();
        let lexer_dfa = deterministic.to_lexer_dfa_impl();
        let minimized = lexer_dfa.minimize();
        Self::from_lexer_dfa_impl(&minimized, true)
    }

    pub fn epsilon() -> Self {
        Self::with_flags(
            vec![State {
                transitions: Vec::new(),
                epsilon_transitions: Vec::new(),
                is_end: true,
            }],
            0,
            true,
            true,
        )
    }

    pub fn from_minimal_lexer_dfa(dfa: &LexerDfa) -> Self {
        Self::from_lexer_dfa_impl(dfa, true)
    }

    pub fn to_lexer_dfa(&self) -> LexerDfa {
        self.to_lexer_dfa_impl()
    }

    pub fn concatenate(&self, rhs: &Self) -> Self {
        let lhs = self.as_minimal();
        let rhs = rhs.as_minimal();

        let rhs_offset = lhs.states.len() as u32;
        let rhs_start = rhs.start_state + rhs_offset;
        let mut states = lhs.states.clone();
        states.extend(rhs.states.iter().cloned().map(|mut state| {
            for (_, target) in &mut state.transitions {
                *target += rhs_offset;
            }
            for target in &mut state.epsilon_transitions {
                *target += rhs_offset;
            }
            state
        }));

        for state in &mut states[..lhs.states.len()] {
            if state.is_end {
                state.is_end = false;
                state.epsilon_transitions.push(rhs_start);
            }
        }

        let mut result = Self::with_flags(states, lhs.start_state, false, false);
        result.set_start_states(lhs.start_states().collect());
        result
    }

    pub fn union(&self, rhs: &Self) -> Self {
        let lhs = self.as_minimal();
        let rhs = rhs.as_minimal();

        let lhs_offset = 1u32;
        let rhs_offset = lhs_offset + lhs.states.len() as u32;
        let lhs_start = lhs.start_state + lhs_offset;
        let rhs_start = rhs.start_state + rhs_offset;

        let mut states = vec![State::default()];
        states.extend(lhs.states.iter().cloned().map(|mut state| {
            for (_, target) in &mut state.transitions {
                *target += lhs_offset;
            }
            for target in &mut state.epsilon_transitions {
                *target += lhs_offset;
            }
            state
        }));
        states.extend(rhs.states.iter().cloned().map(|mut state| {
            for (_, target) in &mut state.transitions {
                *target += rhs_offset;
            }
            for target in &mut state.epsilon_transitions {
                *target += rhs_offset;
            }
            state
        }));
        states[0].epsilon_transitions.push(lhs_start);
        states[0].epsilon_transitions.push(rhs_start);

        let mut result = Self::with_flags(states, 0, false, false);
        for start in lhs.start_states() {
            result.add_start_state(start + lhs_offset);
        }
        for start in rhs.start_states() {
            result.add_start_state(start + rhs_offset);
        }
        result
    }

    pub fn subtract(&self, rhs: &Self) -> Self {
        let lhs = self.as_minimal();
        let rhs = rhs.as_minimal();

        let lhs_tables = lhs.transition_tables();
        let rhs_tables = rhs.transition_tables();

        let start_products: Vec<ProductState> = lhs
            .start_states()
            .flat_map(|left| {
                rhs.start_states().map(move |right| ProductState {
                    left,
                    right: Some(right),
                })
            })
            .collect();
        let start = start_products[0];
        let mut state_ids = HashMap::<ProductState, u32>::new();
        let mut worklist = VecDeque::<ProductState>::new();
        let mut out = Nfa::new(1);
        out.deterministic = true;
        out.minimal = false;

        state_ids.insert(start, 0);
        worklist.push_back(start);
        for product in start_products.into_iter().skip(1) {
            let state = Self::product_state_id(product, &mut state_ids, &mut worklist, &mut out);
            out.add_start_state(state);
        }

        while let Some(product) = worklist.pop_front() {
            let out_state = state_ids[&product];
            out.set_end(out_state, Self::product_accepts(product, &lhs, &rhs));

            for (next_product, bytes) in Self::product_successors(product, &lhs_tables, &rhs_tables)
            {
                let next_state = Self::product_state_id(
                    next_product,
                    &mut state_ids,
                    &mut worklist,
                    &mut out,
                );
                out.add_u8set_transition(out_state, bytes, next_state);
            }
        }

        out.minimize()
    }

    fn transition_tables(&self) -> Vec<TransitionTable> {
        debug_assert!(self.deterministic);
        self.states
            .iter()
            .map(|state| {
                let mut table = [DEAD; 256];
                for (set, target) in &state.transitions {
                    for byte in set.iter() {
                        table[byte as usize] = *target;
                    }
                }
                table
            })
            .collect()
    }

    fn to_lexer_nfa(&self) -> LexerNfa {
        let mut nfa = LexerNfa::new(self.states.len());
        nfa.set_start_states(self.start_states().collect());

        for (state_id, state) in self.states.iter().enumerate() {
            for (set, target) in &state.transitions {
                nfa.add_u8set_transition(state_id as u32, *set, *target);
            }
            for &target in &state.epsilon_transitions {
                nfa.add_epsilon(state_id as u32, target);
            }
            if state.is_end {
                nfa.add_finalizer(state_id as u32, 0);
            }
        }

        nfa
    }

    fn to_lexer_dfa_impl(&self) -> LexerDfa {
        debug_assert!(self.deterministic);

        let mut dfa = LexerDfa::new(self.states.len());
        dfa.ensure_group_capacity(1);

        for (state_id, state) in self.states.iter().enumerate() {
            let entries = Self::sorted_transition_entries(Self::group_target_bytes(
                &state.transitions,
            ));
            dfa.set_transitions_from_sorted_entries(state_id as u32, entries);

            let mut finalizers = BitSet::new(1);
            if state.is_end {
                finalizers.set(0);
            }
            dfa.overwrite_state_metadata(state_id as u32, finalizers, BitSet::new(1));
        }

        let start_u8set = dfa.get_u8set(self.start_state);
        dfa.set_group_u8set(0, start_u8set);
        dfa.set_start_states(self.start_states().collect());
        dfa
    }

    fn from_lexer_dfa_impl(dfa: &LexerDfa, minimal: bool) -> Self {
        let mut states = Vec::with_capacity(dfa.num_states());
        for (state_id, state) in dfa.states().iter().enumerate() {
            let mut transitions = Self::group_dfa_transition_bytes(state)
                .into_iter()
                .map(|(target, bytes)| (bytes, target))
                .collect::<Vec<_>>();
            transitions.sort_unstable_by_key(|(_, target)| *target);
            states.push(State {
                transitions,
                epsilon_transitions: Vec::new(),
                is_end: dfa.finalizers(state_id as u32).contains(0),
            });
        }

        let starts = dfa.start_states();
        Self {
            states,
            start_state: starts[0],
            additional_start_states: starts[1..].to_vec(),
            deterministic: true,
            minimal,
        }
    }

    fn as_deterministic(&self) -> Self {
        if self.deterministic {
            self.clone()
        } else {
            self.determinize()
        }
    }

    fn as_minimal(&self) -> Self {
        if self.minimal {
            self.clone()
        } else {
            self.minimize()
        }
    }

    fn product_accepts(product: ProductState, lhs: &Self, rhs: &Self) -> bool {
        let lhs_accepting = lhs.states[product.left as usize].is_end;
        let rhs_accepting = product
            .right
            .map(|state| rhs.states[state as usize].is_end)
            .unwrap_or(false);
        lhs_accepting && !rhs_accepting
    }

    fn product_successors(
        product: ProductState,
        lhs_tables: &[TransitionTable],
        rhs_tables: &[TransitionTable],
    ) -> HashMap<ProductState, U8Set> {
        let mut bytes_by_target = HashMap::<ProductState, U8Set>::new();
        for byte in 0u8..=255 {
            let left_next = lhs_tables[product.left as usize][byte as usize];
            if left_next == DEAD {
                continue;
            }

            let right_next = product.right.and_then(|state| {
                let next = rhs_tables[state as usize][byte as usize];
                (next != DEAD).then_some(next)
            });
            bytes_by_target
                .entry(ProductState {
                    left: left_next,
                    right: right_next,
                })
                .or_insert_with(U8Set::empty)
                .insert(byte);
        }
        bytes_by_target
    }

    fn product_state_id(
        product: ProductState,
        state_ids: &mut HashMap<ProductState, u32>,
        worklist: &mut VecDeque<ProductState>,
        out: &mut Nfa,
    ) -> u32 {
        if let Some(&existing) = state_ids.get(&product) {
            existing
        } else {
            let new_state = out.add_state();
            state_ids.insert(product, new_state);
            worklist.push_back(product);
            new_state
        }
    }

    fn group_target_bytes(transitions: &[(U8Set, u32)]) -> HashMap<u32, BTreeSet<u8>> {
        let mut target_bytes = HashMap::<u32, BTreeSet<u8>>::new();
        for (set, target) in transitions {
            for byte in set.iter() {
                target_bytes.entry(*target).or_default().insert(byte);
            }
        }
        target_bytes
    }

    fn sorted_transition_entries(target_bytes: HashMap<u32, BTreeSet<u8>>) -> Vec<(u8, u32)> {
        let mut entries = Vec::new();
        for (target, bytes) in target_bytes {
            for byte in bytes {
                entries.push((byte, target));
            }
        }
        entries.sort_unstable_by_key(|(byte, _)| *byte);
        entries
    }

    fn group_dfa_transition_bytes(state: &super::super::dfa::DFAState) -> HashMap<u32, U8Set> {
        let mut target_bytes = HashMap::<u32, U8Set>::new();
        for (byte, &target) in state.transitions.iter() {
            target_bytes
                .entry(target)
                .or_insert_with(U8Set::empty)
                .insert(byte);
        }
        target_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinize_and_minimize_preserve_selectable_entry_states() {
        let mut nfa = Nfa::new(4);
        nfa.add_transition(0, b'a', 2);
        nfa.add_transition(1, b'b', 3);
        nfa.set_end(2, true);
        nfa.set_end(3, true);
        nfa.set_start_states(vec![0, 1]);

        let deterministic = nfa.determinize();
        assert!(deterministic.is_deterministic());
        assert_eq!(deterministic.start_states().collect::<Vec<_>>().len(), 2);
        assert!(deterministic.step(deterministic.start_state, b'a').is_some());
        let auxiliary = deterministic.additional_start_states[0];
        assert!(deterministic.step(auxiliary, b'b').is_some());

        let minimized = deterministic.minimize();
        assert!(minimized.is_minimal());
        assert_eq!(minimized.start_states().collect::<Vec<_>>().len(), 2);
        assert!(minimized.step(minimized.start_state, b'a').is_some());
        let auxiliary = minimized.additional_start_states[0];
        assert!(minimized.step(auxiliary, b'b').is_some());
    }
}
