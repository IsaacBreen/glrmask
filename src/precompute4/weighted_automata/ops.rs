// src/precompute4/weighted_automata/ops.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{Label, StateID, Weight, STOCHASTIC_DEBUG};
use super::dwa::DWA;
use super::nwa::{NWABody, NWAStates, NWA};
use std::collections::VecDeque;

impl DWA {
    pub fn eval_word_weight(&self, word: &[Label]) -> Weight {
        if self.states.0.is_empty() {
            return Weight::zeros();
        }
        let mut s = self.body.start_state;
        let mut acc = Weight::all();

        if s < self.states.len() {
            if let Some(sw) = &self.states[s].state_weight {
                acc &= sw;
                if acc.is_empty() {
                    return Weight::zeros();
                }
            }
        } else {
            return Weight::zeros();
        }

        for &ch in word {
            if s >= self.states.len() {
                return Weight::zeros();
            }
            let st = &self.states[s];
            if let Some((t, w)) = st.get_transition(ch) {
                acc &= w;
                if acc.is_empty() {
                    return Weight::zeros();
                }
                s = t;
                if let Some(sw) = &self.states[s].state_weight {
                    acc &= sw;
                    if acc.is_empty() {
                        return Weight::zeros();
                    }
                }
            } else {
                return Weight::zeros();
            }
        }

        if s >= self.states.len() {
            return Weight::zeros();
        }
        match &self.states[s].final_weight {
            Some(fw) => {
                let res = &acc & fw;
                if res.is_empty() {
                    Weight::zeros()
                } else {
                    res
                }
            }
            None => Weight::zeros(),
        }
    }

    pub fn union(&self, other: &DWA) -> DWA {
        let nwa1 = NWA::from_dwa(self);
        let nwa2 = NWA::from_dwa(other);

        let mut combined_states = NWAStates::default();
        let (start1, _) = combined_states.copy_subgraph_from(&nwa1.states, nwa1.body.start_state);
        let (start2, _) = combined_states.copy_subgraph_from(&nwa2.states, nwa2.body.start_state);

        let body1 = NWABody { start_state: start1 };
        let body2 = NWABody { start_state: start2 };
        let union_body = NWA::union_components(&mut combined_states, &body1, &body2);
        let union_nwa = NWA { states: combined_states, body: union_body };
        let result_dwa = union_nwa.determinize_to_dwa();

        if STOCHASTIC_DEBUG {
            DWA::stochastic_validate_union(self, other, &result_dwa);
        }
        result_dwa
    }

    pub fn concatenate(&self, other: &DWA) -> DWA {
        let nwa1 = NWA::from_dwa(self);
        let nwa2 = NWA::from_dwa(other);

        let mut combined_states = NWAStates::default();
        let (start1, _) = combined_states.copy_subgraph_from(&nwa1.states, nwa1.body.start_state);
        let (start2, _) = combined_states.copy_subgraph_from(&nwa2.states, nwa2.body.start_state);

        let body1 = NWABody { start_state: start1 };
        let body2 = NWABody { start_state: start2 };
        let concat_body = NWA::concatenate_components(&mut combined_states, &body1, &body2, &Weight::all());
        let concat_nwa = NWA { states: combined_states, body: concat_body };
        let result_dwa = concat_nwa.determinize_to_dwa();

        if STOCHASTIC_DEBUG {
            DWA::stochastic_validate_concatenate(self, other, &result_dwa, &Weight::all());
        }
        result_dwa
    }

    pub fn apply_weight(&mut self, weight: &Weight) -> StateID {
        let old_start_id = self.body.start_state;
        let new_start_id = self.states.copy_state(old_start_id);
        self.states.apply_weight(new_start_id, weight);
        self.body.start_state = new_start_id;
        new_start_id
    }

    pub fn apply_weight_inplace(&mut self, weight: &Weight) {
        let start_state = self.states.0.get_mut(self.body.start_state).unwrap();
        if let Some(sw) = &start_state.state_weight {
            start_state.state_weight = Some(sw & weight);
        } else {
            start_state.state_weight = Some(weight.clone());
        }
        if let Some(sw) = &start_state.final_weight {
            start_state.final_weight = Some(sw & weight);
        }
        for (_, tw) in start_state.trans_weights.iter_mut() {
            *tw &= weight;
        }
    }
}

impl NWA {
    pub fn from_dwa(dwa: &DWA) -> Self {
        let mut nwa = NWA::new();
        nwa.states.0.clear();
        for _ in 0..dwa.states.len() {
            nwa.states.add_state();
        }
        nwa.body.start_state = dwa.body.start_state;

        for (i, st) in dwa.states.0.iter().enumerate() {
            nwa.states[i].final_weight = st.final_weight.clone();
            for (lbl, to) in &st.transitions {
                let w = st.trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);
                nwa.states.add_transition(i, *lbl, *to, w).unwrap();
            }
        }
        nwa
    }

    fn internal_union_components(states: &mut NWAStates, body1: &NWABody, body2: &NWABody) -> NWABody {
        let new_start = states.add_state();
        states.add_epsilon(new_start, body1.start_state, Weight::all());
        states.add_epsilon(new_start, body2.start_state, Weight::all());
        NWABody { start_state: new_start }
    }

    pub fn union_components(states: &mut NWAStates, body1: &NWABody, body2: &NWABody) -> NWABody {
        if STOCHASTIC_DEBUG {
            let nwa1 = NWA { states: states.clone(), body: body1.clone() };
            let nwa2 = NWA { states: states.clone(), body: body2.clone() };

            let mut states_after_union = states.clone();
            let union_body = Self::internal_union_components(&mut states_after_union, body1, body2);
            let union_nwa = NWA { states: states_after_union, body: union_body };

            let dwa1 = nwa1.determinize_to_dwa();
            let dwa2 = nwa2.determinize_to_dwa();
            let result_dwa = union_nwa.determinize_to_dwa();
            DWA::stochastic_validate_union(&dwa1, &dwa2, &result_dwa);
        }
        Self::internal_union_components(states, body1, body2)
    }

    pub fn union(&mut self, other: &Self) {
        let other_copied = self.states.copy_subgraph_from_and_return_body(&other.states, other.body.clone());
        self.body = Self::internal_union_components(&mut self.states, &self.body, &other_copied);
    }

    fn internal_concatenate_components(
        states: &mut NWAStates,
        left: &NWABody,
        right: &NWABody,
        eps_weight: &Weight,
    ) -> NWABody {
        let mut visited = vec![false; states.len()];
        let mut q = VecDeque::new();
        if left.start_state < states.len() {
            visited[left.start_state] = true;
            q.push_back(left.start_state);
        }
        while let Some(u) = q.pop_front() {
            for &(v, _) in &states[u].epsilons {
                if v < states.len() && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
            for (_, targets) in states[u].transitions.iter() {
                for (v, _) in targets {
                    if *v < states.len() && !visited[*v] {
                        visited[*v] = true;
                        q.push_back(*v);
                    }
                }
            }
        }

        for sid in 0..states.len() {
            if !visited[sid] {
                continue;
            }
            if let Some(fw) = states[sid].final_weight.clone() {
                let w = &fw & eps_weight;
                if !w.is_empty() {
                    states.add_epsilon(sid, right.start_state, w);
                }
                states[sid].final_weight = None;
            }
        }

        NWABody { start_state: left.start_state }
    }

    pub fn concatenate_components(
        states: &mut NWAStates,
        left: &NWABody,
        right: &NWABody,
        eps_weight: &Weight,
    ) -> NWABody {
        let new_left = states.copy_subgraph_in_place_and_return_body(*left);
        Self::_concatenate_components(states, &new_left, right, eps_weight)
    }

    pub fn _concatenate_components(
        states: &mut NWAStates,
        left: &NWABody,
        right: &NWABody,
        eps_weight: &Weight,
    ) -> NWABody {
        if STOCHASTIC_DEBUG {
            let nwa1 = NWA { states: states.clone(), body: left.clone() };
            let nwa2 = NWA { states: states.clone(), body: right.clone() };

            let mut states_after_concat = states.clone();
            let concat_body =
                Self::internal_concatenate_components(&mut states_after_concat, left, right, eps_weight);
            let concat_nwa = NWA { states: states_after_concat, body: concat_body };

            let dwa1 = nwa1.determinize_to_dwa();
            let dwa2 = nwa2.determinize_to_dwa();
            let result_dwa = concat_nwa.determinize_to_dwa();
            DWA::stochastic_validate_concatenate(&dwa1, &dwa2, &result_dwa, eps_weight);
        }
        Self::internal_concatenate_components(states, &left, right, eps_weight)
    }

    pub fn determinize_components(states: &NWAStates, body: &NWABody) -> DWA {
        let tmp = NWA { states: states.clone(), body: body.clone() };
        tmp.determinize_to_dwa()
    }
}
