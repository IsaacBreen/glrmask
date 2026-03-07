


#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: `DWA` and `DWAState` are the direct glrmask analogue of sep1's weighted deterministic automaton in `dwa_i32/dwa.rs`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::nwa::Label;
use crate::ds::weight::Weight;


#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DWAState {
    
    pub transitions: BTreeMap<Label, (u32, Weight)>,
    
    pub final_weight: Option<Weight>,
}





#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DWA {
    
    pub states: Vec<DWAState>,
    
    pub start_state: u32,
}

impl DWA {
    
    pub fn new(num_tsids: u32, max_token: u32) -> Self {
        let _ = (num_tsids, max_token);
        Self {
            states: vec![DWAState::default()],
            start_state: 0,
        }
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
        self.states.iter().map(|state| state.transitions.len()).sum()
    }

    
    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.final_weight = Some(weight);
        }
    }

    
    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
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

    
    pub fn labels(&self) -> Vec<Label> {
        self.states
            .iter()
            .flat_map(|state| state.transitions.keys().copied())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    
    pub fn is_acyclic(&self) -> bool {
        let num_states = self.states.len();

        for (state_id, state) in self.states.iter().enumerate() {
            for (target, _) in state.transitions.values() {
                if *target as usize == state_id {
                    return false;
                }
            }
        }

        fn visit(state_id: usize, states: &[DWAState], colors: &mut [u8]) -> bool {
            colors[state_id] = 1;
            for (target, _) in states[state_id].transitions.values() {
                let target = *target as usize;
                if target >= colors.len() {
                    continue;
                }
                match colors[target] {
                    1 => return false,
                    0 => {
                        if !visit(target, states, colors) {
                            return false;
                        }
                    }
                    _ => {}
                }
            }
            colors[state_id] = 2;
            true
        }

        let mut colors = vec![0u8; num_states];
        for state_id in 0..num_states {
            if colors[state_id] == 0 && !visit(state_id, &self.states, &mut colors) {
                return false;
            }
        }
        true
    }

    
    
    
    pub fn display_with_symbols<'a>(
        &'a self,
        symbols: &'a BTreeMap<Label, String>,
    ) -> DWADisplayWithSymbols<'a> {
        DWADisplayWithSymbols { dwa: self, symbols }
    }

    
    
    pub fn display_with_all_maps<'a>(
        &'a self,
        symbols: &'a BTreeMap<Label, String>,
        tsid_names: &'a std::collections::BTreeMap<u32, String>,
        token_names: &'a std::collections::BTreeMap<u32, String>,
    ) -> DWADisplayWithAllMaps<'a> {
        DWADisplayWithAllMaps {
            dwa: self,
            symbols,
            tsid_names,
            token_names,
        }
    }
}






fn fmt_dwa_states(
    dwa: &DWA,
    f: &mut std::fmt::Formatter<'_>,
    label_fn: &dyn Fn(Label) -> String,
    weight_fn: &dyn Fn(&Weight) -> String,
) -> std::fmt::Result {
    for (i, st) in dwa.states.iter().enumerate() {
        if st.transitions.is_empty() && st.final_weight.is_none() {
            continue;
        }

        let start_mark = if i as u32 == dwa.start_state { " [START]" } else { "" };
        writeln!(f, "  State {i}{start_mark}")?;

        if let Some(w) = &st.final_weight {
            writeln!(f, "    final: {}", weight_fn(w))?;
        }

        for (label, (tgt, w)) in &st.transitions {
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


pub struct DWADisplayWithSymbols<'a> {
    dwa: &'a DWA,
    symbols: &'a BTreeMap<Label, String>,
}

impl std::fmt::Display for DWADisplayWithSymbols<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dwa = self.dwa;
        writeln!(f, "DWA: {} states, start=State {}", dwa.states.len(), dwa.start_state)?;
        let syms = self.symbols;
        fmt_dwa_states(dwa, f,
            &|label| match syms.get(&label) {
                Some(name) => name.clone(),
                None => format!("{label}"),
            },
            &|w| format!("{w}"),
        )
    }
}


pub struct DWADisplayWithAllMaps<'a> {
    dwa: &'a DWA,
    symbols: &'a BTreeMap<Label, String>,
    tsid_names: &'a std::collections::BTreeMap<u32, String>,
    token_names: &'a std::collections::BTreeMap<u32, String>,
}

impl std::fmt::Display for DWADisplayWithAllMaps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _ = (self.tsid_names, self.token_names);
        let dwa = self.dwa;
        let syms = self.symbols;
        writeln!(f, "DWA: {} states, start=State {}", dwa.states.len(), dwa.start_state)?;
        fmt_dwa_states(dwa, f,
            &|label| syms.get(&label).cloned().unwrap_or_else(|| label.to_string()),
            &|weight| format!("{weight}"),
        )
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
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;

    #[test]
    fn test_dwa_eval_word() {
        
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = DWA::new(nt, max_tok);
        let s1 = dwa.add_state();

        let w_trans = Weight::all();
        let w_final = Weight::all();
        dwa.add_transition(0, 0, s1, w_trans);
        dwa.set_final_weight(s1, w_final);

        let result = dwa.eval_word(&[0]);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_dwa_eval_word_reject() {
        let nt = 1u32;
        let dwa = DWA::new(nt, 5);
        
        let result = dwa.eval_word(&[0]);
        assert!(result.is_empty());
    }
}
