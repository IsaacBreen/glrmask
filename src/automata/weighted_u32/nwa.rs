







#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::ds::weight::Weight;


pub type Label = i32;


#[derive(Debug, Clone, Default)]
pub struct NWAState {
    
    
    pub final_weight: Option<Weight>,
    
    pub transitions: BTreeMap<Label, Vec<(u32, Weight)>>,
    
    pub epsilons: Vec<(u32, Weight)>,
}


#[derive(Debug, Clone)]
pub struct NWA {
    
    pub states: Vec<NWAState>,
    
    pub start_states: Vec<u32>,
}

impl NWA {
    
    pub fn new(num_tsids: u32, max_token: u32) -> Self {
        unimplemented!()
    }

    
    pub fn add_state(&mut self) -> u32 {
        unimplemented!()
    }

    
    pub fn num_states(&self) -> u32 {
        unimplemented!()
    }

    
    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        unimplemented!()
    }

    
    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
        unimplemented!()
    }

    
    pub fn add_epsilon(&mut self, from: u32, to: u32, weight: Weight) {
        unimplemented!()
    }

    
    pub fn num_transitions(&self) -> usize {
        unimplemented!()
    }

    
    pub fn is_acyclic(&self) -> bool {
        let num_states = self.states.len();

        for (state_id, state) in self.states.iter().enumerate() {
            if state
                .transitions
                .values()
                .flatten()
                .any(|(target, _)| *target as usize == state_id)
            {
                return false;
            }
            if state
                .epsilons
                .iter()
                .any(|(target, _)| *target as usize == state_id)
            {
                return false;
            }
        }

        fn visit(state_id: usize, states: &[NWAState], colors: &mut [u8]) -> bool {
            colors[state_id] = 1;

            for (target, _) in states[state_id].transitions.values().flatten() {
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

            for (target, _) in &states[state_id].epsilons {
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

    
    pub fn max_position(&self) -> u32 {
        unimplemented!()
    }

    
    
    
    pub fn display_with_symbols<'a>(
        &'a self,
        symbols: &'a std::collections::BTreeMap<Label, String>,
    ) -> NWADisplayWithSymbols<'a> {
        unimplemented!()
    }

    
    
    pub fn display_with_all_maps<'a>(
        &'a self,
        symbols: &'a std::collections::BTreeMap<Label, String>,
        tsid_names: &'a std::collections::BTreeMap<u32, String>,
        token_names: &'a std::collections::BTreeMap<u32, String>,
    ) -> NWADisplayWithAllMaps<'a> {
        unimplemented!()
    }
}






fn fmt_nwa_states(
    nwa: &NWA,
    f: &mut std::fmt::Formatter<'_>,
    label_fn: &dyn Fn(Label) -> String,
    weight_fn: &dyn Fn(&Weight) -> String,
) -> std::fmt::Result {
    let start_set: std::collections::BTreeSet<u32> = nwa.start_states.iter().copied().collect();

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
        writeln!(f, "NWA: {} states, start={:?}", self.states.len(), self.start_states)?;
        fmt_nwa_states(self, f, &|l| l.to_string(), &|w| format!("{w}"))
    }
}


pub struct NWADisplayWithSymbols<'a> {
    nwa: &'a NWA,
    symbols: &'a std::collections::BTreeMap<Label, String>,
}

impl std::fmt::Display for NWADisplayWithSymbols<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nwa = self.nwa;
        writeln!(f, "NWA: {} states, start={:?}", nwa.states.len(), nwa.start_states)?;
        let syms = self.symbols;
        fmt_nwa_states(nwa, f,
            &|label| match syms.get(&label) {
                Some(name) => name.clone(),
                None => format!("{label}"),
            },
            &|w| format!("{w}"),
        )
    }
}


pub struct NWADisplayWithAllMaps<'a> {
    nwa: &'a NWA,
    symbols: &'a std::collections::BTreeMap<Label, String>,
    tsid_names: &'a std::collections::BTreeMap<u32, String>,
    token_names: &'a std::collections::BTreeMap<u32, String>,
}

impl std::fmt::Display for NWADisplayWithAllMaps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;

    #[test]
    fn test_nwa_basic() {
        let mut nwa = NWA::new(2, 10);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();

        let w = Weight::all();
        nwa.add_transition(s0, 0, s1, w.clone());
        nwa.add_epsilon(s1, s2, w.clone());
        nwa.set_final_weight(s2, w);

        assert_eq!(nwa.num_states(), 3);
        assert_eq!(nwa.num_transitions(), 2);
        assert!(nwa.states[s2 as usize].final_weight.is_some());
    }
}
