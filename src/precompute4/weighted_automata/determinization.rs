use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWA, NWAStateID};
use crate::r#macro::is_debug_level_enabled;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::Instant;

// A "macrostate" in the determinized automaton is a set of NWA states, each with an
// associated weight. The weight represents the union of weights of all paths from the
// NWA start state that could lead to this NWA state for a given input string.
type WeightedMacrostate = BTreeMap<NWAStateID, Weight>;

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();
        let mut nwa = self.clone();
        nwa.simplify();

        if is_debug_level_enabled(5) {
            eprintln!("NWA after simplify:\n{}", nwa);
        }
        if nwa.states.0.is_empty() {
            return DWA::new();
        }

        let mut determinizer = Determinizer::new(&nwa);
        let dwa = determinizer.run();

        if is_debug_level_enabled(5) {
            eprintln!("NWA::determinize_to_dwa result DWA stats:\n{}", dwa.stats());
            eprintln!("NWA::determinize_to_dwa took: {:?}", now.elapsed());
        }
        dwa
    }
}

struct Determinizer<'a> {
    nwa: &'a NWA,
    dwa: DWA,
    macrostate_to_dwa_state: HashMap<WeightedMacrostate, usize>,
    work_queue: VecDeque<WeightedMacrostate>,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA) -> Self {
        Self {
            nwa,
            dwa: DWA::default(),
            macrostate_to_dwa_state: HashMap::new(),
            work_queue: VecDeque::new(),
        }
    }

    fn run(&mut self) -> DWA {
        let mut start_macrostate = WeightedMacrostate::new();
        if self.nwa.body.start_state < self.nwa.states.len() {
            start_macrostate.insert(self.nwa.body.start_state, Weight::all());
        } else {
            return DWA::new();
        }

        let initial_macrostate = self.epsilon_closure(&start_macrostate);

        if initial_macrostate.is_empty() {
            return DWA::new();
        }

        let start_id = self.get_or_create_dwa_state(&initial_macrostate);
        self.dwa.body.start_state = start_id;

        while let Some(current_macrostate) = self.work_queue.pop_front() {
            let current_dwa_id = *self.macrostate_to_dwa_state.get(&current_macrostate).unwrap();

            // Final weight
            let final_weight = self.calculate_final_weight(&current_macrostate);
            if let Some(fw) = final_weight {
                if !fw.is_empty() {
                    self.dwa.set_final_weight(current_dwa_id, fw).unwrap();
                }
            }

            // Transitions
            self.calculate_transitions(&current_macrostate);
        }

        std::mem::take(&mut self.dwa)
    }

    /// Gets the DWA state ID for a given macrostate. If the macrostate hasn't been seen
    /// before, it creates a new DWA state, adds it to the work queue, and returns the new ID.
    fn get_or_create_dwa_state(&mut self, macrostate: &WeightedMacrostate) -> usize {
        if let Some(&id) = self.macrostate_to_dwa_state.get(macrostate) {
            return id;
        }
        let new_id = self.dwa.add_state();
        self.macrostate_to_dwa_state.insert(macrostate.clone(), new_id);
        self.work_queue.push_back(macrostate.clone());
        new_id
    }

    /// Computes the epsilon-closure of a set of weighted NWA states.
    fn epsilon_closure(&self, start_states: &WeightedMacrostate) -> WeightedMacrostate {
        let mut closure = start_states.clone();
        let mut queue: VecDeque<NWAStateID> = start_states.keys().copied().collect();

        while let Some(u) = queue.pop_front() {
            let w_u = if let Some(w) = closure.get(&u) {
                w.clone()
            } else {
                continue;
            };

            if u >= self.nwa.states.len() {
                continue;
            }
            for (v, w_uv) in &self.nwa.states[u].epsilons {
                if *v >= self.nwa.states.len() {
                    continue;
                }
                let propagated_weight = &w_u & w_uv;
                if propagated_weight.is_empty() {
                    continue;
                }

                let v_weight = closure.entry(*v).or_default();
                let old_v_weight = v_weight.clone();
                *v_weight |= &propagated_weight;

                if *v_weight != old_v_weight {
                    queue.push_back(*v);
                }
            }
        }
        closure
    }

    /// Calculates the final weight for a DWA state (macrostate).
    fn calculate_final_weight(&self, macrostate: &WeightedMacrostate) -> Option<Weight> {
        let mut final_weight = Weight::zeros();
        for (&id, weight) in macrostate {
            if id >= self.nwa.states.len() {
                continue;
            }
            if let Some(fw) = &self.nwa.states[id].final_weight {
                final_weight |= &(weight & fw);
            }
        }
        if final_weight.is_empty() {
            None
        } else {
            Some(final_weight)
        }
    }

    /// Computes the set of next NWA states and the total transition weight after consuming a symbol.
    fn move_on_symbol(&self, macrostate: &WeightedMacrostate, symbol: i16) -> (WeightedMacrostate, Weight) {
        let mut next_macrostate = WeightedMacrostate::new();
        let mut total_weight = Weight::zeros();

        for (&id, weight) in macrostate {
            if id >= self.nwa.states.len() {
                continue;
            }
            let state = &self.nwa.states[id];
            let mut took_explicit = false;

            // Labeled transitions
            if let Some(transitions) = state.transitions.get(&symbol) {
                took_explicit = true;
                for (target, trans_weight) in transitions {
                    if *target >= self.nwa.states.len() {
                        continue;
                    }
                    let new_weight = weight & trans_weight;
                    if !new_weight.is_empty() {
                        total_weight |= &new_weight;
                        *next_macrostate.entry(*target).or_default() |= &new_weight;
                    }
                }
            }

            // Default transitions
            for def in &state.default {
                if !took_explicit && !def.exceptions.contains(&symbol) {
                    if def.target >= self.nwa.states.len() {
                        continue;
                    }
                    let new_weight = weight & &def.weight;
                    if !new_weight.is_empty() {
                        total_weight |= &new_weight;
                        *next_macrostate.entry(def.target).or_default() |= &new_weight;
                    }
                }
            }
        }
        (next_macrostate, total_weight)
    }

    /// Calculates and adds the default and exception transitions for a DWA state.
    fn calculate_transitions(&mut self, current_macrostate: &WeightedMacrostate) {
        let current_dwa_id = *self.macrostate_to_dwa_state.get(current_macrostate).unwrap();

        let mut critical_symbols = BTreeSet::new();
        for &id in current_macrostate.keys() {
            if id >= self.nwa.states.len() {
                continue;
            }
            let state = &self.nwa.states[id];
            for &symbol in state.transitions.keys() {
                critical_symbols.insert(symbol);
            }
            for def in &state.default {
                for &symbol in &def.exceptions {
                    critical_symbols.insert(symbol);
                }
            }
        }

        // Find a generic symbol that is not "critical" to determine the default transition.
        let mut generic_symbol: i16 = 0;
        while critical_symbols.contains(&generic_symbol) {
            generic_symbol = generic_symbol.wrapping_add(1);
        }

        let (default_pre_closure, default_weight) = self.move_on_symbol(current_macrostate, generic_symbol);
        let default_target_macrostate = self.epsilon_closure(&default_pre_closure);

        if !default_target_macrostate.is_empty() {
            let target_dwa_id = self.get_or_create_dwa_state(&default_target_macrostate);
            self.dwa.set_default_transition(current_dwa_id, target_dwa_id, default_weight).unwrap();
        }

        for &symbol in &critical_symbols {
            let (pre_closure, weight) = self.move_on_symbol(current_macrostate, symbol);
            let target_macrostate = self.epsilon_closure(&pre_closure);

            if target_macrostate != default_target_macrostate {
                let target_dwa_id = self.get_or_create_dwa_state(&target_macrostate);
                self.dwa.add_transition(current_dwa_id, symbol, target_dwa_id, weight).unwrap();
            }
        }
    }
}
