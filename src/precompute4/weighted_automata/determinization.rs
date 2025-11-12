use super::common::Weight;
use super::dwa::DWA;
use super::nwa::NWA;
use crate::precompute4::weighted_automata::NWAStateID;
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
        nwa.simplify(); // Simplification can reduce NWA size before determinization.

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
    memo: HashMap<WeightedMacrostate, usize>,
    worklist: VecDeque<WeightedMacrostate>,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA) -> Self {
        Self {
            nwa,
            dwa: DWA::default(),
            memo: HashMap::new(),
            worklist: VecDeque::new(),
        }
    }

    fn run(&mut self) -> DWA {
        if self.nwa.body.start_state >= self.nwa.states.len() {
            return DWA::new();
        }

        let mut start_macrostate = WeightedMacrostate::new();
        start_macrostate.insert(self.nwa.body.start_state, Weight::all());
        let initial_macrostate = self.epsilon_closure(&start_macrostate);

        if initial_macrostate.is_empty() {
            return DWA::new();
        }

        self.dwa.body.start_state = self.get_or_create_dwa_state(&initial_macrostate);

        while let Some(macrostate) = self.worklist.pop_front() {
            let dwa_state_id = self.memo[&macrostate];

            // Set final weight for the DWA state
            let final_weight = macrostate
                .iter()
                .filter_map(|(&id, weight)| {
                    self.nwa.states.get(id).and_then(|s| s.final_weight.as_ref()).map(|fw| weight & fw)
                })
                .fold(Weight::zeros(), |acc, w| acc | w);

            if !final_weight.is_empty() {
                self.dwa.set_final_weight(dwa_state_id, final_weight).unwrap();
            }

            // Determine all outgoing transitions by checking critical symbols
            let mut transitions = BTreeMap::new();
            let mut critical_symbols = self.get_critical_symbols(&macrostate);

            // Add a generic symbol to determine the default transition
            let generic_symbol = {
                let mut s: i16 = 0;
                while critical_symbols.contains(&s) {
                    s = s.wrapping_add(1);
                }
                s
            };
            critical_symbols.insert(generic_symbol);

            for symbol in critical_symbols {
                let (next_pre_closure, trans_weight) = self.move_on_symbol(&macrostate, symbol);
                let next_macrostate = self.epsilon_closure(&next_pre_closure);
                transitions.insert(symbol, (next_macrostate, trans_weight));
            }

            // Set default and exception transitions in DWA
            let (default_macrostate, default_weight) =
                transitions.remove(&generic_symbol).unwrap_or_else(|| (WeightedMacrostate::new(), Weight::zeros()));

            if !default_macrostate.is_empty() {
                let default_target_id = self.get_or_create_dwa_state(&default_macrostate);
                self.dwa.set_default_transition(dwa_state_id, default_target_id, default_weight.clone()).unwrap();
            }

            for (symbol, (macrostate, weight)) in transitions {
                if macrostate != default_macrostate || weight != default_weight {
                    let target_id = self.get_or_create_dwa_state(&macrostate);
                    self.dwa.add_transition(dwa_state_id, symbol, target_id, weight).unwrap();
                }
            }
        }

        std::mem::take(&mut self.dwa)
    }

    fn get_critical_symbols(&self, macrostate: &WeightedMacrostate) -> BTreeSet<i16> {
        let mut symbols = BTreeSet::new();
        for &id in macrostate.keys() {
            if let Some(state) = self.nwa.states.get(id) {
                symbols.extend(state.transitions.keys());
                for def in &state.default {
                    symbols.extend(&def.exceptions);
                }
            }
        }
        symbols
    }

    fn get_or_create_dwa_state(&mut self, macrostate: &WeightedMacrostate) -> usize {
        if let Some(&id) = self.memo.get(macrostate) {
            return id;
        }
        let new_id = self.dwa.add_state();
        self.memo.insert(macrostate.clone(), new_id);
        self.worklist.push_back(macrostate.clone());
        new_id
    }

    fn epsilon_closure(&self, start_states: &WeightedMacrostate) -> WeightedMacrostate {
        let mut closure = start_states.clone();
        let mut queue: VecDeque<NWAStateID> = start_states.keys().copied().collect();

        while let Some(u) = queue.pop_front() {
            let w_u = match closure.get(&u) {
                Some(w) => w.clone(),
                None => continue,
            };

            if let Some(state) = self.nwa.states.get(u) {
                for (v, w_uv) in &state.epsilons {
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
        }
        closure
    }

    fn move_on_symbol(&self, macrostate: &WeightedMacrostate, symbol: i16) -> (WeightedMacrostate, Weight) {
        let mut next_macrostate = WeightedMacrostate::new();

        for (&id, weight) in macrostate {
            if let Some(state) = self.nwa.states.get(id) {
                let mut took_explicit = false;
                if let Some(transitions) = state.transitions.get(&symbol) {
                    took_explicit = true;
                    for (target, trans_weight) in transitions {
                        let new_weight = weight & trans_weight;
                        if !new_weight.is_empty() {
                            *next_macrostate.entry(*target).or_default() |= &new_weight;
                        }
                    }
                }
                for def in &state.default {
                    if !took_explicit && !def.exceptions.contains(&symbol) {
                        let new_weight = weight & &def.weight;
                        if !new_weight.is_empty() {
                            *next_macrostate.entry(def.target).or_default() |= &new_weight;
                        }
                    }
                }
            }
        }
        let total_weight = next_macrostate.values().fold(Weight::zeros(), |acc, w| acc | w);
        (next_macrostate, total_weight)
    }
}
