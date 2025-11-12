// src/precompute4/weighted_automata/determinization.rs

use super::common::{StateID, Weight};
use super::dwa::DWA;
use super::nwa::NWA;
use crate::precompute4::weighted_automata::NWAStateID;
use hashbrown::HashMap;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const DETERMINIZATION_STATE_LIMIT: usize = 10000;

type WeightedStateSet = BTreeSet<(NWAStateID, Weight)>;

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let mut dwa = DWA::new();
        dwa.states.0.clear(); // Start with no states, not even a start state.

        let mut macrostate_map: HashMap<WeightedStateSet, StateID> = HashMap::new();
        let mut queue: VecDeque<WeightedStateSet> = VecDeque::new();

        // 1. Initial state
        let start_closure = self.epsilon_closure(&[(self.body.start_state, Weight::all())]);
        let initial_macrostate: WeightedStateSet = start_closure.into_iter().collect();

        if initial_macrostate.is_empty() {
            // NWA accepts nothing, return empty DWA.
            dwa.states.add_state(); // DWA needs at least a start state.
            return dwa;
        }

        let d_start_id = dwa.states.add_state();
        dwa.body.start_state = d_start_id;
        macrostate_map.insert(initial_macrostate.clone(), d_start_id);
        queue.push_back(initial_macrostate);

        let start_time = Instant::now();

        while let Some(current_macrostate) = queue.pop_front() {
            if macrostate_map.len() > DETERMINIZATION_STATE_LIMIT {
                eprintln!(
                    "Determinization state limit ({}) exceeded. Dumping NWA and panicking.",
                    DETERMINIZATION_STATE_LIMIT
                );
                let json = serde_json::to_string_pretty(&self).unwrap();
                let mut file = File::create("nwa_dump.json").unwrap();
                file.write_all(json.as_bytes()).unwrap();
                panic!(
                    "Determinization did not terminate. NWA dumped to nwa_dump.json for analysis."
                );
            }

            let from_dwa_id = *macrostate_map.get(&current_macrostate).unwrap();

            // Update final weight for the current DWA state
            let mut final_weight = Weight::zeros();
            for (nwa_id, weight) in &current_macrostate {
                if let Some(fw) = &self.states[*nwa_id].final_weight {
                    final_weight |= &(weight & fw);
                }
            }
            if !final_weight.is_empty() {
                dwa.states[from_dwa_id].final_weight = Some(final_weight);
            }

            // Group transitions by symbol
            let mut transitions_by_symbol: BTreeMap<i16, Vec<(NWAStateID, Weight)>> =
                BTreeMap::new();
            let mut default_transitions: Vec<(NWAStateID, Weight, BTreeSet<i16>)> = Vec::new();

            for (nwa_id, weight) in &current_macrostate {
                let nwa_state = &self.states[*nwa_id];
                // Labeled transitions
                for (symbol, targets) in &nwa_state.transitions {
                    for (target_id, trans_weight) in targets {
                        let new_weight = weight & trans_weight;
                        if !new_weight.is_empty() {
                            transitions_by_symbol
                                .entry(*symbol)
                                .or_default()
                                .push((*target_id, new_weight));
                        }
                    }
                }
                // Default transitions
                for def in &nwa_state.default {
                    let new_weight = weight & &def.weight;
                    if !new_weight.is_empty() {
                        default_transitions.push((def.target, new_weight, def.exceptions.clone()));
                    }
                }
            }

            let mut all_explicit_symbols: BTreeSet<i16> =
                transitions_by_symbol.keys().copied().collect();
            for (_, _, exceptions) in &default_transitions {
                all_explicit_symbols.extend(exceptions);
            }

            // Create DWA transitions for all "interesting" symbols
            for symbol in &all_explicit_symbols {
                let mut next_macrostate_vec =
                    transitions_by_symbol.get(symbol).cloned().unwrap_or_default();

                for (target, weight, exceptions) in &default_transitions {
                    if !exceptions.contains(symbol) {
                        next_macrostate_vec.push((*target, weight.clone()));
                    }
                }

                if next_macrostate_vec.is_empty() {
                    continue;
                }

                let next_closure = self.epsilon_closure(&next_macrostate_vec);
                let next_macrostate: WeightedStateSet = next_closure.into_iter().collect();

                if next_macrostate.is_empty() {
                    continue;
                }

                let to_dwa_id =
                    *macrostate_map.entry(next_macrostate.clone()).or_insert_with(|| {
                        let new_id = dwa.states.add_state();
                        queue.push_back(next_macrostate);
                        new_id
                    });

                let mut trans_weight = Weight::zeros();
                for (_, w) in &next_macrostate_vec {
                    trans_weight |= w;
                }

                if !trans_weight.is_empty() {
                    dwa.add_transition(from_dwa_id, *symbol, to_dwa_id, trans_weight)
                        .unwrap();
                }
            }

            // Handle default transition for DWA state
            let mut default_targets = Vec::new();
            for (target, weight, _) in &default_transitions {
                default_targets.push((*target, weight.clone()));
            }

            if !default_targets.is_empty() {
                let next_closure = self.epsilon_closure(&default_targets);
                let next_macrostate: WeightedStateSet = next_closure.into_iter().collect();

                if !next_macrostate.is_empty() {
                    let to_dwa_id =
                        *macrostate_map.entry(next_macrostate.clone()).or_insert_with(|| {
                            let new_id = dwa.states.add_state();
                            queue.push_back(next_macrostate);
                            new_id
                        });

                    let mut trans_weight = Weight::zeros();
                    for (_, w) in &default_targets {
                        trans_weight |= w;
                    }

                    if !trans_weight.is_empty() {
                        dwa.set_default_transition(from_dwa_id, to_dwa_id, trans_weight)
                            .unwrap();
                    }
                }
            }
        }

        eprintln!("Determinization finished in {:?}, created {} states.", start_time.elapsed(), macrostate_map.len());
        dwa
    }

    fn epsilon_closure(&self, initial_states: &[(NWAStateID, Weight)]) -> Vec<(NWAStateID, Weight)> {
        let mut closure: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
        let mut worklist: VecDeque<NWAStateID> = VecDeque::new();

        for (id, w) in initial_states {
            if closure.entry(*id).or_default().bitor(w) != *closure.get(id).unwrap() {
                worklist.push_back(*id);
            }
        }

        while let Some(from_id) = worklist.pop_front() {
            let from_weight = closure.get(&from_id).unwrap().clone();

            for (to_id, eps_weight) in &self.states[from_id].epsilons {
                let new_weight = &from_weight & eps_weight;
                if new_weight.is_empty() {
                    continue;
                }

                let current_to_weight = closure.entry(*to_id).or_default();
                let old_to_weight = current_to_weight.clone();
                *current_to_weight |= &new_weight;

                if *current_to_weight != old_to_weight {
                    worklist.push_back(*to_id);
                }
            }
        }
        closure.into_iter().filter(|(_, w)| !w.is_empty()).collect()
    }
}