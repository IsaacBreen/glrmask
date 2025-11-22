#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::common::{Label, NWAStateID, Weight};
use super::dwa::DWA;
use super::nwa::NWA;

impl NWA {
    pub fn determinize(&self) -> DWA {
        let mut dwa = DWA::new();
        dwa.states.0.clear();

        let mut subset_map: HashMap<BTreeMap<NWAStateID, Weight>, NWAStateID> = HashMap::new();
        let mut worklist: VecDeque<BTreeMap<NWAStateID, Weight>> = VecDeque::new();

        let mut start_subset = BTreeMap::new();
        for &s in &self.body.start_states {
            if s < self.states.len() {
                start_subset.insert(s, Weight::all());
            }
        }

        let initial_subset = self.epsilon_closure(&start_subset);

        if !initial_subset.is_empty() {
            let start_id = dwa.add_state();
            dwa.body.start_state = start_id;
            subset_map.insert(initial_subset.clone(), start_id);
            worklist.push_back(initial_subset);
        } else {
            let start_id = dwa.add_state();
            dwa.body.start_state = start_id;
        }

        while let Some(subset) = worklist.pop_front() {
            let from_dwa_id = *subset_map.get(&subset).unwrap();

            let mut final_weight = Weight::zeros();
            for (nwa_id, path_weight) in &subset {
                if let Some(fw) = &self.states[*nwa_id].final_weight {
                    final_weight |= &(path_weight & fw);
                }
            }
            if !final_weight.is_empty() {
                dwa.states[from_dwa_id].final_weight = Some(final_weight);
            }

            let mut transitions: BTreeMap<Label, BTreeMap<NWAStateID, Weight>> = BTreeMap::new();
            for (nwa_id, path_weight) in &subset {
                for (label, targets) in &self.states[*nwa_id].transitions {
                    for (target_nwa_id, trans_weight) in targets {
                        let next_path_weight = path_weight & trans_weight;
                        if !next_path_weight.is_empty() {
                            let entry = transitions.entry(*label).or_default();
                            *entry.entry(*target_nwa_id).or_insert_with(Weight::zeros) |= &next_path_weight;
                        }
                    }
                }
            }

            for (label, next_subset_pre_closure) in transitions {
                let next_subset = self.epsilon_closure(&next_subset_pre_closure);
                if next_subset.is_empty() {
                    continue;
                }
                let to_dwa_id = *subset_map.entry(next_subset.clone()).or_insert_with(|| {
                    let new_id = dwa.add_state();
                    worklist.push_back(next_subset);
                    new_id
                });
                dwa.add_transition(from_dwa_id, label, to_dwa_id, Weight::all()).unwrap();
            }
        }
        dwa
    }

    fn epsilon_closure(&self, subset: &BTreeMap<NWAStateID, Weight>) -> BTreeMap<NWAStateID, Weight> {
        let mut closure = subset.clone();
        let mut worklist: VecDeque<NWAStateID> = subset.keys().copied().collect();

        while let Some(u) = worklist.pop_front() {
            let u_weight = closure.get(&u).unwrap().clone();
            if u >= self.states.len() {
                continue;
            }
            for (v, eps_weight) in &self.states[u].epsilons {
                let v_new_weight = &u_weight & eps_weight;
                if !v_new_weight.is_empty() {
                    let v_current_weight = closure.entry(*v).or_insert_with(Weight::zeros);
                    let combined = &*v_current_weight | &v_new_weight;
                    if combined != *v_current_weight {
                        *v_current_weight = combined;
                        worklist.push_back(*v);
                    }
                }
            }
        }
        closure
    }
}