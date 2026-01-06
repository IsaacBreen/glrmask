//! Rebuild DWA from partition (after minimization).

use super::super::common::Partition;
use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::{DWAStates, DWA};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
struct DwaStateBuilder {
    final_weight: Option<Weight>,
    trans: BTreeMap<Label, (StateID, Weight)>,
}

impl DWA {
    pub(super) fn rebuild_cyclic_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        if n == 0 {
            return;
        }
        let mut class_to_new: FxHashMap<usize, StateID> = FxHashMap::default();
        let mut builders: Vec<DwaStateBuilder> = Vec::new();

        for s in 0..n {
            let c = partition.class_of[s];
            class_to_new.entry(c).or_insert_with(|| {
                let id = builders.len();
                builders.push(DwaStateBuilder::default());
                id
            });
        }

        for old_s in 0..n {
            let c = partition.class_of[old_s];
            let new_id = class_to_new[&c];
            let builder = &mut builders[new_id];
            let st = &self.states[old_s];

            debug_assert!(st.state_weight.is_none());

            if let Some(ref fw) = st.final_weight {
                if !fw.is_empty() {
                    match &mut builder.final_weight {
                        Some(existing) => *existing |= fw,
                        None => builder.final_weight = Some(fw.clone()),
                    }
                }
            }

            for (&label, &dest) in &st.transitions {
                let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
                if w.is_empty() {
                    continue;
                }
                let dest_class = partition.class_of[dest];
                let dest_new = class_to_new[&dest_class];
                use std::collections::btree_map::Entry;
                match builder.trans.entry(label) {
                    Entry::Vacant(e) => {
                        e.insert((dest_new, w));
                    }
                    Entry::Occupied(mut e) => {
                        let (existing_dest, existing_w) = e.get_mut();
                        debug_assert_eq!(
                            *existing_dest, dest_new,
                            "Determinism violated while rebuilding DWA: multiple destinations for label {} in class {}",
                            label, c
                        );
                        *existing_w |= &w;
                    }
                }
            }
        }

        let mut new_states = DWAStates::default();
        for _ in 0..builders.len() {
            new_states.add_state();
        }

        for (new_id, builder) in builders.into_iter().enumerate() {
            let st = &mut new_states[new_id];
            st.state_weight = None;
            st.final_weight = builder.final_weight;
            st.transitions.clear();
            st.trans_weights.clear();
            for (label, (dest_new, weight)) in builder.trans {
                st.transitions.insert(label, dest_new);
                st.trans_weights.insert(label, weight);
            }
        }

        let start_class = partition.class_of[self.body.start_state];
        let new_start = class_to_new[&start_class];
        self.states = new_states;
        self.body.start_state = new_start;
    }
}
