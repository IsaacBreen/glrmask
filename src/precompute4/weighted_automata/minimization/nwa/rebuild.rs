//! Rebuild NWA from partition (after minimization).

use super::super::common::Partition;
use crate::precompute4::weighted_automata::common::{Label, NWAStateID, Weight};
use crate::precompute4::weighted_automata::nwa::{NWAStates, NWA};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
struct NwaStateBuilder {
    final_weight: Option<Weight>,
    eps: BTreeMap<NWAStateID, Weight>,
    trans: BTreeMap<Label, BTreeMap<NWAStateID, Weight>>,
}

impl NWA {
    pub(super) fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        let mut class_to_new: FxHashMap<usize, NWAStateID> = FxHashMap::default();
        let mut builders: Vec<NwaStateBuilder> = Vec::new();

        for s in 0..n {
            let c = partition.class_of[s];
            class_to_new.entry(c).or_insert_with(|| {
                let id = builders.len();
                builders.push(NwaStateBuilder::default());
                id
            });
        }

        for old_s in 0..n {
            let c = partition.class_of[old_s];
            let new_id = class_to_new[&c];
            let builder = &mut builders[new_id];
            let st = &self.states[old_s];

            if let Some(ref fw) = st.final_weight {
                if !fw.is_empty() {
                    match &mut builder.final_weight {
                        Some(existing) => *existing |= fw,
                        None => builder.final_weight = Some(fw.clone()),
                    }
                }
            }

            for &(dest, ref w) in &st.epsilons {
                if w.is_empty() {
                    continue;
                }
                let dest_class = partition.class_of[dest];
                let new_dest = class_to_new[&dest_class];
                let entry = builder.eps.entry(new_dest).or_insert_with(Weight::zeros);
                *entry |= w;
            }

            for (&lbl, targets) in &st.transitions {
                for &(dest, ref w) in targets {
                    if w.is_empty() {
                        continue;
                    }
                    let dest_class = partition.class_of[dest];
                    let new_dest = class_to_new[&dest_class];
                    let per_label = builder.trans.entry(lbl).or_insert_with(BTreeMap::new);
                    let entry = per_label.entry(new_dest).or_insert_with(Weight::zeros);
                    *entry |= w;
                }
            }
        }

        let mut new_states = NWAStates::default();
        for _ in 0..builders.len() {
            new_states.add_state();
        }

        for (new_id, builder) in builders.into_iter().enumerate() {
            let st = &mut new_states[new_id];
            st.final_weight = builder.final_weight;
            st.epsilons.clear();
            for (dest_new, w) in builder.eps {
                if !w.is_empty() {
                    st.epsilons.push((dest_new, w));
                }
            }
            st.transitions.clear();
            for (lbl, dests_map) in builder.trans {
                let mut dests_vec: Vec<(NWAStateID, Weight)> = Vec::new();
                for (dest_new, w) in dests_map {
                    if !w.is_empty() {
                        dests_vec.push((dest_new, w));
                    }
                }
                if !dests_vec.is_empty() {
                    st.transitions.insert(lbl, dests_vec);
                }
            }
        }

        let mut new_start_states = Vec::new();
        for &old_start in &self.body.start_states {
            let start_class = partition.class_of[old_start];
            let new_start = class_to_new[&start_class];
            if !new_start_states.contains(&new_start) {
                new_start_states.push(new_start);
            }
        }
        self.states = new_states;
        self.body.start_states = new_start_states;
    }
}
