//! DWA state minimization via partition refinement.

use super::super::common::Partition;
use crate::dwa_i32::common::{Label, StateID, Weight};
use crate::dwa_i32::dwa::{DWAStates, DWA};
use rustc_hash::FxHashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaTransitionSig {
    dest_class: usize,
    weight: Weight,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaStateSignature {
    final_weight: Option<Weight>,
    outgoing: Vec<(Label, DwaTransitionSig)>,
}

impl DwaStateSignature {
    fn from_state(state_id: StateID, states: &DWAStates, classes: &[usize]) -> Self {
        let st = &states[state_id];

        let mut outgoing: Vec<(Label, DwaTransitionSig)> = st
            .transitions
            .iter()
            .filter_map(|(&label, &dest)| {
                let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
                if w.is_empty() {
                    None
                } else {
                    Some((
                        label,
                        DwaTransitionSig {
                            dest_class: classes[dest],
                            weight: w,
                        },
                    ))
                }
            })
            .collect();
        outgoing.sort_by_key(|(label, sig)| (*label, sig.dest_class));

        DwaStateSignature {
            final_weight: st.final_weight.clone(),
            outgoing,
        }
    }
}

/// DWA minimization using partition refinement.
pub(super) fn minimize_dwa_partition(states: &DWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition { class_of: vec![], num_classes: 0 };
    }

    let mut partition = Partition::new(n);
    loop {
        let mut sig_to_class: FxHashMap<DwaStateSignature, usize> = FxHashMap::default();
        let mut new_classes = vec![0; n];
        let mut next_class = 0;

        for s in 0..n {
            let sig = DwaStateSignature::from_state(s, states, &partition.class_of);
            let entry = sig_to_class.entry(sig).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
            new_classes[s] = *entry;
        }

        if new_classes == partition.class_of {
            partition.num_classes = next_class;
            return partition;
        }

        partition.class_of = new_classes;
        partition.num_classes = next_class;
    }
}

impl DWA {
    pub fn minimize_states_cyclic(&mut self) -> bool {
        let n = self.states.len();
        if n < 3 {
            return false;
        }
        
        // Quick check: count distinct final weights
        let mut fw_count = 0;
        let mut seen_final_weights: rustc_hash::FxHashSet<Option<Weight>> = rustc_hash::FxHashSet::default();
        for s in 0..n {
            if seen_final_weights.insert(self.states[s].final_weight.clone()) {
                fw_count += 1;
            }
        }
        if fw_count == n {
            return false;
        }
        
        let partition = minimize_dwa_partition(&self.states);
        if partition.num_classes() >= n {
            return false;
        }
        self.rebuild_cyclic_from_partition(partition);
        true
    }
}
