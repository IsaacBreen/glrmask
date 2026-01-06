//! NWA state minimization via partition refinement.

use super::super::common::Partition;
use crate::precompute4::weighted_automata::common::{Label, NWAStateID, Weight};
use crate::precompute4::weighted_automata::nwa::{NWAStates, NWA};
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ArcLabel {
    Eps,
    Label(Label),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaTransitionSig {
    label: ArcLabel,
    dest_class: usize,
    weight: Weight,
}

impl NwaTransitionSig {
    fn sort_key(&self) -> (u8, Label, usize) {
        let label_tag = match self.label {
            ArcLabel::Eps => 0,
            ArcLabel::Label(_) => 1,
        };
        let label_val = match self.label {
            ArcLabel::Eps => 0,
            ArcLabel::Label(v) => v,
        };
        (label_tag, label_val, self.dest_class)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaStateSignature {
    final_weight: Option<Weight>,
    outgoing: Vec<NwaTransitionSig>,
}

impl NwaStateSignature {
    fn from_state(state_id: NWAStateID, states: &NWAStates, classes: &[usize]) -> Self {
        let st = &states[state_id];

        let mut num_out = st.epsilons.len();
        for targets in st.transitions.values() {
            num_out += targets.len();
        }
        let mut tmp: Vec<NwaTransitionSig> = Vec::with_capacity(num_out);

        // Epsilon transitions
        for &(dest, ref w) in &st.epsilons {
            if w.is_empty() {
                continue;
            }
            tmp.push(NwaTransitionSig {
                label: ArcLabel::Eps,
                dest_class: classes[dest],
                weight: w.clone(),
            });
        }

        // Labeled transitions
        for (&lbl, targets) in &st.transitions {
            let label = ArcLabel::Label(lbl);
            for &(dest, ref w) in targets {
                if w.is_empty() {
                    continue;
                }
                tmp.push(NwaTransitionSig {
                    label,
                    dest_class: classes[dest],
                    weight: w.clone(),
                });
            }
        }

        if tmp.is_empty() {
            return NwaStateSignature {
                final_weight: st.final_weight.clone(),
                outgoing: Vec::new(),
            };
        }

        tmp.sort_by_key(|sig| sig.sort_key());

        // Compress runs with the same (label, dest_class)
        let mut outgoing: Vec<NwaTransitionSig> = Vec::new();
        let mut iter = tmp.into_iter();
        if let Some(mut cur) = iter.next() {
            for sig in iter {
                if cur.label == sig.label && cur.dest_class == sig.dest_class {
                    cur.weight |= &sig.weight;
                } else {
                    if !cur.weight.is_empty() {
                        outgoing.push(cur);
                    }
                    cur = sig;
                }
            }
            if !cur.weight.is_empty() {
                outgoing.push(cur);
            }
        }

        NwaStateSignature {
            final_weight: st.final_weight.clone(),
            outgoing,
        }
    }
}

pub(super) fn minimize_nwa_partition(states: &NWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition { class_of: vec![], num_classes: 0 };
    }

    let mut partition = Partition::new(n);
    loop {
        let mut sig_to_class: FxHashMap<NwaStateSignature, usize> = FxHashMap::default();
        let mut new_classes = vec![0; n];
        let mut next_class = 0;

        for s in 0..n {
            let sig = NwaStateSignature::from_state(s, states, &partition.class_of);
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

impl NWA {
    pub fn minimize_states(&mut self) -> bool {
        crate::debug!(7, "[NWA] Minimizing states...");
        let n = self.states.len();
        if n <= 1 {
            return false;
        }
        let partition = minimize_nwa_partition(&self.states);
        if partition.num_classes() >= n {
            return false;
        }
        self.rebuild_from_partition(partition);
        true
    }
}
