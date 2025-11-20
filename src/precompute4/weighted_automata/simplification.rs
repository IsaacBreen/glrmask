#![allow(dead_code)]

use super::common::{BENCHMARK_DEBUG, OPTIMIZE_DEBUG, NWAStateID, StateID, Weight, Label};
use super::dwa::{DWAState, DWAStates, DWA};
use super::nwa::{NWAState, NWAStates, NWA};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use rustfst::algorithms::{minimize, minimize_with_config, MinimizeConfig};

const MAX_OPTIMIZE_ITERATIONS: usize = 1000;

#[derive(Clone, Debug)]
struct Partition {
    class_of: Vec<usize>,
    num_classes: usize,
}

impl Partition {
    fn new(num_states: usize) -> Self {
        Self {
            class_of: vec![0; num_states],
            num_classes: if num_states == 0 { 0 } else { 1 },
        }
    }
    fn num_classes(&self) -> usize { self.num_classes }
}

// ---------------- DWA minimization ----------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaTransitionSig {
    label: Label,
    dest_class: usize,
    weight: Weight,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaStateSignature {
    final_weight: Option<Weight>,
    outgoing: Vec<DwaTransitionSig>,
}

impl DwaStateSignature {
    fn from_state(state_id: StateID, states: &DWAStates, classes: &[usize]) -> Self {
        let st = &states[state_id];

        // For a DWA, there is at most one transition per (state, label).
        let mut outgoing = Vec::with_capacity(st.transitions.len());
        for (&label, &dest) in &st.transitions {
            let w = st
                .trans_weights
                .get(&label)
                .cloned()
                .unwrap_or_else(Weight::all);
            if w.is_empty() {
                continue;
            }
            let dest_class = classes[dest];
            outgoing.push(DwaTransitionSig {
                label,
                dest_class,
                weight: w,
            });
        }
        // Iteration over the BTreeMap `transitions` yields labels in a
        // canonical sorted order, so `outgoing` is canonical without any
        // additional sorting.
        DwaStateSignature {
            final_weight: st.final_weight.clone(),
            outgoing,
        }
    }
}

fn minimize_dwa_partition(states: &DWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition { class_of: vec![], num_classes: 0 };
    }

    let mut partition = Partition::new(n);
    loop {
        let mut sig_to_class: HashMap<DwaStateSignature, usize> = HashMap::new();
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

#[derive(Clone, Debug, Default)]
struct DwaStateBuilder {
    final_weight: Option<Weight>,
    trans: BTreeMap<Label, (StateID, Weight)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DwaPass {
    PruneUnreachable,
    PruneDeadEnds,
    PushWeights,
    Minimize,
}

const DWA_PASS_ORDERINGS: &[&[DwaPass]] = &[
    &[DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::Minimize],
    &[DwaPass::Minimize, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::PushWeights],
    &[DwaPass::PushWeights, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::Minimize],
    &[DwaPass::PushWeights, DwaPass::Minimize, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds],
    &[DwaPass::PruneUnreachable, DwaPass::PushWeights, DwaPass::Minimize, DwaPass::PruneDeadEnds],
    &[DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::Minimize, DwaPass::PruneUnreachable],

    &[DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::Minimize, DwaPass::PushWeights],
    &[DwaPass::PruneUnreachable, DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::Minimize],
    &[DwaPass::PruneUnreachable, DwaPass::Minimize, DwaPass::PruneDeadEnds, DwaPass::PushWeights],
    &[DwaPass::PruneUnreachable, DwaPass::Minimize, DwaPass::PushWeights, DwaPass::PruneDeadEnds],
    &[DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::PushWeights, DwaPass::Minimize],
    &[DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::Minimize, DwaPass::PushWeights],
    &[DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::PruneUnreachable, DwaPass::Minimize],
    &[DwaPass::PruneDeadEnds, DwaPass::Minimize, DwaPass::PruneUnreachable, DwaPass::PushWeights],
    &[DwaPass::PruneDeadEnds, DwaPass::Minimize, DwaPass::PushWeights, DwaPass::PruneUnreachable],
    &[DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::Minimize],
    &[DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::Minimize, DwaPass::PruneUnreachable],
    &[DwaPass::PushWeights, DwaPass::Minimize, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable],
    &[DwaPass::Minimize, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::PushWeights],
    &[DwaPass::Minimize, DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::PruneUnreachable],
    &[DwaPass::Minimize, DwaPass::PushWeights, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds],
    &[DwaPass::Minimize, DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable],
];

impl DWA {
    pub fn simplify(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        if OPTIMIZE_DEBUG {
            self.run_optimization_experiment();
        } else if BENCHMARK_DEBUG {
            let initial_states = self.states.len();
            let mut internal = self.clone();
            let internal_start = std::time::Instant::now();
            internal.simplify_internal();
            let internal_time = internal_start.elapsed();
            let internal_states = internal.states.len();

            let mut rustfst = self.clone();
            let rustfst_start = std::time::Instant::now();
            rustfst.simplify_with_rustfst();
            let rustfst_time = rustfst_start.elapsed();
            let rustfst_states = rustfst.states.len();

            if internal_time + rustfst_time > std::time::Duration::from_secs(1) {
                let state_cmp = match internal_states.cmp(&rustfst_states) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };
                let time_cmp = match internal_time.cmp(&rustfst_time) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };

                crate::debug!(5, "[DWA Simplify({})] Internal: t={:.2?}, s={} | RustFST: t={:.2?}, s={}. [s: {}, t: {}]", initial_states, internal_time, internal_states, rustfst_time, rustfst_states, state_cmp, time_cmp);
            }

            *self = internal;
        } else {
            self.simplify_internal();
        }
    }

    pub fn simplify_lightweight(&mut self) {
         let ordering = &[
            DwaPass::PruneDeadEnds,
            DwaPass::PushWeights,
            DwaPass::PruneUnreachable,
        ];

        for _ in 0..10 {
            let mut changed_in_iteration = false;
            for &pass in ordering {
                let pass_changed = match pass {
                    DwaPass::PruneUnreachable => self.prune_unreachable(),
                    DwaPass::PruneDeadEnds => self.prune_dead_ends(),
                    DwaPass::PushWeights => self.push_weights_into_transitions_and_finals(),
                    DwaPass::Minimize => unreachable!(),
                };
                changed_in_iteration |= pass_changed;
            }
            if !changed_in_iteration {
                break;
            }
        }
    }

    pub fn minimize_with_rustfst(&mut self) {
        let mut fst = self.to_rustfst();
        minimize(&mut fst).unwrap();
        *self = DWA::from_rustfst(&fst);
    }

    fn run_optimization_experiment(&mut self) {
        let initial_clone = self.clone();
        let initial_states = self.states.len();
        println!("[DWA Optimize] Starting experiment with {} states.", initial_states);

        let mut best_result: Option<(DWA, std::time::Duration, usize)> = None;

        for (i, &ordering) in DWA_PASS_ORDERINGS.iter().enumerate() {
            let mut dwa = initial_clone.clone();
            let start_time = std::time::Instant::now();
            let mut iterations = 0;
            let mut timed_out = false;
            let mut last_changing_passes: Vec<DwaPass> = Vec::new();

            loop {
                if iterations >= MAX_OPTIMIZE_ITERATIONS {
                    timed_out = true;
                    break;
                }
                iterations += 1;

                let mut changed_in_iteration = false;
                let mut current_changing_passes: Vec<DwaPass> = Vec::new();
                for &pass in ordering {
                    let changed = match pass {
                        DwaPass::PruneUnreachable => dwa.prune_unreachable(),
                        DwaPass::PruneDeadEnds => dwa.prune_dead_ends(),
                        DwaPass::PushWeights => dwa.push_weights_into_transitions_and_finals(),
                        DwaPass::Minimize => dwa.minimize_states(),
                    };
                    if changed {
                        current_changing_passes.push(pass);
                    }
                    changed_in_iteration |= changed;
                }
                if !changed_in_iteration {
                    last_changing_passes.clear();
                    break;
                } else {
                    last_changing_passes = current_changing_passes;
                }
            }

            let elapsed = start_time.elapsed();
            let final_states = dwa.states.len();

            let ordering_str = format!("{:?}", ordering);
            let timeout_str = if timed_out {
                format!(" (TIMED OUT, changing: {:?})", last_changing_passes)
            } else {
                "".to_string()
            };
            println!("[DWA Optimize] Ordering #{}: {}, Time: {:.2?}, States: {}{}", i, ordering_str, elapsed, final_states, timeout_str);

            if !timed_out && best_result.as_ref().map_or(true, |(_, best_time, best_states)| {
                final_states < *best_states || (final_states == *best_states && elapsed < *best_time)
            }) {
                best_result = Some((dwa, elapsed, final_states));
            }
        }

        if let Some((best_dwa, _, _)) = best_result {
            *self = best_dwa;
        }
    }

    fn simplify_with_rustfst(&mut self) -> bool {
        let min_config = MinimizeConfig::default();
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, min_config).unwrap();
        *self = DWA::from_rustfst(&fst);
        true
    }

    fn simplify_internal(&mut self) -> bool {
        crate::debug!(5, "[DWA::simplify] Starting simplification. Initial stats: {}", self.stats());
        let mut total_changed = false;
        let ordering = &[
            DwaPass::PruneDeadEnds,
            DwaPass::Minimize,
            DwaPass::PushWeights,
            DwaPass::PruneUnreachable,
        ];
        let mut last_changing_passes: Vec<DwaPass> = vec![];
        let mut converged = false;

        for _ in 0..MAX_OPTIMIZE_ITERATIONS {
            let mut current_changing_passes = vec![];
            let mut changed_in_iteration = false;
            for &pass in ordering {
                let pass_changed = match pass {
                    DwaPass::PruneUnreachable => self.prune_unreachable(),
                    DwaPass::PruneDeadEnds => self.prune_dead_ends(),
                    DwaPass::PushWeights => self.push_weights_into_transitions_and_finals(),
                    DwaPass::Minimize => self.minimize_states(),
                };
                if pass_changed {
                    current_changing_passes.push(pass);
                }
                changed_in_iteration |= pass_changed;
            }

            total_changed |= changed_in_iteration;
            if !changed_in_iteration {
                converged = true;
                break;
            }
            last_changing_passes = current_changing_passes;
        }

        if !converged {
            crate::debug!(3,
                "DWA simplification did not converge after {} iterations. Still changing: {:?}",
                MAX_OPTIMIZE_ITERATIONS, last_changing_passes
            );
        }

        crate::debug!(5, "[DWA::simplify] Simplification finished. Total changed: {}. Final stats: {}", total_changed, self.stats());
        total_changed
    }

    fn push_weights_into_transitions_and_finals(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }
        let start = self.body.start_state;
        if start >= n {
            return false;
        }

        let mut changed = false;
        let mut preds: Vec<Vec<(StateID, Label)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for (&label, &v) in &st.transitions {
                if v < n {
                    preds[v].push((u, label));
                }
            }
        }

        for v in 0..n {
            if v == start {
                continue;
            }
            if let Some(sw) = self.states[v].state_weight.take() {
                if sw.is_empty() {
                    changed = true;
                    for (u, label) in &preds[v] {
                        if let Some(w) = self.states[*u].trans_weights.get_mut(label) {
                            *w &= &sw;
                        }
                    }
                } else if sw != Weight::all() {
                    changed = true;
                    for (u, label) in &preds[v] {
                        if let Some(w) = self.states[*u].trans_weights.get_mut(label) {
                            *w &= &sw;
                        }
                    }
                } else {
                    changed = true;
                }
            }
        }

        if let Some(sw0) = self.states[start].state_weight.take() {
            if !sw0.is_empty() && sw0 != Weight::all() {
                changed = true;
                for st in &mut self.states.0 {
                    if let Some(ref mut fw) = st.final_weight {
                        *fw &= &sw0;
                    }
                }
            } else if sw0.is_empty() {
                changed = true;
                for st in &mut self.states.0 {
                    if let Some(ref mut fw) = st.final_weight {
                        *fw &= &sw0;
                    }
                }
            } else {
                changed = true;
            }
        }

        changed
    }

    fn minimize_states(&mut self) -> bool {
        let n = self.states.len();
        if n <= 1 {
            return false;
        }
        let partition = minimize_dwa_partition(&self.states);
        if partition.num_classes() >= n {
            return false;
        }
        self.rebuild_from_partition(partition);
        true
    }

    fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        if n == 0 {
            return;
        }
        let mut class_to_new: HashMap<usize, StateID> = HashMap::new();
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

    pub fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }
        if self.body.start_state >= n {
            let changed = n > 0;
            if changed {
                self.states = DWAStates::default();
                let start = self.states.add_state();
                self.body.start_state = start;
            }
            return changed;
        }

        let mut visited = vec![false; n];
        let mut q: VecDeque<StateID> = VecDeque::new();
        visited[self.body.start_state] = true;
        q.push_back(self.body.start_state);

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];
            for &v in st.transitions.values() {
                if v < n && !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }

        if visited.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = DWAStates::default();

        for i in 0..n {
            if visited[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            let mut new_transitions: BTreeMap<Label, StateID> = BTreeMap::new();
            let mut new_trans_weights: BTreeMap<Label, Weight> = BTreeMap::new();
            for (&label, &old_dest) in &st.transitions {
                if old_dest < n && visited[old_dest] {
                    let new_dest = map[old_dest];
                    new_transitions.insert(label, new_dest);
                    if let Some(w) = st.trans_weights.get(&label) {
                        new_trans_weights.insert(label, w.clone());
                    }
                }
            }
            st.transitions = new_transitions;
            st.trans_weights = new_trans_weights;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }

    pub fn prune_dead_ends(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        let mut live = vec![false; n];
        let mut q: VecDeque<StateID> = VecDeque::new();
        let mut rev: Vec<Vec<StateID>> = vec![vec![]; n];

        for u in 0..n {
            let st = &self.states[u];
            for (&label, &v) in &st.transitions {
                if v >= n {
                    continue;
                }
                let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
                if !w.is_empty() {
                    rev[v].push(u);
                }
            }
        }

        for s in 0..n {
            if self.states[s].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                live[s] = true;
                q.push_back(s);
            }
        }

        while let Some(v) = q.pop_front() {
            for &u in &rev[v] {
                if !live[u] {
                    live[u] = true;
                    q.push_back(u);
                }
            }
        }

        if self.body.start_state >= n || !live[self.body.start_state] {
            if n == 1 && self.states[0] == DWAState::default() {
                return false;
            }

            let changed = n > 0;
            if changed {
                self.states = DWAStates::default();
                let start = self.states.add_state();
                self.body.start_state = start;
            }
            return changed;
        }

        if live.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = DWAStates::default();

        for i in 0..n {
            if live[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            let mut new_transitions: BTreeMap<Label, StateID> = BTreeMap::new();
            let mut new_trans_weights: BTreeMap<Label, Weight> = BTreeMap::new();
            for (&label, &old_dest) in &st.transitions {
                if old_dest < n && live[old_dest] {
                    let new_dest = map[old_dest];
                    new_transitions.insert(label, new_dest);
                    if let Some(w) = st.trans_weights.get(&label) {
                        new_trans_weights.insert(label, w.clone());
                    }
                }
            }
            st.transitions = new_transitions;
            st.trans_weights = new_trans_weights;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }
}

// ---------------- NWA minimization ----------------

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

        // Sort by (label, dest_class) to make equal-keys contiguous.
        tmp.sort_by_key(|sig| sig.sort_key());

        // Compress runs with the same (label, dest_class) by OR-ing the weights.
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

fn minimize_nwa_partition(states: &NWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition { class_of: vec![], num_classes: 0 };
    }

    let mut partition = Partition::new(n);
    loop {
        let mut sig_to_class: HashMap<NwaStateSignature, usize> = HashMap::new();
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

#[derive(Clone, Debug, Default)]
struct NwaStateBuilder {
    final_weight: Option<Weight>,
    // Use flat Vectors instead of BTreeMap for fast accumulation and bulk sort/merge
    eps: Vec<(NWAStateID, Weight)>,
    trans: Vec<(Label, NWAStateID, Weight)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NwaPass {
    PruneUnreachable,
    PruneDeadEnds,
    PushFinalWeights,
    CompressTransitions,
    Minimize,
}

const NWA_PASS_ORDERINGS: &[&[NwaPass]] = &[
    &[NwaPass::PruneUnreachable, NwaPass::PushFinalWeights, NwaPass::CompressTransitions, NwaPass::PruneDeadEnds, NwaPass::Minimize],
    &[NwaPass::Minimize, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds, NwaPass::PushFinalWeights, NwaPass::CompressTransitions],
    &[NwaPass::CompressTransitions, NwaPass::PushFinalWeights, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds, NwaPass::Minimize],
    &[NwaPass::PruneUnreachable, NwaPass::CompressTransitions, NwaPass::PushFinalWeights, NwaPass::PruneDeadEnds, NwaPass::Minimize],
    &[NwaPass::Minimize, NwaPass::CompressTransitions, NwaPass::PushFinalWeights, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds],
    &[NwaPass::PushFinalWeights, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds, NwaPass::CompressTransitions, NwaPass::Minimize],
    &[NwaPass::PruneUnreachable, NwaPass::PushFinalWeights, NwaPass::Minimize, NwaPass::CompressTransitions, NwaPass::PruneDeadEnds],
    &[NwaPass::CompressTransitions, NwaPass::PushFinalWeights, NwaPass::Minimize, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds],
    &[NwaPass::PushFinalWeights, NwaPass::CompressTransitions, NwaPass::Minimize, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds],
    &[NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds, NwaPass::CompressTransitions, NwaPass::PushFinalWeights, NwaPass::Minimize],
];

impl NWA {
    pub fn simplify(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        if OPTIMIZE_DEBUG {
            self.run_optimization_experiment();
        } else if BENCHMARK_DEBUG {
            let initial_states = self.states.len();
            let mut internal = self.clone();
            let internal_start = std::time::Instant::now();
            internal.simplify_internal();
            let internal_time = internal_start.elapsed();
            let internal_states = internal.states.len();

            let mut rustfst = self.clone();
            let rustfst_start = std::time::Instant::now();
            rustfst.simplify_with_rustfst();
            let rustfst_time = rustfst_start.elapsed();
            let rustfst_states = rustfst.states.len();

            if internal_time + rustfst_time > std::time::Duration::from_secs(1) {
                let state_cmp = match internal_states.cmp(&rustfst_states) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };
                let time_cmp = match internal_time.cmp(&rustfst_time) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };

                crate::debug!(5, "[NWA Simplify({})] Internal: t={:.2?}, s={} | RustFST: t={:.2?}, s={}. [s: {}, t: {}]", initial_states, internal_time, internal_states, rustfst_time, rustfst_states, state_cmp, time_cmp);
            }

            *self = internal;
        } else {
            self.simplify_internal();
        }
    }

    pub fn minimize_with_rustfst(&mut self) {
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, MinimizeConfig::default().with_allow_nondet(true)).unwrap();
        *self = NWA::from_rustfst(&fst);
    }

    fn run_optimization_experiment(&mut self) {
        let initial_clone = self.clone();
        let initial_states = self.states.len();
        println!("[NWA Optimize] Starting experiment with {} states.", initial_states);

        let mut best_result: Option<(NWA, std::time::Duration, usize)> = None;

        for (i, &ordering) in NWA_PASS_ORDERINGS.iter().enumerate() {
            let mut nwa = initial_clone.clone();
            let start_time = std::time::Instant::now();
            let mut iterations = 0;
            let mut timed_out = false;
            let mut last_changing_passes: Vec<NwaPass> = Vec::new();

            loop {
                if iterations >= MAX_OPTIMIZE_ITERATIONS {
                    timed_out = true;
                    break;
                }
                iterations += 1;

                let mut changed_in_iteration = false;
                let mut current_changing_passes: Vec<NwaPass> = Vec::new();
                for &pass in ordering {
                    let changed = match pass {
                        NwaPass::PruneUnreachable => nwa.prune_unreachable(),
                        NwaPass::PruneDeadEnds => nwa.prune_dead_ends(),
                        NwaPass::PushFinalWeights => nwa.push_final_weights_along_epsilons(),
                        NwaPass::CompressTransitions => nwa.compress_transitions(),
                        NwaPass::Minimize => nwa.minimize_states(),
                    };
                    if changed {
                        current_changing_passes.push(pass);
                    }
                    changed_in_iteration |= changed;
                }
                if !changed_in_iteration {
                    last_changing_passes.clear();
                    break;
                } else {
                    last_changing_passes = current_changing_passes;
                }
            }

            let elapsed = start_time.elapsed();
            let final_states = nwa.states.len();

            let ordering_str = format!("{:?}", ordering);
            let timeout_str = if timed_out {
                format!(" (TIMED OUT, changing: {:?})", last_changing_passes)
            } else {
                "".to_string()
            };
            println!("[NWA Optimize] Ordering #{}: {}, Time: {:.2?}, States: {}{}", i, ordering_str, elapsed, final_states, timeout_str);

            if !timed_out && best_result.as_ref().map_or(true, |(_, best_time, best_states)| {
                final_states < *best_states || (final_states == *best_states && elapsed < *best_time)
            }) {
                best_result = Some((nwa, elapsed, final_states));
            }
        }

        if let Some((best_nwa, _, _)) = best_result {
            *self = best_nwa;
        }
    }

    pub fn simplify_with_rustfst(&mut self) -> bool {
        let min_config = MinimizeConfig::default().with_allow_nondet(true);
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, min_config).unwrap();
        *self = NWA::from_rustfst(&fst);
        true
    }

    pub fn simplify_internal(&mut self) -> bool {
        crate::debug!(5, "[NWA::simplify] Starting simplification. Initial stats: {}", self.stats());
        let mut total_changed = false;
        // PruneUnreachable, CompressTransitions, PushFinalWeights, PruneDeadEnds, Minimize
        let ordering = &[
            NwaPass::PruneUnreachable,
            NwaPass::CompressTransitions,
            NwaPass::PushFinalWeights,
            NwaPass::PruneDeadEnds,
            NwaPass::Minimize,
        ];
        let mut last_changing_passes: Vec<NwaPass> = vec![];
        let mut converged = false;

        for _ in 0..MAX_OPTIMIZE_ITERATIONS {
            let mut current_changing_passes = vec![];
            let mut changed_in_iteration = false;
            for &pass in ordering {
                let pass_changed = match pass {
                    NwaPass::PruneUnreachable => self.prune_unreachable(),
                    NwaPass::PruneDeadEnds => self.prune_dead_ends(),
                    NwaPass::PushFinalWeights => self.push_final_weights_along_epsilons(),
                    NwaPass::CompressTransitions => self.compress_transitions(),
                    NwaPass::Minimize => self.minimize_states(),
                };
                if pass_changed {
                    current_changing_passes.push(pass);
                }
                changed_in_iteration |= pass_changed;
            }

            total_changed |= changed_in_iteration;
            if !changed_in_iteration {
                converged = true;
                break;
            }
            last_changing_passes = current_changing_passes;
        }

        if !converged {
            crate::debug!(3,
                "NWA simplification did not converge after {} iterations. Still changing: {:?}",
                MAX_OPTIMIZE_ITERATIONS, last_changing_passes
            );
        }

        crate::debug!(5, "[NWA::simplify] Simplification finished. Total changed: {}. Final stats: {}", total_changed, self.stats());
        total_changed
    }

    fn minimize_states(&mut self) -> bool {
        crate::debug!(6, "[NWA] Minimizing states...");
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

    /// Canonicalize NWA transitions by merging parallel transitions via in-place sort-and-scan.
    /// This avoids allocating BTreeMaps for aggregation, providing a significant speedup.
    fn compress_transitions(&mut self) -> bool {
        crate::debug!(6, "[NWA] Compressing transitions...");
        let mut changed = false;

        for st in &mut self.states.0 {
            // Compress epsilons
            if !st.epsilons.is_empty() {
                st.epsilons.sort_by(|a, b| a.0.cmp(&b.0));
                
                let len = st.epsilons.len();
                // Scan to remove empty and merge duplicates
                let mut w = 0;
                let mut r = 0;

                // Skip initial empty weights
                while r < len && st.epsilons[r].1.is_empty() {
                    r += 1;
                }

                if r < len {
                    if r != w {
                        st.epsilons.swap(r, w);
                    }
                    // Current tail is at w. Scan from next.
                    r += 1;
                    
                    while r < len {
                        if st.epsilons[r].1.is_empty() {
                            r += 1;
                            continue;
                        }
                        
                        if st.epsilons[r].0 == st.epsilons[w].0 {
                            // Merge into w
                            let weight = std::mem::take(&mut st.epsilons[r].1);
                            st.epsilons[w].1 |= &weight;
                        } else {
                            // New distinct dest
                            w += 1;
                            if r != w {
                                st.epsilons.swap(r, w);
                            }
                        }
                        r += 1;
                    }
                    
                    let new_len = w + 1;
                    if new_len < len {
                        st.epsilons.truncate(new_len);
                        changed = true;
                    }
                } else {
                    // All were empty
                    st.epsilons.clear();
                    changed = true;
                }
            }

            // Compress labeled transitions
            if !st.transitions.is_empty() {
                for targets in st.transitions.values_mut() {
                    if targets.is_empty() { continue; }

                    targets.sort_by(|a, b| a.0.cmp(&b.0));

                    let len = targets.len();
                    let mut w = 0;
                    let mut r = 0;

                    while r < len && targets[r].1.is_empty() {
                        r += 1;
                    }

                    if r < len {
                        if r != w {
                            targets.swap(r, w);
                        }
                        r += 1;

                        while r < len {
                            if targets[r].1.is_empty() {
                                r += 1;
                                continue;
                            }

                            if targets[r].0 == targets[w].0 {
                                let weight = std::mem::take(&mut targets[r].1);
                                targets[w].1 |= &weight;
                            } else {
                                w += 1;
                                if r != w {
                                    targets.swap(r, w);
                                }
                            }
                            r += 1;
                        }

                        let new_len = w + 1;
                        if new_len < len {
                            targets.truncate(new_len);
                            changed = true;
                        }
                    } else {
                        targets.clear();
                        changed = true;
                    }
                }
                
                // Clean up empty keys (rare, but possible if all targets were empty weights)
                let old_len = st.transitions.len();
                st.transitions.retain(|_, v| !v.is_empty());
                if st.transitions.len() != old_len {
                    changed = true;
                }
            }
        }

        changed
    }

    fn push_final_weights_along_epsilons(&mut self) -> bool {
        crate::debug!(6, "[NWA] Pushing final weights along epsilons...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        // Build reverse epsilon adjacency: for each target v, collect (u, w_uv) with u --w_uv--> v.
        let mut rev_eps: Vec<Vec<(NWAStateID, Weight)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for &(v, ref w) in &st.epsilons {
                if v < n && !w.is_empty() {
                    rev_eps[v].push((u, w.clone()));
                }
            }
        }

        // Initialise final_weights with the existing final weights.
        let mut final_weights: Vec<Weight> = Vec::with_capacity(n);
        let mut queue: VecDeque<NWAStateID> = VecDeque::new();
        for i in 0..n {
            let w = self.states.0[i]
                .final_weight
                .clone()
                .unwrap_or_else(Weight::zeros);
            if !w.is_empty() {
                queue.push_back(i);
            }
            final_weights.push(w);
        }

        let mut changed = false;

        // Worklist fixpoint for F(q) = f0(q) ∨ ⋁_{q --w--> r} (w ∧ F(r)).
        while let Some(v) = queue.pop_front() {
            let w_v = final_weights[v].clone();
            if w_v.is_empty() {
                continue;
            }

            for &(u, ref w_uv) in &rev_eps[v] {
                let candidate = &w_v & w_uv;
                if candidate.is_empty() {
                    continue;
                }
                let new_w = &final_weights[u] | &candidate;
                if new_w != final_weights[u] {
                    final_weights[u] = new_w;
                    queue.push_back(u);
                }
            }
        }

        for i in 0..n {
            let new_w = &final_weights[i];
            let new_final = if new_w.is_empty() {
                None
            } else {
                Some(new_w.clone())
            };
            if self.states.0[i].final_weight != new_final {
                self.states.0[i].final_weight = new_final;
                changed = true;
            }
        }

        changed
    }

    fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        let mut class_to_new: HashMap<usize, NWAStateID> = HashMap::new();
        let mut builders: Vec<NwaStateBuilder> = Vec::new();

        for s in 0..n {
            let c = partition.class_of[s];
            class_to_new.entry(c).or_insert_with(|| {
                let id = builders.len();
                builders.push(NwaStateBuilder::default());
                id
            });
        }

        // Phase 1: Accumulate transitions into flat vectors
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
                if !w.is_empty() {
                    let dest_class = partition.class_of[dest];
                    let new_dest = class_to_new[&dest_class];
                    builder.eps.push((new_dest, w.clone()));
                }
            }

            for (&lbl, targets) in &st.transitions {
                for &(dest, ref w) in targets {
                    if !w.is_empty() {
                        let dest_class = partition.class_of[dest];
                        let new_dest = class_to_new[&dest_class];
                        builder.trans.push((lbl, new_dest, w.clone()));
                    }
                }
            }
        }

        let mut new_states = NWAStates::default();
        for _ in 0..builders.len() {
            new_states.add_state();
        }

        // Phase 2: Sort, merge, and build final states
        for (new_id, mut builder) in builders.into_iter().enumerate() {
            let st = &mut new_states[new_id];
            st.final_weight = builder.final_weight;

            // --- Epsilons ---
            if !builder.eps.is_empty() {
                builder.eps.sort_by(|a, b| a.0.cmp(&b.0));
                
                let len = builder.eps.len();
                let mut w = 0;
                let mut r = 1;
                
                // In-place merge
                while r < len {
                    if builder.eps[r].0 == builder.eps[w].0 {
                         // Merge
                         let weight = std::mem::take(&mut builder.eps[r].1);
                         builder.eps[w].1 |= &weight;
                    } else {
                        w += 1;
                        if r != w {
                            builder.eps.swap(r, w);
                        }
                    }
                    r += 1;
                }
                builder.eps.truncate(w + 1);
                st.epsilons = builder.eps;
            }

            // --- Transitions ---
            if !builder.trans.is_empty() {
                // Sort by (Label, Dest)
                builder.trans.sort_by(|a, b| {
                    match a.0.cmp(&b.0) {
                        std::cmp::Ordering::Equal => a.1.cmp(&b.1),
                        other => other,
                    }
                });

                // In-place merge duplicates
                let len = builder.trans.len();
                let mut w = 0;
                let mut r = 1;

                while r < len {
                    if builder.trans[r].0 == builder.trans[w].0 && builder.trans[r].1 == builder.trans[w].1 {
                        let weight = std::mem::take(&mut builder.trans[r].2);
                        builder.trans[w].2 |= &weight;
                    } else {
                        w += 1;
                        if r != w {
                            builder.trans.swap(r, w);
                        }
                    }
                    r += 1;
                }
                builder.trans.truncate(w + 1);

                // Group by Label into BTreeMap
                st.transitions.clear();
                let mut i = 0;
                while i < builder.trans.len() {
                    let current_lbl = builder.trans[i].0;
                    let mut vec = Vec::new();
                    
                    while i < builder.trans.len() && builder.trans[i].0 == current_lbl {
                        vec.push((builder.trans[i].1, builder.trans[i].2.clone()));
                        i += 1;
                    }
                    st.transitions.insert(current_lbl, vec);
                }
            }
        }

        let start_class = partition.class_of[self.body.start_state];
        let new_start = class_to_new[&start_class];
        self.states = new_states;
        self.body.start_state = new_start;
    }

    fn prune_unreachable(&mut self) -> bool {
        crate::debug!(6, "[NWA] Pruning unreachable states...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        if self.body.start_state >= n {
            let changed = n > 0;
            if changed {
                self.states = NWAStates::default();
                let start = self.states.add_state();
                self.body.start_state = start;
            }
            return changed;
        }

        let mut reachable = vec![false; n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        reachable[self.body.start_state] = true;
        q.push_back(self.body.start_state);

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];

            for &(v, ref w) in &st.epsilons {
                if v < n && !reachable[v] && !w.is_empty() {
                    reachable[v] = true;
                    q.push_back(v);
                }
            }

            for (_, targets) in &st.transitions {
                for &(v, ref w) in targets {
                    if v < n && !reachable[v] && !w.is_empty() {
                        reachable[v] = true;
                        q.push_back(v);
                    }
                }
            }
        }

        if reachable.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = NWAStates::default();

        for i in 0..n {
            if reachable[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            st.epsilons.retain(|(v, w)| *v < n && !w.is_empty());
            for (v, _) in &mut st.epsilons {
                *v = map[*v];
            }

            let mut new_transitions: BTreeMap<Label, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
            for (&lbl, targets) in &st.transitions {
                let mut new_targets = Vec::new();
                for &(v, ref w) in targets {
                    if v < n && !w.is_empty() && reachable[v] {
                        new_targets.push((map[v], w.clone()));
                    }
                }
                if !new_targets.is_empty() {
                    new_transitions.insert(lbl, new_targets);
                }
            }
            st.transitions = new_transitions;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }

    fn prune_dead_ends(&mut self) -> bool {
        crate::debug!(6, "[NWA] Pruning dead ends...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        let mut live = vec![false; n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        let mut rev: Vec<Vec<NWAStateID>> = vec![vec![]; n];

        for p in 0..n {
            let st = &self.states[p];
            for &(t, ref w) in &st.epsilons {
                if t < n && !w.is_empty() {
                    rev[t].push(p);
                }
            }
            for (_, targets) in &st.transitions {
                for &(t, ref w) in targets {
                    if t < n && !w.is_empty() {
                        rev[t].push(p);
                    }
                }
            }
        }

        for s in 0..n {
            if self.states[s].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                if !live[s] {
                    live[s] = true;
                    q.push_back(s);
                }
            }
        }

        while let Some(v) = q.pop_front() {
            for &p in &rev[v] {
                if !live[p] {
                    live[p] = true;
                    q.push_back(p);
                }
            }
        }

        if self.body.start_state >= n || !live[self.body.start_state] {
            if n == 1 && self.states[0] == NWAState::default() {
                return false;
            }

            let changed = n > 0;
            if changed {
                self.states = NWAStates::default();
                let start = self.states.add_state();
                self.body.start_state = start;
            }
            return changed;
        }

        if live.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = NWAStates::default();

        for i in 0..n {
            if live[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            st.epsilons.retain(|(v, w)| *v < n && !w.is_empty() && live[*v]);
            for (v, _) in &mut st.epsilons {
                *v = map[*v];
            }

            let mut new_transitions: BTreeMap<Label, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
            for (&lbl, targets) in &st.transitions {
                let mut new_targets = Vec::new();
                for &(v, ref w) in targets {
                    if v < n && !w.is_empty() && live[v] {
                        new_targets.push((map[v], w.clone()));
                    }
                }
                if !new_targets.is_empty() {
                    new_transitions.insert(lbl, new_targets);
                }
            }
            st.transitions = new_transitions;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }
}
