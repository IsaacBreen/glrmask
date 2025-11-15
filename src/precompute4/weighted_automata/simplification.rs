#![allow(dead_code)]

use super::common::{StateID, Weight, NWAStateID};
use super::dwa::{DWAState, DWAStates, DWA};
use super::nwa::{NWAState, NWAStates, NWA};
use std::cmp::Ordering;
use std::collections::{hash_map::DefaultHasher, BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

/// Partition of states into equivalence classes.
#[derive(Clone, Debug)]
struct Partition {
    /// Class id of each state (by index).
    classes: Vec<usize>,
    /// Number of distinct classes.
    num_classes: usize,
}

impl Partition {
    fn new(num_states: usize) -> Self {
        Self {
            classes: vec![0; num_states],
            num_classes: if num_states == 0 { 0 } else { 1 },
        }
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Helper: hash any Hash + Eq type into a u64.
fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

// ============================================================================
// DWA MINIMIZATION
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaTransitionSig {
    label: i16,
    dest_class: usize,
    weight: Weight,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaStateSignature {
    state_weight: Option<Weight>,
    final_weight: Option<Weight>,
    outgoing: Vec<DwaTransitionSig>,
}

impl DwaStateSignature {
    fn from_state(state_id: StateID, states: &DWAStates, classes: &[usize]) -> Self {
        let st = &states[state_id];

        // Use a BTreeMap to get a deterministic ordering by label.
        let mut temp: BTreeMap<i16, (usize, Weight)> = BTreeMap::new();
        for (&label, &dest) in &st.transitions {
            let w = st
                .trans_weights
                .get(&label)
                .cloned()
                .unwrap_or_else(Weight::all);
            let dest_class = classes[dest];
            // Determinism: at most one dest per label; if multiple appear,
            // we union weights and assert same dest_class.
            match temp.get_mut(&label) {
                None => {
                    temp.insert(label, (dest_class, w));
                }
                Some((existing_dest_class, existing_w)) => {
                    debug_assert_eq!(
                        *existing_dest_class, dest_class,
                        "DWA determinism violated: multiple destinations for label {}",
                        label
                    );
                    *existing_w |= &w;
                }
            }
        }

        let mut outgoing = Vec::with_capacity(temp.len());
        for (label, (dest_class, weight)) in temp {
            outgoing.push(DwaTransitionSig {
                label,
                dest_class,
                weight,
            });
        }

        DwaStateSignature {
            state_weight: st.state_weight.clone(),
            final_weight: st.final_weight.clone(),
            outgoing,
        }
    }
}

/// Compute a forward-bisimulation style partition for a DWA.
fn minimize_dwa_partition(states: &DWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition {
            classes: vec![],
            num_classes: 0,
        };
    }

    // Intern all (ArcLabel, Weight) pairs to cheap integer IDs.
    let mut interner: HashMap<(ArcLabel, Weight), usize> = HashMap::new();
    for s in 0..n {
        let st = &states[s];
        // Epsilon transitions
        for &(_, ref w) in &st.epsilons {
            if !w.is_empty() {
                let key = (ArcLabel::Eps, w.clone());
                if !interner.contains_key(&key) {
                    let id = interner.len();
                    interner.insert(key, id);
                }
            }
        }
        // Labeled transitions
        for (&lbl, targets) in &st.transitions {
            let label = ArcLabel::Label(lbl);
            for &(_, ref w) in targets {
                if !w.is_empty() {
                    let key = (label, w.clone());
                    if !interner.contains_key(&key) {
                        let id = interner.len();
                        interner.insert(key, id);
                    }
                }
            }
        }
    }

    let mut partition = Partition::new(n);

    loop {
        let mut sig_to_class: HashMap<DwaStateSignature, usize> = HashMap::new();
        let mut new_classes = vec![0; n];
        let mut next_class = 0;

        for s in 0..n {
            let sig = DwaStateSignature::from_state(s, states, &partition.classes);
            let entry = sig_to_class.entry(sig).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
            new_classes[s] = *entry;
        }

        if new_classes == partition.classes {
            partition.num_classes = next_class;
            return partition;
        }

        partition.classes = new_classes;
        partition.num_classes = next_class;
    }
}

#[derive(Clone, Debug, Default)]
struct DwaStateBuilder {
    state_weight: Option<Weight>,
    final_weight: Option<Weight>,
    // label -> (dest_new, weight)
    trans: BTreeMap<i16, (StateID, Weight)>,
}

impl DWA {
    /// Simplify the DWA in-place by:
    /// 1. Pruning unreachable and dead-end states.
    /// 2. Computing a forward-bisimulation partition over weighted states.
    /// 3. Building the quotient automaton by merging equivalent states.
    /// 4. Pruning again.
    pub fn simplify(&mut self) {
        let _ = self.simplify_internal();
    }

    fn simplify_internal(&mut self) -> bool {
        let mut changed = self.prune_unreachable();
        changed |= self.prune_dead_ends();

        let n = self.states.len();
        if n <= 1 {
            return changed;
        }

        let partition = minimize_dwa_partition(&self.states);
        if partition.num_classes() >= n {
            // No merges.
            return changed;
        }

        self.rebuild_from_partition(partition);
        changed = true;

        // Final cleanup.
        changed |= self.prune_unreachable();
        changed |= self.prune_dead_ends();
        changed
    }

    fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Map each class to a new state id (0..num_classes-1).
        let mut class_to_new: HashMap<usize, StateID> = HashMap::new();
        let mut builders: Vec<DwaStateBuilder> = Vec::new();

        for s in 0..n {
            let c = partition.classes[s];
            class_to_new.entry(c).or_insert_with(|| {
                let id = builders.len();
                builders.push(DwaStateBuilder::default());
                id
            });
        }

        // Aggregate information from all states in each class.
        for old_s in 0..n {
            let c = partition.classes[old_s];
            let new_id = class_to_new[&c];
            let builder = &mut builders[new_id];
            let st = &self.states[old_s];

            if let Some(ref sw) = st.state_weight {
                if !sw.is_empty() {
                    match &mut builder.state_weight {
                        Some(existing) => *existing |= sw,
                        None => builder.state_weight = Some(sw.clone()),
                    }
                }
            }

            if let Some(ref fw) = st.final_weight {
                if !fw.is_empty() {
                    match &mut builder.final_weight {
                        Some(existing) => *existing |= fw,
                        None => builder.final_weight = Some(fw.clone()),
                    }
                }
            }

            for (&label, &dest) in &st.transitions {
                let w = st
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                if w.is_empty() {
                    continue;
                }
                let dest_class = partition.classes[dest];
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
                            "DWA determinism violated while rebuilding: multiple destinations for label {}",
                            label
                        );
                        *existing_w |= &w;
                    }
                }
            }
        }

        // Construct new states vector.
        let mut new_states = DWAStates::default();
        for _ in 0..builders.len() {
            new_states.add_state();
        }

        for (new_id, builder) in builders.into_iter().enumerate() {
            let st = &mut new_states[new_id];
            st.state_weight = builder.state_weight;
            st.final_weight = builder.final_weight;
            st.transitions.clear();
            st.trans_weights.clear();
            for (label, (dest_new, weight)) in builder.trans {
                st.transitions.insert(label, dest_new);
                st.trans_weights.insert(label, weight);
            }
        }

        let start_class = partition.classes[self.body.start_state];
        let new_start = class_to_new[&start_class];

        self.states = new_states;
        self.body.start_state = new_start;
    }

    /// Remove states that are not reachable from the start state.
    fn prune_unreachable(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        if self.body.start_state >= n {
            // Invalid start; reset to a singleton empty DWA.
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
            let mut new_transitions: BTreeMap<i16, StateID> = BTreeMap::new();
            let mut new_trans_weights: BTreeMap<i16, Weight> = BTreeMap::new();
            for (&label, &old_dest) in &st.transitions {
                let new_dest = map[old_dest];
                new_transitions.insert(label, new_dest);
                if let Some(w) = st.trans_weights.get(&label) {
                    new_trans_weights.insert(label, w.clone());
                }
            }
            st.transitions = new_transitions;
            st.trans_weights = new_trans_weights;
        }

        self.body.start_state = map[self.body.start_state];
        self.states = new_states;
        true
    }

    /// Remove states that cannot reach any state with a non-empty final weight.
    fn prune_dead_ends(&mut self) -> bool {
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
                let w = st
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                if !w.is_empty() {
                    rev[v].push(u);
                }
            }
        }

        // Any state with non-empty final weight is live.
        for s in 0..n {
            if self.states[s]
                .final_weight
                .as_ref()
                .map_or(false, |w| !w.is_empty())
            {
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

        // If start state is not live, the language is empty.
        if self.body.start_state >= n || !live[self.body.start_state] {
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
            let mut new_transitions: BTreeMap<i16, StateID> = BTreeMap::new();
            let mut new_trans_weights: BTreeMap<i16, Weight> = BTreeMap::new();
            for (&label, &old_dest) in &st.transitions {
                if live[old_dest] {
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

// ============================================================================
// NWA MINIMIZATION
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ArcLabel {
    Eps,
    Label(i16),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaTransitionSig {
    trans_id: usize,
    // Sorted list of destination classes.
    dest_classes: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaStateSignature {
    final_weight: Option<Weight>,
    outgoing: Vec<NwaTransitionSig>,
}

impl NwaStateSignature {
    fn from_state(
        state_id: NWAStateID,
        states: &NWAStates,
        classes: &[usize],
        interner: &HashMap<(ArcLabel, Weight), usize>,
    ) -> Self {
        let st = &states[state_id];

        // Map trans_id -> set of destination classes.
        let mut temp: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();

        // Epsilon transitions.
        for &(dest, ref w) in &st.epsilons {
            if w.is_empty() {
                continue;
            }
            let key = (ArcLabel::Eps, w.clone());
            let trans_id = interner[&key];
            temp.entry(trans_id).or_default().insert(classes[dest]);
        }

        // Labeled transitions.
        for (&lbl, targets) in &st.transitions {
            let label = ArcLabel::Label(lbl);
            for &(dest, ref w) in targets {
                if w.is_empty() {
                    continue;
                }
                let key = (label, w.clone());
                let trans_id = interner[&key];
                temp.entry(trans_id).or_default().insert(classes[dest]);
            }
        }

        let mut outgoing = Vec::with_capacity(temp.len());
        for (trans_id, dest_set) in temp {
            let dest_classes: Vec<usize> = dest_set.into_iter().collect();
            outgoing.push(NwaTransitionSig {
                trans_id,
                dest_classes,
            });
        }

        NwaStateSignature {
            final_weight: st.final_weight.clone(),
            outgoing,
        }
    }
}

/// Compute a forward-bisimulation style partition for an NWA.
fn minimize_nwa_partition(states: &NWAStates) -> Partition {
    let n = states.len();
    if n == 0 {
        return Partition {
            classes: vec![],
            num_classes: 0,
        };
    }

    let mut partition = Partition::new(n);

    loop {
        let mut sig_to_class: HashMap<NwaStateSignature, usize> = HashMap::new();
        let mut new_classes = vec![0; n];
        let mut next_class = 0;

        for s in 0..n {
            let sig = NwaStateSignature::from_state(s, states, &partition.classes, &interner);
            let entry = sig_to_class.entry(sig).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
            new_classes[s] = *entry;
        }

        if new_classes == partition.classes {
            partition.num_classes = next_class;
            return partition;
        }

        partition.classes = new_classes;
        partition.num_classes = next_class;
    }
}

#[derive(Clone, Debug, Default)]
struct NwaStateBuilder {
    final_weight: Option<Weight>,
    // eps: dest_new -> weight
    eps: BTreeMap<NWAStateID, Weight>,
    // trans: label -> (dest_new -> weight)
    trans: BTreeMap<i16, BTreeMap<NWAStateID, Weight>>,
}

impl NWA {
    /// Simplify the NWA in-place:
    /// 1. Prune unreachable and dead-end states.
    /// 2. Compute a weighted forward-bisimulation partition.
    /// 3. Build the quotient NWA by merging equivalent states.
    /// 4. Prune again.
    ///
    /// Returns `true` if the automaton changed structurally.
    pub fn simplify(&mut self) -> bool {
        println!("prune_unreachable");
        let mut changed = self.prune_unreachable();
        println!("prune_dead_ends");
        changed |= self.prune_dead_ends();

        let n = self.states.len();
        if n <= 1 {
            println!("done");
            return changed;
        }

        println!("minimize_nwa_partition");
        let partition = minimize_nwa_partition(&self.states);
        if partition.num_classes() >= n {
            // No merges.
            println!("done");
            return changed;
        }

        println!("rebuild_from_partition");
        self.rebuild_from_partition(partition);
        changed = true;

        // Final cleanup.
        println!("final prune_unreachable");
        changed |= self.prune_unreachable();
        println!("final prune_dead_ends");
        changed |= self.prune_dead_ends();
        println!("done");
        changed
    }

    fn rebuild_from_partition(&mut self, partition: Partition) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Map each class to a new state id.
        let mut class_to_new: HashMap<usize, NWAStateID> = HashMap::new();
        let mut builders: Vec<NwaStateBuilder> = Vec::new();

        for s in 0..n {
            let c = partition.classes[s];
            class_to_new.entry(c).or_insert_with(|| {
                let id = builders.len();
                builders.push(NwaStateBuilder::default());
                id
            });
        }

        // Aggregate information from states in the same class.
        for old_s in 0..n {
            let c = partition.classes[old_s];
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
                let dest_class = partition.classes[dest];
                let new_dest = class_to_new[&dest_class];
                let entry = builder.eps.entry(new_dest).or_insert_with(Weight::zeros);
                *entry |= w;
            }

            for (&lbl, targets) in &st.transitions {
                for &(dest, ref w) in targets {
                    if w.is_empty() {
                        continue;
                    }
                    let dest_class = partition.classes[dest];
                    let new_dest = class_to_new[&dest_class];
                    let per_label = builder.trans.entry(lbl).or_insert_with(BTreeMap::new);
                    let entry = per_label.entry(new_dest).or_insert_with(Weight::zeros);
                    *entry |= w;
                }
            }
        }

        // Build the new NWAStates.
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

        let start_class = partition.classes[self.body.start_state];
        let new_start = class_to_new[&start_class];

        self.states = new_states;
        self.body.start_state = new_start;
    }

    /// Remove states not reachable from the start state.
    fn prune_unreachable(&mut self) -> bool {
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

            let mut new_transitions: BTreeMap<i16, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
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

    /// Remove states that cannot reach any state with non-empty final weight.
    fn prune_dead_ends(&mut self) -> bool {
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
            if self.states[s]
                .final_weight
                .as_ref()
                .map_or(false, |w| !w.is_empty())
            {
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

        // If the start state is not live, the language is empty.
        if self.body.start_state >= n || !live[self.body.start_state] {
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

            let mut new_transitions: BTreeMap<i16, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
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
