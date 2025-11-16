#![allow(dead_code)]

use super::common::{NWAStateID, StateID, Weight};
use super::dwa::{DWAState, DWAStates, DWA};
use super::nwa::{NWAState, NWAStates, NWA};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use rustfst::algorithms::{minimize_with_config, MinimizeConfig};

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

fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

// ---------------- DWA minimization ----------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DwaTransitionSig {
    label: i16,
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

        // Aggregate transitions by (label, dest_class), summing (unioning) their weights.
        // This is semantically justified since in the bitset semiring we have:
        //   (w1 & x) | (w2 & x) = (w1 | w2) & x
        // so multiple parallel transitions to the same equivalence class under the same
        // label are equivalent to a single transition whose weight is the union.
        let mut agg: BTreeMap<(i16, usize), Weight> = BTreeMap::new();
        for (&label, &dest) in &st.transitions {
            let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
            if w.is_empty() {
                continue;
            }
            let dest_class = classes[dest];
            let key = (label, dest_class);
            agg.entry(key)
                .and_modify(|acc| *acc |= &w)
                .or_insert(w);
        }

        let mut outgoing = Vec::with_capacity(agg.len());
        for ((label, dest_class), weight) in agg {
            if !weight.is_empty() {
                outgoing.push(DwaTransitionSig { label, dest_class, weight });
            }
        }
        outgoing.sort_by(|a, b| {
            a.label
                .cmp(&b.label)
                .then_with(|| a.dest_class.cmp(&b.dest_class))
                .then_with(|| hash_value(&a.weight).cmp(&hash_value(&b.weight)))
        });
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
    trans: BTreeMap<i16, (StateID, Weight)>,
}

impl DWA {
    pub fn simplify(&mut self) {
        crate::debug!(4, "[DWA::simplify] Num states before simplification: {}", self.states.len());
        let instant = std::time::Instant::now();
        let mut internal = self.clone();
        internal.simplify_internal();
        crate::debug!(
            4,
            "[DWA::simplify] Simplification took {:.3} seconds. Num states: {}",
            instant.elapsed().as_secs_f64(),
            internal.states.len()
        );
        let mut rustfst = self.clone();
        rustfst.simplify_with_rustfst();
        crate::debug!(
            4,
            "[DWA::simplify] RustFST minimization took {:.3} seconds. Num states: {}",
            instant.elapsed().as_secs_f64(),
            rustfst.states.len()
        );
        *self = internal;
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
        let mut changed = false;
        changed |= self.prune_unreachable();
        crate::debug!(5, "[DWA::simplify] After prune_unreachable (1): {}", self.stats());
        changed |= self.prune_dead_ends();
        crate::debug!(5, "[DWA::simplify] After prune_dead_ends (1): {}", self.stats());
        changed |= self.push_weights_into_transitions_and_finals();
        crate::debug!(5, "[DWA::simplify] After pushing weights: {}", self.stats());
        changed |= self.minimize_states();
        crate::debug!(5, "[DWA::simplify] After minimizing: {}", self.stats());
        changed |= self.prune_unreachable();
        crate::debug!(5, "[DWA::simplify] After prune_unreachable (2): {}", self.stats());
        changed |= self.prune_dead_ends();
        crate::debug!(5, "[DWA::simplify] After prune_dead_ends (2): {}", self.stats());
        changed
    }
}

// ---------------- NWA minimization ----------------
        let start = self.body.start_state;
        if start >= n {
            return false;
        }

        let mut changed = false;
        let mut preds: Vec<Vec<(StateID, i16)>> = vec![Vec::new(); n];
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

    fn prune_unreachable(&mut self) -> bool {
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
            let mut new_transitions: BTreeMap<i16, StateID> = BTreeMap::new();
            let mut new_trans_weights: BTreeMap<i16, Weight> = BTreeMap::new();
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ArcLabel {
    Eps,
    Label(i16),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NwaTransitionSig {
    label: ArcLabel,
    dest_class: usize,
    weight: Weight,
}

impl NwaTransitionSig {
    fn sort_key(&self) -> (u8, i16, usize, u64) {
        let label_tag = match self.label {
            ArcLabel::Eps => 0,
            ArcLabel::Label(_) => 1,
        };
        let label_val = match self.label {
            ArcLabel::Eps => 0,
            ArcLabel::Label(v) => v,
        };
        let w_hash = hash_value(&self.weight);
        (label_tag, label_val, self.dest_class, w_hash)
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
        // Aggregate transitions by (label, dest_class), summing (unioning) their weights.
        // This matches the semantic equation:
        //   L_s(a v) = ⋁_C ( W_s(a, C) & L_C(v) )
        // where W_s(a, C) is the union of weights of all transitions s -a,w-> t with t in class C.
        // Using aggregated weights per (label, dest_class) gives a right-invariant equivalence
        // that is exactly language-preserving for the bitset-weight semantics.
        let mut temp: BTreeMap<(ArcLabel, usize), Weight> = BTreeMap::new();

        // Epsilon transitions
        for &(dest, ref w) in &st.epsilons {
            if w.is_empty() {
                continue;
            }
            let key = (ArcLabel::Eps, classes[dest]);
            temp.entry(key)
                .and_modify(|acc| *acc |= w)
                .or_insert(w.clone());
        }

        // Labeled transitions
        for (&lbl, targets) in &st.transitions {
            let label = ArcLabel::Label(lbl);
            for &(dest, ref w) in targets {
                if w.is_empty() {
                    continue;
                }
                let key = (label, classes[dest]);
                temp.entry(key)
                    .and_modify(|acc| *acc |= w)
                    .or_insert(w.clone());
            }
        }

        let mut outgoing = Vec::with_capacity(temp.len());
        for ((label, dest_class), weight) in temp {
            if !weight.is_empty() {
                outgoing.push(NwaTransitionSig {
                    label,
                    dest_class,
                    weight,
                });
            }
        }
        outgoing.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));

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
    eps: BTreeMap<NWAStateID, Weight>,
    trans: BTreeMap<i16, BTreeMap<NWAStateID, Weight>>,
}

impl NWA {
    pub fn simplify(&mut self) {
        crate::debug!(4, "[NWA::simplify] Num states before simplification: {}", self.states.len());
        let instant = std::time::Instant::now();
        let mut internal = self.clone();
        internal.simplify_internal();
        crate::debug!(
            4,
            "[NWA::simplify] Simplification took {:.3} seconds. Num states: {}",
            instant.elapsed().as_secs_f64(),
            internal.states.len()
        );
        let mut rustfst = self.clone();
        rustfst.simplify_with_rustfst();
        crate::debug!(
            4,
            "[NWA::simplify] RustFST minimization took {:.3} seconds. Num states: {}",
            instant.elapsed().as_secs_f64(),
            rustfst.states.len()
        );
        *self = internal;
    }

    /// Canonicalize NWA transitions by merging parallel transitions:
    ///  - For each state and epsilon edge, merge multiple (to, w) by unioning weights per `to`.
    ///  - For each state, label, and destination, merge multiple (label, to, w) by unioning weights.
    /// Transitions with empty weight are removed.
    ///
    /// This is sound because the weight semiring is (bitset, |, &, ∅, U), so:
    ///   (w1 & x) | (w2 & x) = (w1 | w2) & x
    /// for all bitsets w1, w2, x. Thus, multiple parallel transitions are semantically
    /// equivalent to a single transition with the union of their weights.
    fn compress_transitions(&mut self) -> bool {
        let mut changed = false;

        for st in &mut self.states.0 {
            // Compress epsilons: (to, w) -> union of weights per `to`.
            if !st.epsilons.is_empty() {
                let mut eps_map: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
                for &(to, ref w) in &st.epsilons {
                    if w.is_empty() {
                        continue;
                    }
                    eps_map
                        .entry(to)
                        .and_modify(|acc| *acc |= w)
                        .or_insert(w.clone());
                }
                if eps_map.len() != st.epsilons.len() {
                    changed = true;
                }
                st.epsilons = eps_map
                    .into_iter()
                    .filter(|(_, w)| !w.is_empty())
                    .collect();
            }

            // Compress labeled transitions: per (label, to) aggregate weights by union.
            if !st.transitions.is_empty() {
                let mut new_transitions: BTreeMap<i16, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
                for (&lbl, targets) in &st.transitions {
                    let mut per_dest: BTreeMap<NWAStateID, Weight> = BTreeMap::new();
                    for &(to, ref w) in targets {
                        if w.is_empty() {
                            continue;
                        }
                        per_dest
                            .entry(to)
                            .and_modify(|acc| *acc |= w)
                            .or_insert(w.clone());
                    }
                    if per_dest.len() != targets.len() {
                        changed = true;
                    }
                    let merged: Vec<(NWAStateID, Weight)> = per_dest
                        .into_iter()
                        .filter(|(_, w)| !w.is_empty())
                        .collect();
                    if !merged.is_empty() {
                        new_transitions.insert(lbl, merged);
                    }
                }
                if new_transitions.len() != st.transitions.len() {
                    changed = true;
                }
                st.transitions = new_transitions;
            }
        }

        changed
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
        let mut changed = false;
        changed |= self.prune_unreachable();
        crate::debug!(5, "[NWA::simplify] After prune_unreachable (1): {}", self.stats());

        changed |= self.push_final_weights_along_epsilons();
        crate::debug!(5, "[NWA::simplify] After pushing final weights along epsilons: {}", self.stats());

        changed |= self.compress_transitions();
        crate::debug!(5, "[NWA::simplify] After compress_transitions: {}", self.stats());

        changed |= self.prune_dead_ends();
        crate::debug!(5, "[NWA::simplify] After prune_dead_ends (1): {}", self.stats());

        let n = self.states.len();
        if n > 1 {
            let partition = minimize_nwa_partition(&self.states);
            if partition.num_classes() < n {
                crate::debug!(5, "[NWA::simplify] Minimizing states ({} -> {})...", n, partition.num_classes());
                self.rebuild_from_partition(partition);
                changed = true;
                crate::debug!(5, "[NWA::simplify] After minimizing: {}", self.stats());
            }
        }

        changed |= self.prune_unreachable();
        crate::debug!(5, "[NWA::simplify] After prune_unreachable (2): {}", self.stats());

        changed |= self.prune_dead_ends();
        crate::debug!(5, "[NWA::simplify] After prune_dead_ends (2): {}", self.stats());
        crate::debug!(
            5,
            "[NWA::simplify] Simplification finished. Total changed: {}. Final stats: {}",
            changed,
            self.stats()
        );
        changed
    }

    fn push_final_weights_along_epsilons(&mut self) -> bool {
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

        let start_class = partition.class_of[self.body.start_state];
        let new_start = class_to_new[&start_class];
        self.states = new_states;
        self.body.start_state = new_start;
    }

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
