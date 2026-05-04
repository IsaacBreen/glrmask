        use std::collections::{BTreeMap, BTreeSet, VecDeque};
        use std::hash::Hash;
        use std::marker::PhantomData;
        use std::ops::Index;

        use rustc_hash::{FxHashMap, FxHashSet};
        use serde::{Deserialize, Serialize};
        use serde::de::{MapAccess, Visitor};
        use serde::ser::SerializeMap;
        use smallvec::SmallVec;

        use super::analysis::{EOF, AnalyzedGrammar};
        use crate::grammar::flat::{NonterminalID, Rule, Symbol, TerminalID};

        const INLINE_ROW_CAPACITY: usize = 8;

        #[derive(Debug, Clone)]
        pub(crate) enum SparseRow<K: Copy + Eq + Hash, V: Clone> {
            Inline(SmallVec<[(K, V); INLINE_ROW_CAPACITY]>),
            Large(FxHashMap<K, V>),
        }

        impl<K: Copy + Eq + Hash, V: Clone> Default for SparseRow<K, V> {
            fn default() -> Self {
                Self::Inline(SmallVec::new())
            }
        }

        impl<K: Copy + Eq + Hash, V: Clone> SparseRow<K, V> {
            #[inline]
            pub(crate) fn len(&self) -> usize {
                match self {
                    Self::Inline(entries) => entries.len(),
                    Self::Large(entries) => entries.len(),
                }
            }

            #[inline]
            pub(crate) fn is_empty(&self) -> bool {
                self.len() == 0
            }

            #[inline]
            pub(crate) fn get(&self, key: &K) -> Option<&V> {
                match self {
                    Self::Inline(entries) => entries.iter().find(|(entry_key, _)| entry_key == key).map(|(_, value)| value),
                    Self::Large(entries) => entries.get(key),
                }
            }

            #[inline]
            pub(crate) fn get_mut(&mut self, key: &K) -> Option<&mut V> {
                match self {
                    Self::Inline(entries) => entries.iter_mut().find(|(entry_key, _)| entry_key == key).map(|(_, value)| value),
                    Self::Large(entries) => entries.get_mut(key),
                }
            }

            pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
                match self {
                    Self::Inline(entries) => {
                        for (entry_key, entry_value) in entries.iter_mut() {
                            if *entry_key == key {
                                return Some(std::mem::replace(entry_value, value));
                            }
                        }
                        if entries.len() < INLINE_ROW_CAPACITY {
                            entries.push((key, value));
                            None
                        } else {
                            let mut large = FxHashMap::default();
                            for (entry_key, entry_value) in entries.drain(..) {
                                large.insert(entry_key, entry_value);
                            }
                            let previous = large.insert(key, value);
                            *self = Self::Large(large);
                            previous
                        }
                    }
                    Self::Large(entries) => entries.insert(key, value),
                }
            }

            pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
                match self {
                    Self::Inline(entries) => {
                        let position = entries.iter().position(|(entry_key, _)| entry_key == key)?;
                        Some(entries.swap_remove(position).1)
                    }
                    Self::Large(entries) => entries.remove(key),
                }
            }

            #[inline]
            pub(crate) fn contains_key(&self, key: &K) -> bool {
                self.get(key).is_some()
            }

            #[inline]
            pub(crate) fn iter(&self) -> SparseRowIter<'_, K, V> {
                match self {
                    Self::Inline(entries) => SparseRowIter::Inline(entries.iter()),
                    Self::Large(entries) => SparseRowIter::Large(entries.iter()),
                }
            }

            #[inline]
            pub(crate) fn keys(&self) -> SparseRowKeys<'_, K, V> {
                match self {
                    Self::Inline(entries) => SparseRowKeys::Inline(entries.iter()),
                    Self::Large(entries) => SparseRowKeys::Large(entries.keys()),
                }
            }

            #[inline]
            pub(crate) fn values(&self) -> SparseRowValues<'_, K, V> {
                match self {
                    Self::Inline(entries) => SparseRowValues::Inline(entries.iter()),
                    Self::Large(entries) => SparseRowValues::Large(entries.values()),
                }
            }
        }

impl<K: Copy + Eq + Hash, V: Clone + PartialEq> PartialEq for SparseRow<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|(key, value)| other.get(key) == Some(value))
    }
}

impl<K: Copy + Eq + Hash, V: Clone + Eq> Eq for SparseRow<K, V> {}

impl<K, V> Serialize for SparseRow<K, V>
where
    K: Copy + Eq + Hash + Serialize,
    V: Clone + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (key, value) in self.iter() {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<'de, K, V> Deserialize<'de> for SparseRow<K, V>
where
    K: Copy + Eq + Hash + Deserialize<'de>,
    V: Clone + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct SparseRowVisitor<K, V>(PhantomData<(K, V)>);

        impl<'de, K, V> Visitor<'de> for SparseRowVisitor<K, V>
        where
            K: Copy + Eq + Hash + Deserialize<'de>,
            V: Clone + Deserialize<'de>,
        {
            type Value = SparseRow<K, V>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a sparse row map")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut row = SparseRow::default();
                while let Some((key, value)) = map.next_entry()? {
                    row.insert(key, value);
                }
                Ok(row)
            }
        }

        deserializer.deserialize_map(SparseRowVisitor::<K, V>(PhantomData))
    }
}

impl<'a, K: Copy + Eq + Hash, V: Clone> IntoIterator for &'a SparseRow<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = SparseRowIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Copy + Eq + Hash, V: Clone> Index<&K> for SparseRow<K, V> {
    type Output = V;

    fn index(&self, index: &K) -> &Self::Output {
        self.get(index).expect("sparse row index missing key")
    }
}

impl<K: Copy + Eq + Hash, V: Clone> FromIterator<(K, V)> for SparseRow<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut row = Self::default();
        for (key, value) in iter {
            row.insert(key, value);
        }
        row
    }
}

pub(crate) enum SparseRowIter<'a, K: Copy + Eq + Hash, V: Clone> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(std::collections::hash_map::Iter<'a, K, V>),
}

impl<'a, K: Copy + Eq + Hash, V: Clone> Iterator for SparseRowIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(entries) => entries.next().map(|(key, value)| (key, value)),
            Self::Large(entries) => entries.next(),
        }
    }
}

pub(crate) enum SparseRowKeys<'a, K: Copy + Eq + Hash, V: Clone> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(std::collections::hash_map::Keys<'a, K, V>),
}

impl<'a, K: Copy + Eq + Hash, V: Clone> Iterator for SparseRowKeys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(entries) => entries.next().map(|(key, _)| key),
            Self::Large(entries) => entries.next(),
        }
    }
}

pub(crate) enum SparseRowValues<'a, K: Copy + Eq + Hash, V: Clone> {
    Inline(std::slice::Iter<'a, (K, V)>),
    Large(std::collections::hash_map::Values<'a, K, V>),
}

impl<'a, K: Copy + Eq + Hash, V: Clone> Iterator for SparseRowValues<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(entries) => entries.next().map(|(_, value)| value),
            Self::Large(entries) => entries.next(),
        }
    }
}

pub(crate) type ActionRow = SparseRow<TerminalID, Action>;
pub(crate) type GotoRow = SparseRow<NonterminalID, (u32, bool)>;

fn replace_shifts_enabled() -> bool {
    true
}

fn replace_gotos_enabled() -> bool {
    true
}

std::thread_local! {
    pub(crate) static LOCAL_FORWARD_REPLACE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn local_forward_replace_enabled() -> bool {
    LOCAL_FORWARD_REPLACE_OVERRIDE.with(|c| {
        if let Some(v) = c.get() {
            return v;
        }
        false
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackShift {
    pub pop: u32,
    pub pushes: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StackShiftGuard {
    pub pop: u32,
    pub states: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GuardedStackShift {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guards: Vec<StackShiftGuard>,
    pub pop: u32,
    pub pushes: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Shift(u32, bool),
    StackShifts(Vec<StackShift>),
    GuardedStackShifts(Vec<GuardedStackShift>),
    Reduce(NonterminalID, u32),
    Split {
        shift: Option<(u32, bool)>,
        reduces: Vec<(NonterminalID, u32)>,
        accept: bool,
    },
    Accept,
}

impl Action {
    /// The shift target, if any. Works for Shift, Split, and single-effect
    /// StackShifts that are representable as ordinary shift or replace-shift.
    #[inline]
    pub fn shift_target(&self) -> Option<u32> {
        match self {
            Action::Shift(t, _) => Some(*t),
            Action::Split { shift: Some((t, _)), .. } => Some(*t),
            Action::StackShifts(shifts)
                if shifts.len() == 1
                    && shifts[0].pushes.len() == 1
                    && shifts[0].pop <= 1 =>
            {
                Some(shifts[0].pushes[0])
            }
            Action::GuardedStackShifts(_) => None,
            _ => None,
        }
    }

    /// Whether the shift is a replace (pop + push instead of just push).
    #[inline]
    pub fn shift_is_replace(&self) -> bool {
        match self {
            Action::Shift(_, r) => *r,
            Action::Split { shift: Some((_, r)), .. } => *r,
            Action::StackShifts(shifts) if shifts.len() == 1 => {
                shifts[0].pop == 1 && shifts[0].pushes.len() == 1
            }
            Action::GuardedStackShifts(_) => false,
            _ => false,
        }
    }

    #[inline]
    pub fn for_each_stack_shift(&self, mut f: impl FnMut(u32, &[u32])) {
        match self {
            Action::Shift(target, false) => f(0, std::slice::from_ref(target)),
            Action::Shift(target, true) => f(1, std::slice::from_ref(target)),
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    f(shift.pop, &shift.pushes);
                }
            }
            Action::GuardedStackShifts(_) => {}
            Action::Split { shift: Some((target, false)), .. } => {
                f(0, std::slice::from_ref(target));
            }
            Action::Split { shift: Some((target, true)), .. } => {
                f(1, std::slice::from_ref(target));
            }
            _ => {}
        }
    }

    /// Iterate over reduce (lhs_nonterminal, reduce_length) pairs.
    #[inline]
    pub fn for_each_reduce(&self, mut f: impl FnMut(NonterminalID, u32)) {
        match self {
            Action::Reduce(nt, len) => f(*nt, *len),
            Action::Split { reduces, .. } => {
                for &(nt, len) in reduces {
                    f(nt, len);
                }
            }
            _ => {}
        }
    }

    /// Number of reduce alternatives.
    #[inline]
    pub fn reduce_count(&self) -> usize {
        match self {
            Action::Reduce(..) => 1,
            Action::Split { reduces, .. } => reduces.len(),
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLRTable {
    pub action: Vec<ActionRow>,
    pub goto: Vec<GotoRow>,
    pub num_states: u32,
    pub num_terminals: u32,
    pub num_rules: u32,
    pub rules: Vec<Rule>,
    /// Set of (state, terminal) pairs where the shift was created by the
    /// transfer mechanism.  The characterization should treat these as
    /// non-replace to avoid creating pop-0 reduces in the template NFA.
    pub forwarded_shifts: FxHashSet<(u32, TerminalID)>,
}

impl GLRTable {
    pub fn build(grammar: &AnalyzedGrammar) -> Self {
        Self::build_with_unit_reduction_inlining(grammar, false)
    }

    pub(crate) fn build_with_unit_reduction_inlining(
        grammar: &AnalyzedGrammar,
        _inline_unit_reductions: bool,
    ) -> Self {
        let (item_sets, transitions) = build_lr1_item_sets(grammar);
        build_lr1_table(grammar, &item_sets, &transitions)
    }

    #[inline]
    pub fn action(&self, state: u32, terminal: TerminalID) -> Option<&Action> {
        self.action
            .get(state as usize)
            .and_then(|by_terminal| by_terminal.get(&terminal))
    }

    #[inline]
    pub fn goto_target(&self, state: u32, nt: NonterminalID) -> Option<(u32, bool)> {
        self.goto
            .get(state as usize)
            .and_then(|by_nt| by_nt.get(&nt).copied())
    }

}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CellUpdate {
    Set(Action),
    Remove,
}

fn build_state_predecessors(
    table: &GLRTable,
    original_num_states: u32,
    constituent_sets: &[BTreeSet<u32>],
) -> Vec<BTreeSet<u32>> {
    let nstates = table.num_states as usize;
    let mut predecessors: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); nstates];

    for src in 0..original_num_states as usize {
        for action in table.action[src].values() {
            if let Some(dst) = action.shift_target() {
                predecessors[dst as usize].extend(constituent_sets[src].iter().copied());
            }
        }
        for &(dst, _) in table.goto[src].values() {
            predecessors[dst as usize].extend(constituent_sets[src].iter().copied());
        }
    }

    predecessors
}

fn build_runtime_state_predecessors(
    table: &GLRTable,
    original_num_states: u32,
    constituent_sets: &[BTreeSet<u32>],
) -> Vec<BTreeSet<u32>> {
    let mut predecessors = vec![BTreeSet::new(); table.num_states as usize];

    for src in 0..table.num_states as usize {
        for action in table.action[src].values() {
            match action {
                Action::Shift(dst, false) => {
                    predecessors[*dst as usize].extend(constituent_sets[src].iter().copied());
                }
                Action::Split { shift: Some((dst, false)), .. } => {
                    predecessors[*dst as usize].extend(constituent_sets[src].iter().copied());
                }
                _ => {}
            }
        }
        for &(dst, replace) in table.goto[src].values() {
            if !replace {
                predecessors[dst as usize].extend(constituent_sets[src].iter().copied());
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for src in 0..original_num_states as usize {
            let src_preds = predecessors[src].clone();
            for action in table.action[src].values() {
                match action {
                    Action::Shift(dst, true) => {
                        let before = predecessors[*dst as usize].len();
                        predecessors[*dst as usize].extend(src_preds.iter().copied());
                        changed |= predecessors[*dst as usize].len() != before;
                    }
                    Action::Split { shift: Some((dst, true)), .. } => {
                        let before = predecessors[*dst as usize].len();
                        predecessors[*dst as usize].extend(src_preds.iter().copied());
                        changed |= predecessors[*dst as usize].len() != before;
                    }
                    _ => {}
                }
            }
            for &(dst, replace) in table.goto[src].values() {
                if replace {
                    let before = predecessors[dst as usize].len();
                    predecessors[dst as usize].extend(src_preds.iter().copied());
                    changed |= predecessors[dst as usize].len() != before;
                }
            }
        }
    }

    predecessors
}

fn subset_key(subset: &BTreeSet<u32>) -> Vec<u32> {
    subset.iter().copied().collect()
}

fn union_state_subsets(
    states: impl IntoIterator<Item = u32>,
    constituent_sets: &[BTreeSet<u32>],
) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for state in states {
        out.extend(constituent_sets[state as usize].iter().copied());
    }
    out
}

fn merge_shift_into_pending(
    pending: &mut PendingAction,
    target: u32,
    replace: bool,
    table: &mut GLRTable,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<(), ()> {
    match pending.shift {
        None => {
            pending.shift = Some((target, replace));
            Ok(())
        }
        Some((existing_target, existing_replace)) => {
            if existing_target == target {
                return if existing_replace == replace { Ok(()) } else { Err(()) };
            }
            if existing_replace != replace {
                return Err(());
            }
            let merged_subset = union_state_subsets([existing_target, target], constituent_sets);
            let merged_target = ensure_subset_state(
                table,
                &merged_subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
            )?;
            pending.shift = Some((merged_target, replace));
            Ok(())
        }
    }
}

fn merge_action_into_pending(
    pending: &mut PendingAction,
    action: &Action,
    table: &mut GLRTable,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<(), ()> {
    match action {
        Action::Shift(target, replace) => merge_shift_into_pending(
            pending,
            *target,
            *replace,
            table,
            constituent_sets,
            subset_to_state,
            failed_subsets,
        ),
        Action::StackShifts(_) => Err(()),
        Action::GuardedStackShifts(_) => Err(()),
        Action::Reduce(nt, len) => {
            pending.push_reduce(*nt, *len);
            Ok(())
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            if let Some((target, replace)) = shift {
                merge_shift_into_pending(
                    pending,
                    *target,
                    *replace,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                )?;
            }
            for &(nt, len) in reduces {
                pending.push_reduce(nt, len);
            }
            if *accept {
                pending.push_accept();
            }
            Ok(())
        }
        Action::Accept => {
            pending.push_accept();
            Ok(())
        }
    }
}

fn build_merged_action_row(
    table: &mut GLRTable,
    subset: &BTreeSet<u32>,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<ActionRow, ()> {
    let mut terminals = BTreeSet::new();
    for &state in subset {
        for &tid in table.action[state as usize].keys() {
            terminals.insert(tid);
        }
    }

    let mut row = ActionRow::default();
    for tid in terminals {
        let mut pending = PendingAction::default();
        for &state in subset {
            if let Some(action) = table.action[state as usize].get(&tid).cloned() {
                merge_action_into_pending(
                    &mut pending,
                    &action,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                )?;
            }
        }
        if let Some(action) = pending.maybe_finish() {
            row.insert(tid, action);
        }
    }

    Ok(row)
}

fn build_merged_goto_row(
    table: &mut GLRTable,
    subset: &BTreeSet<u32>,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<GotoRow, ()> {
    let mut nts = BTreeSet::new();
    for &state in subset {
        for &nt in table.goto[state as usize].keys() {
            nts.insert(nt);
        }
    }

    let mut row = GotoRow::default();
    for nt in nts {
        let mut replace: Option<bool> = None;
        let mut target_subset = BTreeSet::new();
        let mut saw_target = false;

        for &state in subset {
            if let Some(&(target, is_replace)) = table.goto[state as usize].get(&nt) {
                saw_target = true;
                match replace {
                    None => replace = Some(is_replace),
                    Some(existing) if existing == is_replace => {}
                    Some(_) => return Err(()),
                }
                target_subset.extend(constituent_sets[target as usize].iter().copied());
            }
        }

        if !saw_target {
            continue;
        }

        let merged_target = ensure_subset_state(
            table,
            &target_subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
        )?;
        row.insert(nt, (merged_target, replace.unwrap()));
    }

    Ok(row)
}

fn ensure_subset_state(
    table: &mut GLRTable,
    subset: &BTreeSet<u32>,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<u32, ()> {
    debug_assert!(!subset.is_empty());
    if subset.len() == 1 {
        return Ok(*subset.iter().next().unwrap());
    }

    let key = subset_key(subset);
    if let Some(&state) = subset_to_state.get(&key) {
        return Ok(state);
    }
    if failed_subsets.contains(&key) {
        return Err(());
    }

    let state = table.num_states;
    table.num_states += 1;
    table.action.push(ActionRow::default());
    table.goto.push(GotoRow::default());
    constituent_sets.push(subset.clone());
    subset_to_state.insert(key.clone(), state);

    let built = (|| {
        let action_row = build_merged_action_row(
            table,
            subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
        )?;
        let goto_row = build_merged_goto_row(
            table,
            subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
        )?;
        Ok::<_, ()>((action_row, goto_row))
    })();

    match built {
        Ok((action_row, goto_row)) => {
            table.action[state as usize] = action_row;
            table.goto[state as usize] = goto_row;
            Ok(state)
        }
        Err(()) => {
            subset_to_state.remove(&key);
            failed_subsets.insert(key);
            table.action.pop();
            table.goto.pop();
            table.num_states -= 1;
            constituent_sets.pop();
            Err(())
        }
    }
}

fn refresh_merged_states(
    table: &mut GLRTable,
    original_num_states: u32,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) {
    let mut state = original_num_states as usize;
    while state < table.num_states as usize {
        let subset = constituent_sets[state].clone();
        let rebuilt = (|| {
            let action_row = build_merged_action_row(
                table,
                &subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
            )?;
            let goto_row = build_merged_goto_row(
                table,
                &subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
            )?;
            Ok::<_, ()>((action_row, goto_row))
        })();

        if let Ok((action_row, goto_row)) = rebuilt {
            table.action[state] = action_row;
            table.goto[state] = goto_row;
        }

        state += 1;
    }
}

fn unit_reduce_destination(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    lhs: NonterminalID,
) -> Option<u32> {
    let preds = &predecessors[state as usize];
    assert!(!preds.is_empty());

    let relevant_preds: Vec<u32> = preds
        .iter()
        .copied()
        .filter(|&pred| table.goto[pred as usize].contains_key(&lhs))
        .collect();
    if relevant_preds.is_empty() {
        return None;
    }

    let mut reduce_dst: Option<u32> = None;
    for pred in relevant_preds {
        let (dst, is_replace) = table.goto[pred as usize][&lhs];
        if is_replace {
            return None;
        }
        if table.goto[dst as usize] != table.goto[state as usize] {
            return None;
        }
        match reduce_dst {
            None => reduce_dst = Some(dst),
            Some(existing) if existing == dst => {}
            Some(_) => return None,
        }
    }

    reduce_dst
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StackEffectFrame {
    pop: u32,
    pushes: Vec<u32>,
    guards: Vec<StackShiftGuard>,
}

enum ReduceFrameResult {
    Dead,
    Frames(Vec<StackEffectFrame>),
}

fn states_at_depth(
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    depth: u32,
) -> Option<BTreeSet<u32>> {
    let mut states = BTreeSet::from([origin_state]);
    for _ in 0..depth {
        let mut next = BTreeSet::new();
        for state in states {
            next.extend(predecessors.get(state as usize)?.iter().copied());
        }
        if next.is_empty() {
            return None;
        }
        states = next;
    }
    Some(states)
}

fn normalize_states(mut states: Vec<u32>) -> Vec<u32> {
    states.sort_unstable();
    states.dedup();
    states
}

fn add_guard_to_frame(
    frame: &mut StackEffectFrame,
    pop: u32,
    states: impl IntoIterator<Item = u32>,
) -> bool {
    let states = normalize_states(states.into_iter().collect());
    if states.is_empty() {
        return false;
    }

    if let Some(existing) = frame.guards.iter_mut().find(|guard| guard.pop == pop) {
        let wanted: BTreeSet<u32> = states.into_iter().collect();
        existing.states.retain(|state| wanted.contains(state));
        return !existing.states.is_empty();
    }

    frame.guards.push(StackShiftGuard { pop, states });
    frame.guards.sort_by_key(|guard| guard.pop);
    true
}

fn pop_frame(frame: &mut StackEffectFrame, pop: u32) {
    if pop as usize <= frame.pushes.len() {
        let keep = frame.pushes.len() - pop as usize;
        frame.pushes.truncate(keep);
    } else {
        frame.pop += pop - frame.pushes.len() as u32;
        frame.pushes.clear();
    }
}

fn push_transition_to_frame(frame: &mut StackEffectFrame, target: u32, replace: bool) {
    if replace {
        if let Some(top) = frame.pushes.last_mut() {
            *top = target;
        } else {
            frame.pop += 1;
            frame.pushes.push(target);
        }
    } else {
        frame.pushes.push(target);
    }
}

fn frame_to_guarded_shift(frame: StackEffectFrame) -> GuardedStackShift {
    GuardedStackShift {
        guards: frame.guards,
        pop: frame.pop,
        pushes: frame.pushes,
    }
}

fn apply_reduce_to_frame(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    mut frame: StackEffectFrame,
    nt: NonterminalID,
    len: u32,
) -> Option<ReduceFrameResult> {
    pop_frame(&mut frame, len);

    let goto_froms = if let Some(&state) = frame.pushes.last() {
        BTreeSet::from([state])
    } else {
        states_at_depth(predecessors, origin_state, frame.pop)?
    };

    let guard_pop = frame.pop;
    let mut target: Option<u32> = None;
    let mut by_replace: BTreeMap<bool, BTreeSet<u32>> = BTreeMap::new();
    let mut missing = 0usize;
    for goto_from in goto_froms {
        let Some((next_target, replace)) = table.goto[goto_from as usize].get(&nt).copied() else {
            missing += 1;
            continue;
        };
        match target {
            None => target = Some(next_target),
            Some(existing) if existing == next_target => {}
            Some(_) => return None,
        }
        by_replace.entry(replace).or_default().insert(goto_from);
    }

    if missing > 0 && by_replace.is_empty() {
        return Some(ReduceFrameResult::Dead);
    }

    let target = target?;
    let needs_guard = missing > 0 || by_replace.len() > 1;
    let mut frames = Vec::new();
    for (replace, froms) in by_replace {
        let mut next_frame = frame.clone();
        if needs_guard && !add_guard_to_frame(&mut next_frame, guard_pop, froms.into_iter()) {
            continue;
        }
        push_transition_to_frame(&mut next_frame, target, replace);
        frames.push(next_frame);
    }

    if frames.is_empty() {
        Some(ReduceFrameResult::Dead)
    } else {
        frames.sort();
        frames.dedup();
        Some(ReduceFrameResult::Frames(frames))
    }
}

fn compose_guarded_shift_with_frame(
    mut frame: StackEffectFrame,
    shift: &GuardedStackShift,
) -> Option<Option<StackEffectFrame>> {
    let pushed_len = frame.pushes.len() as u32;

    for guard in &shift.guards {
        if guard.states.is_empty() {
            return Some(None);
        }

        if guard.pop < pushed_len {
            let idx = (pushed_len - 1 - guard.pop) as usize;
            let known_state = frame.pushes[idx];
            if guard.states.binary_search(&known_state).is_err() {
                return Some(None);
            }
        } else {
            let translated_pop = frame.pop + (guard.pop - pushed_len);
            if !add_guard_to_frame(&mut frame, translated_pop, guard.states.iter().copied()) {
                return Some(None);
            }
        }
    }

    if shift.pop < shift.guards.iter().map(|guard| guard.pop).max().unwrap_or(0) {
        return None;
    }

    pop_frame(&mut frame, shift.pop);
    frame.pushes.extend_from_slice(&shift.pushes);
    Some(Some(frame))
}

fn stack_effects_for_action(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    tid: TerminalID,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    visiting: &mut BTreeSet<(u32, TerminalID, u8, u32, Vec<u32>, Vec<StackShiftGuard>)>,
) -> Option<Vec<GuardedStackShift>> {
    let action_tag = match action {
        Action::Shift(..) => 0,
        Action::StackShifts(_) => 1,
        Action::GuardedStackShifts(_) => 2,
        Action::Reduce(..) => 3,
        Action::Split { .. } => 4,
        Action::Accept => 5,
    };
    let key = (state, tid, action_tag, frame.pop, frame.pushes.clone(), frame.guards.clone());
    if !visiting.insert(key.clone()) {
        return None;
    }

    let mut out = Vec::new();
    match action {
        Action::Shift(target, replace) => {
            let mut frame = frame;
            let effective_replace = *replace && !table.forwarded_shifts.contains(&(state, tid));
            push_transition_to_frame(&mut frame, *target, effective_replace);
            out.push(frame_to_guarded_shift(frame));
        }
        Action::StackShifts(shifts) => {
            for shift in shifts {
                let mut frame = frame.clone();
                pop_frame(&mut frame, shift.pop);
                frame.pushes.extend_from_slice(&shift.pushes);
                out.push(frame_to_guarded_shift(frame));
            }
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts {
                match compose_guarded_shift_with_frame(frame.clone(), shift)? {
                    None => {}
                    Some(frame) => out.push(frame_to_guarded_shift(frame)),
                }
            }
        }
        Action::Reduce(nt, len) => {
            let frames = match apply_reduce_to_frame(table, predecessors, origin_state, frame, *nt, *len)? {
                ReduceFrameResult::Dead => {
                    visiting.remove(&key);
                    return Some(Vec::new());
                }
                ReduceFrameResult::Frames(frames) => frames,
            };
            for frame in frames {
                let Some(&next_state) = frame.pushes.last() else {
                    continue;
                };
                let Some(next) = table.action[next_state as usize].get(&tid) else {
                    continue;
                };
                out.extend(stack_effects_for_action(
                    table,
                    predecessors,
                    origin_state,
                    tid,
                    next_state,
                    next,
                    frame,
                    visiting,
                )?);
            }
        }
        Action::Split { shift, reduces, accept } => {
            if *accept {
                visiting.remove(&key);
                return None;
            }
            if let Some((target, replace)) = shift {
                let shift_action = Action::Shift(*target, *replace);
                out.extend(stack_effects_for_action(
                    table,
                    predecessors,
                    origin_state,
                    tid,
                    state,
                    &shift_action,
                    frame.clone(),
                    visiting,
                )?);
            }
            for &(nt, len) in reduces {
                let reduce_action = Action::Reduce(nt, len);
                out.extend(stack_effects_for_action(
                    table,
                    predecessors,
                    origin_state,
                    tid,
                    state,
                    &reduce_action,
                    frame.clone(),
                    visiting,
                )?);
            }
        }
        Action::Accept => {
            visiting.remove(&key);
            return None;
        }
    }

    visiting.remove(&key);
    out.sort();
    out.dedup();
    Some(out)
}

fn normalize_guarded_effects(effects: &mut Vec<GuardedStackShift>) {
    for effect in effects.iter_mut() {
        for guard in effect.guards.iter_mut() {
            guard.states.sort_unstable();
            guard.states.dedup();
        }
        effect.guards.retain(|guard| !guard.states.is_empty());
        effect.guards.sort_by_key(|guard| guard.pop);
        effect.guards.dedup();
    }
    effects.retain(|effect| !effect.pushes.is_empty());
    effects.sort();
    effects.dedup();
}

fn stack_effect_action(mut effects: Vec<GuardedStackShift>) -> Option<Action> {
    normalize_guarded_effects(&mut effects);
    if effects.is_empty() {
        return None;
    }
    if effects.iter().all(|effect| effect.guards.is_empty()) {
        let shifts = effects
            .into_iter()
            .map(|effect| StackShift {
                pop: effect.pop,
                pushes: effect.pushes,
            })
            .collect();
        return stack_shift_action(shifts);
    }
    Some(Action::GuardedStackShifts(effects))
}

fn effects_can_be_delayed(effects: &[GuardedStackShift]) -> bool {
    effects.len() > 1
        && effects.iter().all(|effect| effect.pop > 0 && !effect.pushes.is_empty())
        && effects.iter().any(|effect| effect.guards.is_empty())
}

fn try_inline_action_to_stack_shifts(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
) -> Option<Action> {
    let Action::Split {
        reduces,
        accept: false,
        ..
    } = action
    else {
        return None;
    };
    if reduces.is_empty() {
        return None;
    }

    let effects = stack_effects_for_action(
        table,
        predecessors,
        state,
        tid,
        state,
        action,
        StackEffectFrame {
            pop: 0,
            pushes: Vec::new(),
            guards: Vec::new(),
        },
        &mut BTreeSet::new(),
    )?;
    if effects.is_empty() {
        return None;
    }
    if effects_can_be_delayed(&effects) {
        if effects.iter().any(|effect| effect.pop == 0) {
            return None;
        }
        if let Some(delay_state) = try_create_delayed_stack_shift_state(
            table,
            predecessors,
            state,
            &effects,
            constituent_sets,
            0,
        ) {
            return Some(Action::Shift(delay_state, true));
        }
    }
    stack_effect_action(effects)
}

fn stack_shift_action(shifts: Vec<StackShift>) -> Option<Action> {
    if shifts.is_empty() {
        return None;
    }
    if shifts.len() == 1 {
        let shift = &shifts[0];
        if shift.pushes.len() == 1 {
            match shift.pop {
                0 => return Some(Action::Shift(shift.pushes[0], false)),
                1 => return Some(Action::Shift(shift.pushes[0], true)),
                _ => {}
            }
        }
    }
    Some(Action::StackShifts(shifts))
}

fn try_create_delayed_stack_shift_state(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    effects: &[GuardedStackShift],
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    depth: u32,
) -> Option<u32> {
    if depth >= 8 || effects.len() <= 1 || effects.iter().any(|effect| effect.pop == 0 || effect.pushes.is_empty()) {
        return None;
    }

    let mut terminals = BTreeSet::new();
    for effect in effects {
        let top = *effect.pushes.last()?;
        for &terminal in table.action.get(top as usize)?.keys() {
            terminals.insert(terminal);
        }
    }

    let mut row = ActionRow::default();
    for terminal in terminals {
        let mut composed = Vec::new();
        for effect in effects {
            let top = *effect.pushes.last()?;
            let Some(action) = table.action[top as usize].get(&terminal).cloned() else {
                continue;
            };
            composed.extend(stack_effects_for_action(
                table,
                predecessors,
                origin_state,
                terminal,
                top,
                &action,
                StackEffectFrame {
                    pop: effect.pop,
                    pushes: effect.pushes.clone(),
                    guards: effect.guards.clone(),
                },
                &mut BTreeSet::new(),
            )?);
        }
        normalize_guarded_effects(&mut composed);
        let action = if effects_can_be_delayed(&composed) {
            if let Some(next_state) = try_create_delayed_stack_shift_state(
                table,
                predecessors,
                origin_state,
                &composed,
                constituent_sets,
                depth + 1,
            ) {
                Action::Shift(next_state, true)
            } else {
                stack_effect_action(composed)?
            }
        } else {
            stack_effect_action(composed)?
        };
        row.insert(terminal, action);
    }

    if row.is_empty() {
        return None;
    }

    let state = table.num_states;
    table.num_states += 1;
    table.action.push(row);
    table.goto.push(GotoRow::default());
    constituent_sets.push(BTreeSet::from([state]));
    Some(state)
}

fn try_inline_unit_reductions_for_cell(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<Option<CellUpdate>, ()> {
    if let Some(action) = try_inline_action_to_stack_shifts(
        table,
        predecessors,
        state,
        tid,
        action,
        constituent_sets,
    ) {
        return Ok(Some(CellUpdate::Set(action)));
    }

    match action {
        Action::Split {
            shift: Some(_),
            accept: false,
            ..
        }
        | Action::Shift(_, _) => {}
        _ => return Ok(None),
    }

    let mut visiting = BTreeSet::new();
    try_inline_unit_reductions_for_cell_inner(
        table,
        predecessors,
        state,
        tid,
        action,
        constituent_sets,
        subset_to_state,
        failed_subsets,
        &mut visiting,
    )
}

fn try_inline_unit_reductions_for_cell_inner(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    visiting: &mut BTreeSet<(u32, TerminalID)>,
) -> Result<Option<CellUpdate>, ()> {
    if !visiting.insert((state, tid)) {
        return Ok(None);
    }

    let mut pending = PendingAction::default();
    let mut reduces: Vec<(NonterminalID, u32)> = Vec::new();

    match action {
        Action::Shift(target, replace) => pending.push_shift(*target, *replace),
        Action::StackShifts(_) => return Ok(None),
        Action::GuardedStackShifts(_) => return Ok(None),
        Action::Reduce(nt, len) => reduces.push((*nt, *len)),
        Action::Split {
            shift,
            reduces: action_reduces,
            accept,
        } => {
            if let Some((target, replace)) = shift {
                pending.push_shift(*target, *replace);
            }
            reduces.extend(action_reduces.iter().copied());
            if *accept {
                pending.push_accept();
            }
        }
        Action::Accept => pending.push_accept(),
    }

    let mut changed = false;
    for (lhs, pop_len) in reduces {
        if pop_len != 1 {
            pending.push_reduce(lhs, pop_len);
            continue;
        }

        let Some(reduce_dst) = unit_reduce_destination(table, predecessors, state, lhs) else {
            pending.push_reduce(lhs, pop_len);
            continue;
        };

        match table.action[reduce_dst as usize].get(&tid).cloned() {
            None => {
                pending.push_reduce(lhs, pop_len);
            }
            Some(inline_action) => {
                let resolved_inline = match try_inline_unit_reductions_for_cell_inner(
                    table,
                    predecessors,
                    reduce_dst,
                    tid,
                    &inline_action,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                    visiting,
                )? {
                    Some(CellUpdate::Set(action)) => Some(action),
                    Some(CellUpdate::Remove) => None,
                    None => Some(inline_action),
                };

                let Some(resolved_inline) = resolved_inline else {
                    changed = true;
                    continue;
                };

                merge_action_into_pending(
                    &mut pending,
                    &resolved_inline,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                )?;
                changed = true;
            }
        }
    }

    let result = if !changed {
        Ok(None)
    } else {
        Ok(match pending.maybe_finish() {
            Some(action) => Some(CellUpdate::Set(action)),
            None => Some(CellUpdate::Remove),
        })
    };
    visiting.remove(&(state, tid));
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Item {
    rule: u32,
    dot: u32,
    stack_depth: u32,
}

impl Item {
    fn new(rule: u32, dot: u32, stack_depth: u32) -> Self {
        Self { rule, dot, stack_depth }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

fn lr0_closure(items: &BTreeSet<Item>, rules: &[Rule]) -> BTreeSet<Item> {
    let mut result = items.clone();
    let mut queue: VecDeque<Item> = items.iter().copied().collect();

    while let Some(item) = queue.pop_front() {
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            for (i, r) in rules.iter().enumerate() {
                if r.lhs == *nt {
                    let new_item = Item::new(i as u32, 0, r.rhs.len() as u32);
                    if result.insert(new_item) {
                        queue.push_back(new_item);
                    }
                }
            }
        }
    }
    result
}

fn lr0_goto_set(items: &BTreeSet<Item>, sym: &Symbol, rules: &[Rule]) -> BTreeSet<Item> {
    let mut kernel = BTreeSet::new();
    for item in items {
        if item.next_symbol(rules) == Some(sym) {
            kernel.insert(Item::new(item.rule, item.dot + 1, item.stack_depth));
        }
    }
    lr0_closure(&kernel, rules)
}

fn build_item_sets<ItemT, NextSymbol, GotoSet>(
    initial: BTreeSet<ItemT>,
    next_symbol: NextSymbol,
    goto_set: GotoSet,
) -> (Vec<BTreeSet<ItemT>>, Vec<BTreeMap<Symbol, u32>>)
where
    ItemT: Copy + Ord + std::hash::Hash,
    NextSymbol: Fn(&ItemT) -> Option<Symbol>,
    GotoSet: Fn(&BTreeSet<ItemT>, &Symbol) -> BTreeSet<ItemT>,
{
    let mut item_sets = vec![initial.clone()];
    let mut transitions = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<ItemT>, u32> = FxHashMap::default();
    set_to_id.insert(initial.iter().copied().collect(), 0);

    let mut queue = VecDeque::from([0u32]);
    while let Some(state_id) = queue.pop_front() {
        let symbols: BTreeSet<Symbol> = item_sets[state_id as usize]
            .iter()
            .filter_map(&next_symbol)
            .collect();

        for symbol in &symbols {
            let target_items = goto_set(&item_sets[state_id as usize], symbol);
            if target_items.is_empty() {
                continue;
            }

            let key: Vec<ItemT> = target_items.iter().copied().collect();
            let target_id = if let Some(&existing_id) = set_to_id.get(&key) {
                existing_id
            } else {
                let new_id = item_sets.len() as u32;
                set_to_id.insert(key, new_id);
                item_sets.push(target_items);
                transitions.push(BTreeMap::new());
                queue.push_back(new_id);
                new_id
            };

            transitions[state_id as usize].insert(symbol.clone(), target_id);
        }
    }

    (item_sets, transitions)
}

#[allow(dead_code)]
fn build_lr0_item_sets(grammar: &AnalyzedGrammar) -> (Vec<BTreeSet<Item>>, Vec<BTreeMap<Symbol, u32>>) {
    let rules = &grammar.rules;

    let initial = {
        let mut s = BTreeSet::new();
        s.insert(Item::new(0, 0, rules[0].rhs.len() as u32)); 
        lr0_closure(&s, rules)
    };

    build_item_sets(
        initial,
        |item| item.next_symbol(rules).cloned(),
        |items, sym| lr0_goto_set(items, sym, rules),
    )
}

#[derive(Debug, Default, Clone)]
struct PendingAction {
    shift: Option<(u32, bool)>,
    reduces: Vec<(NonterminalID, u32)>,
    accept: bool,
}

impl PendingAction {
    fn push_shift(&mut self, target: u32, is_replace: bool) {
        match self.shift {
            Some((existing, _)) => debug_assert_eq!(existing, target),
            None => self.shift = Some((target, is_replace)),
        }
    }

    fn push_reduce(&mut self, nt: NonterminalID, len: u32) {
        self.reduces.push((nt, len));
    }

    fn push_accept(&mut self) {
        self.accept = true;
    }

    fn maybe_finish(mut self) -> Option<Action> {
        self.reduces.sort_unstable();
        self.reduces.dedup();
        match (self.shift, self.reduces.len(), self.accept) {
            (None, 0, false) => None,
            (Some((target, replace)), 0, false) => Some(Action::Shift(target, replace)),
            (None, 1, false) => Some(Action::Reduce(self.reduces[0].0, self.reduces[0].1)),
            (None, 0, true) => Some(Action::Accept),
            (shift, _, accept) => Some(Action::Split {
                shift,
                reduces: self.reduces,
                accept,
            }),
        }
    }

    fn finish(self) -> Action {
        self.maybe_finish()
            .expect("PendingAction::finish called on an empty action")
    }
}

fn initialize_pending_and_goto(
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> (
    Vec<BTreeMap<TerminalID, PendingAction>>,
    Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    FxHashSet<(u32, TerminalID)>,
) {
    let mut pending = std::iter::repeat_with(BTreeMap::<TerminalID, PendingAction>::new)
        .take(transitions.len())
        .collect::<Vec<_>>();
    let mut goto: Vec<FxHashMap<NonterminalID, (u32, bool)>> = (0..transitions.len()).map(|_| FxHashMap::default()).collect();
    let mut forwarded_shifts = FxHashSet::default();

    for (state_id, by_symbol) in transitions.iter().enumerate() {
        for (symbol, &(target, is_replace, is_forwarded)) in by_symbol {
            match symbol {
                Symbol::Terminal(terminal) => {
                    pending[state_id]
                        .entry(*terminal)
                        .or_default()
                        .push_shift(target, is_replace);
                    if is_forwarded {
                        forwarded_shifts.insert((state_id as u32, *terminal));
                    }
                }
                Symbol::Nonterminal(nonterminal) => {
                    goto[state_id].insert(*nonterminal, (target, is_replace));
                }
            }
        }
    }

    (pending, goto, forwarded_shifts)
}

fn finish_table(
    grammar: &AnalyzedGrammar,
    pending: Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    forwarded_shifts: FxHashSet<(u32, TerminalID)>,
) -> GLRTable {
    let action: Vec<ActionRow> = pending
        .into_iter()
        .map(|by_terminal| {
            by_terminal
                .into_iter()
                .map(|(terminal, pending)| (terminal, pending.finish()))
                .collect()
        })
        .collect();
    let goto: Vec<GotoRow> = goto.into_iter().map(IntoIterator::into_iter).map(Iterator::collect).collect();
    let num_states = action.len() as u32;

    GLRTable {
        action,
        goto,
        num_states,
        num_terminals: grammar.num_terminals,
        num_rules: grammar.rules.len() as u32,
        rules: grammar.rules.clone(),
        forwarded_shifts,
    }
}

#[allow(dead_code)]
fn build_slr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<Item>],
    transitions: &[BTreeMap<Symbol, u32>],
) -> GLRTable {
    // Convert old-style transitions to new format with replace=false, forwarded=false
    let transitions_with_replace: Vec<BTreeMap<Symbol, (u32, bool, bool)>> = transitions
        .iter()
        .map(|m| m.iter().map(|(s, &t)| (s.clone(), (t, false, false))).collect())
        .collect();
    let (mut pending, goto, forwarded_shifts) = initialize_pending_and_goto(&transitions_with_replace);

    for (state_id, items) in item_sets.iter().enumerate() {

        for item in items {
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }

            if item.rule == 0 {
                pending[state_id].entry(EOF).or_default().push_accept();
                continue;
            }

            for &lookahead in &grammar.follow[rule.lhs as usize] {
                pending[state_id]
                    .entry(lookahead)
                    .or_default()
                    .push_reduce(rule.lhs, item.stack_depth);
            }
        }
    }

    finish_table(grammar, pending, goto, forwarded_shifts)
}

// LR(1) item set construction.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LR1Item {
    rule: u32,
    dot: u32,
    lookahead: TerminalID,
    stack_depth: u32,
    /// When true, this item was "transferred" from a parent state to provide
    /// goto information.  Transferred items do NOT participate in closure,
    /// shift actions, or reduce actions — only goto transitions.
    transferred: bool,
}

impl LR1Item {
    fn new(rule: u32, dot: u32, lookahead: TerminalID, stack_depth: u32) -> Self {
        Self { rule, dot, lookahead, stack_depth, transferred: false }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

/// Compute FIRST set for a sequence of symbols followed by a lookahead terminal.
fn first_of_sequence(
    symbols: &[Symbol],
    lookahead: TerminalID,
    first: &[BTreeSet<TerminalID>],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeSet<TerminalID> {
    let mut result = BTreeSet::new();
    let mut all_nullable = true;
    for sym in symbols {
        match sym {
            Symbol::Terminal(t) => {
                result.insert(*t);
                all_nullable = false;
                break;
            }
            Symbol::Nonterminal(nt) => {
                result.extend(&first[*nt as usize]);
                if !nullable.contains(nt) {
                    all_nullable = false;
                    break;
                }
            }
        }
    }
    if all_nullable {
        result.insert(lookahead);
    }
    result
}

fn lr1_closure(
    items: &BTreeSet<LR1Item>,
    grammar: &AnalyzedGrammar,
) -> BTreeSet<LR1Item> {
    let rules = &grammar.rules;
    let mut result = items.clone();
    let mut queue: VecDeque<LR1Item> = items.iter().copied().collect();

    while let Some(item) = queue.pop_front() {
        // Transferred items do not participate in closure.
        if item.transferred {
            continue;
        }
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            let rhs = &rules[item.rule as usize].rhs;
            let beta = &rhs[(item.dot as usize + 1)..];

            let lookaheads = first_of_sequence(beta, item.lookahead, &grammar.first, &grammar.nullable);

            for &i in &grammar.rules_by_lhs[*nt as usize] {
                let sd = grammar.rules[i as usize].rhs.len() as u32;
                for &la in &lookaheads {
                    let new_item = LR1Item::new(i, 0, la, sd);
                    if result.insert(new_item) {
                        queue.push_back(new_item);
                    }
                }
            }
        }
    }
    result
}

#[allow(dead_code)]
fn lr1_goto_set(
    items: &BTreeSet<LR1Item>,
    sym: &Symbol,
    grammar: &AnalyzedGrammar,
) -> BTreeSet<LR1Item> {
    let rules = &grammar.rules;
    let mut kernel = BTreeSet::new();
    for item in items {
        if item.next_symbol(rules) == Some(sym) {
            kernel.insert(LR1Item::new(item.rule, item.dot + 1, item.lookahead, item.stack_depth));
        }
    }
    lr1_closure(&kernel, grammar)
}

/// Compute transferred items for the local-forward replace optimisation.
///
/// For each dot-1 item `[A → X . rest, la]` in `kernel`, find "foo items" in
/// `source_items` — items whose symbol-after-dot is `Nonterminal(A)`.  These
/// are the items that generate gotos for `A` in the source state.  Transferring
/// them into the target kernel provides the same gotos at the target so the
/// transition can be marked replace.
///
/// Returns `None` if:
/// - any dot-1 item belongs to `rule == 0` (augmented start), or
/// - any dot-1 item is NOT completed (i.e., not a single-symbol production), or
/// - any dot-1 item's LHS nonterminal has NO foo items in the source.
///
/// Recursively follows single-symbol production chains: when a foo item is
/// itself a single-symbol production at dot=0, its LHS nonterminal also needs
/// foo items in the source.
///
/// Returns `Some(transferred)` with the set of transferred items otherwise.

/// Eagerly advance transferred items past completed nonterminals in the
/// same kernel.  For example, if the kernel has:
///   [LPAREN → '('. , sd=0]  (completed, non-transferred)
///   [f → .LPAREN e ')' , transferred]
/// then the transferred item can be directly advanced to [f → LPAREN.e ')']
/// because LPAREN is already completed.  This avoids creating a zero-pop
/// reduce + goto in the table.
fn eagerly_advance_transferred(
    kernel: &BTreeSet<LR1Item>,
    rules: &[Rule],
) -> BTreeSet<LR1Item> {
    // Collect completed nonterminals (LHS of completed non-transferred items
    // with sd == 0).
    let mut completed_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    for item in kernel {
        if item.transferred {
            continue;
        }
        let rule = &rules[item.rule as usize];
        if item.dot as usize == rule.rhs.len() && item.stack_depth == 0 {
            completed_nts.insert(rule.lhs);
        }
    }

    if completed_nts.is_empty() {
        return kernel.clone();
    }

    let mut advanced_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    let mut result = BTreeSet::new();
    for item in kernel {
        if item.transferred {
            if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                if completed_nts.contains(nt) {
                    // Advance the transferred item past the completed NT.
                    // Keep sd unchanged: the advancement replaces a
                    // reduce+goto pair that was a no-op in stack terms
                    // (zero-pop reduce followed by non-replace goto).
                    result.insert(LR1Item {
                        rule: item.rule,
                        dot: item.dot + 1,
                        lookahead: item.lookahead,
                        stack_depth: item.stack_depth,
                        transferred: false,
                    });
                    advanced_nts.insert(*nt);
                    continue;
                }
            }
        }
        result.insert(*item);
    }

    // Remove completed items whose nonterminal was consumed by advancement.
    // They would produce zero-pop reduces that are now unnecessary.
    let advanced_nts_copy = advanced_nts;
    result.retain(|item| {
        if item.transferred {
            return true;
        }
        let rule = &rules[item.rule as usize];
        if item.dot as usize == rule.rhs.len() && item.stack_depth == 0 {
            !advanced_nts_copy.contains(&rule.lhs)
        } else {
            true
        }
    });

    result
}

fn compute_transfer_items(
    kernel: &BTreeSet<LR1Item>,
    source_items: &BTreeSet<LR1Item>,
    rules: &[Rule],
) -> Option<Vec<LR1Item>> {
    let mut transferred = Vec::new();

    // Collect the LHS nonterminals of all dot-1 items.
    // Only completed dot-1 items (single-symbol productions) are eligible.
    let mut needed_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    for item in kernel.iter().filter(|it| it.dot == 1) {
        if item.rule == 0 {
            return None;
        }
        let rule = &rules[item.rule as usize];
        if (item.dot as usize) != rule.rhs.len() {
            return None;
        }
        needed_nts.insert(rule.lhs);
    }

    if needed_nts.is_empty() {
        return None;
    }

    // Iteratively find foo items, following the nonterminal chain.
    // The chain extends through ALL foo items' LHS nonterminals so that
    // gotos for every nonterminal in the reduce chain are available.
    let mut all_needed = needed_nts.clone();
    let mut found_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    loop {
        let mut new_needed: BTreeSet<NonterminalID> = BTreeSet::new();
        for item in source_items {
            if item.transferred {
                continue;
            }
            if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                if all_needed.contains(nt) && !found_nts.contains(nt) {
                    transferred.push(LR1Item {
                        transferred: true,
                        ..*item
                    });
                    found_nts.insert(*nt);
                    // Add this foo item's LHS to the chain so that gotos
                    // for it are also generated in the target state.
                    if item.dot == 0 {
                        let foo_rule = &rules[item.rule as usize];
                        let chain_nt = foo_rule.lhs;
                        if !all_needed.contains(&chain_nt) {
                            new_needed.insert(chain_nt);
                        }
                    }
                }
            }
        }
        if new_needed.is_empty() {
            break;
        }
        all_needed.extend(&new_needed);
    }

    // ALL initially needed nonterminals must have at least one foo item.
    // Chain-extended nonterminals may not have foo items (e.g. the
    // augmented start nonterminal) which is fine.
    if !needed_nts.is_subset(&found_nts) {
        return None;
    }

    if transferred.is_empty() {
        return None;
    }

    Some(transferred)
}

fn build_lr1_item_sets(
    grammar: &AnalyzedGrammar,
) -> (Vec<BTreeSet<LR1Item>>, Vec<BTreeMap<Symbol, (u32, bool, bool)>>) {
    let rules = &grammar.rules;

    let initial = {
        let mut s = BTreeSet::new();
        let sd = rules[0].rhs.len() as u32;
        s.insert(LR1Item::new(0, 0, EOF, sd));
        lr1_closure(&s, grammar)
    };

    let mut item_sets = vec![initial.clone()];
    let mut transitions: Vec<BTreeMap<Symbol, (u32, bool, bool)>> = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<LR1Item>, u32> = FxHashMap::default();
    set_to_id.insert(initial.iter().copied().collect(), 0);

    // Pre-compute safety checks for the transfer mechanism only when the
    // mechanism is enabled. Recursion detection can be expensive on large
    // grammars, so avoid paying that cost on the default path.
    let transfer_safe = if local_forward_replace_enabled() {
        !grammar.nullable.iter().any(|&n| n != 0) && !grammar_has_recursion(rules)
    } else {
        false
    };

    let mut queue = VecDeque::from([0u32]);
    while let Some(state_id) = queue.pop_front() {
        let source_items = item_sets[state_id as usize].clone();

        // Build all goto kernels in a single pass over items.
        let mut kernels: BTreeMap<Symbol, BTreeSet<LR1Item>> = BTreeMap::new();
        for item in &source_items {
            // Transferred items only advance through nonterminal gotos.
            if item.transferred {
                if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                    kernels
                        .entry(Symbol::Nonterminal(*nt))
                        .or_default()
                        .insert(LR1Item {
                            rule: item.rule,
                            dot: item.dot + 1,
                            lookahead: item.lookahead,
                            stack_depth: item.stack_depth,
                            transferred: false, // un-transfer on goto
                        });
                }
                continue;
            }
            if let Some(sym) = item.next_symbol(rules) {
                kernels
                    .entry(sym.clone())
                    .or_default()
                    .insert(LR1Item::new(item.rule, item.dot + 1, item.lookahead, item.stack_depth));
            }
        }

        for (symbol, mut kernel) in kernels {
            let base_kernel = kernel.clone();
            // Check replace condition: is_replace iff no item in the kernel
            // has dot at position 1 (i.e., all items came from items that
            // already had dot > 0).
            let has_dot_1 = kernel.iter().any(|item| item.dot == 1);
            let mut is_replace = match &symbol {
                Symbol::Terminal(_) => !has_dot_1 && replace_shifts_enabled(),
                Symbol::Nonterminal(_) => !has_dot_1 && replace_gotos_enabled(),
            };
            let mut used_transfer = false;

            // Local-forward transfer: when has_dot_1 but transfer is enabled,
            // try to transfer foo items from the source state into the kernel
            // so the transition can still be marked as replace.
            if has_dot_1 && !is_replace && local_forward_replace_enabled() && transfer_safe {
                let enable_for_this = match &symbol {
                    Symbol::Terminal(_) => replace_shifts_enabled(),
                    Symbol::Nonterminal(_) => replace_gotos_enabled(),
                };
                if enable_for_this {
                    if let Some(transferred) =
                        compute_transfer_items(&kernel, &source_items, rules)
                    {
                        kernel.extend(transferred);
                        // De-duplicate: remove transferred items that also
                        // exist as non-transferred.
                        let to_remove: Vec<LR1Item> = kernel
                            .iter()
                            .filter(|it| it.transferred)
                            .filter(|it| {
                                kernel.contains(&LR1Item {
                                    transferred: false,
                                    ..**it
                                })
                            })
                            .copied()
                            .collect();
                        for item in to_remove {
                            kernel.remove(&item);
                        }
                        is_replace = true;
                        used_transfer = true;
                    }
                }
            }

            let mut kernel_has_transferred = used_transfer && kernel.iter().any(|it| it.transferred);

            // If replace, decrement stack_depth for non-transferred kernel
            // items — the replace absorbs one stack level for items that
            // went through the shift. Transferred items didn't go through
            // the shift; they were copied from the source state and await
            // nonterminal gotos, so their sd is already correct.
            let mut adjusted_kernel: BTreeSet<LR1Item> = if is_replace {
                kernel
                    .iter()
                    .map(|item| {
                        if item.transferred {
                            *item
                        } else {
                            LR1Item {
                                stack_depth: item.stack_depth.saturating_sub(1),
                                ..*item
                            }
                        }
                    })
                    .collect()
            } else {
                kernel
            };

            let mut target_items = lr1_closure(&adjusted_kernel, grammar);
            if used_transfer && matches!(symbol, Symbol::Terminal(_)) {
                let base_target_items = lr1_closure(&base_kernel, grammar);
                let base_has_zero_pop_completed = base_target_items.iter().any(|item| {
                    let rule = &rules[item.rule as usize];
                    (item.dot as usize) == rule.rhs.len() && item.stack_depth == 0
                });
                let transferred_has_zero_pop_completed = target_items.iter().any(|item| {
                    let rule = &rules[item.rule as usize];
                    (item.dot as usize) == rule.rhs.len() && item.stack_depth == 0
                });
                if transferred_has_zero_pop_completed && !base_has_zero_pop_completed {
                    kernel = base_kernel;
                    is_replace = false;
                    kernel_has_transferred = false;
                    adjusted_kernel = kernel;
                    target_items = lr1_closure(&adjusted_kernel, grammar);
                }
            }

            // Track whether this replace was created by the transfer mechanism.
            let is_forwarded = is_replace && kernel_has_transferred;

            if target_items.is_empty() {
                continue;
            }

            let key: Vec<LR1Item> = target_items.iter().copied().collect();
            let target_id = if let Some(&existing_id) = set_to_id.get(&key) {
                existing_id
            } else {
                let new_id = item_sets.len() as u32;
                set_to_id.insert(key, new_id);
                item_sets.push(target_items);
                transitions.push(BTreeMap::new());
                queue.push_back(new_id);
                new_id
            };

            transitions[state_id as usize].insert(symbol, (target_id, is_replace, is_forwarded));
        }
    }

    (item_sets, transitions)
}

fn current_unique_reduce_len(
    pending: &[BTreeMap<TerminalID, PendingAction>],
    state: u32,
    lookahead: TerminalID,
    nonterminal: NonterminalID,
) -> Option<u32> {
    let pending_action = pending.get(state as usize)?.get(&lookahead)?;
    let mut unique_len = None;
    for &(reduce_nt, reduce_len) in &pending_action.reduces {
        if reduce_nt != nonterminal {
            continue;
        }
        match unique_len {
            None => unique_len = Some(reduce_len),
            Some(existing) if existing == reduce_len => {}
            Some(_) => return None,
        }
    }
    unique_len
}

/// Check if any nonterminal can transitively reach itself through the
/// grammar's production rules. Returns true if any recursion exists.
fn grammar_has_recursion(rules: &[Rule]) -> bool {
    let max_nt = rules.iter().map(|r| r.lhs).max().unwrap_or(0) as usize + 1;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); max_nt];
    for rule in rules {
        let lhs = rule.lhs as usize;
        for sym in &rule.rhs {
            if let Symbol::Nonterminal(nt) = sym {
                adj[lhs].push(*nt as usize);
            }
        }
    }

    // 0 = unvisited, 1 = visiting, 2 = done.
    let mut color = vec![0u8; max_nt];
    fn dfs(node: usize, adj: &[Vec<usize>], color: &mut [u8]) -> bool {
        color[node] = 1;
        for &next in &adj[node] {
            match color[next] {
                1 => return true,
                0 => {
                    if dfs(next, adj, color) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        color[node] = 2;
        false
    }

    for nt in 0..max_nt {
        if color[nt] == 0 && dfs(nt, &adj, &mut color) {
            return true;
        }
    }
    false
}

fn apply_local_forward_replace(
    pending: &mut Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: &mut Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
    rules: &[Rule],
) {
    // Build incoming-transition count per state so we can check whether
    // a reduce-state is uniquely reachable.  If multiple transitions
    // lead into the reduce state, rewriting its reduce length would
    // corrupt other paths (e.g. recursive grammars).
    let mut in_count = vec![0u32; item_sets.len()];
    for t in transitions {
        for &(target, _, _) in t.values() {
            in_count[target as usize] += 1;
        }
    }

    loop {
        let mut changed = false;

        for source in 0..item_sets.len() {
            for (symbol, &(target, _, _)) in &transitions[source] {
                let currently_non_replace = match symbol {
                    Symbol::Terminal(terminal) => pending[source]
                        .get(terminal)
                        .and_then(|pending_action| pending_action.shift)
                        .is_some_and(|(_, is_replace)| !is_replace),
                    Symbol::Nonterminal(nonterminal) => goto[source]
                        .get(nonterminal)
                        .is_some_and(|&(_, is_replace)| !is_replace),
                };
                if !currently_non_replace {
                    continue;
                }

                let dot1_items: Vec<_> = item_sets[target as usize]
                    .iter()
                    .copied()
                    .filter(|item| item.dot == 1)
                    .collect();
                if dot1_items.is_empty() {
                    continue;
                }

                let mut forwarded: BTreeMap<(u32, NonterminalID), u32> = BTreeMap::new();
                let mut rewrites = Vec::new();
                let mut valid = true;

                for item in dot1_items {
                    if item.rule == 0 {
                        valid = false;
                        break;
                    }

                    let rule = &rules[item.rule as usize];
                    let reduce_nt = rule.lhs;
                    let Some(&(forward_target, _)) = goto[source].get(&reduce_nt) else {
                        valid = false;
                        break;
                    };

                    let mut reduce_state = target;
                    let mut chain_unique = in_count[target as usize] <= 1;
                    for next_symbol in &rule.rhs[item.dot as usize..] {
                        let Some(&(next_state, _, _)) = transitions[reduce_state as usize].get(next_symbol) else {
                            valid = false;
                            break;
                        };
                        reduce_state = next_state;
                        // If a state in the chain is reachable from
                        // multiple predecessors, we can't safely rewrite
                        // its reduce because other paths share it.
                        if in_count[reduce_state as usize] > 1 {
                            chain_unique = false;
                        }
                    }
                    if !valid {
                        break;
                    }
                    if !chain_unique {
                        valid = false;
                        break;
                    }

                    let Some(current_len) = current_unique_reduce_len(
                        pending,
                        reduce_state,
                        item.lookahead,
                        reduce_nt,
                    ) else {
                        valid = false;
                        break;
                    };
                    if current_len != 1 {
                        valid = false;
                        break;
                    }

                    match forwarded.get(&(reduce_state, reduce_nt)) {
                        Some(&existing_target) if existing_target != forward_target => {
                            valid = false;
                            break;
                        }
                        Some(_) => {}
                        None => {
                            if let Some(&(existing_target, existing_replace)) =
                                goto[reduce_state as usize].get(&reduce_nt)
                            {
                                if existing_target != forward_target || !existing_replace {
                                    valid = false;
                                    break;
                                }
                            }
                            forwarded.insert((reduce_state, reduce_nt), forward_target);
                        }
                    }

                    rewrites.push((reduce_state, item.lookahead, reduce_nt));
                }

                if !valid || rewrites.is_empty() {
                    continue;
                }

                changed = true;

                match symbol {
                    Symbol::Terminal(terminal) => {
                        if let Some(pending_action) = pending[source].get_mut(terminal) {
                            if let Some((_, is_replace)) = pending_action.shift.as_mut() {
                                *is_replace = true;
                            }
                        }
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        if let Some((_, is_replace)) = goto[source].get_mut(nonterminal) {
                            *is_replace = true;
                        }
                    }
                }

                for &(reduce_state, lookahead, reduce_nt) in &rewrites {
                    if let Some(pending_action) = pending[reduce_state as usize].get_mut(&lookahead) {
                        for (existing_nt, reduce_len) in pending_action.reduces.iter_mut() {
                            if *existing_nt == reduce_nt && *reduce_len == 1 {
                                *reduce_len = 0;
                            }
                        }
                    }
                }

                for ((reduce_state, reduce_nt), forward_target) in forwarded {
                    goto[reduce_state as usize].insert(reduce_nt, (forward_target, true));
                }
            }
        }

        if !changed {
            break;
        }
    }
}

/// Inline zero-pop reduces: when a state has Reduce(nt, 0) on some terminal,
/// follow the goto to the target state and copy the target's action for that
/// terminal into the current state. This eliminates the zero-pop reduce
/// entirely, replacing it with a direct shift or accept.
///
/// Iterates until no more inlining is possible (handles chains of zero-pop
/// reduces).
fn inline_zero_pop_reduces(
    pending: &mut Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: &mut Vec<FxHashMap<NonterminalID, (u32, bool)>>,
) {
    loop {
        let mut changed = false;

        for state in 0..pending.len() {
            // Collect (terminal, nt, target_state) triples for zero-pop reduces.
            let mut to_inline: Vec<(TerminalID, NonterminalID, u32)> = Vec::new();
            if let Some(by_terminal) = pending.get(state) {
                for (&terminal, pa) in by_terminal {
                    for &(reduce_nt, reduce_len) in &pa.reduces {
                        if reduce_len == 0 {
                            if let Some(&(target, _)) = goto[state].get(&reduce_nt) {
                                to_inline.push((terminal, reduce_nt, target));
                            }
                        }
                    }
                }
            }

            for (terminal, reduce_nt, target) in to_inline {
                if target as usize == state {
                    continue; // avoid self-reference
                }

                // Read action at target state for the same terminal.
                let target_pa = pending[target as usize].get(&terminal).cloned();

                if let Some(pa) = pending[state].get_mut(&terminal) {
                    // Remove the zero-pop reduce.
                    pa.reduces.retain(|&(nt, len)| !(nt == reduce_nt && len == 0));

                    // Inline target's actions.
                    if let Some(tpa) = target_pa {
                        if let Some((shift_target, _)) = tpa.shift {
                            pa.push_shift(shift_target, true);
                        }
                        for (nt, len) in &tpa.reduces {
                            pa.push_reduce(*nt, *len);
                        }
                        if tpa.accept {
                            pa.push_accept();
                        }

                        // Propagate gotos needed for any zero-pop reduces
                        // we just inlined from the target state.
                        for &(nt, len) in &tpa.reduces {
                            if len == 0 {
                                if let Some(&goto_entry) = goto[target as usize].get(&nt) {
                                    goto[state].entry(nt).or_insert(goto_entry);
                                }
                            }
                        }
                    }

                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }
}

fn build_lr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> GLRTable {
    let (pending, goto, forwarded_shifts) = initialize_pending_and_goto(transitions);

    // Replace flags are now carried in the transitions map from
    // build_lr1_item_sets, so we don't need to recompute them here.

    let mut pending = pending;
    let goto = goto;
    for (state_id, items) in item_sets.iter().enumerate() {

        for item in items {
            // Transferred items do not generate reduces.
            if item.transferred {
                continue;
            }
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }

            if item.rule == 0 {
                pending[state_id].entry(item.lookahead).or_default().push_accept();
                continue;
            }

            pending[state_id]
                .entry(item.lookahead)
                .or_default()
                .push_reduce(rule.lhs, item.stack_depth);
        }
    }

    finish_table(grammar, pending, goto, forwarded_shifts)
}

// IELR-style merge.

