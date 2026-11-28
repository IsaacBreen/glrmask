//! Flattened tokenizer-DFA views for the equivalence-analysis passes.

use crate::automata::lexer::Lexer;
use std::sync::Arc;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

fn normalize_group_ids(mut groups: Vec<usize>) -> Vec<usize> {
    groups.sort_unstable();
    groups.dedup();
    groups
}

fn collect_group_ids(groups: impl Iterator<Item = u32>) -> Vec<usize> {
    normalize_group_ids(groups.map(|group| group as usize).collect())
}

fn collect_filtered_group_ids(
    groups: impl Iterator<Item = u32>,
    active_groups: &[bool],
) -> Vec<usize> {
    normalize_group_ids(
        groups
            .map(|group| group as usize)
            .filter(|&group| group < active_groups.len() && active_groups[group])
            .collect(),
    )
}

/// Per-state metadata: finalizers and reachable groups.
/// Transitions are stored separately in `FlatDfa::transitions` for sharing.
#[derive(Debug, Clone)]
pub struct FlatDfaState {
    /// Sorted list of group IDs that finalize at this state.
    pub finalizers: Vec<usize>,
    /// Sorted list of group IDs reachable from this state.
    pub possible_future_group_ids: Vec<usize>,
}

/// An exact byte action on the represented alphabet. Dense physical rows and
/// finite-vocabulary quotient rows differ only in their column map. The rows
/// always exist; derived transposes must never stand in for missing topology.
#[derive(Debug, Clone)]
pub struct ByteRows {
    pub(super) values: Arc<[u32]>,
    pub(super) byte_to_column: [u8; 256],
    pub(super) width: usize,
}

pub(super) fn column_representatives(classes: &[u8; 256]) -> Vec<u8> {
    let width = classes.iter().copied().max().unwrap() as usize + 1;
    let mut representatives = vec![0; width];
    for byte in (0..=255u8).rev() {
        representatives[classes[byte as usize] as usize] = byte;
    }
    representatives
}

/// Give every relevant byte a column, reserving zero for the unused domain.
/// A wider alphabet can use the dense representation without expansion.
pub(super) fn conservative_byte_classes(relevant: &[bool; 256]) -> Option<[u8; 256]> {
    if relevant.iter().filter(|&&active| active).count() >= 255 {
        return None;
    }
    let mut classes = [0; 256];
    let mut next = 1u8;
    for (byte, &active) in relevant.iter().enumerate() {
        if active {
            classes[byte] = next;
            next += 1;
        }
    }
    Some(classes)
}

impl ByteRows {
    pub fn dense(values: Arc<[u32]>) -> Self {
        Self::new(values, std::array::from_fn(|byte| byte as u8))
    }

    pub fn new(values: Arc<[u32]>, byte_to_column: [u8; 256]) -> Self {
        let width = byte_to_column.iter().copied().max().unwrap() as usize + 1;
        assert_eq!(values.len() % width, 0, "invalid byte rows");
        Self { values, byte_to_column, width }
    }

    pub fn num_states(&self) -> usize {
        self.values.len() / self.width
    }

    #[inline]
    pub fn row(&self, state: usize) -> &[u32] {
        let base = state * self.width;
        &self.values[base..base + self.width]
    }

    #[inline]
    pub fn trans(&self, state: usize, byte: usize) -> u32 {
        self.values[state * self.width + self.byte_to_column[byte] as usize]
    }

    pub fn same_topology(&self, other: &Self) -> bool {
        self.byte_to_column == other.byte_to_column
            && (Arc::ptr_eq(&self.values, &other.values) || self.values == other.values)
    }

    /// Materialize an alternative exact layout, or borrow the existing row
    /// storage by Arc when the requested coordinate is already present.
    pub(super) fn repack(&self, classes: &[u8; 256], transposed: bool) -> Arc<[u32]> {
        if !transposed && self.byte_to_column == *classes {
            return Arc::clone(&self.values);
        }
        let representatives = column_representatives(classes);
        let width = representatives.len();
        let states = self.num_states();
        let mut values = vec![u32::MAX; states * width];
        for state in 0..states {
            let row = self.row(state);
            for (class, &byte) in representatives.iter().enumerate() {
                let index = if transposed { class * states + state } else { state * width + class };
                values[index] = row[self.byte_to_column[byte as usize] as usize];
            }
        }
        Arc::from(values)
    }
}

#[derive(Debug, Clone)]
pub struct FlatDfa {
    pub states: Vec<FlatDfaState>,
    pub start_state: usize,
    pub transitions: ByteRows,
}
pub(super) const BYTE_COLUMN_HASH_MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;

pub(super) fn byte_classes_from_column_hashes(
    num_states: usize,
    column_hashes: &[u64; 256],
    relevant: &[bool; 256],
    target: impl Fn(usize, usize) -> u32,
) -> [u8; 256] {
    let mut sorted: Vec<u8> = (0..=255).filter(|&byte| relevant[byte as usize]).collect();
    sorted.sort_unstable_by_key(|&byte| column_hashes[byte as usize]);

    let mut classes = [0; 256];
    // Keep one unused catch-all column only for restricted alphabets.
    let mut next = if sorted.len() == 256 { 0 } else { 1 };
    for (i, &byte) in sorted.iter().enumerate() {
        let hash = column_hashes[byte as usize];
        let prior = sorted[..i].iter().rev()
            .take_while(|&&previous| column_hashes[previous as usize] == hash)
            .find(|&&previous| (0..num_states).all(|state|
                target(state, byte as usize) == target(state, previous as usize)));
        classes[byte as usize] = if let Some(&previous) = prior {
            classes[previous as usize]
        } else {
            let class = u8::try_from(next).expect("byte class domain");
            next += 1;
            class
        };
    }
    classes
}
pub fn compute_physical_byte_classes(tokenizer: &Tokenizer, transitions: &[u32]) -> [u8; 256] {
    let num_states = tokenizer.num_states() as usize;
    assert_eq!(transitions.len(), num_states * 256);

    let hashes = physical_column_hashes(tokenizer, Some(transitions));
    byte_classes_from_column_hashes(num_states, &hashes, &[true; 256],
        |state, byte| transitions[state * 256 + byte])
}

/// Polynomial column hashes derived from sparse rows. The default is the
/// all-dead column; each present edge changes just its weighted term.
fn physical_column_hashes(tokenizer: &Tokenizer, flat: Option<&[u32]>) -> [u64; 256] {
    let mut power = 1u64;
    let mut dead_hash = 0u64;
    let mut deltas = [0u64; 256];
    for state in (0..tokenizer.num_states()).rev() {
        dead_hash = dead_hash.wrapping_add((u32::MAX as u64).wrapping_mul(power));
        for (byte, target) in tokenizer.transitions_from(state) {
            debug_assert!(flat.is_none_or(|rows| rows[state as usize * 256 + byte as usize] == target));
            deltas[byte as usize] = deltas[byte as usize]
                .wrapping_add((target as u64).wrapping_sub(u32::MAX as u64).wrapping_mul(power));
        }
        power = power.wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER);
    }
    deltas.map(|delta| delta.wrapping_add(dead_hash))
}

/// Same physical observation without allocating a dense table.
pub(super) fn compute_tokenizer_byte_classes(tokenizer: &Tokenizer) -> [u8; 256] {
    byte_classes_from_column_hashes(tokenizer.num_states() as usize,
        &physical_column_hashes(tokenizer, None), &[true; 256],
        |state, byte| tokenizer.get_transition(state as u32, byte as u8))
}

/// Scan the represented byte action once. Dense physical rows have no
/// indirection in the inner loop; a finite alphabet visits only its columns.
pub(super) fn scan_byte_columns(
    rows: &ByteRows,
    relevant: &[bool; 256],
    mut observe: impl FnMut(usize, usize, u32),
) {
    let bytes: Vec<usize> = (0..256).filter(|&byte| relevant[byte]).collect();
    if bytes.len() == 256 && rows.width == 256
        && rows.byte_to_column.iter().enumerate().all(|(byte, &column)| byte == column as usize)
    {
        for (state, row) in rows.values.chunks_exact(256).enumerate() {
            for (byte, &target) in row.iter().enumerate() {
                observe(state, byte, target);
            }
        }
    } else {
        for state in 0..rows.num_states() {
            let row = rows.row(state);
            for &byte in &bytes {
                observe(state, byte, row[rows.byte_to_column[byte] as usize]);
            }
        }
    }
}

pub fn compute_byte_classes(dfa: &FlatDfa) -> [u8; 256] {
    let mut hashes = [0u64; 256];
    scan_byte_columns(&dfa.transitions, &[true; 256], |_, byte, target| {
        hashes[byte] = hashes[byte].wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER)
            .wrapping_add(target as u64);
    });
    byte_classes_from_column_hashes(dfa.states.len(), &hashes, &[true; 256],
        |state, byte| dfa.trans(state, byte))
}

/// Exact byte congruence of the physical lexer projected onto the selected
/// terminal languages. Dead sources and destinations have no active action.
pub fn compute_active_physical_byte_classes_from_flat_trans(
    tokenizer: &Tokenizer,
    active_groups: &[bool],
    flat_trans: &[u32],
) -> [u8; 256] {
    let states = tokenizer.num_states() as usize;
    assert_eq!(flat_trans.len(), states * 256);
    let active_language = (0..states)
        .map(|state| {
            tokenizer
                .matched_terminals_iter(state as u32)
                .chain(tokenizer.possible_future_terminals_iter(state as u32))
                .any(|terminal| active_groups.get(terminal as usize).copied().unwrap_or(false))
        })
        .collect::<Vec<_>>();

    #[inline]
    fn projected_target(
        flat_trans: &[u32],
        active_language: &[bool],
        source: usize,
        byte: usize,
    ) -> u32 {
        if !active_language[source] {
            return u32::MAX;
        }
        let target = flat_trans[source * 256 + byte];
        if target == u32::MAX || !active_language[target as usize] {
            u32::MAX
        } else {
            target
        }
    }

    let mut hashes = [0u64; 256];
    for source in 0..states {
        for byte in 0..256 {
            hashes[byte] = hashes[byte]
                .wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER)
                .wrapping_add(projected_target(flat_trans, &active_language, source, byte) as u64);
        }
    }
    byte_classes_from_column_hashes(states, &hashes, &[true; 256],
        |source, byte| projected_target(flat_trans, &active_language, source, byte))
}
#[inline]
fn state_has_active_language(state: &FlatDfaState) -> bool {
    !state.finalizers.is_empty() || !state.possible_future_group_ids.is_empty()
}

fn tokenizer_state(tokenizer: &Tokenizer, state: u32, active: Option<&[bool]>) -> FlatDfaState {
    let finalizers = tokenizer.matched_terminals_iter(state);
    let futures = tokenizer.possible_future_terminals_iter(state);
    let (finalizers, possible_future_group_ids) = match active {
        Some(active) => (collect_filtered_group_ids(finalizers, active), collect_filtered_group_ids(futures, active)),
        None => (collect_group_ids(finalizers), collect_group_ids(futures)),
    };
    FlatDfaState { finalizers, possible_future_group_ids }
}

impl FlatDfa {
    #[inline]
    pub fn trans(&self, state: usize, byte: usize) -> u32 {
        self.transitions.trans(state, byte)
    }

    #[inline]
    pub fn transitions_for(&self, state: usize) -> &[u32] {
        self.transitions.row(state)
    }

    fn from_tokenizer_parts(tokenizer: &Tokenizer, active: Option<&[bool]>, flat: Option<&Arc<[u32]>>) -> Self {
        let num_states = tokenizer.num_states() as usize;
        let states: Vec<_> = (0..num_states).map(|state| tokenizer_state(tokenizer, state as u32, active)).collect();
        let transitions = if let Some(flat) = flat.filter(|_| active.is_none()) {
            Arc::clone(flat)
        } else {
            let alive: Vec<_> = states.iter().map(state_has_active_language).collect();
            let mut rows = vec![u32::MAX; num_states * 256];
            for state in 0..num_states {
                if active.is_some() && !alive[state] { continue; }
                let row = &mut rows[state * 256..(state + 1) * 256];
                if let Some(flat) = flat {
                    row.copy_from_slice(&flat[state * 256..(state + 1) * 256]);
                    if active.is_some() {
                        for target in row {
                            if *target != u32::MAX && !alive[*target as usize] { *target = u32::MAX; }
                        }
                    }
                } else {
                    for (byte, target) in tokenizer.transitions_from(state as u32) {
                        if active.is_none() || alive[target as usize] {
                            row[byte as usize] = target;
                        }
                    }
                }
            }
            Arc::from(rows)
        };
        Self { states, start_state: tokenizer.start_state() as usize, transitions: ByteRows::dense(transitions) }
    }

    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        Self::from_tokenizer_parts(tokenizer, None, None)
    }

    pub fn from_tokenizer_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {
        Self::from_tokenizer_parts(tokenizer, Some(active_groups), None)
    }

    pub fn from_flat_trans(flat_trans: &Arc<[u32]>, tokenizer: &Tokenizer) -> Self {
        Self::from_tokenizer_parts(tokenizer, None, Some(flat_trans))
    }

    pub fn from_flat_trans_filtered(flat_trans: &Arc<[u32]>, tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {
        Self::from_tokenizer_parts(tokenizer, Some(active_groups), Some(flat_trans))
    }
}
/// A thin wrapper around glrmask's `Tokenizer` that exposes the flattened DFA.
///
/// The equivalence-analysis code calls `dfa()` and accesses `.states` and
/// `.start_state` directly.
/// This wrapper pre-extracts all data into `FlatDfa` on construction.
pub struct TokenizerView {
    pub flat_dfa: FlatDfa,
}

impl TokenizerView {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_tokenizer(tokenizer),
        }
    }

    /// Build a view that filters finalizers and futures to only active groups.
    pub fn new_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_tokenizer_filtered(tokenizer, active_groups),
        }
    }

    /// Build a view using a pre-built shared flat transition table (no group filtering).
    /// Shares transition data via Arc — zero-copy.
    pub fn new_from_flat_trans(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
    ) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_flat_trans(flat_trans, tokenizer),
        }
    }

    /// Build an active-terminal view using a pre-built shared raw transition
    /// table. The projected transition table is owned by this view.
    pub fn new_filtered_from_flat_trans(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
        active_groups: &[bool],
    ) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_flat_trans_filtered(flat_trans, tokenizer, active_groups),
        }
    }

    pub fn dfa(&self) -> &FlatDfa {
        &self.flat_dfa
    }

    pub fn initial_state_id(&self) -> usize {
        self.flat_dfa.start_state
    }

    /// Materialize the same certified quotient in either the dense byte
    /// coordinate or the finite-vocabulary column coordinate. The irrelevant
    /// catch-all is dead; the certificate concerns only the observed alphabet.
    pub fn new_filtered_quotient_from_flat_trans_with_observation_cache(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
        active_groups: &[bool],
        state_map: &ManyToOneIdMap,
        raw_observation_ids: &[u32],
        observation_representatives: &[u32],
        relevant_bytes: Option<&[bool; 256]>,
    ) -> (Self, f64) {
        let raw_states = tokenizer.num_states() as usize;
        assert_eq!(flat_trans.len(), raw_states * 256, "invalid raw transition table");
        assert_eq!(state_map.original_to_internal.len(), raw_states, "invalid state map");
        assert_eq!(raw_observation_ids.len(), raw_states, "invalid observation IDs");
        let started_at = std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some().then(Instant::now);
        let observations: Vec<_> = observation_representatives.iter()
            .map(|&state| tokenizer_state(tokenizer, state, Some(active_groups))).collect();
        let states: Vec<_> = state_map.representative_original_ids.iter()
            .map(|&state| observations[raw_observation_ids[state as usize] as usize].clone()).collect();
        let compact = relevant_bytes.and_then(conservative_byte_classes);
        let classes = compact.unwrap_or_else(|| std::array::from_fn(|byte| byte as u8));
        let representatives = column_representatives(&classes);
        let width = representatives.len();
        let alive: Vec<_> = states.iter().map(state_has_active_language).collect();
        let mut rows = vec![u32::MAX; states.len() * width];
        for (internal, &raw) in state_map.representative_original_ids.iter().enumerate() {
            if !alive[internal] { continue; }
            let source = raw as usize * 256;
            let row = &mut rows[internal * width..(internal + 1) * width];
            for (class, &byte) in representatives.iter().enumerate() {
                if compact.is_some() && class == 0 { continue; }
                let target = flat_trans[source + byte as usize];
                if target == u32::MAX { continue; }
                let mapped = state_map.original_to_internal[target as usize];
                if mapped != u32::MAX && alive[mapped as usize] {
                    row[class] = mapped;
                }
            }
        }
        let start_state = state_map.original_to_internal[tokenizer.start_state() as usize];
        assert_ne!(start_state, u32::MAX, "quotient start state must be mapped");
        (
            Self { flat_dfa: FlatDfa { states, start_state: start_state as usize,
                transitions: ByteRows::new(Arc::from(rows), classes) } },
            started_at.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0),
        )
    }
    /// Verify that `state_map` is an output-labelled right congruence for every
    /// byte that may occur in the active vocabulary. This is the exact condition
    /// needed to evaluate token paths in the quotient DFA.
    pub fn is_relevant_byte_congruent(
        &self,
        state_map: &ManyToOneIdMap,
        relevant_bytes: &[bool; 256],
    ) -> bool {
        let dfa = self.dfa();
        if state_map.original_to_internal.len() != dfa.states.len() {
            return false;
        }
        for (internal, members) in state_map.internal_to_originals.iter().enumerate() {
            let Some(&representative) = state_map.representative_original_ids.get(internal) else {
                return false;
            };
            let representative = representative as usize;
            if representative >= dfa.states.len()
                || state_map.original_to_internal[representative] != internal as u32
            {
                return false;
            }
            let representative_state = &dfa.states[representative];
            for &raw_state in members {
                let raw_state = raw_state as usize;
                if raw_state >= dfa.states.len()
                    || state_map.original_to_internal[raw_state] != internal as u32
                {
                    return false;
                }
                let state = &dfa.states[raw_state];
                if state.finalizers != representative_state.finalizers
                    || state.possible_future_group_ids
                        != representative_state.possible_future_group_ids
                {
                    return false;
                }
                for (byte, &relevant) in relevant_bytes.iter().enumerate() {
                    if !relevant {
                        continue;
                    }
                    let representative_target = dfa.trans(representative, byte);
                    let target = dfa.trans(raw_state, byte);
                    let mapped_representative = if representative_target == u32::MAX {
                        u32::MAX
                    } else {
                        state_map.original_to_internal[representative_target as usize]
                    };
                    let mapped_target = if target == u32::MAX {
                        u32::MAX
                    } else {
                        state_map.original_to_internal[target as usize]
                    };
                    if mapped_target != mapped_representative {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Materialize the certified quotient without changing the byte coordinate.
    pub fn quotient_by_state_map(&self, state_map: &ManyToOneIdMap) -> Self {
        let dfa = self.dfa();
        let states: Vec<_> = state_map.representative_original_ids.iter()
            .map(|&raw| dfa.states[raw as usize].clone()).collect();
        let alive: Vec<_> = states.iter().map(state_has_active_language).collect();
        let width = dfa.transitions.width;
        let mut rows = vec![u32::MAX; states.len() * width];
        for (internal, &raw) in state_map.representative_original_ids.iter().enumerate() {
            if !alive[internal] { continue; }
            for (column, &target) in dfa.transitions.row(raw as usize).iter().enumerate() {
                if target == u32::MAX { continue; }
                let mapped = state_map.original_to_internal[target as usize];
                assert_ne!(mapped, u32::MAX, "quotient target must be mapped");
                if alive[mapped as usize] { rows[internal * width + column] = mapped; }
            }
        }
        let start_state = state_map.original_to_internal[dfa.start_state];
        assert_ne!(start_state, u32::MAX, "quotient start state must be mapped");
        Self { flat_dfa: FlatDfa { states, start_state: start_state as usize,
            transitions: ByteRows::new(Arc::from(rows), dfa.transitions.byte_to_column) } }
    }
}
#[cfg(test)]
mod sparse_transition_cache_tests {
    use super::*;

    #[test]
    fn compact_rows_share_storage_and_compare_topology_not_shape() {
        let mut classes = [0; 256];
        classes[b'a' as usize] = 1;
        classes[b'b' as usize] = 1;
        classes[b'c' as usize] = 2;
        let values: Arc<[u32]> = Arc::from(vec![u32::MAX, 1, 0, u32::MAX, 0, 1]);
        let rows = ByteRows::new(Arc::clone(&values), classes);
        assert!(Arc::ptr_eq(&rows.repack(&classes, false), &values));
        assert_eq!(rows.repack(&classes, true).as_ref(), &[u32::MAX, u32::MAX, 1, 0, 0, 1]);
        for state in 0..2 {
            for byte in 0..256 {
                let expected = match byte as u8 {
                    b'a' | b'b' => 1 - state as u32,
                    b'c' => state as u32,
                    _ => u32::MAX,
                };
                assert_eq!(rows.trans(state, byte), expected);
            }
        }
        let mut different = classes;
        different[b'a' as usize] = 2;
        assert!(!rows.same_topology(&ByteRows::new(values, different)));
    }

    #[test]
    fn filtered_byte_classes_ignore_inactive_only_transition_topology() {
        use crate::automata::lexer::ast::Expr;
        use crate::automata::lexer::compile::build_regex_monolithic as build_regex;

        fn tokenizer(expressions: Vec<Expr>) -> Tokenizer {
            let terminal_count = expressions.len() as u32;
            build_regex(&expressions).into_tokenizer(
                terminal_count,
                Some(Arc::from(expressions.into_boxed_slice())),
            )
        }

        fn same_partition(left: &[u8; 256], right: &[u8; 256]) -> bool {
            (0..256).all(|a| {
                (0..256).all(|b| (left[a] == left[b]) == (right[a] == right[b]))
            })
        }

        let with_inactive = tokenizer(vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"xq".to_vec()),
            Expr::U8Seq(b"yrr".to_vec()),
        ]);
        let active_only = tokenizer(vec![Expr::U8Seq(b"ab".to_vec())]);

        let filtered = TokenizerView::new_filtered(&with_inactive, &[true, false, false]);
        let reference = TokenizerView::new(&active_only);
        let filtered_classes = compute_byte_classes(filtered.dfa());
        let reference_classes = compute_byte_classes(reference.dfa());

        assert_eq!(filtered.dfa().trans(filtered.dfa().start_state, b'x' as usize), u32::MAX);
        assert_eq!(filtered.dfa().trans(filtered.dfa().start_state, b'y' as usize), u32::MAX);
        assert_eq!(filtered_classes[b'x' as usize], filtered_classes[b'y' as usize]);
        assert!(
            same_partition(&filtered_classes, &reference_classes),
            "inactive-only lexer topology changed the active-language byte congruence",
        );
    }

    #[test]
    fn direct_active_physical_byte_classes_match_materialized_filtered_view() {
        use crate::automata::lexer::ast::Expr;
        use crate::automata::lexer::compile::build_regex_monolithic as build_regex;

        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"acx".to_vec()),
            Expr::U8Seq(b"yrr".to_vec()),
        ];
        let terminal_count = expressions.len() as u32;
        let tokenizer = build_regex(&expressions).into_tokenizer(
            terminal_count,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let mut flat_values = Vec::with_capacity(tokenizer.num_states() as usize * 256);
        for state in 0..tokenizer.num_states() {
            for byte in 0u16..=255 {
                flat_values.push(tokenizer.step(state, byte as u8).unwrap_or(u32::MAX));
            }
        }
        let flat = Arc::<[u32]>::from(flat_values);
        let active = [true, true, false];
        let direct = compute_active_physical_byte_classes_from_flat_trans(
            &tokenizer,
            &active,
            flat.as_ref(),
        );
        let materialized = TokenizerView::new_filtered_from_flat_trans(&flat, &tokenizer, &active);
        let materialized_classes = compute_byte_classes(materialized.dfa());
        assert!((0..256).all(|left| {
            (0..256).all(|right| {
                (direct[left] == direct[right])
                    == (materialized_classes[left] == materialized_classes[right])
            })
        }));
    }

    #[test]
    fn sparse_column_hashes_preserve_exact_byte_classes() {
        let num_states = 4usize;
        let dead = u32::MAX;
        let mut transitions = vec![dead; num_states * 256];
        let rows = [
            vec![(b'a', 1), (b'b', 1), (b'c', 2)],
            vec![(b'a', 2), (b'b', 2), (b'd', 3)],
            vec![(b'a', 3), (b'b', 3), (b'c', 1)],
            vec![(b'a', 3), (b'b', 3), (b'd', 1)],
        ];
        let mut row_weight = vec![0u64; num_states];
        let mut power = 1u64;
        for state in (0..num_states).rev() {
            row_weight[state] = power;
            power = power.wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER);
        }
        let mut all_dead_hash = 0u64;
        for _ in 0..num_states {
            all_dead_hash = all_dead_hash
                .wrapping_mul(BYTE_COLUMN_HASH_MULTIPLIER)
                .wrapping_add(dead as u64);
        }
        let mut sparse_hashes = [all_dead_hash; 256];
        for (state, row) in rows.iter().enumerate() {
            for &(byte, target) in row {
                transitions[state * 256 + byte as usize] = target;
                sparse_hashes[byte as usize] = sparse_hashes[byte as usize].wrapping_add(
                    (target as u64)
                        .wrapping_sub(dead as u64)
                        .wrapping_mul(row_weight[state]),
                );
            }
        }
        let dfa = FlatDfa {
            states: (0..num_states)
                .map(|_| FlatDfaState {
                    finalizers: Vec::new(),
                    possible_future_group_ids: Vec::new(),
                })
                .collect(),
            start_state: 0,
            transitions: ByteRows::dense(Arc::from(transitions)),
        };
        let classes = compute_byte_classes(&dfa);
        for left in 0..256 {
            for right in 0..256 {
                assert_eq!(classes[left] == classes[right],
                    (0..num_states).all(|state| dfa.trans(state, left) == dfa.trans(state, right)));
            }
        }
        assert_eq!(
            byte_classes_from_column_hashes(num_states, &sparse_hashes, &[true; 256], |state, byte| dfa.trans(state, byte)),
            compute_byte_classes(&dfa),
        );
    }
}
