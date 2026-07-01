//! Exact directed terminal subsumption for the L2+ terminal-DWA reference path.
//!
//! A representative terminal can subsume a member when the member's
//! one-terminal residual DFA is a quotient of a forward-closed part of the
//! representative's one-terminal residual DFA. The map is onto destination
//! residual partitions, but it may leave source-only partitions unmapped. This
//! strictly generalizes strict terminal interchangeability: an interchangeable
//! pair has maps both ways, each a bijection on residual partitions.
//!
//! The reference builder restores one member at a time. A member transport runs
//! only the representative terminal and relabels just that terminal to the
//! member. Consequently, a directed witness need preserve the representative
//! terminal's future matching behaviour exactly; it need not provide a global
//! permutation of every active terminal label.

use std::collections::{BTreeMap, BTreeSet};

use super::nwa_builder::TerminalNwaTransportMode;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::grammar::flat::TerminalID;

const NO_STATE: u32 = u32::MAX;

/// A partial, surjective residual-partition homomorphism from a representative
/// terminal's one-terminal DFA to a member terminal's one-terminal DFA.
///
/// `source_block_to_target_block[p] == None` means that the source residual
/// partition `p` is source-only. Mapped source partitions are transition-closed
/// and cover every destination partition. The inverse selection is the concrete
/// scanner state used to simulate a destination/original lexer state.
#[derive(Clone, Debug)]
struct SubsumptionMap {
    source_block_to_target_block: Vec<Option<u32>>,
    scanner_state_for_target_state: Vec<Option<u32>>,
}

struct RestrictedDfa<'a> {
    tokenizer: &'a Tokenizer,
    bytes: Vec<u8>,
    real_state_count: usize,
}

impl<'a> RestrictedDfa<'a> {
    fn new(tokenizer: &'a Tokenizer, relevant_bytes: &[bool; 256]) -> Self {
        Self {
            tokenizer,
            bytes: (0..=255u8)
                .filter(|&byte| relevant_bytes[byte as usize])
                .collect(),
            real_state_count: tokenizer.num_states() as usize,
        }
    }

    fn state_count(&self) -> usize {
        self.real_state_count + 1
    }

    fn dead_state(&self) -> usize {
        self.real_state_count
    }

    fn terminal_output(&self, state: usize, terminal: TerminalID) -> bool {
        state != self.dead_state()
            && self
                .tokenizer
                .matched_terminals_iter(state as u32)
                .any(|matched| matched == terminal)
    }

    fn successor(&self, state: usize, byte_slot: usize) -> usize {
        if state == self.dead_state() {
            return state;
        }
        let next = self
            .tokenizer
            .get_transition(state as u32, self.bytes[byte_slot]);
        if next == NO_STATE {
            self.dead_state()
        } else {
            next as usize
        }
    }

    fn minimize_terminal(&self, terminal: TerminalID) -> Vec<u32> {
        let state_count = self.state_count();
        let output = (0..state_count)
            .map(|state| self.terminal_output(state, terminal))
            .collect::<Vec<_>>();
        let mut blocks = classify_keys(output.iter().copied());
        loop {
            let keys = (0..state_count)
                .map(|state| {
                    let successors = (0..self.bytes.len())
                        .map(|slot| blocks[self.successor(state, slot)])
                        .collect::<Vec<_>>();
                    (output[state], successors)
                })
                .collect::<Vec<_>>();
            let next = classify_keys(keys);
            if next == blocks {
                return blocks;
            }
            blocks = next;
        }
    }

    /// Minimize two one-terminal observations together. Equal combined blocks
    /// are exactly equal future match languages after replacing `source` by
    /// `target`; the underlying byte transition graph is shared.
    fn minimize_source_and_target(&self, source: TerminalID, target: TerminalID) -> Vec<u32> {
        let state_count = self.state_count();
        let combined_count = state_count * 2;
        let output = (0..combined_count)
            .map(|combined| {
                let copy = combined / state_count;
                let state = combined % state_count;
                self.terminal_output(state, if copy == 0 { source } else { target })
            })
            .collect::<Vec<_>>();
        let mut blocks = classify_keys(output.iter().copied());
        loop {
            let keys = (0..combined_count)
                .map(|combined| {
                    let copy = combined / state_count;
                    let state = combined % state_count;
                    let successors = (0..self.bytes.len())
                        .map(|slot| {
                            blocks[copy * state_count + self.successor(state, slot)]
                        })
                        .collect::<Vec<_>>();
                    (output[combined], successors)
                })
                .collect::<Vec<_>>();
            let next = classify_keys(keys);
            if next == blocks {
                return blocks;
            }
            blocks = next;
        }
    }

    /// Return a directed source-to-target partition map if `source` subsumes
    /// `target`. The source domain may omit residual partitions, while every
    /// target partition must have at least one source preimage.
    fn subsumption_map(
        &self,
        source: TerminalID,
        target: TerminalID,
    ) -> Option<SubsumptionMap> {
        let state_count = self.state_count();
        let source_blocks = self.minimize_terminal(source);
        let target_blocks = self.minimize_terminal(target);
        let combined_blocks = self.minimize_source_and_target(source, target);

        let source_count = source_blocks
            .iter()
            .copied()
            .max()
            .map_or(0, |block| block as usize + 1);
        let target_count = target_blocks
            .iter()
            .copied()
            .max()
            .map_or(0, |block| block as usize + 1);

        // A source residual partition has one combined residual class. If it
        // had several, its supposedly equal source futures would be split by a
        // target observation and the witness is invalid.
        let mut combined_for_source_block = vec![None::<u32>; source_count];
        for state in 0..state_count {
            let source_block = source_blocks[state] as usize;
            let combined = combined_blocks[state];
            match combined_for_source_block[source_block] {
                Some(existing) if existing != combined => return None,
                Some(_) => {}
                None => combined_for_source_block[source_block] = Some(combined),
            }
        }

        let mut target_blocks_by_combined = BTreeMap::<u32, BTreeSet<u32>>::new();
        let mut target_representative_state = vec![self.dead_state(); target_count];
        for state in 0..state_count {
            let target_block = target_blocks[state] as usize;
            target_blocks_by_combined
                .entry(combined_blocks[state_count + state])
                .or_default()
                .insert(target_block as u32);
            if state < self.real_state_count {
                target_representative_state[target_block] = state;
            }
        }

        let mut source_block_to_target_block = vec![None::<u32>; source_count];
        for source_block in 0..source_count {
            let combined = combined_for_source_block[source_block]?;
            let Some(candidates) = target_blocks_by_combined.get(&combined) else {
                // This is precisely the asymmetric case: the representative
                // owns a residual partition that the member does not need.
                continue;
            };
            if candidates.len() == 1 {
                source_block_to_target_block[source_block] = candidates.iter().next().copied();
            }
        }

        // Validate the partial homomorphism directly. Its mapped domain is
        // automatically forward-closed: once a representative residual has a
        // target image, each byte successor has the matching target successor.
        for source_state in 0..state_count {
            let source_block = source_blocks[source_state] as usize;
            let Some(target_block) = source_block_to_target_block[source_block] else {
                continue;
            };
            let target_state = target_representative_state[target_block as usize];
            if self.terminal_output(source_state, source)
                != self.terminal_output(target_state, target)
            {
                return None;
            }
            for slot in 0..self.bytes.len() {
                let source_next = self.successor(source_state, slot);
                let target_next = self.successor(target_state, slot);
                let source_next_block = source_blocks[source_next] as usize;
                if source_block_to_target_block[source_next_block]
                    != Some(target_blocks[target_next])
                {
                    return None;
                }
            }
        }

        // Select a concrete source scanner state for each target partition. A
        // target partition whose only preimage is the artificial dead state has
        // no possible member match; represent that with `None` and omit that
        // mode/root in the NWA builder.
        let mut source_representative_for_target_block = vec![None::<u32>; target_count];
        for source_state in 0..self.real_state_count {
            let source_block = source_blocks[source_state] as usize;
            if let Some(target_block) = source_block_to_target_block[source_block] {
                source_representative_for_target_block[target_block as usize]
                    .get_or_insert(source_state as u32);
            }
        }

        // Surjectivity is required at the residual-partition level, including
        // the artificial dead partition. The combined construction guarantees
        // a source preimage exactly when the target language is representable.
        for target_block in 0..target_count {
            let has_source_preimage = source_block_to_target_block
                .iter()
                .any(|&mapped| mapped == Some(target_block as u32));
            if !has_source_preimage {
                return None;
            }
        }

        let scanner_state_for_target_state = (0..self.real_state_count)
            .map(|target_state| {
                let target_block = target_blocks[target_state] as usize;
                source_representative_for_target_block[target_block]
            })
            .collect::<Vec<_>>();

        Some(SubsumptionMap {
            source_block_to_target_block,
            scanner_state_for_target_state,
        })
    }
}

fn classify_keys<K: Ord>(keys: impl IntoIterator<Item = K>) -> Vec<u32> {
    let mut ids = BTreeMap::<K, u32>::new();
    keys.into_iter()
        .map(|key| {
            let next = ids.len() as u32;
            *ids.entry(key).or_insert(next)
        })
        .collect()
}

/// Planning data for exact directed terminal subsumption.
///
/// The historical name is kept because this remains the implementation behind
/// `GLRMASK_L2P_TERMINAL_INTERCHANGEABILITY`; a group may now be a directed
/// representative family rather than an equivalence class.
#[derive(Clone, Debug)]
pub(crate) struct TerminalInterchangeability {
    original_active: Vec<bool>,
    active_representatives: Vec<bool>,
    representative_for: Vec<TerminalID>,
    members_by_representative: Vec<Vec<TerminalID>>,
    maps_by_representative_member: BTreeMap<(TerminalID, TerminalID), SubsumptionMap>,
}

impl TerminalInterchangeability {
    pub(crate) fn identity(active: &[bool]) -> Self {
        let terminal_count = active.len();
        Self {
            original_active: active.to_vec(),
            active_representatives: active.to_vec(),
            representative_for: (0..terminal_count as u32).collect(),
            members_by_representative: (0..terminal_count as u32)
                .map(|terminal| vec![terminal])
                .collect(),
            maps_by_representative_member: BTreeMap::new(),
        }
    }

    pub(crate) fn build(
        tokenizer: &Tokenizer,
        active_terminals: &[bool],
        relevant_bytes: &[bool; 256],
        ignore_terminal: Option<TerminalID>,
    ) -> Self {
        let candidates = active_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &active)| active.then_some(terminal as TerminalID))
            .filter(|&terminal| Some(terminal) != ignore_terminal)
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            return Self::identity(active_terminals);
        }

        let restricted = RestrictedDfa::new(tokenizer, relevant_bytes);
        let mut accepted = BTreeMap::<(TerminalID, TerminalID), SubsumptionMap>::new();
        for &source in &candidates {
            for &target in &candidates {
                if source == target {
                    continue;
                }
                if let Some(map) = restricted.subsumption_map(source, target) {
                    accepted.insert((source, target), map);
                }
            }
        }

        let mut result = Self::identity(active_terminals);
        let mut remaining = candidates.iter().copied().collect::<BTreeSet<_>>();
        while let Some(&first_remaining) = remaining.iter().next() {
            let mut best_representative = first_remaining;
            let mut best_members = vec![first_remaining];
            for &representative in &remaining {
                let members = remaining
                    .iter()
                    .copied()
                    .filter(|&member| {
                        member == representative
                            || accepted.contains_key(&(representative, member))
                    })
                    .collect::<Vec<_>>();
                if members.len() > best_members.len()
                    || (members.len() == best_members.len()
                        && representative < best_representative)
                {
                    best_representative = representative;
                    best_members = members;
                }
            }

            for &member in &best_members {
                remaining.remove(&member);
            }
            if best_members.len() < 2 {
                continue;
            }

            result.members_by_representative[best_representative as usize] = best_members.clone();
            for &member in &best_members {
                result.representative_for[member as usize] = best_representative;
                if member != best_representative {
                    result.active_representatives[member as usize] = false;
                    let map = accepted
                        .get(&(best_representative, member))
                        .expect("selected terminal-subsumption member missing direct map")
                        .clone();
                    result
                        .maps_by_representative_member
                        .insert((best_representative, member), map);
                }
            }
        }

        if std::env::var_os("GLRMASK_DEBUG_TERMINAL_INTERCHANGEABILITY").is_some() {
            for (representative, members) in result.members_by_representative.iter().enumerate() {
                if members.len() < 2 {
                    continue;
                }
                eprintln!(
                    "[glrmask/debug][terminal_subsumption] representative={} members={:?}",
                    representative, members,
                );
                if std::env::var_os("GLRMASK_DEBUG_TERMINAL_INTERCHANGEABILITY_MAPS").is_some()
                {
                    for &member in members {
                        if member == representative as TerminalID {
                            continue;
                        }
                        let map = result
                            .maps_by_representative_member
                            .get(&(representative as TerminalID, member))
                            .expect("debug transport missing");
                        eprintln!(
                            "[glrmask/debug][terminal_subsumption_transport] representative={} member={} source_block_to_target_block={:?} scanner_state_for_target_state={:?}",
                            representative,
                            member,
                            map.source_block_to_target_block,
                            map.scanner_state_for_target_state,
                        );
                    }
                }
            }
        }
        result
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.representative_for
            .iter()
            .enumerate()
            .all(|(terminal, &representative)| terminal as TerminalID == representative)
    }

    pub(crate) fn active_representatives(&self) -> &[bool] {
        &self.active_representatives
    }

    pub(crate) fn active_terminal_count_before(&self) -> usize {
        self.original_active.iter().filter(|&&active| active).count()
    }

    pub(crate) fn active_terminal_count_after(&self) -> usize {
        self.active_representatives
            .iter()
            .filter(|&&active| active)
            .count()
    }

    /// Build an identity mode plus one independent representative-only mode for
    /// every hidden member. Member modes intentionally drop all other source
    /// labels, so their only contribution is the exact restored member terminal.
    pub(crate) fn terminal_nwa_transport_modes(&self) -> Option<Vec<TerminalNwaTransportMode>> {
        if self.is_identity() {
            return None;
        }
        let terminal_count = self.original_active.len();
        let state_count = self
            .maps_by_representative_member
            .values()
            .next()
            .map(|map| map.scanner_state_for_target_state.len())?;
        let identity_states = (0..state_count as u32).map(Some).collect::<Vec<_>>();
        let identity_labels = (0..terminal_count as u32).map(Some).collect::<Vec<_>>();
        let mut modes = vec![TerminalNwaTransportMode {
            scanner_state_for_original: identity_states,
            terminal_map: identity_labels,
        }];

        for (representative, members) in self.members_by_representative.iter().enumerate() {
            let representative = representative as TerminalID;
            for &member in members {
                if member == representative {
                    continue;
                }
                let map = self
                    .maps_by_representative_member
                    .get(&(representative, member))?;
                let mut terminal_map = vec![None; terminal_count];
                terminal_map[representative as usize] = Some(member);
                modes.push(TerminalNwaTransportMode {
                    scanner_state_for_original: map.scanner_state_for_target_state.clone(),
                    terminal_map,
                });
            }
        }
        Some(modes)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;

    fn tokenizer(expressions: Vec<Expr>) -> Tokenizer {
        build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        )
    }

    #[test]
    fn strict_interchangeability_is_two_way_subsumption() {
        let expressions = vec![
            Expr::Seq(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())),
                    min: 0,
                    max: None,
                },
            ]),
            Expr::Seq(vec![
                Expr::U8Seq(b"aaa".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"aaaa".to_vec())),
                    min: 0,
                    max: None,
                },
            ]),
        ];
        let tokenizer = tokenizer(expressions);
        let dfa = RestrictedDfa::new(&tokenizer, &[true; 256]);
        assert!(dfa.subsumption_map(0, 1).is_some());
        assert!(dfa.subsumption_map(1, 0).is_some());
        let plan = TerminalInterchangeability::build(&tokenizer, &[true, true], &[true; 256], None);
        assert_eq!(plan.active_terminal_count_before(), 2);
        assert_eq!(plan.active_terminal_count_after(), 1);
        assert!(plan.terminal_nwa_transport_modes().is_some());
    }

    #[test]
    fn directed_subsumption_allows_source_only_residual_partitions() {
        // Terminal 0 can match either `ab` or `xab`; after the source-only
        // `x` prefix, its residual language is exactly terminal 1's `ab`
        // language. The source initial residual is intentionally unmapped.
        let expressions = vec![
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"x".to_vec())),
                    min: 0,
                    max: Some(1),
                },
                Expr::U8Seq(b"ab".to_vec()),
            ]),
            Expr::U8Seq(b"ab".to_vec()),
        ];
        let tokenizer = tokenizer(expressions);
        let dfa = RestrictedDfa::new(&tokenizer, &[true; 256]);
        let forward = dfa
            .subsumption_map(0, 1)
            .expect("optional-prefix terminal must subsume its core terminal");
        assert!(
            forward
                .source_block_to_target_block
                .iter()
                .any(Option::is_none),
            "the representative needs a source-only residual partition",
        );
        assert!(dfa.subsumption_map(1, 0).is_none());

        let plan = TerminalInterchangeability::build(&tokenizer, &[true, true], &[true; 256], None);
        assert_eq!(plan.active_terminal_count_before(), 2);
        assert_eq!(plan.active_terminal_count_after(), 1);
        let modes = plan
            .terminal_nwa_transport_modes()
            .expect("directed mode expected");
        assert_eq!(modes.len(), 2);
        assert_eq!(modes[1].terminal_map[0], Some(1));
        assert_eq!(modes[1].terminal_map[1], None);
    }

    #[test]
    fn one_representative_can_cover_incomparable_members() {
        // Terminal 0 has two source-only entry branches. Its residual after
        // `x` is terminal 1, and its residual after `y` is terminal 2.
        // Terminals 1 and 2 are incomparable, but terminal 0 subsumes both.
        let tokenizer = tokenizer(vec![
            Expr::Choice(vec![
                Expr::Seq(vec![Expr::U8Seq(b"x".to_vec()), Expr::U8Seq(b"ab".to_vec())]),
                Expr::Seq(vec![Expr::U8Seq(b"y".to_vec()), Expr::U8Seq(b"ac".to_vec())]),
            ]),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ac".to_vec()),
        ]);
        let dfa = RestrictedDfa::new(&tokenizer, &[true; 256]);
        assert!(dfa.subsumption_map(0, 1).is_some());
        assert!(dfa.subsumption_map(0, 2).is_some());
        assert!(dfa.subsumption_map(1, 2).is_none());
        assert!(dfa.subsumption_map(2, 1).is_none());

        let plan = TerminalInterchangeability::build(
            &tokenizer,
            &[true, true, true],
            &[true; 256],
            None,
        );
        assert_eq!(plan.active_terminal_count_before(), 3);
        assert_eq!(plan.active_terminal_count_after(), 1);
        let modes = plan
            .terminal_nwa_transport_modes()
            .expect("star family requires member transports");
        assert_eq!(modes.len(), 3);
        assert_eq!(modes[1].terminal_map[0], Some(1));
        assert_eq!(modes[2].terminal_map[0], Some(2));
    }

    #[test]
    fn byte_preserving_subsumption_rejects_distinct_literal_bytes() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ]);
        let dfa = RestrictedDfa::new(&tokenizer, &[true; 256]);
        assert!(dfa.subsumption_map(0, 1).is_none());
        assert!(dfa.subsumption_map(1, 0).is_none());
        let plan = TerminalInterchangeability::build(&tokenizer, &[true, true], &[true; 256], None);
        assert!(plan.is_identity());
    }

    #[test]
    fn inactive_outputs_do_not_affect_one_terminal_subsumption() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
        ]);
        let dfa = RestrictedDfa::new(&tokenizer, &[true; 256]);
        assert!(dfa.subsumption_map(0, 2).is_some());
        assert!(dfa.subsumption_map(2, 0).is_some());
    }

    #[test]
    fn metadata_only_terminal_filter_preserves_state_ids_and_byte_transitions() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"aba".to_vec()),
        ]);
        let active = [true, false, true];
        let filtered = tokenizer.deactivate_terminals_without_minimizing(&active);
        assert_eq!(filtered.num_states(), tokenizer.num_states());
        for state in 0..tokenizer.num_states() {
            for byte in 0..=255u8 {
                assert_eq!(
                    filtered.get_transition(state, byte),
                    tokenizer.get_transition(state, byte),
                );
            }
            let expected_matches = tokenizer
                .matched_terminals_iter(state)
                .filter(|&terminal| active[terminal as usize])
                .collect::<Vec<_>>();
            assert_eq!(
                filtered.matched_terminals_iter(state).collect::<Vec<_>>(),
                expected_matches,
            );
            let expected_futures = tokenizer
                .possible_future_terminals_iter(state)
                .filter(|&terminal| active[terminal as usize])
                .collect::<Vec<_>>();
            assert_eq!(
                filtered
                    .possible_future_terminals_iter(state)
                    .collect::<Vec<_>>(),
                expected_futures,
            );
        }
    }

    #[test]
    fn restricted_byte_alphabet_omits_unlisted_transitions() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::Seq(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"z".to_vec())),
                    min: 0,
                    max: Some(1),
                },
            ]),
        ]);
        let after_a = tokenizer.get_transition(tokenizer.initial_state_id(), b'a');
        assert_ne!(tokenizer.get_transition(after_a, b'z'), NO_STATE);

        let mut only_a = [false; 256];
        only_a[b'a' as usize] = true;
        let restricted = RestrictedDfa::new(&tokenizer, &only_a);
        assert_eq!(restricted.bytes, vec![b'a']);
        assert_eq!(restricted.bytes.len(), 1);
        assert_ne!(
            restricted.successor(after_a as usize, 0),
            tokenizer.get_transition(after_a, b'z') as usize,
        );

        let unrestricted = RestrictedDfa::new(&tokenizer, &[true; 256]);
        assert_eq!(unrestricted.bytes.len(), 256);
        assert_eq!(
            unrestricted.successor(after_a as usize, b'z' as usize),
            tokenizer.get_transition(after_a, b'z') as usize,
        );
    }
}
