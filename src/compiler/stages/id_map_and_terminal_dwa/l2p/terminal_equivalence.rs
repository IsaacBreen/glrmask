//! Exact terminal equivalence classes for one L2P vocabulary partition.
//!
//! A terminal DWA is indexed by one shared tokenizer-state map at runtime.
//! Consequently, the quotient used here is deliberately state preserving:
//! two terminals are equivalent when, at every tokenizer state, they have the
//! same final and possible-future bits.  Because all terminals share the same byte
//! transition graph, those two columns determine every future match event on
//! every byte path.  This is the coarsest exact terminal quotient compatible
//! with the existing single-TSID runtime representation.
//!
//! The quotient is partition-local because only active L2P terminals enter it.
//! Non-representatives are hidden from the state/vocab-equivalence pass and
//! from terminal-NWA construction, then restored on the completed NWA before
//! grammar-specific follow processing.

use std::collections::BTreeMap;

use rustc_hash::FxHashMap;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::nwa::NWA;
use crate::grammar::flat::TerminalID;

#[derive(Clone, Debug, Default)]
pub(crate) struct TerminalEquivalence {
    representative_for_terminal: Vec<TerminalID>,
    members_by_representative: Vec<Vec<TerminalID>>,
    active_representatives: Vec<bool>,
    active_terminal_count: usize,
    class_count: usize,
    quotient_hits: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TerminalEquivalenceProfile {
    pub(crate) active_terminals: usize,
    pub(crate) classes: usize,
    pub(crate) quotient_hits: usize,
    pub(crate) expanded_transition_copies: usize,
}

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
struct TerminalObservationSignature {
    final_states: Vec<u32>,
    future_states: Vec<u32>,
}

impl TerminalEquivalence {
    pub(crate) fn identity(active_terminals: &[bool]) -> Self {
        let num_terminals = active_terminals.len();
        let representative_for_terminal = (0..num_terminals as u32).collect::<Vec<_>>();
        let members_by_representative = (0..num_terminals as u32)
            .map(|terminal| vec![terminal])
            .collect();
        Self {
            representative_for_terminal,
            members_by_representative,
            active_representatives: active_terminals.to_vec(),
            active_terminal_count: active_terminals.iter().filter(|&&active| active).count(),
            class_count: active_terminals.iter().filter(|&&active| active).count(),
            quotient_hits: 0,
        }
    }

    pub(crate) fn build(
        tokenizer: &Tokenizer,
        active_terminals: &[bool],
        ignore_terminal: Option<TerminalID>,
    ) -> Self {
        let num_terminals = tokenizer.num_terminals() as usize;
        assert_eq!(
            active_terminals.len(),
            num_terminals,
            "L2P terminal-equivalence mask must cover every tokenizer terminal"
        );

        let active_ids: Vec<TerminalID> = (0..num_terminals)
            .filter(|&terminal| active_terminals[terminal])
            .map(|terminal| terminal as TerminalID)
            .collect();
        if active_ids.len() < 2 {
            return Self::identity(&active_terminals[..num_terminals]);
        }

        let mut signatures = vec![
            TerminalObservationSignature {
                final_states: Vec::new(),
                future_states: Vec::new(),
            };
            num_terminals
        ];
        let mut live_counts = vec![0usize; num_terminals];
        let terminal_is_quotient_active = |terminal: TerminalID| {
            active_terminals.get(terminal as usize).copied().unwrap_or(false)
                && Some(terminal) != ignore_terminal
        };

        for state in 0..tokenizer.num_states() {
            for terminal in tokenizer.matched_terminals_iter(state) {
                if terminal_is_quotient_active(terminal) {
                    signatures[terminal as usize].final_states.push(state);
                }
            }
            for terminal in tokenizer.possible_future_terminals_iter(state) {
                if terminal_is_quotient_active(terminal) {
                    signatures[terminal as usize].future_states.push(state);
                    live_counts[terminal as usize] += 1;
                }
            }
        }

        let mut representative_for_terminal =
            (0..num_terminals as u32).collect::<Vec<TerminalID>>();
        let mut members_by_representative = (0..num_terminals as u32)
            .map(|terminal| vec![terminal])
            .collect::<Vec<_>>();
        let mut active_representatives = vec![false; num_terminals];
        let mut groups = FxHashMap::<TerminalObservationSignature, Vec<TerminalID>>::default();

        for &terminal in &active_ids {
            if Some(terminal) == ignore_terminal {
                active_representatives[terminal as usize] = true;
                continue;
            }
            let signature = std::mem::take(&mut signatures[terminal as usize]);
            groups.entry(signature).or_default().push(terminal);
        }

        let mut class_count = 0usize;
        let mut quotient_hits = 0usize;
        for mut members in groups.into_values() {
            members.sort_unstable();
            let representative = *members
                .iter()
                .min_by_key(|&&terminal| (live_counts[terminal as usize], terminal))
                .expect("terminal equivalence class must be non-empty");
            quotient_hits += members.len().saturating_sub(1);
            class_count += 1;
            for &terminal in &members {
                representative_for_terminal[terminal as usize] = representative;
            }
            members_by_representative[representative as usize] = members;
            active_representatives[representative as usize] = true;
        }

        if let Some(ignore_terminal) = ignore_terminal {
            if active_terminals.get(ignore_terminal as usize).copied().unwrap_or(false) {
                class_count += 1;
            }
        }

        Self {
            representative_for_terminal,
            members_by_representative,
            active_representatives,
            active_terminal_count: active_ids.len(),
            class_count,
            quotient_hits,
        }
    }

    pub(crate) fn representative_active_terminals(&self) -> &[bool] {
        &self.active_representatives
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.quotient_hits == 0
    }

    pub(crate) fn profile(&self) -> TerminalEquivalenceProfile {
        TerminalEquivalenceProfile {
            active_terminals: self.active_terminal_count,
            classes: self.class_count,
            quotient_hits: self.quotient_hits,
            expanded_transition_copies: 0,
        }
    }

    pub(crate) fn expand_nwa(&self, nwa: &mut NWA) -> TerminalEquivalenceProfile {
        let mut profile = TerminalEquivalenceProfile {
            active_terminals: self.active_terminal_count,
            classes: self.class_count,
            quotient_hits: self.quotient_hits,
            ..TerminalEquivalenceProfile::default()
        };
        if self.is_identity() {
            return profile;
        }

        for state in nwa.states_mut() {
            let original = std::mem::take(&mut state.transitions);
            let mut expanded = BTreeMap::new();
            for (label, targets) in original {
                let Some(terminal) = u32::try_from(label).ok() else {
                    expanded.insert(label, targets);
                    continue;
                };
                let representative = self
                    .representative_for_terminal
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(terminal);
                if representative != terminal {
                    expanded.insert(label, targets);
                    continue;
                }

                let members = self
                    .members_by_representative
                    .get(representative as usize)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                if members.len() <= 1 {
                    expanded.insert(label, targets);
                    continue;
                }

                for &member in members {
                    if member != representative {
                        profile.expanded_transition_copies += targets.len();
                    }
                    let entry = expanded.entry(member as i32).or_insert_with(Vec::new);
                    for (target, weight) in &targets {
                        if let Some((_, existing)) = entry.iter_mut().find(|(existing_target, _)| *existing_target == *target) {
                            *existing = existing.union(weight);
                        } else {
                            entry.push((*target, weight.clone()));
                        }
                    }
                }
            }
            state.transitions = expanded;
        }

        profile
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;

    #[test]
    fn identity_preserves_the_active_mask() {
        let equivalence = TerminalEquivalence::identity(&[true, false, true]);
        assert_eq!(equivalence.representative_active_terminals(), &[true, false, true]);
        assert!(equivalence.is_identity());
    }

    #[test]
    fn duplicate_terminal_columns_share_one_representative() {
        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        let equivalence = TerminalEquivalence::build(&tokenizer, &[true, true, true], None);

        assert_eq!(equivalence.representative_for_terminal[0], 0);
        assert_eq!(equivalence.representative_for_terminal[1], 0);
        assert_eq!(equivalence.members_by_representative[0], vec![0, 1]);
        assert_eq!(equivalence.representative_active_terminals(), &[true, false, true]);
        assert_eq!(equivalence.profile().quotient_hits, 1);
    }

    #[test]
    fn ignored_terminal_is_never_merged() {
        let expressions = vec![Expr::U8Seq(b"ab".to_vec()), Expr::U8Seq(b"ab".to_vec())];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        let equivalence = TerminalEquivalence::build(&tokenizer, &[true, true], Some(1));

        assert_eq!(equivalence.representative_active_terminals(), &[true, true]);
        assert!(equivalence.is_identity());
    }

    #[test]
    fn expansion_restores_each_hidden_terminal_label_with_the_same_weight() {
        let equivalence = TerminalEquivalence {
            representative_for_terminal: vec![0, 0],
            members_by_representative: vec![vec![0, 1], vec![1]],
            active_representatives: vec![true, false],
            active_terminal_count: 2,
            class_count: 1,
            quotient_hits: 1,
        };
        let mut nwa = NWA::new(1, 0);
        let source = nwa.add_state();
        let target = nwa.add_state();
        nwa.add_transition(source, 0, target, crate::ds::weight::Weight::all());

        let profile = equivalence.expand_nwa(&mut nwa);

        assert_eq!(profile.expanded_transition_copies, 1);
        assert_eq!(
            nwa.states()[source as usize].transitions.get(&0),
            nwa.states()[source as usize].transitions.get(&1)
        );
    }
}
