//! Direct dynamic mask generation.
//!
//! This implementation intentionally does not consult the parser DWA. It walks
//! the vocabulary byte trie while advancing the lexer and GLR parser directly.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use crate::automata::lexer::Lexer;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{advance_stacks, stack_may_advance_on, ParserGSS};
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use crate::grammar::flat::TerminalID;

use super::artifact::Constraint;
use super::state::ConstraintState;

#[derive(Clone)]
struct TraverseWork<'a> {
    node: &'a VocabPrefixTreeNode,
    tokenizer_state: u32,
    gss: ParserGSS,
    exclusion: Option<(u32, TerminalID)>,
}

#[inline]
fn set_mask_bit(buf: &mut [u32], token_id: u32) {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    if let Some(slot) = buf.get_mut(word) {
        *slot |= 1u32 << bit;
    }
}

fn update_eos_mask(state: &ConstraintState<'_>, buf: &mut [u32]) {
    let Some(token_id) = state.constraint.eos_token_id else {
        return;
    };
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    let Some(slot) = buf.get_mut(word) else {
        return;
    };
    *slot &= !(1u32 << bit);
    if state.is_complete() {
        *slot |= 1u32 << bit;
    }
}

/// Match a terminal at the current lexer state, dropping accumulator branches
/// which have explicitly forbidden that terminal. Once a terminal is selected,
/// its pending follow restrictions are consumed, just as in commit.
fn prune_for_terminal(
    gss: &ParserGSS,
    tokenizer_state: u32,
    terminal: TerminalID,
    execution_end_state: Option<u32>,
) -> ParserGSS {
    // This is the accumulator half of normal byte commit: reject branches that
    // have forbidden this terminal at the current lexer state, and carry a
    // surviving restriction forward to the lexer state at the end of the bytes
    // being tested. Restrictions for any other lexer state belong to other
    // tokenization branches and are intentionally not copied.
    gss.apply_and_prune_no_promote(|disallowed: &TerminalsDisallowed| {
        if disallowed
            .get(&tokenizer_state)
            .is_some_and(|set| set.contains(&terminal))
        {
            return None;
        }

        let mut remapped = BTreeMap::new();
        if let Some(end_state) = execution_end_state {
            if let Some(blocked) = disallowed.get(&tokenizer_state) {
                remapped.insert(end_state, blocked.clone());
            }
        }
        Some(TerminalsDisallowed(Arc::new(remapped)))
    })
}

/// Keep the branch that has not yet committed a terminal while the lexer
/// consumes the whole segment. This mirrors the accumulator remapping done by
/// the ordinary byte-commit path.
fn remap_continuation(
    constraint: &Constraint,
    gss: &ParserGSS,
    tokenizer_state: u32,
    end_state: u32,
    matched_terminals: &[TerminalID],
) -> ParserGSS {
    let actionable: Vec<TerminalID> = matched_terminals
        .iter()
        .copied()
        .filter(|&terminal| {
            Some(terminal) != constraint.ignore_terminal
                && stack_may_advance_on(&constraint.table, gss, terminal)
        })
        .collect();

    if actionable.is_empty()
        && gss.all_accs_satisfy(|disallowed: &TerminalsDisallowed| disallowed.is_empty())
    {
        return gss.clone();
    }

    gss.apply_and_prune_no_promote(|disallowed: &TerminalsDisallowed| {
        if let Some(blocked) = disallowed.get(&tokenizer_state) {
            if !actionable.is_empty() && actionable.iter().all(|terminal| blocked.contains(terminal)) {
                return None;
            }
        }

        let mut remapped = BTreeMap::new();
        if let Some(blocked) = disallowed.get(&tokenizer_state) {
            remapped.insert(end_state, blocked.clone());
        }
        Some(TerminalsDisallowed(Arc::new(remapped)))
    })
}

fn parser_child(
    constraint: &Constraint,
    gss: &ParserGSS,
    tokenizer_state: u32,
    terminal: TerminalID,
    execution_end_state: Option<u32>,
) -> Option<ParserGSS> {
    // Ignore terminals reset the lexer but deliberately leave the parser alone.
    if Some(terminal) == constraint.ignore_terminal {
        return Some(gss.clone());
    }

    // This is the same future-terminal restriction attached by the ordinary
    // commit path. It prevents a branch that matched `terminal` earlier in a
    // token from matching the same still-live terminal again at this segment's
    // end state.
    let future_disallow_state = execution_end_state.filter(|&end_state| {
        constraint
            .tokenizer
            .possible_future_terminals(end_state)
            .contains(terminal as usize)
    });
    let pruned = prune_for_terminal(
        gss,
        tokenizer_state,
        terminal,
        execution_end_state,
    );
    if pruned.is_empty() || !stack_may_advance_on(&constraint.table, &pruned, terminal) {
        return None;
    }

    let mut advanced = advance_stacks(&constraint.table, &pruned, terminal);
    if advanced.is_empty() {
        return None;
    }
    if let Some(end_state) = future_disallow_state {
        advanced = advanced.apply(|disallowed: &TerminalsDisallowed| {
            disallowed.with_insert(end_state, terminal)
        });
    }

    Some(advanced)
}

fn continuation_gss(
    constraint: &Constraint,
    gss: &ParserGSS,
    tokenizer_state: u32,
    end_state: u32,
    matched_terminals: &[TerminalID],
 ) -> Option<ParserGSS> {
    let remapped = remap_continuation(
        constraint,
        gss,
        tokenizer_state,
        end_state,
        matched_terminals,
    );
    if remapped.is_empty() {
        return None;
    }
    Some(remapped)
}

fn token_boundary_allowed(
    constraint: &Constraint,
    tokenizer_state: u32,
    gss: &ParserGSS,
) -> bool {
    constraint
        .tokenizer
        .tokens_accessible_from_state(tokenizer_state)
        .iter()
        .any(|terminal| {
            let terminal = terminal as TerminalID;
            let pruned = prune_for_terminal(gss, tokenizer_state, terminal, None);
            !pruned.is_empty()
                && (Some(terminal) == constraint.ignore_terminal
                    || stack_may_advance_on(&constraint.table, &pruned, terminal))
        })
}

fn excluded_by_first_byte(
    constraint: &Constraint,
    segment: &[u8],
    exclusion: Option<(u32, TerminalID)>,
) -> bool {
    let Some((tokenizer_state, terminal)) = exclusion else {
        return false;
    };
    debug_assert!(!segment.is_empty());
    constraint
        .tokenizer
        .execute_from_state_all_widths(&segment[..1], tokenizer_state)
        .matches
        .iter()
        .any(|matched| matched.id == terminal)
}

pub(crate) fn fill_mask_dynamic(state: &ConstraintState<'_>, buf: &mut [u32]) {
    let vocab = state.constraint.dynamic_mask_vocab.as_ref().expect(
        "dynamic mask generation is unavailable: the lexer persistence property does not hold",
    );

    buf.fill(0);
    let initial_tsid = state.constraint.tokenizer.initial_state();
    let mut traversal = Vec::<TraverseWork<'_>>::new();

    for (&tokenizer_state, gss) in &state.state {
        if gss.is_empty() {
            continue;
        }
        traversal.push(TraverseWork {
            node: &vocab.trie.root,
            tokenizer_state,
            gss: gss.clone(),
            exclusion: None,
        });
    }

    while let Some(current) = traversal.pop() {
        if current.node.has_token()
            && (current.tokenizer_state == initial_tsid
                || token_boundary_allowed(
                    state.constraint,
                    current.tokenizer_state,
                    &current.gss,
                ))
        {
            let canonical_token_id = current.node.token_id() as u32;
            let token_ids = vocab
                .token_ids
                .get(&canonical_token_id)
                .expect("dynamic vocabulary trie node lacks token ids");
            for &token_id in token_ids.iter() {
                set_mask_bit(buf, token_id);
            }
        }

        for (segment, child) in current.node.iter_children() {
            if excluded_by_first_byte(state.constraint, segment, current.exclusion) {
                continue;
            }

            let mut segment_queue = VecDeque::new();
            segment_queue.push_back((0usize, current.tokenizer_state, current.gss.clone()));

            while let Some((position, tokenizer_state, gss)) = segment_queue.pop_front() {
                let execution = state
                    .constraint
                    .tokenizer
                    .execute_from_state_all_widths(&segment[position..], tokenizer_state);
                let matched_terminals: Vec<TerminalID> =
                    execution.matches.iter().map(|matched| matched.id).collect();

                for matched in &execution.matches {
                    debug_assert!(matched.width > 0);
                    let Some(advanced_parser) = parser_child(
                        state.constraint,
                        &gss,
                        tokenizer_state,
                        matched.id,
                        execution.end_state,
                    ) else {
                        continue;
                    };

                    let next_position = position + matched.width;
                    if next_position == segment.len() {
                        traversal.push(TraverseWork {
                            node: child,
                            tokenizer_state: initial_tsid,
                            gss: advanced_parser,
                            exclusion: execution.end_state.map(|end_state| (end_state, matched.id)),
                        });
                    } else {
                        segment_queue.push_back((next_position, initial_tsid, advanced_parser));
                    }
                }

                if let Some(end_state) = execution.end_state {
                    if let Some(continuation) = continuation_gss(
                        state.constraint,
                        &gss,
                        tokenizer_state,
                        end_state,
                        &matched_terminals,
                    ) {
                        traversal.push(TraverseWork {
                            node: child,
                            tokenizer_state: end_state,
                            gss: continuation,
                            exclusion: None,
                        });
                    }
                }
            }
        }
    }

    update_eos_mask(state, buf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Constraint, Vocab};

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word).is_some_and(|word| word & (1u32 << bit) != 0)
    }

    fn direct_mask(state: &ConstraintState<'_>) -> Vec<u32> {
        let mut mask = vec![0u32; state.constraint.mask_len()];
        state.fill_mask_dynamic(&mut mask);
        mask
    }

    fn assert_dynamic_parity(state: &ConstraintState<'_>) {
        assert_eq!(state.mask(), direct_mask(state));
    }

    #[test]
    fn dynamic_mask_matches_normal_for_repeat_and_cross_terminal_tokens() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b"ab".to_vec()),
                (4, b"aab".to_vec()),
                (5, b"aaa".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(constraint.dynamic_mask_available());

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 4));

        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 2));

        state.commit_token(2).unwrap();
        assert!(state.is_finished());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_availability_and_trie_are_rebuilt_after_load() {
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec())],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert!(loaded.dynamic_mask_available());
        assert_dynamic_parity(&loaded.start());
    }

    #[test]
    fn dynamic_mask_keeps_duplicate_byte_token_aliases() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (7, b"a".to_vec()),
                (12, b"ab".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(constraint.dynamic_mask_available());

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        let mask = direct_mask(&state);
        assert!(token_allowed(&mask, 0));
        assert!(token_allowed(&mask, 7));
        assert!(token_allowed(&mask, 12));

        state.commit_token(7).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 1));
    }

    #[test]
    fn dynamic_mask_matches_normal_across_an_ignore_terminal() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b" b".to_vec()),
                (4, b"  b".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(constraint.dynamic_mask_available());

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));
        assert!(token_allowed(&state.mask(), 4));

        state.commit_token(4).unwrap();
        assert!(state.is_finished());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_matches_normal_at_every_reachable_small_state() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b"bb".to_vec()),
                (4, b"c".to_vec()),
                (5, b"ab".to_vec()),
                (6, b"ba".to_vec()),
                (7, b"a c".to_vec()),
                (8, b"b c".to_vec()),
                (9, b" aa".to_vec()),
                (10, b" bb".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+;
t A ::= 'a'+;
t B ::= 'b'+;
t C ::= 'c';
nt start ::= A B C | B A C | A C | B C;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(constraint.dynamic_mask_available());

        fn visit(state: ConstraintState<'_>, depth: usize) {
            assert_dynamic_parity(&state);
            if depth == 3 {
                return;
            }
            let mask = state.mask();
            for token_id in 0..11u32 {
                if !token_allowed(&mask, token_id) {
                    continue;
                }
                let mut next = state.clone();
                next.commit_token(token_id).unwrap();
                visit(next, depth + 1);
            }
        }

        visit(constraint.start(), 0);
    }

    #[test]
    fn dynamic_mask_matches_normal_when_one_repeated_terminal_crosses_tokens() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"aaa".to_vec()),
                (3, b"aaaa".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(constraint.dynamic_mask_available());

        fn visit(state: ConstraintState<'_>, depth: usize) {
            assert_dynamic_parity(&state);
            if depth == 3 {
                return;
            }
            let mask = state.mask();
            for token_id in 0..4u32 {
                if !token_allowed(&mask, token_id) {
                    continue;
                }
                let mut next = state.clone();
                next.commit_token(token_id).unwrap();
                visit(next, depth + 1);
            }
        }

        visit(constraint.start(), 0);
    }

    #[test]
    fn dynamic_mask_matches_normal_for_a_partial_json_string() {
        let vocab = Vocab::new(
            vec![
                (0, b"\"".to_vec()),
                (1, b"a".to_vec()),
                (2, b"b".to_vec()),
                (3, b"\\\"".to_vec()),
                (4, b"\"a".to_vec()),
                (5, b"a\"".to_vec()),
            ],
            None,
        );
        let constraint =
            Constraint::from_json_schema(r#"{"type":"string"}"#, &vocab).unwrap();
        assert!(constraint.dynamic_mask_available());

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_handles_split_json_number_phases() {
        let vocab = Vocab::new(
            vec![
                (0, b"-".to_vec()),
                (1, b"0".to_vec()),
                (2, b"1".to_vec()),
                (3, b"2".to_vec()),
                (4, b"3".to_vec()),
                (5, b".".to_vec()),
                (6, b"e".to_vec()),
                (7, b"+".to_vec()),
            ],
            None,
        );
        let constraint =
            Constraint::from_json_schema(r#"{"type":"number"}"#, &vocab).unwrap();
        assert!(constraint.dynamic_mask_available());

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        for bytes in [b"1".as_slice(), b".".as_slice(), b"2".as_slice(), b"e".as_slice(), b"-".as_slice(), b"3".as_slice()] {
            state.commit_bytes(bytes).unwrap();
            assert_dynamic_parity(&state);
        }
        assert!(state.is_complete());
    }

    #[test]
    fn dynamic_mask_is_unavailable_for_a_live_cross_terminal_prefix() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"abc".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'abc';
nt start ::= A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(!constraint.dynamic_mask_available());

        let state = constraint.start();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = direct_mask(&state);
        }))
        .is_err());
    }
}
