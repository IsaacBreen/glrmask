//! Direct dynamic mask generation.
//!
//! This implementation intentionally does not consult the parser DWA. It walks
//! the vocabulary byte trie while advancing the lexer and GLR parser directly.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use crate::automata::lexer::Lexer;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{advance_stacks, stack_may_advance_on, ParserGSS};
use crate::ds::leveled_gss::LeveledGSS;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use crate::grammar::flat::TerminalID;

use super::artifact::{Constraint, DynamicMaskVocab};
use super::state::ConstraintState;

type ExclusionMap = BTreeMap<u32, BTreeSet<TerminalID>>;
type Exclusions = Arc<ExclusionMap>;
type ParserStacks = LeveledGSS<u32, ()>;

#[derive(Clone)]
struct TraverseWork<'a> {
    node: &'a VocabPrefixTreeNode,
    tokenizer_state: u32,
    gss: ParserStacks,
    exclusions: Exclusions,
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

/// Dynamic masking keeps terminal restrictions in `Exclusions`. The parser
/// table routines still use `ParserGSS`, so give their stack operations an
/// otherwise-unused empty accumulator.
fn with_empty_accumulators(stacks: &ParserStacks) -> ParserGSS {
    stacks.apply(|_| TerminalsDisallowed::new())
}

/// Advance every outstanding exclusion through one compressed vocabulary-trie
/// edge. If any excluded terminal matches anywhere on the edge, this traversal
/// branch would duplicate that terminal and is rejected. Otherwise each entry
/// follows its lexer state and keeps only terminals still accessible there.
fn advance_exclusions(
    constraint: &Constraint,
    segment: &[u8],
    exclusions: &Exclusions,
) -> Option<Exclusions> {
    if exclusions.is_empty() {
        return Some(exclusions.clone());
    }

    let mut advanced = ExclusionMap::new();
    for (&tokenizer_state, blocked) in exclusions.iter() {
        let execution = constraint
            .tokenizer
            .execute_from_state_all_widths(segment, tokenizer_state);
        if execution
            .matches
            .iter()
            .any(|matched| blocked.contains(&matched.id))
        {
            return None;
        }

        let Some(end_state) = execution.end_state else {
            continue;
        };
        let accessible = constraint.tokenizer.tokens_accessible_from_state(end_state);
        let next_blocked = advanced.entry(end_state).or_default();
        next_blocked.extend(
            blocked
                .iter()
                .copied()
                .filter(|terminal| accessible.contains(*terminal as usize)),
        );
    }
    Some(Arc::new(advanced))
}

/// Record that a terminal committed at this token boundary cannot be matched
/// again by the parallel lexer continuation carried in `exclusions`.
fn with_excluded_terminal(
    exclusions: &Exclusions,
    tokenizer_state: u32,
    terminal: TerminalID,
) -> Exclusions {
    let mut next = (**exclusions).clone();
    next.entry(tokenizer_state).or_default().insert(terminal);
    Arc::new(next)
}

fn parser_child(
    constraint: &Constraint,
    stacks: &ParserStacks,
    terminal: TerminalID,
) -> Option<ParserStacks> {
    // Ignore terminals reset the lexer but deliberately leave the parser alone.
    if Some(terminal) == constraint.ignore_terminal {
        return Some(stacks.clone());
    }
    let parser_gss = with_empty_accumulators(stacks);
    if !stack_may_advance_on(&constraint.table, &parser_gss, terminal) {
        return None;
    }
    let advanced = advance_stacks(&constraint.table, &parser_gss, terminal).apply(|_| ());
    (!advanced.is_empty()).then_some(advanced)
}

fn token_boundary_allowed(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
) -> bool {
    let parser_gss = with_empty_accumulators(stacks);
    constraint
        .tokenizer
        .tokens_accessible_from_state(tokenizer_state)
        .iter()
        .any(|terminal| {
            let terminal = terminal as TerminalID;
            Some(terminal) == constraint.ignore_terminal
                || stack_may_advance_on(&constraint.table, &parser_gss, terminal)
        })
}

fn mark_reachable_tokens(
    vocab: &DynamicMaskVocab,
    node: &VocabPrefixTreeNode,
    buf: &mut [u32],
) {
    for canonical_token_id in node.reachable_token_ids().iter() {
        let token_ids = vocab
            .token_ids
            .get(&(canonical_token_id as u32))
            .expect("dynamic vocabulary trie node lacks token ids");
        for &token_id in token_ids.iter() {
            set_mask_bit(buf, token_id);
        }
    }
}

/// The terminal-wise loop map is a quotient loop, not necessarily a raw DFA
/// loop.  If every possible remaining byte is a loop for every terminal that
/// the parser can currently admit, then the no-finalization continuation has
/// exactly the same terminal language at every descendant vocabulary node.
/// Consequently every token below `node` is admissible through that
/// continuation and no lexer/parser walk is needed.
///
/// `Exclusions` carry parallel lexer continuations that reject a branch if one
/// of their blocked terminals matches.  They may only be skipped when each
/// blocked terminal has the same quotient loop property.  A currently accepting
/// blocked terminal can only reject descendants after one more byte, not the
/// token ending at the current node; a nonaccepting one cannot newly match
/// anywhere in the subtree.
enum TerminalLoopSubtree {
    CannotSkip,
    MarkAllTokens,
    MarkCurrentNodeOnly,
}

fn terminal_loop_subtree(
    constraint: &Constraint,
    vocab: &DynamicMaskVocab,
    node: &VocabPrefixTreeNode,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    exclusions: &Exclusions,
) -> TerminalLoopSubtree {
    let future_terminals = constraint.tokenizer.tokens_accessible_from_state(tokenizer_state);
    if future_terminals.is_empty() {
        return TerminalLoopSubtree::CannotSkip;
    }

    let subtree_bytes = U8Set::from_words(*node.subtree_bytes());
    let loops_by_state = vocab.terminal_self_loop_bytes(&constraint.tokenizer);
    let Some(loops) = loops_by_state.get(tokenizer_state as usize) else {
        return TerminalLoopSubtree::CannotSkip;
    };

    // `advance_row_allows` is the table's cheap, conservative admission set.
    // It may include a guarded shift whose lower-stack guard later fails, but
    // that only makes this optimization decline to fire.  The exact
    // `token_boundary_allowed` check below establishes the one live parser
    // continuation required before we mark a whole subtree.
    let top_states = stacks.peek_values();
    let mut has_candidate_terminal = false;
    for terminal in future_terminals.iter() {
        let terminal = terminal as TerminalID;
        if Some(terminal) != constraint.ignore_terminal
            && !top_states
                .iter()
                .any(|&parser_state| constraint.table.advance_row_allows(parser_state, terminal))
        {
            continue;
        }
        has_candidate_terminal = true;
        let Some(loop_bytes) = loops.get(terminal as usize) else {
            return TerminalLoopSubtree::CannotSkip;
        };
        if !subtree_bytes.is_subset(loop_bytes) {
            return TerminalLoopSubtree::CannotSkip;
        }
    }
    if !has_candidate_terminal || !token_boundary_allowed(constraint, tokenizer_state, stacks) {
        return TerminalLoopSubtree::CannotSkip;
    }

    for (&excluded_state, blocked_terminals) in exclusions.iter() {
        let Some(exclusion_loops) = loops_by_state.get(excluded_state as usize) else {
            return TerminalLoopSubtree::CannotSkip;
        };
        for &terminal in blocked_terminals {
            let Some(loop_bytes) = exclusion_loops.get(terminal as usize) else {
                return TerminalLoopSubtree::CannotSkip;
            };
            if !subtree_bytes.is_subset(loop_bytes) {
                return TerminalLoopSubtree::CannotSkip;
            }
            if constraint
                .tokenizer
                .matched_terminal_bitset(excluded_state)
                .contains(terminal as usize)
            {
                return TerminalLoopSubtree::MarkCurrentNodeOnly;
            }
        }
    }

    TerminalLoopSubtree::MarkAllTokens
}

pub(crate) fn fill_mask_dynamic(state: &ConstraintState<'_>, buf: &mut [u32]) {
    let vocab = &state.constraint.dynamic_mask_vocab;

    buf.fill(0);
    let initial_tsid = state.constraint.tokenizer.initial_state();
    let mut traversal = Vec::<TraverseWork<'_>>::new();

    for (&tokenizer_state, gss) in &state.state {
        for (stacks, exclusions) in gss.partition_by_accumulator() {
            traversal.push(TraverseWork {
                node: &vocab.trie.root,
                tokenizer_state,
                gss: stacks,
                exclusions: exclusions.0,
            });
        }
    }

    while let Some(current) = traversal.pop() {
        let subtree_action = terminal_loop_subtree(
            state.constraint,
            vocab,
            current.node,
            current.tokenizer_state,
            &current.gss,
            &current.exclusions,
        );
        if matches!(subtree_action, TerminalLoopSubtree::MarkAllTokens) {
            mark_reachable_tokens(vocab, current.node, buf);
            continue;
        }

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

        if matches!(subtree_action, TerminalLoopSubtree::MarkCurrentNodeOnly) {
            continue;
        }

        for (segment, child) in current.node.iter_children() {
            let Some(segment_exclusions) =
                advance_exclusions(state.constraint, segment, &current.exclusions)
            else {
                continue;
            };

            let mut segment_queue = VecDeque::new();
            segment_queue.push_back((0usize, current.tokenizer_state, current.gss.clone()));

            while let Some((position, tokenizer_state, gss)) = segment_queue.pop_front() {
                let execution = state
                    .constraint
                    .tokenizer
                    .execute_from_state_all_widths(&segment[position..], tokenizer_state);

                for matched in &execution.matches {
                    debug_assert!(matched.width > 0);
                    let Some(advanced_parser) = parser_child(state.constraint, &gss, matched.id)
                    else {
                        continue;
                    };

                    let next_position = position + matched.width;
                    if next_position == segment.len() {
                        let exclusions = execution.end_state.map_or_else(
                            || segment_exclusions.clone(),
                            |end_state| {
                                with_excluded_terminal(&segment_exclusions, end_state, matched.id)
                            },
                        );
                        traversal.push(TraverseWork {
                            node: child,
                            tokenizer_state: initial_tsid,
                            gss: advanced_parser,
                            exclusions,
                        });
                    } else {
                        segment_queue.push_back((next_position, initial_tsid, advanced_parser));
                    }
                }

                if let Some(end_state) = execution.end_state {
                    traversal.push(TraverseWork {
                        node: child,
                        tokenizer_state: end_state,
                        gss,
                        exclusions: segment_exclusions.clone(),
                    });
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

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));

        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 2));

        state.commit_token(2).unwrap();
        assert!(state.is_finished());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_trie_is_rebuilt_after_load() {
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

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));

        state.commit_token(3).unwrap();
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
    fn dynamic_mask_handles_monolithic_json_number() {
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

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        for bytes in [b"1".as_slice(), b".".as_slice(), b"2".as_slice(), b"e".as_slice(), b"-".as_slice(), b"3".as_slice()] {
            state.commit_bytes(bytes).unwrap();
            assert_dynamic_parity(&state);
        }
        assert!(state.is_complete());
    }

    #[test]
    fn dynamic_mask_keeps_other_gss_paths_when_one_path_excludes_a_terminal() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"ab".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'ab';
t B ::= 'a';
t C ::= 'c';
t D ::= 'b';
nt start ::= A C | B D;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_token(0).unwrap();

        let paths = state
            .state
            .values()
            .flat_map(|gss| gss.to_stacks())
            .collect::<Vec<_>>();
        assert!(paths.iter().any(|(_, exclusions)| exclusions.is_empty()));
        assert!(paths.iter().any(|(_, exclusions)| !exclusions.is_empty()));

        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 1));
        state.commit_token(1).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_handles_overlapping_live_terminal_paths() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"b".to_vec()),
                (3, b"bc".to_vec()),
                (4, b"c".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'ab';
t B ::= 'b' | 'bc';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));
        state.commit_token(3).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_handles_a_live_cross_terminal_prefix() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"abc".to_vec()),
                (3, b"bc".to_vec()),
                (4, b"c".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'abc';
nt start ::= A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));
        assert!(!token_allowed(&state.mask(), 4));
        state.commit_token(3).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn terminal_loop_skip_handles_an_already_accepting_exclusion() {
        use std::collections::{BTreeMap, BTreeSet};

        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"aa".to_vec()), (2, b"aaa".to_vec())],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let state = constraint.start();
        let initial = constraint.tokenizer.initial_state();
        let (stacks, _) = state
            .state
            .get(&initial)
            .unwrap()
            .partition_by_accumulator()
            .into_iter()
            .next()
            .unwrap();
        let continuation = constraint
            .tokenizer
            .execute_from_state_all_widths(b"a", initial)
            .end_state
            .unwrap();
        let (_, node) = constraint
            .dynamic_mask_vocab
            .trie
            .root
            .iter_children()
            .next()
            .unwrap();

        let empty = Arc::new(BTreeMap::new());
        assert!(matches!(
            terminal_loop_subtree(
                &constraint,
                &constraint.dynamic_mask_vocab,
                node,
                continuation,
                &stacks,
                &empty,
            ),
            TerminalLoopSubtree::MarkAllTokens,
        ));

        let exclusions = Arc::new(BTreeMap::from([(
            continuation,
            BTreeSet::from([0]),
        )]));
        assert!(matches!(
            terminal_loop_subtree(
                &constraint,
                &constraint.dynamic_mask_vocab,
                node,
                continuation,
                &stacks,
                &exclusions,
            ),
            TerminalLoopSubtree::MarkCurrentNodeOnly,
        ));
    }

}
