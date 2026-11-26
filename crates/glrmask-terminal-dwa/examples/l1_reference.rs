//! Direct scalar semantics oracle for the projected L1 compiler.

use std::sync::Arc;

use glrmask_glr::__private::glr::analysis::AnalyzedGrammar;
use glrmask_grammar::__private::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};
use glrmask_lexer::__private::automata::lexer::{Lexer, tokenizer::Tokenizer};
#[cfg(test)]
use glrmask_lexer::__private::automata::lexer::{ast::bytes, compile::build_regex};
use glrmask_terminal_dwa::__private::terminal_dwa::{
    l1::{build_flat_transition_table, build_l1_id_map_and_terminal_dwa},
    types::TerminalColoring,
};
use glrmask_vocab::Vocab;

fn grammar() -> AnalyzedGrammar {
    AnalyzedGrammar::from_grammar_def(&GrammarDef {
        rules: vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0)],
        }],
        start: 0,
        terminals: (0..2)
            .map(|id| Terminal::Literal {
                id,
                bytes: vec![b'a' + id as u8],
            })
            .collect(),
        ..GrammarDef::default()
    })
}

fn check(tokenizer: &Tokenizer, vocab: &Vocab, active: &[bool]) {
    let grammar = grammar();
    let coloring = TerminalColoring::identity(2);
    let flat = Arc::from(build_flat_transition_table(tokenizer));
    let result = build_l1_id_map_and_terminal_dwa(
        "reference",
        tokenizer,
        vocab,
        &coloring,
        false,
        None,
        &grammar,
        active,
        &flat,
        None,
        None,
        None,
        None,
        None,
    );

    // Compare raw state/token/terminal triples against a direct lexer walk.
    for raw_state in 0..tokenizer.num_states() {
        for (original_token, bytes) in vocab.iter() {
            let ends = tokenizer.execute_from_state_end_only(bytes, raw_state);
            for (terminal, &enabled) in active.iter().enumerate() {
                let expected = enabled
                    && ends.clone().into_iter().any(|end| {
                        tokenizer
                            .matched_terminals_iter(end)
                            .any(|id| id as usize == terminal)
                            || tokenizer
                                .tokens_accessible_from_state(end)
                                .iter()
                                .any(|id| id == terminal)
                    });
                let actual = result.as_ref().is_some_and(|artifact| {
                    let tsid =
                        artifact.id_map.tokenizer_states.original_to_internal[raw_state as usize];
                    let token = artifact
                        .id_map
                        .internal_token_for_original(original_token)
                        .expect("projected L1 omitted an original vocabulary token");
                    let weight = artifact.dwa.eval_word(&[terminal as i32]);
                    weight.is_full()
                        || weight.raw_range_values().any(|(range, tokens)| {
                            range.contains(&tsid) && tokens.iter().any(|id| id == token)
                        })
                });
                assert_eq!(
                    actual, expected,
                    "L1 mismatch at raw state {raw_state}, token {original_token}, terminal {terminal}"
                );
            }
        }
    }
}

fn main() {
    let tokenizer =
        glrmask_lexer::__private::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
    let vocab = Vocab::from_entries_unchecked(vec![
        (0, vec![]),
        (1, b"a".to_vec()),
        (2, b"a".to_vec()),
        (3, b"aa".to_vec()),
        (4, b"b".to_vec()),
    ]);
    check(&tokenizer, &vocab, &[true, true]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_active_sets(tokenizer: Tokenizer, vocab: Vocab) {
        for active in [[true, true], [true, false], [false, true]] {
            check(&tokenizer, &vocab, &active);
        }
    }

    #[test]
    fn epsilon_nfa_matches_scalar_semantics() {
        check_active_sets(
            glrmask_lexer::__private::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer(),
            Vocab::from_entries_unchecked(vec![(0, vec![]), (1, b"a".to_vec()), (2, b"a".to_vec()), (3, b"aa".to_vec()), (4, b"b".to_vec()), (5, b"x".to_vec())]),
        );
    }

    #[test]
    fn deterministic_matches_scalar_semantics() {
        let exprs = vec![bytes(b"a"), bytes(b"ab")];
        let tokenizer =
            build_regex(&exprs).into_tokenizer(2, Some(Arc::from(exprs.into_boxed_slice())));
        check_active_sets(
            tokenizer,
            Vocab::from_entries_unchecked(vec![
                (0, vec![]),
                (1, b"a".to_vec()),
                (2, b"ab".to_vec()),
                (3, b"aba".to_vec()),
                (4, b"x".to_vec()),
            ]),
        );
    }
}
