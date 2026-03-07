//! Terminal DWA construction.
//!
//! Cargo-check-only skeleton: signatures and module structure are preserved,
//! but implementation bodies are intentionally gutted.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeSet;

use crate::Vocab;
use crate::automata::lexer::tokenizer::TokenizerDfa;
use crate::automata::weighted::nwa::Nwa;
use crate::compiler::debug::TerminalDebug;
use crate::compiler::glr::analysis::GlrGrammar;
use crate::compiler::grammar::ast::TerminalId;
use crate::compiler::pipeline::vocab_pre::VocabPreprocessing;

#[derive(Debug, Clone)]
pub struct TerminalDwa {
    pub nwa: Nwa,
    pub tsid_roots: Vec<u32>,
    #[allow(dead_code)]
    pub non_greedy_terminals_by_tokenizer_state: Vec<BTreeSet<TerminalId>>,
    #[allow(dead_code)]
    pub possible_future_terminals_by_tokenizer_state: Vec<BTreeSet<TerminalId>>,
}

fn build_terminal_dwa_nwa(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    used_terminals: &BTreeSet<TerminalId>,
) -> TerminalDwa {
    unimplemented!()
}

fn compute_ever_allowed_follows(grammar: &GlrGrammar) -> Vec<Vec<TerminalId>> {
    unimplemented!()
}

fn compute_always_allowed_follows(grammar: &GlrGrammar) -> Vec<Vec<TerminalId>> {
    unimplemented!()
}

fn collapse_always_allowed(
    terminal_dwa: &mut TerminalDwa,
    grammar: &GlrGrammar,
) -> bool {
    unimplemented!()
}

fn prune_disallowed_follows(
    terminal_dwa: &mut TerminalDwa,
    grammar: &GlrGrammar,
) -> bool {
    unimplemented!()
}

fn build_terminal_dwa_impl(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    grammar: &GlrGrammar,
    used_terminals: &BTreeSet<TerminalId>,
    capture_debug: bool,
) -> (TerminalDwa, Option<TerminalDebug>) {
    unimplemented!()
}

pub(crate) fn build_terminal_dwa(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    grammar: &GlrGrammar,
    used_terminals: &BTreeSet<TerminalId>,
) -> TerminalDwa {
    unimplemented!()
}

pub(crate) fn build_terminal_dwa_with_debug(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    grammar: &GlrGrammar,
    used_terminals: &BTreeSet<TerminalId>,
) -> (TerminalDwa, TerminalDebug) {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;
    use crate::automata::regex::bytes;
    use crate::automata::lexer::tokenizer::TokenizerDfa;
    use crate::compiler::glr::analysis::GlrGrammar;
    use crate::compiler::grammar::ast::tests::simple_ab_grammar;
    use crate::compiler::pipeline::vocab_pre::VocabPreprocessing;

    #[test]
    fn test_build_terminal_dwa_collapses_always_allowed_follow_path() {
        let grammar = simple_ab_grammar();
        let glr_grammar = GlrGrammar::from_grammar_def(&grammar);
        let tokenizer = TokenizerDfa::from_grammar_def(&grammar);
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"ab".to_vec())], None);
        let vocab_pre = VocabPreprocessing::compute(&tokenizer, &vocab, None);

        let all_terminals: BTreeSet<TerminalId> = (0..glr_grammar.num_terminals).collect();
        let terminal_dwa = build_terminal_dwa(&tokenizer, &vocab, &vocab_pre, &glr_grammar, &all_terminals);
        let initial_tsid = vocab_pre.state_to_tsid[tokenizer.initial_state() as usize] as usize;
        let root = terminal_dwa.tsid_roots[initial_tsid];
        let a_targets = &terminal_dwa.nwa.states[root as usize].transitions[&0];
        assert!(!a_targets.is_empty());

        let mut combined_a = Weight::empty();
        for (_, weight) in a_targets {
            combined_a = combined_a.union(weight);
        }
        assert!(!combined_a.is_empty() || combined_a.is_full());

        for (dest, weight) in a_targets {
            let state = &terminal_dwa.nwa.states[*dest as usize];
            assert!(state.final_weight.is_some());
            assert!(!state.transitions.contains_key(&1));
            if !state.transitions.is_empty() {
                assert!(!weight.is_empty() || weight.is_full());
            }
        }
    }

    #[test]
    fn test_terminal_dwa_carries_tokenizer_greedy_metadata() {
        let grammar = simple_ab_grammar();
        let glr_grammar = GlrGrammar::from_grammar_def(&grammar);
        let tokenizer = TokenizerDfa::from_expr_groups(&[
            crate::automata::regex::ExprGroup {
                expr: bytes(b"a"),
                is_non_greedy: true,
            },
            crate::automata::regex::ExprGroup {
                expr: bytes(b"ab"),
                is_non_greedy: false,
            },
        ]);
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"ab".to_vec())], None);
        let vocab_pre = VocabPreprocessing::compute(&tokenizer, &vocab, None);

        let all_terminals: BTreeSet<TerminalId> = (0..glr_grammar.num_terminals).collect();
        let terminal_dwa = build_terminal_dwa(&tokenizer, &vocab, &vocab_pre, &glr_grammar, &all_terminals);
        let state_after_a = tokenizer.run(b"a") as usize;

        assert!(terminal_dwa.non_greedy_terminals_by_tokenizer_state[state_after_a].contains(&0));
        assert!(terminal_dwa.possible_future_terminals_by_tokenizer_state[state_after_a].contains(&1));
    }
}
