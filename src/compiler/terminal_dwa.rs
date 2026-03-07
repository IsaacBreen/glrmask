//! Terminal DWA construction.
//!
//! Cargo-check-only skeleton: signatures and module structure are preserved,
//! but implementation bodies are intentionally gutted.
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

use std::collections::BTreeSet;

use crate::Vocab;
use crate::automata::weighted::nwa::Nwa;
use crate::compiler::debug::TerminalDebug;
use crate::compiler::glr::grammar::GlrGrammar;
use crate::compiler::grammar_def::TerminalId;
use crate::compiler::tokenizer_dfa::TokenizerDfa;
use crate::compiler::vocab_pre::VocabPreprocessing;

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
    unimplemented!("cargo-check-only stub")
}

fn compute_ever_allowed_follows(grammar: &GlrGrammar) -> Vec<Vec<TerminalId>> {
    unimplemented!("cargo-check-only stub")
}

fn compute_always_allowed_follows(grammar: &GlrGrammar) -> Vec<Vec<TerminalId>> {
    unimplemented!("cargo-check-only stub")
}

fn collapse_always_allowed(
    terminal_dwa: &mut TerminalDwa,
    grammar: &GlrGrammar,
) -> bool {
    unimplemented!("cargo-check-only stub")
}

fn prune_disallowed_follows(
    terminal_dwa: &mut TerminalDwa,
    grammar: &GlrGrammar,
) -> bool {
    unimplemented!("cargo-check-only stub")
}

fn build_terminal_dwa_impl(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    grammar: &GlrGrammar,
    used_terminals: &BTreeSet<TerminalId>,
    capture_debug: bool,
) -> (TerminalDwa, Option<TerminalDebug>) {
    unimplemented!("cargo-check-only stub")
}

pub(crate) fn build_terminal_dwa(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    grammar: &GlrGrammar,
    used_terminals: &BTreeSet<TerminalId>,
) -> TerminalDwa {
    unimplemented!("cargo-check-only stub")
}

pub(crate) fn build_terminal_dwa_with_debug(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    grammar: &GlrGrammar,
    used_terminals: &BTreeSet<TerminalId>,
) -> (TerminalDwa, TerminalDebug) {
    unimplemented!("cargo-check-only stub")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::regex::bytes;
    use crate::compiler::grammar_def::tests::simple_ab_grammar;
    use crate::compiler::glr::grammar::GlrGrammar;
    use crate::compiler::tokenizer_dfa::TokenizerDfa;
    use crate::compiler::vocab_pre::VocabPreprocessing;
    use crate::automata::weighted::weight::TokenSet;

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
        assert_eq!(combined_a.tokens_for_tsid(initial_tsid as u32), TokenSet::from_iter([0..=1u32]));

        for (dest, weight) in a_targets {
            let state = &terminal_dwa.nwa.states[*dest as usize];
            assert!(state.final_weight.is_some());
            assert!(!state.transitions.contains_key(&1));
            if !state.transitions.is_empty() {
                assert_eq!(weight.tokens_for_tsid(initial_tsid as u32), TokenSet::from_iter([1..=1u32]));
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
