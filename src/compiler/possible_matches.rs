#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: this helper is the closest glrmask analogue to the compiler-side
// construction of sep1's `possible_matches` data that later feeds runtime mask
// filtering.

use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::grammar::model::{GrammarDef, TerminalID};

pub(crate) type PossibleMatchesByState = BTreeMap<u32, BTreeMap<TerminalID, RangeSetBlaze<u32>>>;

pub(crate) fn build_possible_matches_by_state(
    grammar: &GrammarDef,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> PossibleMatchesByState {
    let mut possible_matches_by_state = BTreeMap::new();
    for tokenizer_state in 0..tokenizer.num_states() {
        let mut terminal_to_tokens = BTreeMap::new();
        for terminal in &grammar.terminals {
            let token_ids = vocab
                .entries
                .iter()
                .filter_map(|(token_id, bytes)| {
                    let exec = tokenizer.execute_from_state(bytes, tokenizer_state);
                    (exec.end_state.is_some()
                        && exec
                            .matches
                            .iter()
                            .any(|matched| matched.id == terminal.id && matched.width == bytes.len()))
                        .then_some(*token_id)
                })
                .collect::<RangeSetBlaze<u32>>();
            if !token_ids.is_empty() {
                terminal_to_tokens.insert(terminal.id, token_ids);
            }
        }
        possible_matches_by_state.insert(tokenizer_state, terminal_to_tokens);
    }
    possible_matches_by_state
}
