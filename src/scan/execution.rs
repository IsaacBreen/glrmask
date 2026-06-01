//! Runtime lexer-scan execution primitives.
//!
//! Runtime commit uses these helpers to scan the bytes of the selected token
//! from a concrete lexer state.  They do not collect the global CanMatch table;
//! that is a compile-time job handled by `compile::scan_relation`.

use smallvec::SmallVec;

use crate::automata::lexer::tokenizer::{Tokenizer, TokenizerExecResult, TokenizerMatch};

/// Scan `bytes` from `start_state` using pre-flattened tokenizer transitions.
///
/// The returned `TokenizerExecResult` preserves the historical runtime format,
/// but the function lives in the shared scan namespace so the commit code does
/// not appear to own the mathematical definition of lexer scanning.
pub(crate) fn execute_tokenizer_from_state(
    tokenizer: &Tokenizer,
    fast_transitions: &[Box<[u32; 256]>],
    bytes: &[u8],
    start_state: u32,
) -> TokenizerExecResult {
    let mut tokenizer_state = start_state;
    let mut matches = SmallVec::<[(u32, usize, u32); 8]>::new();

    for (index, &byte) in bytes.iter().enumerate() {
        let next_state = fast_transitions
            .get(tokenizer_state as usize)
            .map_or(u32::MAX, |transitions| transitions[byte as usize]);
        if next_state == u32::MAX {
            return TokenizerExecResult {
                end_state: None,
                matches: materialize_matches(matches),
            };
        }

        tokenizer_state = next_state;
        let width = index + 1;
        for terminal in tokenizer.matched_terminals_iter(tokenizer_state) {
            if let Some((_, existing_width, existing_end_state)) =
                matches.iter_mut().find(|(id, _, _)| *id == terminal)
            {
                *existing_width = width;
                *existing_end_state = tokenizer_state;
            } else {
                matches.push((terminal, width, tokenizer_state));
            }
        }
    }

    TokenizerExecResult {
        end_state: Some(tokenizer_state),
        matches: materialize_matches(matches),
    }
}

fn materialize_matches(matches: SmallVec<[(u32, usize, u32); 8]>) -> Vec<TokenizerMatch> {
    matches
        .into_iter()
        .map(|(id, width, end_state)| TokenizerMatch { id, width, end_state })
        .collect()
}
