use std::collections::BTreeSet;

use crate::automata::lexer::tokenizer::Tokenizer;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RemainingMatchSignature {
    pub terminal_id: u32,
    pub width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenizerStateSignature {
    pub remaining_matches: Vec<RemainingMatchSignature>,
    pub terminal_set: Vec<u32>,
}

fn is_active_group(group_id: u32, active_groups: Option<&[bool]>) -> bool {
    active_groups.map_or(true, |groups| {
        groups.get(group_id as usize).copied().unwrap_or(false)
    })
}

pub(crate) fn tokenizer_state_signature_for_token(
    tokenizer: &Tokenizer,
    start_state: u32,
    token_bytes: &[u8],
    active_groups: Option<&[bool]>,
) -> TokenizerStateSignature {
    let exec = tokenizer.execute_from_state(token_bytes, start_state);
    let mut remaining_matches = Vec::new();
    let mut terminal_set = BTreeSet::new();

    if let Some(end_state) = exec.end_state {
        for terminal_id in tokenizer.possible_future_terminals_iter(end_state) {
            if is_active_group(terminal_id, active_groups) {
                terminal_set.insert(terminal_id);
            }
        }
    }

    for matched in exec.matches {
        if !is_active_group(matched.id, active_groups) {
            continue;
        }
        if matched.width == token_bytes.len() {
            terminal_set.insert(matched.id);
        } else {
            remaining_matches.push(RemainingMatchSignature {
                terminal_id: matched.id,
                width: matched.width,
            });
        }
    }

    remaining_matches.sort_unstable();

    TokenizerStateSignature {
        remaining_matches,
        terminal_set: terminal_set.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::ds::u8set::U8Set;

    use super::tokenizer_state_signature_for_token;

    #[test]
    fn exact_width_match_moves_into_terminal_set() {
        let exact_spaces = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::single(b' '))),
            min: 2,
            max: Some(2),
        };
        let quote = Expr::U8Seq(vec![b'"']);
        let regex = build_regex(&[exact_spaces.clone(), quote.clone()]);
        let tokenizer = Tokenizer {
            dfa: regex.dfa,
            num_terminals: 2,
            exprs: Some(Arc::from(vec![exact_spaces, quote].into_boxed_slice())),
        };

        let signature = tokenizer_state_signature_for_token(
            &tokenizer,
            tokenizer.initial_state(),
            b"  ",
            None,
        );

        assert!(signature.remaining_matches.is_empty());
        assert!(signature.terminal_set.contains(&0));
    }
}