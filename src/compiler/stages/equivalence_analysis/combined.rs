
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This corresponds most closely to sep1's `combined_equivalence_analysis.rs`, but only returns the compact internal-ID maps glrmask's compiler consumes.

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::equivalence_analysis::state_analysis::analyze_state_equivalences;
use crate::compiler::stages::equivalence_analysis::vocab_analysis::analyze_vocab_equivalences;


pub(crate) fn analyze_equivalences(tokenizer: &Tokenizer, vocab: &Vocab) -> InternalIdMap {
    InternalIdMap {
        tokenizer_states: analyze_state_equivalences(tokenizer),
        vocab_tokens: analyze_vocab_equivalences(vocab),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};

    #[test]
    fn test_internal_id_map_shape() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal {
                id: 0,
                name: "a".into(),
                pattern: "a".into(),
            }],
        };
        let tok = Tokenizer::from_grammar_def(&gdef);
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"a".to_vec()),
                (2, b"b".to_vec()),
            ],
            None,
        );
        let id_map = analyze_equivalences(&tok, &vocab);

        assert!(id_map.num_tsids() >= 1);
        assert_eq!(id_map.max_token_id(), 2);
    }
}
