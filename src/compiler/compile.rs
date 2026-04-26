#[cfg(test)]
use crate::compiler::grammar::transforms::prepare_grammar_for_compile;
#[cfg(test)]
use crate::Vocab;
#[cfg(test)]
use crate::grammar::flat::GrammarDef;
#[cfg(test)]
use crate::runtime::Constraint;
pub(crate) use super::pipeline::{
    build_tokenizer,
    compile_owned,
    compile_owned_profiled,
    compile_profile_enabled,
    compute_disallowed_follows,
    emit_compile_profile_summary,
};

#[cfg(test)]
pub(crate) use super::pipeline::build_tokenizer_from_exprs;

#[cfg(test)]
pub(crate) fn compile(grammar: &GrammarDef, vocab: &Vocab) -> Constraint {
    let (prepared_grammar, _tokenizer) = prepare_grammar_for_compile(grammar);
    super::pipeline::compile_prepared(prepared_grammar, vocab)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::regex::Expr;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::parser::{ParserGSS, advance_stacks};
    use crate::compiler::glr::table::GLRTable;
    use crate::grammar::flat::tests::*;
    use crate::grammar::flat::{NonterminalID, Rule, Symbol, Terminal};
    use crate::compiler::grammar::transforms::{
        compact_unused_terminals,
        expand_nullable_terminals,
        inline_single_use_nonterminals,
        prepare_grammar_for_compile,
        prepare_owned_grammar_for_compile,
    };
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::combined::analyze_equivalences;
    use crate::import::json_schema::json_schema_to_grammar;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    fn elapsed_ms(started_at: Instant) -> f64 {
        started_at.elapsed().as_secs_f64() * 1000.0
    }

    fn mask_has_token(mask: &[u32], token: u32) -> bool {
        let word = token as usize / 32;
        let bit = token as usize % 32;
        word < mask.len() && (mask[word] & (1u32 << bit)) != 0
    }

    fn kb814_normalized_schema_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/kb814_normalized_schema.json")
    }

    fn kb814_prepared_terminals_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/kb814_prepared_terminals.json")
    }

    fn gpt2_vocab_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../grammars2024/benchmarking/gpt2_vocab.json")
    }

    fn gpt2_token_str_to_bytes(token_str: &str) -> Vec<u8> {
        token_str.chars().map(unicode_char_to_byte).collect()
    }

    fn unicode_char_to_byte(ch: char) -> u8 {
        if let Some(byte) = printable_byte(ch) {
            return byte;
        }

        let codepoint = ch as u32;
        let offset = codepoint
            .checked_sub(256)
            .expect("unsupported GPT-2 vocab char");
        for byte in 0u16..=255 {
            if printable_byte(char::from_u32(byte as u32).unwrap()).is_none() {
                let candidate_offset = non_printable_rank(byte as u8);
                if candidate_offset == offset as usize {
                    return byte as u8;
                }
            }
        }
        panic!("unable to decode GPT-2 vocab char: {ch:?}");
    }

    fn printable_byte(ch: char) -> Option<u8> {
        let codepoint = ch as u32;
        if (33..=126).contains(&codepoint)
            || (161..=172).contains(&codepoint)
            || (174..=255).contains(&codepoint)
        {
            Some(codepoint as u8)
        } else {
            None
        }
    }

    fn non_printable_rank(target: u8) -> usize {
        let mut rank = 0usize;
        for byte in 0u16..target as u16 {
            let byte = byte as u8;
            if printable_byte(char::from_u32(byte as u32).unwrap()).is_none() {
                rank += 1;
            }
        }
        rank
    }

    fn load_gpt2_vocab() -> Vocab {
        let vocab_path = gpt2_vocab_path();
        let vocab_json = fs::read_to_string(&vocab_path)
            .unwrap_or_else(|err| panic!("failed to read GPT-2 vocab at {}: {err}", vocab_path.display()));
        let vocab_map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&vocab_json).expect("parse GPT-2 vocab json");
        let entries = vocab_map
            .into_iter()
            .map(|(token_str, token_id)| {
                let token_id = token_id.as_u64().expect("token id must be integer") as u32;
                (token_id, gpt2_token_str_to_bytes(&token_str))
            })
            .collect();
        Vocab::new(entries, None)
    }

    #[test]
    fn test_compile_simple_ab() {
        let gdef = simple_ab_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
        assert!(!constraint.possible_matches_for_state(0).is_empty());
    }

    #[test]
    fn test_possible_matches_union_covers_all_tokenizer_reachable_tokens() {
        let gdef = simple_ab_grammar();
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
                (3, b"ba".to_vec()),
                (4, b"x".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);

        for tokenizer_state in 0..constraint.tokenizer.num_states() {
            let mut expected = std::collections::BTreeSet::new();
            for (token_id, token_bytes) in &vocab.entries {
                let exec = constraint
                    .tokenizer
                    .execute_from_state(token_bytes, tokenizer_state);
                if !exec.matches.is_empty() {
                    expected.insert(*token_id);
                }
            }

            let actual: std::collections::BTreeSet<u32> = constraint
                .possible_matches_for_state(tokenizer_state)
                .values()
                .flat_map(|token_ids| token_ids.iter())
                .collect();

            assert!(
                expected.is_subset(&actual),
                "possible_matches union should cover all tokenizer-reachable tokens for state {} \
                 (expected {:?} ⊆ actual {:?})",
                tokenizer_state,
                expected,
                actual,
            );
        }
    }

    #[test]
    fn test_compile_choice() {
        let gdef = choice_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
    }

    #[test]
    fn test_compile_two_nt() {
        let gdef = two_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
        assert!(constraint.table.num_states > 0);
    }

    #[test]
    fn test_compile_duplicate_token_bytes_expand_back_to_all_original_tokens() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (10, b"a".to_vec()),
                (20, b"a".to_vec()),
                (30, b"b".to_vec()),
            ],
            None,
        );

        let mask = compile(&gdef, &vocab).start().mask();
        assert!(mask_has_token(&mask, 10));
        assert!(mask_has_token(&mask, 20));
        assert!(!mask_has_token(&mask, 30));
    }

    #[test]
    fn test_compile_duplicate_token_bytes_collapse_in_internal_possible_matches() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (10, b"a".to_vec()),
                (20, b"a".to_vec()),
                (30, b"b".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let tokenizer_state = constraint.tokenizer.initial_state();
        let internal_token = constraint.internal_token_for_original(10);
        assert_eq!(internal_token, constraint.internal_token_for_original(20));

        let internal_matches: std::collections::BTreeSet<u32> = constraint
            .possible_matches_for_state_internal(tokenizer_state)
            .into_iter()
            .flat_map(|m| m.into_values())
            .flat_map(|token_ids| token_ids.into_iter())
            .collect();
        assert_eq!(internal_matches, std::collections::BTreeSet::from([internal_token]));

        let original_matches: std::collections::BTreeSet<u32> = constraint
            .possible_matches_for_state(tokenizer_state)
            .values()
            .flat_map(|token_ids| token_ids.iter())
            .collect();
        assert_eq!(original_matches, std::collections::BTreeSet::from([10, 20]));
    }

    #[test]
    fn test_build_tokenizer_projects_hidden_exclusion_groups() {
        let grammar = GrammarDef {
            rules: vec![],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Exclude {
                        expr: Box::new(Expr::U8Class(crate::ds::u8set::U8Set::from_range(0, 255))),
                        exclude: Box::new(Expr::U8Seq(b"a".to_vec())),
                    },
                },
            ],
            ..Default::default()
        };

        let tokenizer = build_tokenizer(&grammar);

        assert_eq!(tokenizer.matched_terminals(tokenizer.run(b"a")), std::collections::BTreeSet::from([0]));
        assert_eq!(tokenizer.matched_terminals(tokenizer.run(b"b")), std::collections::BTreeSet::from([1]));
    }

    #[test]
    fn test_end_to_end_choice() {
        let gdef = choice_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed");

        state
            .commit_token(0).unwrap();
        assert!(
            state.is_finished(),
            "parse should accept after 'a'"
        );
    }

    #[test]
    fn test_end_to_end_two_nt() {
        let gdef = two_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        state
            .commit_token(0).unwrap();
        assert!(
            !state.is_finished(),
            "not yet accepting after 'a'"
        );

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1).unwrap();
        assert!(state.is_finished(), "should accept after 'ab'");
    }

    #[test]
    fn test_end_to_end_nested_nt() {
        let gdef = nested_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        state
            .commit_token(0).unwrap();
        assert!(!state.is_finished(), "not accepting after 'a'");

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1).unwrap();
        assert!(state.is_finished(), "should accept after 'ab'");
    }

    #[test]
    fn test_end_to_end_three_terminals() {
        let gdef = three_terminal_grammar();
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'c' should NOT be allowed initially");

        state.commit_token(0).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        state.commit_token(1).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        state.commit_token(2).unwrap();
        assert!(state.is_finished(), "should accept after 'abc'");
    }

    #[test]
    fn test_end_to_end_nested_two_rhs() {
        let gdef = nested_two_rhs_grammar();
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'c' should NOT be allowed initially");

        state.commit_token(0).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        state.commit_token(1).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        state.commit_token(2).unwrap();
        assert!(state.is_finished(), "should accept after 'abc'");
    }

    #[test]
    fn test_commit_preserves_longer_terminal_continuation_after_shorter_match() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"ab".to_vec(),
                },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should not be allowed initially");

        state.commit_token(0).unwrap();
        assert!(
            !state.is_finished(),
            "the shorter literal 'a' should not complete a grammar expecting 'ab'"
        );

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token 'b' should remain allowed as a continuation of the longer literal 'ab'"
        );

        state.commit_token(1).unwrap();
        assert!(state.is_finished(), "should accept after committing 'ab' byte by byte");
    }

    // ── Nullable terminal expansion tests ───────────────────────────────────

    #[test]
    fn test_expand_nullable_terminals_no_nullables() {
        let gdef = simple_ab_grammar();
        let nullable = std::collections::BTreeSet::new();
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);
        assert_eq!(rules.len(), gdef.rules.len());
        assert_eq!(rules[0].rhs, gdef.rules[0].rhs);
    }

    #[test]
    fn test_expand_nullable_terminals_single_nullable() {
        // Grammar: S → t0 t1, where t0 is nullable.
        // Expected: fresh NT2, S → NT2 t1, NT2 → ε, NT2 → t0
        let gdef = simple_ab_grammar(); // S → T0 T1, nonterminals: {0}
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        // 1 rewritten original rule + 2 fresh-NT rules = 3 total.
        assert_eq!(rules.len(), 3);

        // The fresh NT id should be grammar.num_nonterminals() = 1.
        let fresh_nt = gdef.num_nonterminals();

        // S → NT_fresh t1
        assert_eq!(rules[0].lhs, 0);
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(fresh_nt), Symbol::Terminal(1)]
        );

        // NT_fresh → ε and NT_fresh → t0
        let fresh_rules: Vec<&Rule> =
            rules.iter().filter(|r| r.lhs == fresh_nt).collect();
        assert_eq!(fresh_rules.len(), 2);

        let rhs_set: std::collections::BTreeSet<Vec<Symbol>> =
            fresh_rules.iter().map(|r| r.rhs.clone()).collect();
        assert!(rhs_set.contains(&vec![])); // ε
        assert!(rhs_set.contains(&vec![Symbol::Terminal(0)])); // t0
    }

    #[test]
    fn test_expand_nullable_terminals_both_nullable() {
        // Grammar: S → t0 t1, where both are nullable.
        // Expected: fresh NT1 for t0, fresh NT2 for t1.
        // S → NT1 NT2, NT1 → ε | t0, NT2 → ε | t1
        let gdef = simple_ab_grammar();
        let nullable = std::collections::BTreeSet::from([0u32, 1u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        // 1 rewritten rule + 2*2 fresh-NT rules = 5 total.
        assert_eq!(rules.len(), 5);

        let nt0 = gdef.num_nonterminals();     // fresh NT for t0
        let nt1 = gdef.num_nonterminals() + 1; // fresh NT for t1

        // S → NT0 NT1
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(nt0), Symbol::Nonterminal(nt1)]
        );
    }

    #[test]
    fn test_expand_nullable_terminals_nonterminal_untouched() {
        // Grammar: S → A t1, A → t0. If t0 is nullable:
        //   - Fresh NT for t0.
        //   - S → A t1 unchanged (A is a nonterminal, not touched).
        //   - A → NT_fresh (rewritten from A → t0).
        let gdef = two_nt_grammar(); // S → N1 T1, N1 → T0
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        let fresh_nt = gdef.num_nonterminals(); // = 2

        // S → N1 T1 — N1 is a nonterminal, not rewritten.
        let s_rules: Vec<&Rule> = rules.iter().filter(|r| r.lhs == 0).collect();
        assert_eq!(s_rules.len(), 1);
        assert_eq!(
            s_rules[0].rhs,
            vec![Symbol::Nonterminal(1), Symbol::Terminal(1)]
        );

        // N1 → NT_fresh (was N1 → T0, T0 is nullable so replaced).
        let n1_rules: Vec<&Rule> = rules.iter().filter(|r| r.lhs == 1).collect();
        assert_eq!(n1_rules.len(), 1);
        assert_eq!(n1_rules[0].rhs, vec![Symbol::Nonterminal(fresh_nt)]);

        // Fresh NT → ε and Fresh NT → T0.
        let fresh_rules: Vec<&Rule> =
            rules.iter().filter(|r| r.lhs == fresh_nt).collect();
        assert_eq!(fresh_rules.len(), 2);
    }

    #[test]
    fn test_expand_nullable_terminals_multiple_occurrences() {
        // Grammar: S → t0 t0, where t0 is nullable.
        // Both occurrences should be replaced by the SAME fresh NT.
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        let fresh_nt = gdef.num_nonterminals(); // = 1
        // S → NT NT (same fresh NT for both positions) + 2 fresh-NT rules = 3.
        assert_eq!(rules.len(), 3);
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(fresh_nt), Symbol::Nonterminal(fresh_nt)]
        );
    }

    #[test]
    fn test_drain_nullable_terminals_from_tokenizer() {
        // Build a tokenizer with a nullable terminal (regex `a*` matches empty string).
        let exprs = vec![
            crate::automata::regex::Expr::Repeat {    // nullable: matches ""
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 0,
                max: None,
            },
            Expr::U8Seq(b"b".to_vec()),                  // not nullable
        ];
        let mut tok = build_tokenizer_from_exprs(&exprs);

        // Before drain: terminal 0 should match at start state.
        assert!(
            tok.matched_terminals(tok.start_state()).contains(&0),
            "terminal 0 should be a start-state finalizer before drain"
        );

        let nullable = tok.isolate_start_state_and_drain_nullable_terminals();
        assert_eq!(nullable, std::collections::BTreeSet::from([0u32]));

        // After drain: terminal 0 should NOT match at start state.
        assert!(
            !tok.matched_terminals(tok.start_state()).contains(&0),
            "terminal 0 should be removed from start-state finalizers after drain"
        );
    }

    #[test]
    fn test_compile_with_nullable_terminal() {
        // S → opt_a b, where opt_a is `a*` (nullable).
        // The grammar should accept both "ab" and "b".
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Pattern {
                    id: 0,
                    pattern: "a*".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"aa".to_vec()),
            ],
            None,
        );
        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);

        // "b" alone should be accepted (opt_a consumed nothing).
        let state = constraint.start();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 1), "'b' should be allowed initially (opt_a is nullable)");
    }

    #[test]
    fn test_compact_unused_terminals_remaps_rules_and_terminal_ids() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };

        compact_unused_terminals(&mut grammar);

        assert_eq!(
            grammar.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            "used terminals should be renumbered densely when a dead terminal is removed from the middle"
        );
        assert_eq!(grammar.terminals.len(), 2);
        assert_eq!(grammar.terminals[0].id(), 0);
        assert_eq!(grammar.terminals[1].id(), 1);
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), "b");
        assert_eq!(grammar.ignore_terminal, None);
    }

    #[test]
    fn test_compact_unused_terminals_preserves_ignore_terminal_and_remaps_it() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(3)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Pattern {
                    id: 2,
                    pattern: " +".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 3,
                    bytes: b"b".to_vec(),
                },
            ],
            ignore_terminal: Some(2),
            ..Default::default()
        };

        compact_unused_terminals(&mut grammar);

        assert_eq!(
            grammar.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            "used terminals should still be renumbered densely when an ignore terminal is retained"
        );
        assert_eq!(grammar.terminals.len(), 3);
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), " +");
        assert_eq!(grammar.terminals[2].name(), "b");
        assert_eq!(grammar.ignore_terminal, Some(1));
    }

    #[test]
    fn test_compact_unused_terminals_merges_identical_terminals() {
        // Terminals 0 and 2 are identical ("a"), terminal 1 is different ("b").
        // After compacting, terminals 0 and 2 should map to the same new ID.
        let mut grammar = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(2)] },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"b".to_vec() },
                Terminal::Literal { id: 2, bytes: b"a".to_vec() },
            ],
            nonterminal_names: BTreeMap::new(),
            terminal_names: BTreeMap::new(),
            ignore_terminal: None,
        };
        compact_unused_terminals(&mut grammar);
        assert_eq!(grammar.terminals.len(), 2, "identical terminals should be merged");
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), "b");
        // Rule 1: T0 → merged "a" (id 0), T1 → "b" (id 1)
        assert_eq!(grammar.rules[0].rhs, vec![Symbol::Terminal(0), Symbol::Terminal(1)]);
        // Rule 2: T2 → merged "a" (id 0)
        assert_eq!(grammar.rules[1].rhs, vec![Symbol::Terminal(0)]);
    }

    #[test]
    fn test_compile_drops_unused_terminals_before_final_tokenizer_build() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Pattern {
                    id: 1,
                    pattern: "x*".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"x".to_vec()),
            ],
            None,
        );

        let (normalized, tokenizer) = prepare_grammar_for_compile(&gdef);
        let constraint = compile_owned(gdef, &vocab);

        assert_eq!(
            tokenizer.num_terminals,
            2,
            "the final tokenizer should be built only from the live compacted terminals"
        );
        assert_eq!(normalized.terminals.len(), 2);
        assert_eq!(
            normalized
                .terminals
                .iter()
                .map(|terminal| terminal.name())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string()],
            "the dead middle terminal should be absent from the normalized grammar"
        );
        assert_eq!(
            normalized.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            "rules should be remapped to the compacted terminal IDs"
        );

        let mut state = constraint.start();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should still be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should not be allowed initially");
        assert!(!mask_has_token(&mask, 2), "dead terminal token 'x' should not leak into the mask");

        state.commit_token(0).unwrap();
        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should not be allowed after committing 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should remain the live continuation after remapping");
        assert!(!mask_has_token(&mask, 2), "dead terminal token 'x' should remain absent after remapping");
    }

    #[test]
    fn test_compile_treats_ignore_terminal_as_epsilon_and_preserves_it_through_compaction() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(3)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Pattern {
                    id: 2,
                    pattern: " +".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 3,
                    bytes: b"b".to_vec(),
                },
            ],
            ignore_terminal: Some(2),
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b" ".to_vec()),
                (2, b"b".to_vec()),
                (3, b" a".to_vec()),
                (4, b" b".to_vec()),
            ],
            None,
        );

        let (normalized, _tokenizer) = prepare_grammar_for_compile(&gdef);
        let constraint = compile_owned(gdef, &vocab);

        assert_eq!(constraint.ignore_terminal, Some(1));
        assert_eq!(normalized.terminals.len(), 3);
        assert_eq!(normalized.ignore_terminal, Some(1));
        assert_eq!(
            normalized
                .terminals
                .iter()
                .map(|terminal| terminal.name())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), " +".to_string(), "b".to_string()],
            "the dead terminal should be removed while the ignore terminal is preserved"
        );
        assert_eq!(
            normalized.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            "live grammar terminals should be remapped around the retained ignore terminal"
        );

        let mut state = constraint.start();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(mask_has_token(&mask, 1), "ignore-only token ' ' should be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'b' should not be allowed before 'a'");
        assert!(mask_has_token(&mask, 3), "token ' a' should be allowed via ignore+terminal composition");
        assert!(!mask_has_token(&mask, 4), "token ' b' should not be allowed before 'a'");

        state.commit_token(3).unwrap();
        assert!(!state.is_finished(), "consuming ignored space plus 'a' should still leave trailing 'b'");

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should no longer be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "ignore-only token ' ' should still be allowed between grammar terminals");
        assert!(mask_has_token(&mask, 2), "token 'b' should be allowed after 'a'");
        assert!(!mask_has_token(&mask, 3), "token ' a' should not be allowed once the grammar expects 'b'");
        assert!(mask_has_token(&mask, 4), "token ' b' should be allowed via ignore+terminal composition after 'a'");

        state.commit_token(4).unwrap();
        assert!(state.is_finished(), "consuming ignored space plus 'b' should finish the grammar");
    }

    #[test]
    fn test_prepare_grammar_for_compile_retains_and_remaps_names() {
        let grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            nonterminal_names: std::collections::BTreeMap::from([(0, "start".to_string())]),
            terminal_names: std::collections::BTreeMap::from([
                (0, "A".to_string()),
                (1, "DEAD".to_string()),
                (2, "B".to_string()),
            ]),
            ignore_terminal: None,
        };

        let (normalized, _tokenizer) = prepare_grammar_for_compile(&grammar);

        assert_eq!(normalized.nonterminal_names.get(&0).map(String::as_str), Some("start"));
        assert_eq!(normalized.terminal_names.get(&0).map(String::as_str), Some("A"));
        assert_eq!(normalized.terminal_names.get(&1).map(String::as_str), Some("B"));
        assert!(!normalized.terminal_names.values().any(|name| name == "DEAD"));
    }

    #[test]
    fn test_inline_single_use_nonterminals_compacts_repetition_tail_chain() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(3)],
            },
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(1), Symbol::Terminal(1)],
            },
            Rule {
                lhs: 3,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Nonterminal(1),
                    Symbol::Nonterminal(4),
                    Symbol::Terminal(1),
                ],
            },
            Rule {
                lhs: 4,
                rhs: vec![Symbol::Nonterminal(5)],
            },
            Rule {
                lhs: 4,
                rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)],
            },
            Rule {
                lhs: 5,
                rhs: vec![Symbol::Nonterminal(6), Symbol::Nonterminal(7)],
            },
            Rule {
                lhs: 6,
                rhs: vec![Symbol::Terminal(2)],
            },
            Rule {
                lhs: 7,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 8,
                rhs: vec![Symbol::Nonterminal(9)],
            },
            Rule {
                lhs: 8,
                rhs: vec![Symbol::Nonterminal(8), Symbol::Nonterminal(9)],
            },
            Rule {
                lhs: 9,
                rhs: vec![Symbol::Nonterminal(6), Symbol::Nonterminal(2)],
            },
            Rule {
                lhs: 10,
                rhs: vec![Symbol::Terminal(3), Symbol::Nonterminal(2), Symbol::Nonterminal(8), Symbol::Terminal(4)],
            },
        ];
        let names = std::collections::BTreeMap::from([
            (0, "start".to_string()),
            (1, "json_kv".to_string()),
            (2, "json_value".to_string()),
            (3, "json_object".to_string()),
            (10, "json_array".to_string()),
        ]);

        let protected: std::collections::BTreeSet<NonterminalID> = names.keys().copied().chain(std::iter::once(0)).collect();

        inline_single_use_nonterminals(&mut rules, &protected);

        assert!(!rules.iter().any(|rule| matches!(rule.lhs, 6 | 7)));
        assert!(rules.contains(&Rule {
            lhs: 5,
            rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(1)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 4,
            rhs: vec![Symbol::Nonterminal(5)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 4,
            rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 9,
            rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(2)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 8,
            rhs: vec![Symbol::Nonterminal(9)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 8,
            rhs: vec![Symbol::Nonterminal(8), Symbol::Nonterminal(9)],
        }));
    }

    #[test]
    #[should_panic]
    fn test_inline_single_use_nonterminals_keeps_multi_symbol_helper_with_multiple_occurrences() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(2)],
            },
            Rule {
                lhs: 2,
                rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(3)],
            },
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Terminal(1)],
            },
        ];
        let names = std::collections::BTreeMap::from([
            (0, "start".to_string()),
            (1, "root".to_string()),
        ]);
        let protected: std::collections::BTreeSet<NonterminalID> = names.keys().copied().chain(std::iter::once(0)).collect();

        inline_single_use_nonterminals(&mut rules, &protected);

        assert!(rules.iter().any(|rule| rule.lhs == 2));
        assert!(rules.contains(&Rule {
            lhs: 1,
            rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(2)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 2,
            rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
        }));
    }

    #[test]
    #[ignore = "fixture generation for kb814 tokenizer/equivalence benchmarking"]
    fn test_write_kb814_prepared_terminals_fixture() {
        let schema_path = kb814_normalized_schema_path();
        let schema_json = fs::read_to_string(&schema_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", schema_path.display()));
        let grammar = json_schema_to_grammar(&schema_json).expect("kb814 schema should import");
        let (prepared_grammar, _tokenizer) = prepare_owned_grammar_for_compile(grammar);

        let terminals_path = kb814_prepared_terminals_path();
        let payload = serde_json::to_vec(&prepared_grammar.terminals)
            .expect("serialize prepared terminals");
        fs::write(&terminals_path, payload)
            .unwrap_or_else(|err| panic!("failed to write {}: {err}", terminals_path.display()));

        eprintln!(
            "[kb814] wrote_prepared_terminals path={} terminals={}",
            terminals_path.display(),
            prepared_grammar.terminals.len(),
        );
    }

    #[test]
    #[ignore = "kb814 tokenizer/equivalence timing benchmark"]
    fn test_kb814_prepared_terminals_gpt2_timings() {
        let terminals_path = kb814_prepared_terminals_path();
        let terminals_json = fs::read_to_string(&terminals_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", terminals_path.display()));
        let terminals: Vec<Terminal> = serde_json::from_str(&terminals_json)
            .expect("parse prepared terminals json");
        let vocab = load_gpt2_vocab();
        let grammar = GrammarDef {
            terminals,
            ..Default::default()
        };

        let tokenizer_started_at = Instant::now();
        let tokenizer = build_tokenizer(&grammar);
        let tokenizer_ms = elapsed_ms(tokenizer_started_at);
        eprintln!(
            "[kb814] terminals_file={} vocab_file={} terminals={} tokenizer_states={} build_tokenizer_ms={:.3}",
            terminals_path.display(),
            gpt2_vocab_path().display(),
            grammar.terminals.len(),
            tokenizer.num_states(),
            tokenizer_ms,
        );

        unsafe {
            std::env::set_var("GLRMASK_PROFILE_COMPILE", "1");
        }
        let equivalence_started_at = Instant::now();
        let id_map = analyze_equivalences(
            &tokenizer,
            &vocab,
            &std::collections::BTreeMap::new(),
            None,
            None,
        );
        let equivalence_ms = elapsed_ms(equivalence_started_at);
        eprintln!(
            "[kb814] tokenizer_state_classes={} vocab_classes={} equivalence_ms={:.3}",
            id_map.tokenizer_states.internal_to_originals.len(),
            id_map.vocab_tokens.internal_to_originals.len(),
            equivalence_ms,
        );
    }

    /// Regression test for o76439: structural import with nested closed objects
    /// must accept cross-token terminal matches (e.g., ` {"` after `,`).
    #[test]
    fn test_o76439_gpt2_vocab_false_negative() {
        let vocab = load_gpt2_vocab();

        // Actual o76439 schema
        let schema = r#"{
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "ignoreSevertiesAtOrBelow": {
                    "type": "string",
                    "enum": ["negligible", "Negligible", "low", "Low",
                             "medium", "Medium", "high", "High"]
                },
                "vulnerabilities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cveId": {"type": "string", "minLength": 1, "maxLength": 512},
                            "rationale": {"type": "string", "minLength": 1, "maxLength": 512}
                        },
                        "required": ["cveId", "rationale"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["ignoreSevertiesAtOrBelow"],
            "additionalProperties": false
        }"#;

        let c = crate::Constraint::from_json_schema(schema, &vocab).unwrap();
        let mut state = c.start();

        // Commit the prefix (token positions 0..49 from the mismatch report)
        let prefix = b"{\"ignoreSevertiesAtOrBelow\": \"Medium\", \"vulnerabilities\": [{\"cveId\": \"CVE-2022-1234\", \"rationale\": \"This vulnerability is not applicable to our system.\"},";
        state.commit_bytes(prefix).expect("prefix should commit");

        let mut state_clone = state.clone();
        state_clone
            .commit_bytes(b" {\"")
            .expect("token bytes ` {\"` should commit after the array-item separator");

        let target_bytes = b" {\"";
        let target_token_id = vocab
            .entries
            .iter()
            .find(|(_, bytes)| bytes.as_slice() == target_bytes)
            .map(|(&id, _)| id)
            .expect("GPT-2 vocab must contain ` {\"`");

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, target_token_id),
            "token {} (` {{\"`) must be in the mask — false negative regression (o76439)",
            target_token_id
        );
    }

    #[test]
    fn test_cross_token_bridge_after_partial_literal_terminal() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"\"".to_vec() },
                Terminal::Literal { id: 1, bytes: b": ".to_vec() },
                Terminal::Literal { id: 2, bytes: b"true".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"\"".to_vec()),
                (1, b"\":".to_vec()),
                (2, b": ".to_vec()),
                (3, b" true".to_vec()),
                (4, b"true".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token '\" :' must be allowed so the compile can stop mid-': ' and continue in the next token"
        );

        state.commit_token(1).unwrap();

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 3),
            "token ' true' must be allowed after '\" :' to bridge ': ' into 'true'"
        );
    }

    #[test]
    fn test_cross_token_bridge_after_complete_literal_terminal() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"}".to_vec() },
                Terminal::Literal { id: 1, bytes: b",".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"}".to_vec()),
                (1, b",".to_vec()),
                (2, b"},".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token '}}' should be allowed initially");
        assert!(
            mask_has_token(&mask, 2),
            "token '}},' must be allowed to bridge a complete '}}' terminal into the following ',' terminal"
        );

        state.commit_token(2).unwrap();
        assert!(state.is_finished(), "state should finish after committing bridged token '}},'");
    }

    #[test]
    fn test_cross_token_bridge_across_reduction_boundary() {
        let gdef = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1), Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"}".to_vec() },
                Terminal::Literal { id: 1, bytes: b",".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"}".to_vec()),
                (1, b",".to_vec()),
                (2, b"},".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token '}}' should be allowed initially");
        assert!(
            mask_has_token(&mask, 2),
            "token '}},' must be allowed even when the ',' only becomes legal after reducing the preceding '}}' item"
        );

        state.commit_token(2).unwrap();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token '}}' should remain allowed for the trailing reduced item");

        state.commit_token(0).unwrap();
        assert!(state.is_finished(), "state should finish after the bridged token and final reduced item");
    }

    #[test]
    fn test_cross_token_bridge_across_nullable_inner_chain() {
        let gdef = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(0),
                        Symbol::Nonterminal(1),
                        Symbol::Terminal(3),
                        Symbol::Terminal(4),
                    ],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(2),
                        Symbol::Terminal(2),
                    ],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![Symbol::Terminal(5)],
                },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"{\"evt\":".to_vec() },
                Terminal::Literal { id: 1, bytes: b" {".to_vec() },
                Terminal::Literal { id: 2, bytes: b"}".to_vec() },
                Terminal::Literal { id: 3, bytes: b", ".to_vec() },
                Terminal::Literal { id: 4, bytes: b"\"next\":\"x\"}".to_vec() },
                Terminal::Literal { id: 5, bytes: b"\"k\":\"v\"".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"{\"evt\":".to_vec()),
                (1, b" {},".to_vec()),
                (2, b" \"next\":\"x\"}".to_vec()),
                (3, b" {\"k\":\"v\"},".to_vec()),
            ],
            None,
        );

        let (prepared_grammar, _) = prepare_grammar_for_compile(&gdef);
        assert!(
            prepared_grammar.rules.iter().any(|rule| {
                rule.rhs
                    .windows(2)
                    .any(|window| {
                        window == [Symbol::Terminal(1), Symbol::Terminal(2)]
                            || window == [Symbol::Terminal(2), Symbol::Terminal(3)]
                    })
            }),
            "prepared grammar should expose direct terminal adjacency across the nullable inner chain"
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();
        let prefix_exec = constraint
            .tokenizer
            .execute_from_state(b"{\"evt\":", constraint.tokenizer.initial_state());
        let prefix_state = prefix_exec.end_state.expect("prefix should leave the tokenizer in a live state");
        let _possible_matches = constraint.possible_matches_for_state(prefix_state);

        state
            .commit_bytes(b"{\"evt\":")
            .expect("prefix token should advance the parser state");

        let mask = state.mask();
        assert!(mask_has_token(&mask, 1), "token ' {{}},' should be allowed after the key prefix when the inner object body reduces through epsilon before '}}'");
        assert!(mask_has_token(&mask, 3), "non-empty object token should also remain allowed");

        state.commit_token(1).unwrap();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 2), "after the bridged empty-object token, the trailing sibling token should remain allowed");

        state.commit_token(2).unwrap();
        assert!(state.is_finished(), "state should finish after the bridged empty-object token and trailing sibling token");
    }

    #[test]
    fn test_cross_token_bridge_after_partial_key_prefix_through_nested_nullable_object_chain() {
        let gdef = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(0),
                        Symbol::Nonterminal(1),
                        Symbol::Terminal(5),
                        Symbol::Terminal(6),
                    ],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(2),
                        Symbol::Terminal(4),
                    ],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(3)],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![Symbol::Nonterminal(4)],
                },
                Rule {
                    lhs: 3,
                    rhs: vec![Symbol::Terminal(3), Symbol::Terminal(7), Symbol::Nonterminal(3)],
                },
                Rule { lhs: 3, rhs: vec![] },
                Rule {
                    lhs: 4,
                    rhs: vec![Symbol::Terminal(7), Symbol::Nonterminal(3)],
                },
                Rule { lhs: 4, rhs: vec![] },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"{\"onRequestExternal\": ".to_vec() },
                Terminal::Literal { id: 1, bytes: b"{".to_vec() },
                Terminal::Literal { id: 2, bytes: b"\"removeRules\": \"x\"".to_vec() },
                Terminal::Literal { id: 3, bytes: b", ".to_vec() },
                Terminal::Literal { id: 4, bytes: b"}".to_vec() },
                Terminal::Literal { id: 5, bytes: b", ".to_vec() },
                Terminal::Literal { id: 6, bytes: b"\"next\":\"x\"}".to_vec() },
                Terminal::Literal { id: 7, bytes: b"\"extra\": \"y\"".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"{\"onRequestExternal\":".to_vec()),
                (1, b" {},".to_vec()),
                (2, b" \"next\":\"x\"}".to_vec()),
                (3, b" {\"removeRules\": \"x\"},".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();
        state
            .commit_bytes(b"{\"onRequestExternal\":")
            .expect("partial key-prefix token should advance the parser state");

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token ' {{}},' should remain allowed when the empty object closes only after a nested nullable chain and the comma-space separator continues in the next token"
        );

        state.commit_token(1).unwrap();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 2), "the trailing sibling token should remain allowed after the bridged empty object token");
    }

    #[test]
    fn test_cross_token_bridge_with_regex_additional_property_alternative() {
        let gdef = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(0),
                        Symbol::Nonterminal(1),
                        Symbol::Terminal(3),
                        Symbol::Terminal(4),
                    ],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(2),
                        Symbol::Terminal(2),
                    ],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![
                        Symbol::Terminal(5),
                        Symbol::Terminal(6),
                        Symbol::Nonterminal(3),
                    ],
                },
                Rule { lhs: 2, rhs: vec![] },
                Rule {
                    lhs: 3,
                    rhs: vec![
                        Symbol::Terminal(3),
                        Symbol::Terminal(5),
                        Symbol::Terminal(6),
                        Symbol::Nonterminal(3),
                    ],
                },
                Rule { lhs: 3, rhs: vec![] },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"{\"sendRequest\":\"x\", \"onRequestExternal\": ".to_vec() },
                Terminal::Literal { id: 1, bytes: b"{".to_vec() },
                Terminal::Literal { id: 2, bytes: b"}".to_vec() },
                Terminal::Literal { id: 3, bytes: b", ".to_vec() },
                Terminal::Literal { id: 4, bytes: b"\"tail\":\"y\"}".to_vec() },
                Terminal::Pattern { id: 5, pattern: r#"\"(?:[^\"\\]|\\.)*\": ?"#.to_string(), utf8: false },
                Terminal::Literal { id: 6, bytes: b"\"z\"".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"{\"sendRequest\":\"x\", \"onRequestExternal\":".to_vec()),
                (1, b" {},".to_vec()),
                (2, b" \"tail\":\"y\"}".to_vec()),
                (3, b" {\"other\":\"z\"},".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();
        state
            .commit_bytes(b"{\"sendRequest\":\"x\", \"onRequestExternal\":")
            .expect("partial key-prefix token should advance the parser state");

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token ' {{}},' should remain allowed even when the empty object competes with a regex additional-property branch"
        );

        state.commit_token(1).unwrap();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 2), "the trailing sibling token should remain allowed after the bridged empty object token");
    }

    #[test]
    fn test_json_schema_open_ordered_object_keeps_additional_property_continuation_after_shared_optional_ref() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "extension": {
                    "type": "object",
                    "properties": {
                        "req0": { "instanceof": "function" },
                        "req1": { "instanceof": "function" },
                        "onRequest": { "$ref": "#/definitions/Event" },
                        "onRequestExternal": { "$ref": "#/definitions/Event" }
                    },
                    "required": ["req0", "req1"]
                }
            },
            "definitions": {
                "Event": {
                    "type": "object",
                    "properties": {
                        "addListener": { "instanceof": "function" },
                        "addRules": { "instanceof": "function" },
                        "getRules": { "instanceof": "function" },
                        "hasListener": { "instanceof": "function" },
                        "hasListeners": { "instanceof": "function" },
                        "removeListener": { "instanceof": "function" },
                        "removeRules": { "instanceof": "function" }
                    }
                }
            }
        }"##;

        let grammar = json_schema_to_grammar(schema).expect("schema should lower to a grammar");
        let vocab = Vocab::new(vec![(0, b" {},".to_vec())], None);
        let constraint = compile(&grammar, &vocab);
        let mut state = constraint.start();

        state
            .commit_bytes(
                b"{\"extension\": {\"req0\": \"function () {}\", \"req1\": \"function () {}\", \"onRequest\": {\"addListener\": \"function () {}\", \"addRules\": \"function () {}\", \"getRules\": \"function () {}\", \"hasListener\": \"function () {}\", \"hasListeners\": \"function () {}\", \"removeListener\": \"function () {}\", \"removeRules\": \"function () {}\"}, \"onRequestExternal\":"
            )
            .expect("prefix should keep the parser state live");

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 0),
            "open ordered objects should still allow an empty shared-ref object token followed by a comma when additional properties remain available"
        );

        state
            .commit_token(0)
            .expect("the bridged empty-object token should stay accepted");
    }

    #[test]
    fn test_json_schema_o62060_minimized_empty_object_bridge_up_to_w() {
        const PREFIX: &[u8] = b"{\"a\": 0, \"b\": 0, \"c\":";

        let tail = (b'e'..=b'w')
            .map(|key| format!("\"{}\":{{}}", key as char))
            .collect::<Vec<_>>()
            .join(",");
        let schema = [
            "{\"type\":\"object\",\"properties\":{\"a\":{},\"b\":{},\"d\":{},\"c\":{\"type\":\"object\"},",
            &tail,
            "},\"required\":[\"a\",\"b\",\"e\",\"c\"],\"additionalProperties\":false}",
        ]
            .concat();

        let grammar = json_schema_to_grammar(&schema).expect("schema should lower to a grammar");
        let vocab = Vocab::new(vec![(0u32, b" {},".to_vec())], None);
        let constraint = compile(&grammar, &vocab);
        let mut mask_state = constraint.start();

        mask_state
            .commit_bytes(PREFIX)
            .expect("minimized prefix bytes should advance the parser state");

        let mask = mask_state.mask();
        let mask_accepts = mask_has_token(&mask, 0);

        let mut commit_state = constraint.start();
        commit_state
            .commit_bytes(PREFIX)
            .expect("minimized prefix bytes should advance the parser state");
        let commit_accepts = commit_state.commit_token(0u32).is_ok();

        assert_eq!(
            (mask_accepts, commit_accepts),
            (true, true),
            "token ' {{}},' should remain both masked-in and committable after the minimized o62060 prefix witness"
        );
    }

    #[test]
    fn test_json_schema_o62060_minimized_empty_object_bridge_up_to_x() {
        const PREFIX: &[u8] = b"{\"a\": 0, \"b\": 0, \"c\":";

        let tail = (b'e'..=b'x')
            .map(|key| format!("\"{}\":{{}}", key as char))
            .collect::<Vec<_>>()
            .join(",");
        let schema = [
            "{\"type\":\"object\",\"properties\":{\"a\":{},\"b\":{},\"d\":{},\"c\":{\"type\":\"object\"},",
            &tail,
            "},\"required\":[\"a\",\"b\",\"e\",\"c\"],\"additionalProperties\":false}",
        ]
        .concat();

        let grammar = json_schema_to_grammar(&schema).expect("schema should lower to a grammar");
        let vocab = Vocab::new(vec![(0u32, b" {},".to_vec())], None);
        let constraint = compile(&grammar, &vocab);
        let mut mask_state = constraint.start();

        mask_state
            .commit_bytes(PREFIX)
            .expect("minimized prefix bytes should advance the parser state");

        let mask = mask_state.mask();
        let mask_accepts = mask_has_token(&mask, 0);

        let mut commit_state = constraint.start();
        commit_state
            .commit_bytes(PREFIX)
            .expect("minimized prefix bytes should advance the parser state");
        let commit_accepts = commit_state.commit_token(0u32).is_ok();

        println!("state before: {:?}", mask_state.debug_parser_stacks());
        println!("state after:  {:?}", commit_state.debug_parser_stacks());

        assert_eq!(
            (mask_accepts, commit_accepts),
            (true, true),
            "token ' {{}},' should remain both masked-in and committable after the minimized o62060 prefix witness"
        );
    }

    #[test]
    fn test_terminal_dwa_path_o62060() {
        let tail = (b'e'..=b'x')
            .map(|key| format!("\"{}\":{{}}", key as char))
            .collect::<Vec<_>>()
            .join(",");
        let schema = [
            "{\"type\":\"object\",\"properties\":{\"a\":{},\"b\":{},\"d\":{},\"c\":{\"type\":\"object\"},",
            &tail,
            "},\"required\":[\"a\",\"b\",\"e\",\"c\"],\"additionalProperties\":false}",
        ]
        .concat();

        let grammar = json_schema_to_grammar(&schema).expect("schema should lower to a grammar");
        let vocab = Vocab::new(vec![(0u32, b" {},".to_vec())], None);
        let constraint = compile(&grammar, &vocab);

        let (prepared_grammar, _) =
            crate::compiler::grammar::transforms::prepare_grammar_for_compile(&grammar);

        use crate::compiler::glr::analysis::AnalyzedGrammar;
        use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};

        let analyzed_grammar = AnalyzedGrammar::from_grammar_def(&prepared_grammar);

        let id_map = InternalIdMap {
            tokenizer_states: ManyToOneIdMap {
                original_to_internal: constraint.state_to_internal_tsid.clone(),
                representative_original_ids: constraint
                    .internal_tsid_to_states
                    .iter()
                    .map(|v| v[0])
                    .collect(),
                internal_to_originals: constraint.internal_tsid_to_states.clone(),
            },
            vocab_tokens: ManyToOneIdMap {
                original_to_internal: constraint.original_token_to_internal.clone(),
                representative_original_ids: constraint
                    .internal_token_to_tokens
                    .iter()
                    .map(|v| v[0])
                    .collect(),
                internal_to_originals: constraint.internal_token_to_tokens.clone(),
            },
        };

        let (terminal_dwa, _) =
            crate::compiler::stages::terminal_dwa_compat::build_terminal_dwa_for_existing_id_map_with_possible_matches_and_coloring(
                &analyzed_grammar,
                &constraint.tokenizer,
                &vocab,
                &id_map,
                &crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalColoring::identity(
                    analyzed_grammar.num_terminals as usize,
                ),
                false,
                constraint.ignore_terminal,
                None,
            );

        let max_path_len = terminal_dwa
            .max_accepting_path_length_with_nonempty_weight()
            .expect("terminal DWA should have an accepting path with nonempty weight");
        assert!(
            max_path_len > vocab.entries.get(&0).unwrap().len(),
            "expected terminal DWA max path length with nonempty weight to exceed the sole token byte length, got max_path_len={} token_bytes_len={}",
            max_path_len,
            vocab.entries.get(&0).unwrap().len(),
        );
    }

    #[test]
    fn test_json_schema_o62060_minimized_empty_object_bridge_commit_full_prefix_and_comma_space() {
        const FULL_TERMINALS: &[u32] = &[
            7,  // {
            1,  // "
            12, // a"
            6,  // : 
            2,  // JSON_INTEGER
            8,  // , 
            1,  // "
            13, // b"
            6,  // : 
            2,  // JSON_INTEGER
            8,  // , 
            1,  // "
            15, // c"
            6,  // : 
            7,  // {
            9,  // }
            8,  // , 
        ];

        let grammar: GrammarDef = serde_json::from_str(include_str!("../../tests/data/o62060_minimized_empty_object_bridge_grammar.json"))
            .expect("fixture grammar json should parse");
        let (prepared, _tokenizer) = prepare_grammar_for_compile(&grammar);
        let analyzed = AnalyzedGrammar::from_grammar_def(&prepared);

        for (label, table) in [
            ("no_inline", GLRTable::build_with_unit_reduction_inlining(&analyzed, false)),
            ("inline", GLRTable::build_with_unit_reduction_inlining(&analyzed, true)),
        ] {
            let mut state = ParserGSS::from_stacks(&[(vec![0u32], TerminalsDisallowed::new())]);

            for (step, &terminal) in FULL_TERMINALS.iter().enumerate() {
                let before_tops = state.peek_values();
                state = advance_stacks(&table, &state, terminal);
                assert!(
                    !state.is_empty(),
                    "{} direct GLR table drive died at step {} on terminal {} with incoming tops {:?}",
                    label,
                    step,
                    terminal,
                    before_tops,
                );
            }

            assert!(
                !state.is_empty(),
                "{} direct GLR table drive should keep the full minimized terminal witness through ' {{}}, ' alive",
                label,
            );
        }
    }
}
