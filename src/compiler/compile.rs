#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::regex::parse_regex;
use crate::automata::lexer::compile::build_regex;
use crate::automata::regex::Expr;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::debug::{AutomataDebug, CompileDebug, TerminalDebug};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::{GrammarDef, NonterminalID, Terminal};
use crate::compiler::grammar_def::{Rule, Symbol, TerminalID};
use crate::compiler::parser_dwa::build_parser_dwa_from_terminal_dwa;
use crate::compiler::possible_matches::build_possible_matches_by_state;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::compiler::stages::templates::Templates;
use crate::compiler::terminal_dwa::build_terminal_dwa;
use crate::runtime::Constraint;

// ── Tokenizer construction ──────────────────────────────────────────────────

/// Build a [`Tokenizer`] from a [`GrammarDef`].
///
/// Each terminal is compiled through the NFA→DFA pipeline via [`build_regex`].
/// The group index matches the terminal ID (guaranteed by construction).
pub(crate) fn build_tokenizer(grammar: &GrammarDef) -> Tokenizer {
    let exprs: Vec<Expr> = grammar
        .terminals
        .iter()
        .map(|terminal| match terminal {
            Terminal::Literal { bytes, .. } => Expr::U8Seq(bytes.clone()),
            Terminal::Pattern { pattern, .. } => parse_regex(pattern),
            Terminal::Expr { expr, .. } => expr.clone(),
        })
        .collect();
    let regex = build_regex(&exprs);
    Tokenizer {
        dfa: regex.dfa,
        num_terminals: grammar.num_terminals(),
    }
}

/// Build a [`Tokenizer`] from a slice of regex expressions.
///
/// Each expression's index becomes its `TerminalID`.
pub(crate) fn build_tokenizer_from_exprs(exprs: &[Expr]) -> Tokenizer {
    let num = exprs.len() as u32;
    let regex = build_regex(exprs);
    Tokenizer {
        dfa: regex.dfa,
        num_terminals: num,
    }
}

fn decode_literal_pattern(pattern: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(pattern.len());
    let bytes = pattern.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' && index + 1 < bytes.len() {
            index += 1;
            out.push(match bytes[index] {
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                other => other,
            });
        } else {
            out.push(bytes[index]);
        }
        index += 1;
    }
    out
}

// ── Nullable terminal expansion ─────────────────────────────────────────────

/// Expand grammar rules so that nullable terminals (those matching the empty
/// string) are treated as optional at every position they appear.
///
/// For each rule with *k* nullable-terminal positions in its RHS, produce 2^k
/// variants covering every subset of those positions.  Duplicate rules are
/// removed.  Rules where all RHS symbols are removed (ε-rules) are kept — the
/// resulting nullable nonterminals are left for downstream handling.
pub(crate) fn expand_nullable_terminals(
    grammar: &GrammarDef,
    nullable_terminals: &std::collections::BTreeSet<TerminalID>,
) -> GrammarDef {
    if nullable_terminals.is_empty() {
        return grammar.clone();
    }

    // Expand rules via power-set removal of nullable-terminal positions.
    // Only terminal positions are considered; nullable nonterminals that
    // arise as a consequence (e.g. `A → nullable_t`) are left in the
    // grammar for downstream handling.
    let mut seen = std::collections::HashSet::<Rule>::new();
    let mut new_rules = Vec::new();
    for rule in &grammar.rules {
        let nullable_positions: Vec<usize> = rule
            .rhs
            .iter()
            .enumerate()
            .filter(|(_, sym)| matches!(sym, Symbol::Terminal(tid) if nullable_terminals.contains(tid)))
            .map(|(i, _)| i)
            .collect();

        let k = nullable_positions.len();
        for mask in 0..(1u64 << k) {
            let new_rhs: Vec<Symbol> = rule
                .rhs
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    if let Ok(idx) = nullable_positions.binary_search(i) {
                        mask & (1u64 << idx) == 0
                    } else {
                        true
                    }
                })
                .map(|(_, sym)| sym.clone())
                .collect();

            let candidate = Rule {
                lhs: rule.lhs,
                rhs: new_rhs,
            };
            if seen.insert(candidate.clone()) {
                new_rules.push(candidate);
            }
        }
    }

    GrammarDef {
        rules: new_rules,
        start: grammar.start,
        terminals: grammar.terminals.clone(),
    }
}

pub fn compile(grammar: &GrammarDef, vocab: &Vocab) -> Constraint {
    // Step 1: Build tokenizer (NFA→DFA – may produce start-state finalizers
    //         for nullable terminals).
    let mut tokenizer = build_tokenizer(grammar);

    // Step 2: Drain nullable terminals from the tokenizer.
    let nullable_terminals = tokenizer.drain_nullable_terminals();

    // Step 3: Expand grammar rules to inline the optionality of nullable
    //         terminals.
    let normalized = expand_nullable_terminals(grammar, &nullable_terminals);

    let glr_grammar = AnalyzedGrammar::from_grammar_def(&normalized);

    // Debug check: verify grammar preconditions before expensive pipeline stages.
    // Violations here indicate the grammar (or its normalization) has shapes that
    // will cause panics or incorrect results later in the pipeline.
    #[cfg(debug_assertions)]
    if let Err(msg) = glr_grammar.debug_check_grammar_preconditions() {
        panic!("[glrmask] grammar precondition violations:\n{}", msg);
    }

    let table = GLRTable::build(&glr_grammar);
    let id_map = InternalIdMap::build(&tokenizer, vocab);

    let possible_matches_by_state = build_possible_matches_by_state(&normalized, &tokenizer, vocab);

    let token_bytes: std::collections::BTreeMap<u32, Vec<u8>> =
        vocab.entries.iter().cloned().collect();
    let terminal_dwa = build_terminal_dwa(
        &glr_grammar,
        &tokenizer,
        vocab,
        &id_map,
    );
    let parser_dwa = build_parser_dwa_from_terminal_dwa(
        &table,
        &glr_grammar,
        &tokenizer,
        &terminal_dwa,
        &id_map,
    );

    Constraint {
        parser_dwa,
        table,
        tokenizer,
        possible_matches: possible_matches_by_state.clone(),
        state_to_internal_tsid: id_map.tokenizer_states.original_to_internal.clone(),
        internal_tsid_to_states: id_map.tokenizer_states.internal_to_originals.clone(),
        eos_token_id: vocab.eos_token_id,
        token_bytes,
    }
}

pub(crate) fn compile_with_debug(grammar: &GrammarDef, vocab: &Vocab) -> (Constraint, CompileDebug) {
    // Step 1: Build tokenizer.
    let mut tokenizer = build_tokenizer(grammar);

    // Step 2: Drain nullable terminals.
    let nullable_terminals = tokenizer.drain_nullable_terminals();

    // Step 3: Expand grammar rules for nullable terminal optionality.
    let normalized = expand_nullable_terminals(grammar, &nullable_terminals);

    let glr_grammar = AnalyzedGrammar::from_grammar_def(&normalized);

    #[cfg(debug_assertions)]
    if let Err(msg) = glr_grammar.debug_check_grammar_preconditions() {
        panic!("[glrmask] grammar precondition violations:\n{}", msg);
    }

    let table = GLRTable::build(&glr_grammar);
    let id_map = InternalIdMap::build(&tokenizer, vocab);

    let possible_matches_by_state = build_possible_matches_by_state(&normalized, &tokenizer, vocab);

    let characterizations = characterize_terminals(&table, &glr_grammar);
    let templates = Templates::from_characterizations(&characterizations);
    let terminal_dwa = build_terminal_dwa(
        &glr_grammar,
        &tokenizer,
        vocab,
        &id_map,
    );
    let parser_dwa = build_parser_dwa_from_terminal_dwa(
        &table,
        &glr_grammar,
        &tokenizer,
        &terminal_dwa,
        &id_map,
    );

    let vocab_entries: Vec<(u32, Vec<u8>)> = vocab.entries.iter().cloned().collect();
    let token_bytes: std::collections::BTreeMap<u32, Vec<u8>> =
        vocab.entries.iter().cloned().collect();
    let constraint = Constraint {
        parser_dwa: parser_dwa.clone(),
        table: table.clone(),
        tokenizer: tokenizer.clone(),
        possible_matches: possible_matches_by_state.clone(),
        state_to_internal_tsid: id_map.tokenizer_states.original_to_internal.clone(),
        internal_tsid_to_states: id_map.tokenizer_states.internal_to_originals.clone(),
        eos_token_id: vocab.eos_token_id,
        token_bytes: token_bytes.clone(),
    };

    let debug = CompileDebug::from_parts(
        grammar.clone(),
        normalized.clone(),
        glr_grammar.clone(),
        table.clone(),
        AutomataDebug {
            characterizations,
            terminal_dwa: terminal_dwa.clone(),
            terminal_debug: TerminalDebug {
                nwa_after_build: terminal_dwa.nwa.clone(),
                nwa_after_collapse: terminal_dwa.nwa.clone(),
            },
            templates,
            parser_nwa_before_resolve: NWA::new(0, 0),
            parser_nwa_after_resolve: NWA::new(0, 0),
            parser_dwa_pre_minimize: parser_dwa.clone(),
            parser_dwa: parser_dwa.clone(),
            id_map: id_map.clone(),
        },
        vocab_entries,
        vocab.eos_token_id,
    );

    (constraint, debug)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::model::tests::*;

    fn mask_has_token(mask: &[u32], token: u32) -> bool {
        let word = token as usize / 32;
        let bit = token as usize % 32;
        word < mask.len() && (mask[word] & (1u32 << bit)) != 0
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

            assert_eq!(
                actual,
                expected,
                "possible_matches union should equal all tokenizer-reachable tokens for state {}",
                tokenizer_state
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
    fn test_end_to_end_simple_ab() {
        
        let gdef = simple_ab_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        state
            .commit_token(0);
        assert!(
            !state.is_finished(),
            "not yet accepting after 'a'"
        );

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1);
        assert!(state.is_finished(), "should accept after 'ab'");
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
            .commit_token(0);
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
            .commit_token(0);
        assert!(
            !state.is_finished(),
            "not yet accepting after 'a'"
        );

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1);
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
            .commit_token(0);
        assert!(!state.is_finished(), "not accepting after 'a'");

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1);
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

        state.commit_token(0);

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        state.commit_token(1);

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        state.commit_token(2);
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

        state.commit_token(0);

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        state.commit_token(1);

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        state.commit_token(2);
        assert!(state.is_finished(), "should accept after 'abc'");
    }

    // ── Nullable terminal expansion tests ───────────────────────────────────

    #[test]
    fn test_expand_nullable_terminals_no_nullables() {
        let gdef = simple_ab_grammar();
        let nullable = std::collections::BTreeSet::new();
        let expanded = expand_nullable_terminals(&gdef, &nullable);
        assert_eq!(expanded.rules.len(), gdef.rules.len());
        assert_eq!(expanded.rules[0].rhs, gdef.rules[0].rhs);
    }

    #[test]
    fn test_expand_nullable_terminals_single_nullable() {
        // Grammar: S → t0 t1, where t0 is nullable.
        // Expected expansion: S → t0 t1 | t1
        let gdef = simple_ab_grammar(); // S → T0 T1
        let nullable = std::collections::BTreeSet::from([0u32]);
        let expanded = expand_nullable_terminals(&gdef, &nullable);

        assert_eq!(expanded.rules.len(), 2, "should have original + one expanded variant");
        let rhs_set: std::collections::BTreeSet<Vec<Symbol>> =
            expanded.rules.iter().map(|r| r.rhs.clone()).collect();
        assert!(rhs_set.contains(&vec![Symbol::Terminal(0), Symbol::Terminal(1)]));
        assert!(rhs_set.contains(&vec![Symbol::Terminal(1)]));
    }

    #[test]
    fn test_expand_nullable_terminals_both_nullable_no_epsilon() {
        // Grammar: S → t0 t1, where both are nullable.
        // Expected: S → t0 t1 | t0 | t1  (no ε-rule)
        let gdef = simple_ab_grammar();
        let nullable = std::collections::BTreeSet::from([0u32, 1u32]);
        let expanded = expand_nullable_terminals(&gdef, &nullable);

        assert_eq!(expanded.rules.len(), 4, "should be 4 variants including ε-rule");
        let rhs_set: std::collections::BTreeSet<Vec<Symbol>> =
            expanded.rules.iter().map(|r| r.rhs.clone()).collect();
        assert!(rhs_set.contains(&vec![Symbol::Terminal(0), Symbol::Terminal(1)]));
        assert!(rhs_set.contains(&vec![Symbol::Terminal(0)]));
        assert!(rhs_set.contains(&vec![Symbol::Terminal(1)]));
        // ε-rule IS present (nullable nonterminals are left for downstream handling).
        assert!(rhs_set.contains(&Vec::<Symbol>::new()));
    }

    #[test]
    fn test_expand_nullable_terminals_cascading_nonterminal() {
        // Grammar: S → A t1, A → t0. If t0 is nullable:
        //   - A → t0 and A → ε (nullable nonterminal left in grammar)
        //   - S → A t1 unchanged (only terminal positions expanded, not NT positions)
        let gdef = two_nt_grammar(); // S → N1 T1, N1 → T0
        let nullable = std::collections::BTreeSet::from([0u32]);
        let expanded = expand_nullable_terminals(&gdef, &nullable);

        // S → N1 T1 only (no nonterminal cascade expansion).
        let s_rules: Vec<&Rule> = expanded.rules.iter().filter(|r| r.lhs == 0).collect();
        let n1_rules: Vec<&Rule> = expanded.rules.iter().filter(|r| r.lhs == 1).collect();
        assert_eq!(s_rules.len(), 1, "S should have 1 variant (no NT cascade)");
        assert_eq!(n1_rules.len(), 2, "N1 should have 2 variants: t0 and ε");
        let n1_rhs_set: std::collections::BTreeSet<Vec<Symbol>> =
            n1_rules.iter().map(|r| r.rhs.clone()).collect();
        assert!(n1_rhs_set.contains(&vec![Symbol::Terminal(0)]));
        assert!(n1_rhs_set.contains(&Vec::<Symbol>::new()), "N1 → ε should be present");
    }

    #[test]
    fn test_expand_nullable_terminals_dedup() {
        // Grammar: S → t0 t0, where t0 is nullable.
        // Power-set would produce: t0 t0, t0 (pos 0 removed), t0 (pos 1 removed), ε.
        // After dedup: t0 t0, t0 (one copy), ε.
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
        };
        let nullable = std::collections::BTreeSet::from([0u32]);
        let expanded = expand_nullable_terminals(&gdef, &nullable);
        assert_eq!(expanded.rules.len(), 3, "should be 3 after dedup: [t0 t0], [t0], and [ε]");
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

        let nullable = tok.drain_nullable_terminals();
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
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"b".to_vec(),
                },
            ],
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
        let mut state = constraint.start();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 1), "'b' should be allowed initially (opt_a is nullable)");
    }
}
