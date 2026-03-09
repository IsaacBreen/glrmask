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
use crate::compiler::glr::analysis::{AnalyzedGrammar, normalize_grammar};
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
            Terminal::Pattern { pattern, utf8, .. } => parse_regex(pattern, *utf8),
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

/// Rewrite grammar rules so that nullable terminals (those matching the empty
/// string) are treated as optional.  Operates in place on owned rule data.
///
/// For each nullable terminal `T`, a fresh nonterminal `NT` is allocated
/// (consuming from `*next_nt`) with two productions: `NT → ε` and `NT → T`.
/// Every occurrence of `T` in the existing rules is replaced by `NT`.  The
/// tokenizer's start-state finalizer for `T` is assumed to already be drained
/// before this function is called.
pub(crate) fn expand_nullable_terminals(
    rules: &mut Vec<Rule>,
    next_nt: &mut u32,
    nullable_terminals: &std::collections::BTreeSet<TerminalID>,
) {
    if nullable_terminals.is_empty() {
        return;
    }

    // Map: nullable terminal id → fresh nonterminal id.
    let mut nt_for_terminal = std::collections::BTreeMap::<TerminalID, NonterminalID>::new();
    let mut extra_rules = Vec::new();

    for &tid in nullable_terminals {
        let fresh_nt = *next_nt;
        *next_nt += 1;
        nt_for_terminal.insert(tid, fresh_nt);

        // NT → ε
        extra_rules.push(Rule {
            lhs: fresh_nt,
            rhs: vec![],
        });
        // NT → T
        extra_rules.push(Rule {
            lhs: fresh_nt,
            rhs: vec![Symbol::Terminal(tid)],
        });
    }

    // Rewrite existing rules in place: replace nullable Terminal(T) with Nonterminal(NT).
    for rule in rules.iter_mut() {
        for sym in rule.rhs.iter_mut() {
            if let Symbol::Terminal(tid) = sym {
                if let Some(&nt) = nt_for_terminal.get(tid) {
                    *sym = Symbol::Nonterminal(nt);
                }
            }
        }
    }

    rules.extend(extra_rules);
}

fn remap_terminal_id(terminal: &Terminal, new_id: TerminalID) -> Terminal {
    match terminal {
        Terminal::Literal { bytes, .. } => Terminal::Literal {
            id: new_id,
            bytes: bytes.clone(),
        },
        Terminal::Pattern { pattern, utf8, .. } => Terminal::Pattern {
            id: new_id,
            pattern: pattern.clone(),
            utf8: *utf8,
        },
        Terminal::Expr { expr, .. } => Terminal::Expr {
            id: new_id,
            expr: expr.clone(),
        },
    }
}

/// Remove terminals that are no longer referenced by any normalized rule and
/// compact the remaining terminal IDs to a dense 0..N-1 range.
pub(crate) fn compact_unused_terminals(
    rules: &mut Vec<Rule>,
    terminals: &[Terminal],
    ignore_terminal: Option<TerminalID>,
) -> (Vec<Terminal>, Option<TerminalID>) {
    let mut used = std::collections::BTreeSet::<TerminalID>::new();
    for rule in rules.iter() {
        for symbol in &rule.rhs {
            if let Symbol::Terminal(terminal_id) = symbol {
                used.insert(*terminal_id);
            }
        }
    }
    if let Some(ignore_terminal) = ignore_terminal {
        used.insert(ignore_terminal);
    }

    let mut remap = std::collections::BTreeMap::<TerminalID, TerminalID>::new();
    let mut compacted = Vec::with_capacity(used.len());

    for old_id in used {
        let new_id = compacted.len() as TerminalID;
        let terminal = terminals.get(old_id as usize).unwrap_or_else(|| {
            panic!("terminal id {} referenced by a rule but missing from grammar.terminals", old_id)
        });
        remap.insert(old_id, new_id);
        compacted.push(remap_terminal_id(terminal, new_id));
    }

    for rule in rules.iter_mut() {
        for symbol in rule.rhs.iter_mut() {
            if let Symbol::Terminal(terminal_id) = symbol {
                *terminal_id = *remap
                    .get(terminal_id)
                    .expect("used terminal must have been assigned a compacted id");
            }
        }
    }

    let remapped_ignore_terminal = ignore_terminal.and_then(|old_id| remap.get(&old_id).copied());

    (compacted, remapped_ignore_terminal)
}

fn prepare_grammar_for_compile(grammar: &GrammarDef) -> (GrammarDef, Tokenizer) {
    // Probe nullability against the original terminal set first; nullable
    // terminals are expanded into optional grammar structure before we compact
    // away any terminals that normalization proves unreachable.
    let mut nullable_probe = build_tokenizer(grammar);
    let nullable_terminals = nullable_probe.drain_nullable_terminals();

    let mut rules = grammar.rules.clone();
    let start = grammar.start;
    let mut next_nt = grammar.num_nonterminals();

    expand_nullable_terminals(&mut rules, &mut next_nt, &nullable_terminals);
    normalize_grammar(&mut rules, start);

    let (terminals, ignore_terminal) = compact_unused_terminals(
        &mut rules,
        &grammar.terminals,
        grammar.ignore_terminal,
    );
    let normalized = GrammarDef {
        rules,
        start,
        terminals,
        ignore_terminal,
    };

    // Build the real tokenizer only from the compacted live terminal set so
    // dead terminals never make it into downstream lexer/parser stages.
    let mut tokenizer = build_tokenizer(&normalized);
    let _ = tokenizer.drain_nullable_terminals();

    (normalized, tokenizer)
}

pub fn compile(grammar: &GrammarDef, vocab: &Vocab) -> Constraint {
    let (normalized, tokenizer) = prepare_grammar_for_compile(grammar);

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
        normalized.ignore_terminal,
    );
    let parser_dwa = build_parser_dwa_from_terminal_dwa(
        &table,
        &glr_grammar,
        &tokenizer,
        &terminal_dwa,
    );

    Constraint {
        parser_dwa,
        table,
        tokenizer,
        ignore_terminal: normalized.ignore_terminal,
        possible_matches: possible_matches_by_state.clone(),
        state_to_internal_tsid: id_map.tokenizer_states.original_to_internal.clone(),
        internal_tsid_to_states: id_map.tokenizer_states.internal_to_originals.clone(),
        eos_token_id: vocab.eos_token_id,
        token_bytes,
    }
}

pub(crate) fn compile_with_debug(grammar: &GrammarDef, vocab: &Vocab) -> (Constraint, CompileDebug) {
    let (normalized, tokenizer) = prepare_grammar_for_compile(grammar);

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
        normalized.ignore_terminal,
    );
    let parser_dwa = build_parser_dwa_from_terminal_dwa(
        &table,
        &glr_grammar,
        &tokenizer,
        &terminal_dwa,
    );

    let vocab_entries: Vec<(u32, Vec<u8>)> = vocab.entries.iter().cloned().collect();
    let token_bytes: std::collections::BTreeMap<u32, Vec<u8>> =
        vocab.entries.iter().cloned().collect();
    let constraint = Constraint {
        parser_dwa: parser_dwa.clone(),
        table: table.clone(),
        tokenizer: tokenizer.clone(),
        ignore_terminal: normalized.ignore_terminal,
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
                nwa_after_build: NWA::new(0, 0),
                nwa_after_collapse: NWA::new(0, 0),
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
            ignore_terminal: None,
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should not be allowed initially");

        state.commit_token(0);
        assert!(
            !state.is_finished(),
            "the shorter literal 'a' should not complete a grammar expecting 'ab'"
        );

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token 'b' should remain allowed as a continuation of the longer literal 'ab'"
        );

        state.commit_token(1);
        assert!(state.is_finished(), "should accept after committing 'ab' byte by byte");
    }

    // ── Nullable terminal expansion tests ───────────────────────────────────

    #[test]
    fn test_expand_nullable_terminals_no_nullables() {
        let gdef = simple_ab_grammar();
        let nullable = std::collections::BTreeSet::new();
        let mut rules = gdef.rules.clone();
        let mut next_nt = gdef.num_nonterminals();
        expand_nullable_terminals(&mut rules, &mut next_nt, &nullable);
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
        let mut next_nt = gdef.num_nonterminals();
        expand_nullable_terminals(&mut rules, &mut next_nt, &nullable);

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
        let mut next_nt = gdef.num_nonterminals();
        expand_nullable_terminals(&mut rules, &mut next_nt, &nullable);

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
        let mut next_nt = gdef.num_nonterminals();
        expand_nullable_terminals(&mut rules, &mut next_nt, &nullable);

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
            ignore_terminal: None,
        };
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        let mut next_nt = gdef.num_nonterminals();
        expand_nullable_terminals(&mut rules, &mut next_nt, &nullable);

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
                    utf8: true,
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"b".to_vec(),
                },
            ],
            ignore_terminal: None,
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

    #[test]
    fn test_compact_unused_terminals_remaps_rules_and_terminal_ids() {
        let terminals = vec![
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
        ];
        let mut rules = vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
        }];

        let (compacted, ignore_terminal) = compact_unused_terminals(&mut rules, &terminals, None);

        assert_eq!(
            rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            "used terminals should be renumbered densely when a dead terminal is removed from the middle"
        );
        assert_eq!(compacted.len(), 2);
        assert_eq!(compacted[0].id(), 0);
        assert_eq!(compacted[1].id(), 1);
        assert_eq!(compacted[0].name(), "a");
        assert_eq!(compacted[1].name(), "b");
        assert_eq!(ignore_terminal, None);
    }

    #[test]
    fn test_compact_unused_terminals_preserves_ignore_terminal_and_remaps_it() {
        let terminals = vec![
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
        ];
        let mut rules = vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0), Symbol::Terminal(3)],
        }];

        let (compacted, ignore_terminal) = compact_unused_terminals(&mut rules, &terminals, Some(2));

        assert_eq!(
            rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            "used terminals should still be renumbered densely when an ignore terminal is retained"
        );
        assert_eq!(compacted.len(), 3);
        assert_eq!(compacted[0].name(), "a");
        assert_eq!(compacted[1].name(), " +");
        assert_eq!(compacted[2].name(), "b");
        assert_eq!(ignore_terminal, Some(1));
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
            ignore_terminal: None,
        };
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"x".to_vec()),
            ],
            None,
        );

        let (constraint, debug) = compile_with_debug(&gdef, &vocab);

        assert_eq!(
            constraint.tokenizer.num_terminals,
            2,
            "the final tokenizer should be built only from the live compacted terminals"
        );
        assert_eq!(debug.normalized_grammar_def.terminals.len(), 2);
        assert_eq!(
            debug
                .normalized_grammar_def
                .terminals
                .iter()
                .map(|terminal| terminal.name())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string()],
            "the dead middle terminal should be absent from the normalized grammar"
        );
        assert_eq!(
            debug.normalized_grammar_def.rules,
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

        state.commit_token(0);
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

        let (constraint, debug) = compile_with_debug(&gdef, &vocab);

        assert_eq!(constraint.ignore_terminal, Some(1));
        assert_eq!(debug.normalized_grammar_def.terminals.len(), 3);
        assert_eq!(debug.normalized_grammar_def.ignore_terminal, Some(1));
        assert_eq!(
            debug
                .normalized_grammar_def
                .terminals
                .iter()
                .map(|terminal| terminal.name())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), " +".to_string(), "b".to_string()],
            "the dead terminal should be removed while the ignore terminal is preserved"
        );
        assert_eq!(
            debug.normalized_grammar_def.rules,
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

        state.commit_token(3);
        assert!(!state.is_finished(), "consuming ignored space plus 'a' should still leave trailing 'b'");

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should no longer be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "ignore-only token ' ' should still be allowed between grammar terminals");
        assert!(mask_has_token(&mask, 2), "token 'b' should be allowed after 'a'");
        assert!(!mask_has_token(&mask, 3), "token ' a' should not be allowed once the grammar expects 'b'");
        assert!(mask_has_token(&mask, 4), "token ' b' should be allowed via ignore+terminal composition after 'a'");

        state.commit_token(4);
        assert!(state.is_finished(), "consuming ignored space plus 'b' should finish the grammar");
    }
}
