//! Tests for the terminal-ended normal form construction.

use std::collections::BTreeSet;

use super::*;
use crate::glr::analysis::AnalyzedGrammar;
use crate::glr::table::{GLRTable, GlrTableConstruction};
use crate::grammar::flat::Terminal;

fn nt(id: u32) -> Symbol {
    Symbol::Nonterminal(id)
}

fn t(id: u32) -> Symbol {
    Symbol::Terminal(id)
}

fn rule(lhs: u32, rhs: Vec<Symbol>) -> Rule {
    Rule { lhs, rhs }
}

/// Independent finite-word fixed point (same universe semantics as the
/// normalization suite's helper: `max_len` bounds the tested word universe,
/// not derivation depth).
fn bounded_language(rules: &[Rule], start: u32, max_len: usize) -> BTreeSet<Vec<u32>> {
    let n = rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                Symbol::Terminal(_) => None,
            }))
        })
        .chain(std::iter::once(start))
        .max()
        .unwrap_or(0) as usize
        + 1;
    let mut languages = vec![BTreeSet::<Vec<u32>>::new(); n];

    loop {
        let mut changed = false;
        for rule in rules {
            let mut words = BTreeSet::from([Vec::<u32>::new()]);
            for symbol in &rule.rhs {
                let suffixes = match symbol {
                    Symbol::Terminal(terminal) => BTreeSet::from([vec![*terminal]]),
                    Symbol::Nonterminal(nonterminal) => languages[*nonterminal as usize].clone(),
                };
                let mut joined = BTreeSet::new();
                for prefix in &words {
                    for suffix in &suffixes {
                        if prefix.len() + suffix.len() <= max_len {
                            let mut word = prefix.clone();
                            word.extend_from_slice(suffix);
                            joined.insert(word);
                        }
                    }
                }
                words = joined;
            }
            let target = &mut languages[rule.lhs as usize];
            let old_len = target.len();
            target.extend(words);
            changed |= target.len() != old_len;
        }
        if !changed {
            return languages[start as usize].clone();
        }
    }
}

fn grammar(rules: Vec<Rule>, start: u32, num_terminals: u32) -> GrammarDef {
    GrammarDef {
        rules,
        start,
        terminals: (0..num_terminals)
            .map(|id| Terminal::Literal {
                id,
                bytes: vec![b'x'],
            })
            .collect(),
        ..Default::default()
    }
}

fn prepare(rules: &[Rule], start: u32, num_terminals: u32) -> PreparedGrammar {
    let source = grammar(rules.to_vec(), start, num_terminals);
    let limits = CompileLimits::default();
    let mut budget = CompileBudget::new();
    prepare_terminal_ended(&source, &limits, &mut budget).expect("terminal-ended preparation")
}

fn prepare_stats(rules: &[Rule], start: u32, num_terminals: u32) -> (PreparedGrammar, TerminalEndedStats) {
    let source = grammar(rules.to_vec(), start, num_terminals);
    let limits = CompileLimits::default();
    let mut budget = CompileBudget::new();
    prepare_terminal_ended_with_stats(&source, &limits, &mut budget)
        .expect("terminal-ended preparation")
}

/// Source and output must generate the same words up to `max_len`, with
/// `accepts_empty` restoring the epsilon the output omits.
fn assert_language_equal(rules: &[Rule], start: u32, num_terminals: u32, max_len: usize) {
    let prepared = prepare(rules, start, num_terminals);
    certify_terminal_ended(&prepared).expect("certificate");
    let source = bounded_language(rules, start, max_len);
    let mut output = bounded_language(&prepared.rules, prepared.start, max_len);
    if prepared.accepts_empty {
        output.insert(Vec::new());
    }
    assert_eq!(output, source, "rules={rules:?} start={start}");
}

// ── §3.6 worked examples ────────────────────────────────────────────────────

#[test]
fn direct_recursion_positive_closure() {
    let rules = vec![rule(0, vec![nt(1), nt(0)]), rule(0, vec![t(1)]), rule(1, vec![t(0)])];
    let prepared = prepare(&rules, 0, 2);
    assert!(!prepared.accepts_empty);
    assert_eq!(bounded_language(&prepared.rules, 0, 6), bounded_language(&rules, 0, 6));
    assert_language_equal(&rules, 0, 2, 6);
}

#[test]
fn nonlinear_coefficients_are_included_directly() {
    for rules in [
        vec![rule(0, vec![nt(0), nt(0)]), rule(0, vec![t(0)])],
        vec![
            rule(0, vec![nt(0), nt(0)]),
            rule(0, vec![nt(0), t(0), t(1)]),
            rule(0, vec![t(2)]),
        ],
    ] {
        assert_language_equal(&rules, 0, 3, 7);
    }
}

// ── §12.1 cases ─────────────────────────────────────────────────────────────

#[test]
fn opaque_noncommuting_prefixes_preserve_order() {
    let rules = vec![
        rule(0, vec![nt(2), nt(1)]),
        rule(0, vec![t(0)]),
        rule(1, vec![nt(3), nt(0)]),
        rule(1, vec![t(1)]),
        rule(2, vec![t(2), t(3)]),
        rule(3, vec![t(4), t(5)]),
    ];
    assert_language_equal(&rules, 0, 6, 7);
}

#[test]
fn unit_scc_with_exits_preserves_language() {
    for rules in [
        vec![rule(0, vec![nt(1)]), rule(1, vec![nt(0)]), rule(1, vec![nt(2)]), rule(2, vec![t(0)])],
        // Unit SCC with no productive exit: empty language, no panic.
        vec![rule(0, vec![nt(1)]), rule(1, vec![nt(0)])],
        // Long unit DAG.
        vec![
            rule(0, vec![nt(1)]),
            rule(1, vec![nt(2)]),
            rule(2, vec![nt(3)]),
            rule(3, vec![t(0)]),
        ],
        // Multiple unit exits.
        vec![
            rule(0, vec![nt(1)]),
            rule(1, vec![nt(0)]),
            rule(0, vec![nt(2)]),
            rule(0, vec![nt(3)]),
            rule(2, vec![t(0)]),
            rule(3, vec![t(1)]),
        ],
        // Unit SCC involving the start.
        vec![rule(0, vec![nt(1)]), rule(1, vec![nt(0)]), rule(1, vec![t(0), nt(1)])],
    ] {
        assert_language_equal(&rules, 0, 2, 5);
    }
}

#[test]
fn nullable_suffixes_changing_boundaries() {
    let rules = vec![
        rule(0, vec![t(0), nt(0), nt(1)]),
        rule(0, vec![t(1)]),
        rule(1, vec![]),
        rule(1, vec![t(2)]),
    ];
    assert_language_equal(&rules, 0, 3, 6);
}

#[test]
fn empty_language_and_empty_word_cases() {
    for rules in [
        vec![rule(0, vec![])],
        vec![rule(0, vec![nt(0)])],
        vec![rule(0, vec![nt(0), nt(0)])],
        vec![rule(0, vec![]), rule(0, vec![t(0)])],
        vec![],
    ] {
        assert_language_equal(&rules, 0, 1, 5);
    }

    let epsilon_only = prepare(&[rule(0, vec![])], 0, 1);
    assert!(epsilon_only.accepts_empty);
    assert!(epsilon_only.rules.is_empty());

    let both = prepare(&[rule(0, vec![]), rule(0, vec![t(0)])], 0, 1);
    assert!(both.accepts_empty);
    assert_eq!(bounded_language(&both.rules, 0, 4), BTreeSet::from([vec![0]]));

    let empty_language = prepare(&[rule(0, vec![nt(0), nt(0)])], 0, 1);
    assert!(!empty_language.accepts_empty);
    assert!(empty_language.rules.is_empty());
}

// ── §12.2 missing table discriminator grammar ───────────────────────────────

#[test]
fn mixed_right_boundary_left_corner_recursion() {
    // A -> a B ; B -> A b | c. Acyclic right-boundary graph, nested recursion.
    let rules = vec![
        rule(0, vec![t(0), nt(1)]),
        rule(1, vec![nt(0), t(1)]),
        rule(1, vec![t(2)]),
    ];
    assert_language_equal(&rules, 0, 3, 8);
}

// ── Addendum: nullable repetition linearity ─────────────────────────────────

#[test]
fn nullable_repetition_preprocessing_stays_linear() {
    // S -> N N ... N a (40 Ns); N -> epsilon | n. Must not enumerate 2^40
    // omission subsets before the context construction.
    let mut rhs = vec![nt(1); 40];
    rhs.push(t(1));
    let rules = vec![rule(0, rhs), rule(1, vec![]), rule(1, vec![t(0)])];
    let (prepared, stats) = prepare_stats(&rules, 0, 2);
    eprintln!(
        "[terminal-ended] nullable-repetition preprocessed binary rules = {}",
        stats.preprocessed_binary_rules
    );
    assert!(stats.preprocessed_binary_rules < 500, "{}", stats.preprocessed_binary_rules);
    assert!(!prepared.accepts_empty);
    assert_language_equal(&rules, 0, 2, 6);
}

// ── Fuzz witnesses ──────────────────────────────────────────────────────────

const WITNESS_DIR: &str = "/Users/isaacbreen/Projects2/worktrees/fuzz-results-901205/cases";

fn embed_witness(seed: u32) -> (u32, Vec<(i64, Vec<i64>)>) {
    match seed {
        385 => (
            0,
            vec![
                (0, vec![-2]),
                (1, vec![-1]),
                (0, vec![1, -1, -1]),
                (0, vec![1]),
                (0, vec![0, -1, 1]),
                (1, vec![0, 0, 1]),
                (1, vec![]),
                (1, vec![-1, 0]),
            ],
        ),
        315 => (
            0,
            vec![
                (0, vec![-3]),
                (1, vec![-1]),
                (2, vec![-1]),
                (3, vec![-2]),
                (0, vec![0, 3]),
                (0, vec![1]),
                (1, vec![0, 2, 2]),
                (2, vec![]),
                (2, vec![-1, -1]),
                (2, vec![-1]),
                (3, vec![]),
                (3, vec![3, 1, 1]),
                (3, vec![-3, 3]),
            ],
        ),
        218 => (
            2,
            vec![
                (0, vec![-2]),
                (1, vec![-2]),
                (2, vec![-3]),
                (0, vec![-3, 1]),
                (0, vec![-2, 2, 1]),
                (1, vec![-3, 0, 1]),
                (1, vec![]),
                (2, vec![]),
                (2, vec![-1, -3, 1]),
                (2, vec![0, 1]),
            ],
        ),
        2633 => (
            2,
            vec![
                (0, vec![-2]),
                (1, vec![-3]),
                (2, vec![-1]),
                (3, vec![-1]),
                (0, vec![3, 3]),
                (1, vec![2, 2]),
                (1, vec![1, 2, 1]),
                (1, vec![]),
                (2, vec![-2]),
                (2, vec![3]),
                (2, vec![0, -2, 3]),
                (3, vec![0, 1]),
                (3, vec![3]),
                (3, vec![1, -3, 1]),
            ],
        ),
        360 => (
            2,
            vec![
                (0, vec![-2]),
                (1, vec![-1]),
                (2, vec![-2]),
                (0, vec![]),
                (0, vec![1, -3, 2]),
                (0, vec![]),
                (1, vec![1, 0, 0]),
                (1, vec![1, 1]),
                (1, vec![]),
                (2, vec![]),
                (2, vec![2, 0, -1]),
                (2, vec![0, 1]),
            ],
        ),
        494 => (
            0,
            vec![
                (0, vec![-3]),
                (1, vec![-3]),
                (2, vec![-3]),
                (0, vec![2, 1, -2]),
                (0, vec![1]),
                (1, vec![2]),
                (1, vec![1, 2]),
                (2, vec![-1, 0]),
                (2, vec![-2, 1]),
            ],
        ),
        1654 => (
            0,
            vec![
                (0, vec![-2]),
                (1, vec![-3]),
                (2, vec![-2]),
                (0, vec![1, 0]),
                (0, vec![-3, 0]),
                (0, vec![2]),
                (1, vec![-2, 2]),
                (1, vec![0, 0]),
                (2, vec![]),
                (2, vec![2, -3, 0]),
                (2, vec![-3, 1, -3]),
            ],
        ),
        2530 => (
            3,
            vec![
                (0, vec![-3]),
                (1, vec![-1]),
                (2, vec![-1]),
                (3, vec![-3]),
                (0, vec![3]),
                (0, vec![]),
                (0, vec![]),
                (1, vec![]),
                (1, vec![-2, -2, -3]),
                (1, vec![-3, -3]),
                (2, vec![]),
                (2, vec![0, 1, 2]),
                (3, vec![-2, 2, 0]),
                (3, vec![0, -1, 0]),
                (3, vec![2, -2, 0]),
            ],
        ),
        other => panic!("no embedded witness for seed {other}"),
    }
}

/// Minimal scanner for the generated `input.json` shape. Only used to
/// cross-check the embedded witnesses on a machine that has the corpus.
fn parse_json_int(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    while *pos < bytes.len() && (bytes[*pos] as char).is_ascii_whitespace() {
        *pos += 1;
    }
    let start = *pos;
    if *pos < bytes.len() && bytes[*pos] == b'-' {
        *pos += 1;
    }
    let digits = *pos;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos == digits {
        return None;
    }
    std::str::from_utf8(&bytes[start..*pos]).ok()?.parse().ok()
}

fn read_witness_file(seed: u32) -> Option<(u32, Vec<(i64, Vec<i64>)>)> {
    let dir = std::fs::read_dir(WITNESS_DIR).ok()?;
    let prefix = format!("{seed}-");
    let mut path = None;
    for entry in dir.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(&prefix) {
            let candidate = entry.path().join("input.json");
            if candidate.exists() {
                path = Some(candidate);
                break;
            }
        }
    }
    let text = std::fs::read(path?).ok()?;
    let bytes = text.as_slice();

    let start_key: &[u8] = b"\"start\"";
    let start_at = bytes.windows(start_key.len()).position(|w| w == start_key)?;
    let mut pos = start_at + start_key.len();
    while pos < bytes.len() && bytes[pos] != b':' {
        pos += 1;
    }
    pos += 1;
    let start = parse_json_int(bytes, &mut pos)? as u32;

    let rules_key: &[u8] = b"\"rules\"";
    let rules_at = bytes.windows(rules_key.len()).position(|w| w == rules_key)?;
    let mut pos = rules_at + rules_key.len();
    while pos < bytes.len() && bytes[pos] != b'[' {
        pos += 1;
    }
    pos += 1;
    let mut rules = Vec::new();
    loop {
        while pos < bytes.len() && bytes[pos] != b'[' && bytes[pos] != b']' {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] == b']' {
            break;
        }
        pos += 1; // outer '['
        let lhs = parse_json_int(bytes, &mut pos)?;
        while pos < bytes.len() && bytes[pos] != b'[' {
            pos += 1;
        }
        pos += 1; // rhs '['
        let mut rhs = Vec::new();
        loop {
            while pos < bytes.len() && bytes[pos] != b']' && bytes[pos] != b'-' && !bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            if pos >= bytes.len() || bytes[pos] == b']' {
                pos += 1;
                break;
            }
            rhs.push(parse_json_int(bytes, &mut pos)?);
        }
        // Close the pair.
        while pos < bytes.len() && bytes[pos] != b']' {
            pos += 1;
        }
        pos += 1;
        rules.push((lhs, rhs));
    }
    Some((start, rules))
}

fn witness(seed: u32) -> (Vec<Rule>, u32, u32) {
    let (start, raw) = embed_witness(seed);
    let mut rules = Vec::new();
    let mut max_terminal = 0u32;
    for (lhs, rhs) in raw {
        let rhs = rhs
            .into_iter()
            .map(|value| {
                if value < 0 {
                    let terminal = (-value - 1) as u32;
                    max_terminal = max_terminal.max(terminal);
                    Symbol::Terminal(terminal)
                } else {
                    Symbol::Nonterminal(value as u32)
                }
            })
            .collect();
        rules.push(Rule {
            lhs: lhs as u32,
            rhs,
        });
    }
    (rules, start, max_terminal + 1)
}

const WITNESS_SEEDS: [u32; 8] = [315, 218, 2633, 360, 494, 1654, 2530, 385];

#[test]
fn embedded_witnesses_match_the_corpus_files() {
    if std::fs::read_dir(WITNESS_DIR).is_err() {
        eprintln!("witness corpus {WITNESS_DIR} not present; skipping cross-check");
        return;
    }
    for seed in WITNESS_SEEDS {
        let embedded = embed_witness(seed);
        let from_file = read_witness_file(seed)
            .unwrap_or_else(|| panic!("witness file for seed {seed}"));
        assert_eq!(embedded.0, from_file.0, "start mismatch for seed {seed}");
        assert_eq!(
            embedded.1, from_file.1,
            "rule mismatch for seed {seed}; update the embedded witness"
        );
    }
}

#[test]
fn fuzz_witnesses_preserve_language() {
    for seed in WITNESS_SEEDS {
        let (rules, start, num_terminals) = witness(seed);
        assert_language_equal(&rules, start, num_terminals, 6);
    }
}

// ── Randomized small grammars ───────────────────────────────────────────────

#[test]
fn random_small_grammars_preserve_language() {
    fn next(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed
    }

    let mut seed = 0x7465_726d_5f65_6e64_u64;
    for case in 0..2_000u32 {
        let n = 2 + (next(&mut seed) % 3) as u32; // 2..=4 nonterminals
        let mut rules = Vec::new();
        for lhs in 0..n {
            let alternatives = 1 + next(&mut seed) % 4;
            for _ in 0..alternatives {
                let len = (next(&mut seed) % 4) as usize; // 0..=3
                let mut rhs = Vec::new();
                for _ in 0..len {
                    if next(&mut seed) % 2 == 0 {
                        rhs.push(t((next(&mut seed) % 3) as u32));
                    } else {
                        rhs.push(nt((next(&mut seed) % n as u64) as u32));
                    }
                }
                rules.push(rule(lhs, rhs));
            }
        }
        let start = (next(&mut seed) % n as u64) as u32;
        let prepared = prepare(&rules, start, 3);
        certify_terminal_ended(&prepared).unwrap_or_else(|error| {
            panic!("certificate failed for case {case}: {error:?}\nrules={rules:?}")
        });
        let source = bounded_language(&rules, start, 5);
        let mut output = bounded_language(&prepared.rules, prepared.start, 5);
        if prepared.accepts_empty {
            output.insert(Vec::new());
        }
        assert_eq!(output, source, "case {case} rules={rules:?} start={start}");
    }
}

// ── Certificate shape ───────────────────────────────────────────────────────

#[test]
fn every_output_rule_is_short_and_terminal_ended() {
    for seed in WITNESS_SEEDS {
        let (rules, start, num_terminals) = witness(seed);
        let prepared = prepare(&rules, start, num_terminals);
        certify_terminal_ended(&prepared).expect("certificate");
        for rule in &prepared.rules {
            assert!((1..=3).contains(&rule.rhs.len()), "{rule:?}");
            assert!(
                matches!(rule.rhs.last(), Some(Symbol::Terminal(_))),
                "{rule:?}"
            );
        }
    }
}

// ── Size table ──────────────────────────────────────────────────────────────

#[test]
fn witness_rule_counts_stay_small() {
    let mut report = Vec::new();
    for seed in [315u32, 218, 2633, 385] {
        let (rules, start, num_terminals) = witness(seed);
        let (prepared, stats) = prepare_stats(&rules, start, num_terminals);
        report.push((seed, prepared.rules.len(), stats));
    }
    for (seed, rules, stats) in &report {
        eprintln!(
            "[terminal-ended] seed={seed} rules={rules} pairs={} candidates={} binary={}",
            stats.demanded_pairs, stats.candidate_work, stats.preprocessed_binary_rules,
        );
    }
    let seed_315 = report.iter().find(|(seed, _, _)| *seed == 315).unwrap().1;
    assert!(seed_315 <= 400, "seed 315 produced {seed_315} rules");
}

// ── Table certificate ───────────────────────────────────────────────────────

fn build_table(rules: &[Rule], start: u32, num_terminals: u32, construction: GlrTableConstruction) -> GLRTable {
    let source = grammar(rules.to_vec(), start, num_terminals);
    let analyzed = AnalyzedGrammar::from_grammar_def(&source);
    GLRTable::build_with_default_construction(&analyzed, construction)
}

#[test]
fn goto_target_certificate_holds_for_witnesses() {
    let mut failures = Vec::new();
    for seed in WITNESS_SEEDS {
        let (rules, start, num_terminals) = witness(seed);
        let prepared = prepare(&rules, start, num_terminals);
        if prepared.rules.is_empty() {
            continue;
        }
        for construction in [GlrTableConstruction::Lalr, GlrTableConstruction::ExperimentalCoreMerged] {
            let table = build_table(&prepared.rules, prepared.start, num_terminals, construction);
            if let Err(error) = table.certify_no_ordinary_goto_target_reduces() {
                failures.push((seed, construction, error));
            }
        }
    }
    for (seed, construction, error) in &failures {
        eprintln!("[terminal-ended] goto certificate FAILED seed={seed} construction={construction:?}: {error}");
    }
    assert!(
        failures.is_empty(),
        "goto-state certificate failed for {} table(s)",
        failures.len()
    );
}

#[test]
fn goto_target_certificate_holds_for_the_discriminator() {
    let rules = vec![
        rule(0, vec![t(0), nt(1)]),
        rule(1, vec![nt(0), t(1)]),
        rule(1, vec![t(2)]),
    ];
    let prepared = prepare(&rules, 0, 3);
    for construction in [GlrTableConstruction::Lalr, GlrTableConstruction::ExperimentalCoreMerged] {
        let table = build_table(&prepared.rules, prepared.start, 3, construction);
        table
            .certify_no_ordinary_goto_target_reduces()
            .unwrap_or_else(|error| panic!("{construction:?}: {error}"));
    }
}
