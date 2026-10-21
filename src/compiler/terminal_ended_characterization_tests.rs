//! Native template characterization over terminal-ended tables.
//!
//! The §5.3 local goto-state certificate is checked on the `GLRTable` itself in
//! `glrmask-glr`. This suite drives the actual native characterization
//! (`glrmask-parser-dwa`) on the same tables and asserts that the nonterminal
//! re-reduction relation is empty — the strongest form of the design's claim
//! that the terminal-ended route needs no HLT/HLR cleanup.
//!
//! The fuzz witnesses live outside the repository. When their corpus directory
//! is present the test reads `input.json` with `serde_json`; otherwise it still
//! exercises the built-in worked examples so the characterization path is
//! always covered.

use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{GLRTable, GlrTableConstruction};
use crate::compiler::glr::terminal_ended::{
    prepare_terminal_ended, CompileBudget, CompileLimits, PreparedGrammar,
};
use crate::compiler::stages::templates::characterize::{
    characterize_completion_domain, try_characterize_selected_terminals_for_terminal_count,
};
use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

const WITNESS_DIR: &str = "/Users/isaacbreen/Projects2/worktrees/fuzz-results-901205/cases";

fn nt(id: u32) -> Symbol {
    Symbol::Nonterminal(id)
}

fn t(id: u32) -> Symbol {
    Symbol::Terminal(id)
}

fn rule(lhs: u32, rhs: Vec<Symbol>) -> Rule {
    Rule { lhs, rhs }
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

fn assert_characterization_clean(
    label: &str,
    prepared: &PreparedGrammar,
    num_terminals: u32,
) -> Result<(), String> {
    if prepared.rules.is_empty() {
        return Ok(());
    }
    let source = grammar(prepared.rules.clone(), prepared.start, num_terminals);
    let analyzed = AnalyzedGrammar::from_grammar_def(&source);
    for construction in [
        GlrTableConstruction::Lalr,
        GlrTableConstruction::ExperimentalCoreMerged,
    ] {
        let table = GLRTable::build_with_default_construction(&analyzed, construction);
        let selected = vec![true; num_terminals as usize];
        let characterizations = try_characterize_selected_terminals_for_terminal_count(
            &table,
            num_terminals,
            &selected,
        )
        .map_err(|error| format!("{label} {construction:?}: characterization failed: {error}"))?;
        for (terminal, characterization) in &characterizations {
            if !characterization.nt_rereduces.is_empty() {
                return Err(format!(
                    "{label} {construction:?} terminal {terminal}: {} NtRereduce edges",
                    characterization.nt_rereduces.len()
                ));
            }
            if let Some(cycle) = characterization.find_cycle() {
                return Err(format!(
                    "{label} {construction:?} terminal {terminal}: NT cycle {cycle:?}"
                ));
            }
        }
        let completion = characterize_completion_domain(&table)
            .map_err(|error| format!("{label} {construction:?}: completion domain: {error}"))?;
        if let Some(cycle) = completion.find_cycle() {
            return Err(format!(
                "{label} {construction:?}: completion NT cycle {cycle:?}"
            ));
        }
    }
    Ok(())
}

#[test]
fn built_in_terminal_ended_tables_have_no_nt_rereduce() {
    let cases: Vec<(&str, Vec<Rule>, u32, u32)> = vec![
        (
            "s_ss_a",
            vec![rule(0, vec![nt(0), nt(0)]), rule(0, vec![t(0)])],
            0,
            1,
        ),
        (
            "s_ss_asb_c",
            vec![
                rule(0, vec![nt(0), nt(0)]),
                rule(0, vec![nt(0), t(0), t(1)]),
                rule(0, vec![t(2)]),
            ],
            0,
            3,
        ),
        (
            "mixed_right_boundary_left_corner",
            vec![
                rule(0, vec![t(0), nt(1)]),
                rule(1, vec![nt(0), t(1)]),
                rule(1, vec![t(2)]),
            ],
            0,
            3,
        ),
        (
            "opaque_prefixes",
            vec![
                rule(0, vec![nt(2), nt(1)]),
                rule(0, vec![t(0)]),
                rule(1, vec![nt(3), nt(0)]),
                rule(1, vec![t(1)]),
                rule(2, vec![t(2), t(3)]),
                rule(3, vec![t(4), t(5)]),
            ],
            0,
            6,
        ),
    ];
    for (label, rules, start, num_terminals) in cases {
        let prepared = prepare(&rules, start, num_terminals);
        assert_characterization_clean(label, &prepared, num_terminals)
            .unwrap_or_else(|error| panic!("{error}"));
    }
}

fn read_witness(seed: u32) -> Option<(Vec<Rule>, u32, u32)> {
    let entries = std::fs::read_dir(WITNESS_DIR).ok()?;
    let prefix = format!("{seed}-");
    let mut path = None;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with(&prefix) {
            let candidate = entry.path().join("input.json");
            if candidate.exists() {
                path = Some(candidate);
                break;
            }
        }
    }
    let bytes = std::fs::read(path?).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let start = value.get("start")?.as_u64()? as u32;
    let raw_rules = value.get("rules")?.as_array()?;
    let mut rules = Vec::new();
    let mut max_terminal = 0u32;
    for raw in raw_rules {
        let pair = raw.as_array()?;
        let lhs = pair.first()?.as_i64()? as u32;
        let rhs = pair.get(1)?.as_array()?;
        let rhs = rhs
            .iter()
            .map(|symbol| {
                let value = symbol.as_i64().expect("witness symbol is an integer");
                if value < 0 {
                    let terminal = (-value - 1) as u32;
                    max_terminal = max_terminal.max(terminal);
                    Symbol::Terminal(terminal)
                } else {
                    Symbol::Nonterminal(value as u32)
                }
            })
            .collect();
        rules.push(Rule { lhs, rhs });
    }
    Some((rules, start, max_terminal + 1))
}

#[test]
fn witness_tables_have_no_nt_rereduce() {
    if std::fs::read_dir(WITNESS_DIR).is_err() {
        eprintln!("witness corpus {WITNESS_DIR} not present; skipping");
        return;
    }
    let mut checked = 0usize;
    for seed in [315u32, 218, 2633, 360, 494, 1654, 2530, 385] {
        let (rules, start, num_terminals) =
            read_witness(seed).unwrap_or_else(|| panic!("missing witness {seed}"));
        let prepared = prepare(&rules, start, num_terminals);
        assert_characterization_clean(&format!("seed {seed}"), &prepared, num_terminals)
            .unwrap_or_else(|error| panic!("{error}"));
        checked += 1;
    }
    assert_eq!(checked, 8);
}
