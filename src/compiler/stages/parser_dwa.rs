//! Parser DWA construction.
//!
//! Converts the GLR parse table into a Nondeterministic Weighted Automaton (NWA)
//! that reads parser state stacks bottom-to-top and signals which tokens are
//! valid at each position. Then determinizes and minimizes to get a DWA.
//!
//! # Architecture
//!
//! The DWA labels are parser state IDs (i32). At runtime, the GLR parser's
//! Graph-Structured Stack (GSS) provides the "word" that the DWA reads.
//! The DWA weights encode which LLM tokens are valid.
//!
//! # Algorithm overview
//!
//! 1. **Characterize** each terminal: find all stack patterns that make it valid.
//! 2. **Bundle equivalent templates** so parser-side structure is shared.
//! 3. **Build NWA** from template bundles (labels = parser state IDs, weights = token sets).
//! 3. **Determinize + minimize** → DWA.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet};

use crate::automata::lexer::tokenizer::TokenizerDfa;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::{DWA, DWAState};
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::GlrGrammar;
use crate::compiler::glr::table::{Action, GlrTable};
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
use crate::compiler::grammar::ast::{NonterminalId, TerminalId};
use crate::compiler::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::compiler::stages::templates::compile::{build_template_bundles, build_template_nwa_from_bundles};
use crate::compiler::terminal_dwa::build_terminal_dwa;
use crate::Vocab;

#[cfg(test)]
use crate::compiler::glr::labels::encode_negative_label;
#[cfg(test)]
use crate::ds::weight::Weight;

fn terminals_present_in_terminal_dwa(terminal_dwa: &crate::compiler::terminal_dwa::TerminalDwa) -> BTreeSet<TerminalId> {
    let mut terminals = BTreeSet::new();
    for state in &terminal_dwa.nwa.states {
        for &label in state.transitions.keys() {
            let Ok(terminal) = TerminalId::try_from(label) else {
                continue;
            };
            terminals.insert(terminal);
        }
    }
    terminals
}

// ---------------------------------------------------------------------------
// NWA construction from characterizations
// ---------------------------------------------------------------------------

/// Build the parser NWA from terminal characterizations and internal ID mappings.
///
/// The NWA labels are parser state IDs (non-negative i32).
/// The weights encode which LLM tokens are valid, in TSID × token space.
///
/// The NWA reads the parser stack **bottom-to-top**. For a reduce at `from_state`
/// with pop_count `p` revealing state `revealed`, the stack word is:
///   [...prefix, revealed, skip{p-1}, from_state]
/// where skip{p-1} are the p-1 intermediate states between revealed and from_state.
///
/// Instead of using explicit per-state self-loops (which create O(num_states)
/// transitions and cause the NWA to be cyclic), we use DEFAULT_LABEL as a
/// single wildcard transition.  The determinizer treats DEFAULT as a normal
/// label.  At runtime, the DWA walker falls back to DEFAULT when no specific
/// transition matches.
///
/// NWA patterns:
/// - **Shift**: `skip_start --[DEFAULT]--> skip_start` (self-loop, matches any prefix)
///              `skip_start --[from_state]--> accept(token_weight)`
/// - **Reduce** (pop_count=P, revealed=R, from_state=S):
///              `skip_start --[R]--> mid_0 --[DEFAULT]--> mid_1 ... --[S]--> accept`
#[allow(dead_code)]
pub fn build_parser_nwa(
    table: &GlrTable,
    grammar: &GlrGrammar,
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> NWA {
    use std::time::Instant;

    let t0 = Instant::now();
    let characterizations = characterize_terminals(table, grammar);
    eprintln!("[glrmask::dwa]   characterize:  {:.3}s", t0.elapsed().as_secs_f64());

    // Pre-compute used terminals from characterizations to filter token iteration.
    let characterized_terminals: std::collections::BTreeSet<TerminalId> =
        characterizations.keys().copied().collect();

    let t0 = Instant::now();
    let terminal_dwa = build_terminal_dwa(tokenizer, vocab, id_map, grammar, &characterized_terminals);
    eprintln!("[glrmask::dwa]   terminal_dwa:  {:.3}s ({} nwa states)", t0.elapsed().as_secs_f64(), terminal_dwa.nwa.states.len());
    debug_assert_eq!(terminal_dwa.nwa.start_states.len(), terminal_dwa.tsid_roots.len());

    let t0 = Instant::now();
    let used_terminals = terminals_present_in_terminal_dwa(&terminal_dwa);
    let template_bundles = build_template_bundles(&characterizations, &used_terminals);
    eprintln!("[glrmask::dwa]   bundles:       {:.3}s ({} bundles, {} used terminals)", t0.elapsed().as_secs_f64(), template_bundles.len(), used_terminals.len());

    let t0 = Instant::now();
    let mut nwa = build_template_nwa_from_bundles(
        &template_bundles,
        &terminal_dwa,
        id_map.num_tsids(),
        id_map.max_token_id(),
    );
    eprintln!("[glrmask::dwa]   compose NWA:   {:.3}s ({} states, {} transitions)", t0.elapsed().as_secs_f64(), nwa.states.len(), nwa.num_transitions());

    // Dump NWA before resolve_negatives if GLRMASK_DUMP_DWA is set
    if std::env::var("GLRMASK_DUMP_DWA").unwrap_or_default() == "1" {
        eprintln!("\n=== NWA BEFORE resolve_negatives ({} states, {} transitions) ===", nwa.states.len(), nwa.num_transitions());
        eprintln!("  start_states: {:?}", nwa.start_states);
        for (i, state) in nwa.states.iter().enumerate() {
            let has_content = !state.transitions.is_empty() || !state.epsilons.is_empty() || state.final_weight.is_some();
            if !has_content { continue; }
            eprintln!("  state {}:", i);
            for (&label, targets) in &state.transitions {
                for (dest, _w) in targets {
                    let label_str = if label == DEFAULT_LABEL {
                        "DEFAULT".to_string()
                    } else if is_negative_label(label) {
                        format!("neg({})", negative_to_positive_label(label))
                    } else {
                        format!("{}", label)
                    };
                    eprintln!("    --[{}]--> {}", label_str, dest);
                }
            }
            for (dest, _w) in &state.epsilons {
                eprintln!("    --[eps]--> {}", dest);
            }
            if state.final_weight.is_some() {
                eprintln!("    FINAL");
            }
        }
    }

    let t0 = Instant::now();
    resolve_negative_codes_in_nwa(&mut nwa);
    eprintln!("[glrmask::dwa]   resolve neg:   {:.3}s", t0.elapsed().as_secs_f64());

    // Dump NWA after resolve_negatives if GLRMASK_DUMP_DWA is set
    if std::env::var("GLRMASK_DUMP_DWA").unwrap_or_default() == "1" {
        eprintln!("\n=== NWA AFTER resolve_negatives ({} states, {} transitions) ===", nwa.states.len(), nwa.num_transitions());
        eprintln!("  start_states: {:?}", nwa.start_states);
        for (i, state) in nwa.states.iter().enumerate() {
            let has_content = !state.transitions.is_empty() || !state.epsilons.is_empty() || state.final_weight.is_some();
            if !has_content { continue; }
            eprintln!("  state {}:", i);
            for (&label, targets) in &state.transitions {
                for (dest, _w) in targets {
                    let label_str = if label == DEFAULT_LABEL {
                        "DEFAULT".to_string()
                    } else {
                        format!("{}", label)
                    };
                    eprintln!("    --[{}]--> {}", label_str, dest);
                }
            }
            for (dest, _w) in &state.epsilons {
                eprintln!("    --[eps]--> {}", dest);
            }
            if state.final_weight.is_some() {
                eprintln!("    FINAL");
            }
        }
    }

    nwa
}

/// Check if a `DWA` is acyclic (no self-loops or back-edges) over *all* states.
///
/// Used to decide whether acyclic minimization can be applied.
fn is_acyclic(dwa: &DWA) -> bool {
    let n = dwa.states.len();
    // Quick check: any self-loops?
    for (i, st) in dwa.states.iter().enumerate() {
        for (target, _) in st.transitions.values() {
            if *target as usize == i {
                return false;
            }
        }
    }
    // Full DFS cycle check.
    let mut color = vec![0u8; n]; // 0=white, 1=gray, 2=black
    fn dfs(u: usize, states: &[DWAState], color: &mut [u8]) -> bool {
        color[u] = 1;
        for (target, _) in states[u].transitions.values() {
            let v = *target as usize;
            if v >= color.len() {
                continue;
            }
            match color[v] {
                1 => return false, // back edge → cycle
                0 => {
                    if !dfs(v, states, color) {
                        return false;
                    }
                }
                _ => {}
            }
        }
        color[u] = 2;
        true
    }
    for i in 0..n {
        if color[i] == 0 && !dfs(i, &dwa.states, &mut color) {
            return false;
        }
    }
    true
}

/// Detect cycles in the non-accepting subgraph that are **reachable from the
/// DWA's start state**.
///
/// This checks whether any cycle can actually arise during constrained
/// generation.  Accepting states are treated as sinks: once the DWA enters
/// an accepting state the characterisation is done.  Unreachable states
/// (disconnected from the start state) are excluded, so structural cycles
/// that can never appear in a real execution do not trigger this check.
///
/// Returns `Some(cycle_path)` if a reachable cycle is found (state indices
/// starting and ending at the cycle entry point), `None` if acyclic.
fn find_cycle_in_non_accepting_states(dwa: &DWA) -> Option<Vec<usize>> {
    let n = dwa.states.len();
    let non_accepting: Vec<bool> = dwa.states.iter().map(|s| s.final_weight.is_none()).collect();
    let start = dwa.start_state as usize;
    if start >= n || !non_accepting[start] {
        return None; // start is accepting or out-of-bounds — nothing to check
    }

    let mut color = vec![0u8; n]; // 0=white, 1=gray(on path), 2=black(done)
    let mut parent = vec![usize::MAX; n];

    fn dfs(
        u: usize,
        states: &[DWAState],
        non_accepting: &[bool],
        color: &mut [u8],
        parent: &mut [usize],
    ) -> Option<usize> {
        color[u] = 1;
        for (v, _) in states[u].transitions.values() {
            let v = *v as usize;
            if v >= color.len() || !non_accepting[v] {
                continue; // accepting states are sinks — don't recurse
            }
            match color[v] {
                1 => {
                    parent[v] = u;
                    return Some(v); // back edge → v is the cycle entry
                }
                0 => {
                    parent[v] = u;
                    if let Some(cs) = dfs(v, states, non_accepting, color, parent) {
                        return Some(cs);
                    }
                }
                _ => {}
            }
        }
        color[u] = 2;
        None
    }

    // Only start from `start_state` — this restricts the search to states
    // actually reachable in execution.
    if let Some(cycle_start) = dfs(start, &dwa.states, &non_accepting, &mut color, &mut parent) {
        // Reconstruct cycle: walk parent pointers back to cycle_start.
        let mut path = vec![cycle_start];
        let mut cur = parent[cycle_start];
        while cur != cycle_start && cur != usize::MAX {
            path.push(cur);
            cur = parent[cur];
        }
        path.push(cycle_start);
        path.reverse();
        return Some(path);
    }
    None
}

/// Build the full parser DWA by constructing the NWA, determinizing, and minimizing.
pub fn build_parser_dwa(
    table: &GlrTable,
    grammar: &GlrGrammar,
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> DWA {
    let (dwa, _) = build_parser_dwa_impl(table, grammar, tokenizer, vocab, id_map, false);
    dwa
}

/// Build the full parser DWA, returning an [`AutomataDebug`] bundle alongside.
#[allow(dead_code)]
pub fn build_parser_dwa_with_debug(
    table: &GlrTable,
    grammar: &GlrGrammar,
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> (DWA, crate::compiler::debug::AutomataDebug) {
    let (dwa, dbg) = build_parser_dwa_impl(table, grammar, tokenizer, vocab, id_map, true);
    (dwa, dbg.expect("debug=true must produce Some"))
}

fn build_parser_dwa_impl(
    table: &GlrTable,
    grammar: &GlrGrammar,
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    capture_debug: bool,
) -> (DWA, Option<crate::compiler::debug::AutomataDebug>) {
    use std::time::Instant;

    // --- Step 1: Characterize & build terminal DWA ---
    let t = Instant::now();
    let characterizations = characterize_terminals(table, grammar);
    let characterized_terminals: std::collections::BTreeSet<TerminalId> =
        characterizations.keys().copied().collect();
    eprintln!("[glrmask::dwa]   characterize:  {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let (terminal_dwa, terminal_debug) = if capture_debug {
        use crate::compiler::terminal_dwa::build_terminal_dwa_with_debug;
        let (dwa, dbg) = build_terminal_dwa_with_debug(tokenizer, vocab, id_map, grammar, &characterized_terminals);
        (dwa, Some(dbg))
    } else {
        (build_terminal_dwa(tokenizer, vocab, id_map, grammar, &characterized_terminals), None)
    };
    eprintln!("[glrmask::dwa]   terminal_dwa:  {:.3}s ({} nwa states)", t.elapsed().as_secs_f64(), terminal_dwa.nwa.states.len());
    debug_assert_eq!(terminal_dwa.nwa.start_states.len(), terminal_dwa.tsid_roots.len());

    // --- Step 2: Template bundles ---
    let t = Instant::now();
    let used_terminals = terminals_present_in_terminal_dwa(&terminal_dwa);
    let template_bundles = build_template_bundles(&characterizations, &used_terminals);
    eprintln!("[glrmask::dwa]   bundles:       {:.3}s ({} bundles, {} used terminals)", t.elapsed().as_secs_f64(), template_bundles.len(), used_terminals.len());

    // --- Step 3: Compose NWA ---
    let t = Instant::now();
    let mut nwa = build_template_nwa_from_bundles(
        &template_bundles,
        &terminal_dwa,
        id_map.num_tsids(),
        id_map.max_token_id(),
    );
    eprintln!("[glrmask::dwa]   compose NWA:   {:.3}s ({} states, {} transitions)", t.elapsed().as_secs_f64(), nwa.states.len(), nwa.num_transitions());

    let nwa_before_resolve = if capture_debug { Some(nwa.clone()) } else { None };

    // --- Step 4: Resolve negatives ---
    let t = Instant::now();
    resolve_negative_codes_in_nwa(&mut nwa);
    eprintln!("[glrmask::dwa]   resolve neg:   {:.3}s", t.elapsed().as_secs_f64());

    let nwa_after_resolve = if capture_debug { Some(nwa.clone()) } else { None };

    let t_build = Instant::now();
    eprintln!("[glrmask::dwa] NWA build:      {:.3}s ({} states, {} transitions)",
        t_build.elapsed().as_secs_f64(), nwa.states.len(),
        nwa.states.iter().map(|s| s.transitions.len()).sum::<usize>());

    // --- Step 5: Determinize ---
    let t = Instant::now();
    let dwa = determinize(&nwa).unwrap();
    eprintln!("[glrmask::dwa] Determinize:    {:.3}s ({} states)", t.elapsed().as_secs_f64(), dwa.num_states());

    // Acyclicity check
    if let Some(cycle) = find_cycle_in_non_accepting_states(&dwa) {
        panic!(
            "parser DWA has a graph-reachable cycle in non-accepting states\n\
             cycle path: {:?}\n{}",
            cycle,
            cycle.iter().map(|&s| {
                let st = &dwa.states[s];
                let accepting = if st.final_weight.is_some() { "ACCEPTING" } else { "non-accepting" };
                let edges: Vec<_> = st.transitions.iter().map(|(k, (t, _))| format!("  --[{}]--> {}", k, t)).collect();
                format!("  s{} [{}]:\n{}", s, accepting, edges.join("\n"))
            }).collect::<Vec<_>>().join("\n")
        );
    }

    let dwa_pre_minimize = if capture_debug { Some(dwa.clone()) } else { None };

    // --- Step 6: Minimize ---
    let t = Instant::now();
    let final_dwa = if is_acyclic(&dwa) {
        let result = minimize(&dwa);
        eprintln!("[glrmask::dwa] Minimize:       {:.3}s ({} → {} states)", t.elapsed().as_secs_f64(), dwa.num_states(), result.num_states());
        result
    } else {
        eprintln!("[glrmask::dwa] Minimize:       skipped (cyclic DWA)");
        dwa
    };

    let debug = if capture_debug {
        Some(crate::compiler::debug::AutomataDebug {
            characterizations: characterizations.clone(),
            terminal_dwa,
            terminal_debug: terminal_debug.unwrap(),
            template_bundles,
            parser_nwa_before_resolve: nwa_before_resolve.unwrap(),
            parser_nwa_after_resolve: nwa_after_resolve.unwrap(),
            parser_dwa_pre_minimize: dwa_pre_minimize.unwrap(),
            parser_dwa: final_dwa.clone(),
            id_map: id_map.clone(),
        })
    } else {
        None
    };

    (final_dwa, debug)
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;
    use crate::Vocab;
    use crate::automata::lexer::tokenizer::TokenizerDfa;
    use crate::compiler::glr::analysis::GlrGrammar;
    use crate::compiler::grammar::ast::GrammarDef;
    use crate::compiler::grammar::ast::tests::*;

    fn make_vocab_and_preprocessing(
        gdef: &GrammarDef,
    ) -> (Vocab, TokenizerDfa, InternalIdMap) {
        let tok = TokenizerDfa::from_grammar_def(gdef);
        // Build vocab: one token per terminal pattern.
        let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
        for (i, td) in gdef.terminals.iter().enumerate() {
            entries.push((i as u32, td.name.as_bytes().to_vec()));
        }
        let vocab = Vocab::new(entries, None);
        let id_map = InternalIdMap::build(&tok, &vocab);
        (vocab, tok, id_map)
    }

    #[test]
    fn test_characterize_simple_ab() {
        let gdef = simple_ab_grammar(); // S → a b
        let gg = GlrGrammar::from_grammar_def(&gdef);
        let table = GlrTable::build(&gg);
        let chars = characterize_terminals(&table, &gg);

        // Terminal 'a' (id=0): should have shift from some state.
        assert!(chars.contains_key(&0));
        assert!(!chars[&0].shifts.is_empty());

        // Terminal 'b' (id=1): might have shift or reduce.
        assert!(chars.contains_key(&1));
    }

    #[test]
    fn test_build_parser_nwa_simple() {
        let gdef = simple_ab_grammar();
        let gg = GlrGrammar::from_grammar_def(&gdef);
        let table = GlrTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let nwa = build_parser_nwa(&table, &gg, &tok, &vocab, &vp);
        assert!(nwa.num_states() > 1);
        assert!(nwa.num_transitions() > 0);
        assert!(nwa.states.iter().all(|state| {
            state
                .transitions
                .keys()
                .all(|label| !is_negative_label(*label))
        }));
    }

    #[test]
    fn test_resolve_negative_codes_simple_cancellation() {
        let mut nwa = NWA::new(1, 2);
        let start = nwa.add_state();
        let mid = nwa.add_state();
        let end = nwa.add_state();
        nwa.start_states.push(start);

        let w = Weight::empty();
        nwa.add_transition(start, 0, mid, w.clone());
        nwa.add_transition(mid, encode_negative_label(0), end, w.clone());
        nwa.set_final_weight(end, w.clone());

        resolve_negative_codes_in_nwa(&mut nwa);

        assert!(nwa.states.iter().all(|state| {
            state
                .transitions
                .keys()
                .all(|label| !is_negative_label(*label))
        }));
        assert_eq!(nwa.states[mid as usize].final_weight.as_ref(), Some(&w));
    }

    #[test]
    fn test_build_parser_dwa_simple() {
        let gdef = simple_ab_grammar();
        let gg = GlrGrammar::from_grammar_def(&gdef);
        let table = GlrTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &tok, &vocab, &vp);
        assert!(dwa.num_states() > 0);
    }

    #[test]
    fn test_build_parser_dwa_choice() {
        let gdef = choice_grammar(); // S → a | b
        let gg = GlrGrammar::from_grammar_def(&gdef);
        let table = GlrTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &tok, &vocab, &vp);
        assert!(dwa.num_states() > 0);
    }
}
