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

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::{DWA, DWAState};
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{Action, GLRTable};
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
use crate::compiler::grammar::model::{NonterminalID, TerminalID};
use crate::compiler::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::compiler::stages::templates::Templates;
use crate::compiler::terminal_dwa::build_terminal_dwa;
use crate::Vocab;
use crate::ds::weight::Weight;

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

    fn visit_cycle_path(
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
                    if let Some(cs) = visit_cycle_path(v, states, non_accepting, color, parent) {
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
    if let Some(cycle_start) = visit_cycle_path(start, &dwa.states, &non_accepting, &mut color, &mut parent) {
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

/// Build the full parser DWA by constructing the parser NWA,
/// resolving negatives, determinizing, and minimizing.
pub fn build_parser_dwa(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> DWA {
    use std::time::Instant;

    let t = Instant::now();
    let characterizations = characterize_terminals(table, grammar);
    let characterized_terminals: BTreeSet<TerminalID> =
        characterizations.keys().copied().collect();
    eprintln!("[glrmask::dwa]   characterize:  {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let terminal_dwa = build_terminal_dwa(grammar, tokenizer, vocab, id_map);
    eprintln!("[glrmask::dwa]   terminal_dwa:  {:.3}s ({} nwa states)", t.elapsed().as_secs_f64(), terminal_dwa.nwa.states.len());
    debug_assert_eq!(terminal_dwa.nwa.start_states.len(), terminal_dwa.tsid_roots.len());

    let t = Instant::now();
    let templates = Templates::from_characterizations(&characterizations);
    eprintln!("[glrmask::dwa]   templates:     {:.3}s ({} terminals)", t.elapsed().as_secs_f64(), templates.by_terminal.len());

    let t = Instant::now();
    let terminal_weights: BTreeMap<TerminalID, Weight> = BTreeMap::new();
    let mut nwa = templates.build_bundle(&terminal_weights);
    eprintln!("[glrmask::dwa]   compose NWA:   {:.3}s ({} states, {} transitions)", t.elapsed().as_secs_f64(), nwa.states.len(), nwa.num_transitions());

    let t = Instant::now();
    resolve_negative_codes_in_nwa(&mut nwa);
    eprintln!("[glrmask::dwa]   resolve neg:   {:.3}s", t.elapsed().as_secs_f64());

    let t_build = Instant::now();
    eprintln!("[glrmask::dwa] NWA build:      {:.3}s ({} states, {} transitions)",
        t_build.elapsed().as_secs_f64(), nwa.states.len(),
        nwa.states.iter().map(|s| s.transitions.len()).sum::<usize>());

    let t = Instant::now();
    let dwa = determinize(&nwa).unwrap();
    eprintln!("[glrmask::dwa] Determinize:    {:.3}s ({} states)", t.elapsed().as_secs_f64(), dwa.num_states());

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

    let t = Instant::now();
    if dwa.is_acyclic() {
        let result = minimize(&dwa);
        eprintln!("[glrmask::dwa] Minimize:       {:.3}s ({} → {} states)", t.elapsed().as_secs_f64(), dwa.num_states(), result.num_states());
        result
    } else {
        eprintln!("[glrmask::dwa] Minimize:       skipped (cyclic DWA)");
        dwa
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;
    use crate::Vocab;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::GrammarDef;
    use crate::compiler::grammar::model::tests::*;

    fn make_vocab_and_preprocessing(
        gdef: &GrammarDef,
    ) -> (Vocab, Tokenizer, InternalIdMap) {
        let tok = Tokenizer::from_grammar_def(gdef);
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
    fn test_build_parser_dwa_simple() {
        let gdef = simple_ab_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &tok, &vocab, &vp);
        assert!(dwa.num_states() > 0);
    }

    #[test]
    fn test_build_parser_dwa_choice() {
        let gdef = choice_grammar(); // S → a | b
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &tok, &vocab, &vp);
        assert!(dwa.num_states() > 0);
    }
}
