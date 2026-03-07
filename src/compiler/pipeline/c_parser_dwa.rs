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
//! 3. **Determinize + minimize** → CompDwa → Dwa.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::{CompDwa, CompDwaState};
use crate::automata::weighted::minimize::minimize_acyclic;
use crate::automata::weighted::nwa::Nwa;
use crate::compiler::glr::grammar::GlrGrammar;
use crate::compiler::glr::table::{Action, GlrTable};
use crate::compiler::grammar_def::{NonterminalId, TerminalId};
use crate::compiler::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::terminal_dwa::build_terminal_dwa;
use crate::compiler::template::{build_template_bundles, build_template_nwa_from_bundles};
use crate::compiler::tokenizer_dfa::TokenizerDfa;
use crate::compiler::vocab_pre::VocabPreprocessing;
use crate::Vocab;

pub use crate::compiler::labels::DEFAULT_LABEL;

#[cfg(test)]
use crate::automata::weighted::weight::Weight;
#[cfg(test)]
use crate::compiler::labels::{encode_negative_label, is_negative_label};

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
// Terminal characterization
// ---------------------------------------------------------------------------

/// Shift: from parser state `from`, terminal T shifts to state `to`.
type InitialShift = (u32, u32);

/// Reduce: from parser state `from`, terminal T reduces rule with
/// `pop_count` states and LHS nonterminal `nt`.
type InitialReduce = (u32, usize, NonterminalId);

/// After reducing to nonterminal `nt_from`, if the revealed state is `revealed`,
/// then goto(revealed, nt_from) = `goto_state`, and from `goto_state`, terminal T
/// can shift to `shift_state`.
type NtEscape = (NonterminalId, u32, u32, u32);

/// After reducing to nonterminal `nt_from`, if the revealed state is `revealed`,
/// goto(revealed, nt_from) = `goto_state`, and from `goto_state`, terminal T
type NtRereduce = (NonterminalId, u32, usize, NonterminalId);

/// Stack pattern characterization for a single terminal.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerminalCharacterization {
    pub shifts: Vec<InitialShift>,
    pub reduces: Vec<InitialReduce>,
    pub nt_escapes: Vec<NtEscape>,
    pub nt_rereduces: Vec<NtRereduce>,
    /// All nonterminals involved in reduce cascades.
    pub all_nts: BTreeSet<NonterminalId>,
}


/// Characterize a single terminal: find all stack patterns that allow it.
pub(crate) fn characterize_terminal(
    table: &GlrTable,
    grammar: &GlrGrammar,
) -> BTreeMap<TerminalId, TerminalCharacterization> {
    let mut result = BTreeMap::new();
    let num_states = table.num_states;

    for t in 0..grammar.num_terminals {
        let mut tc = TerminalCharacterization {
            shifts: Vec::new(),
            reduces: Vec::new(),
            nt_escapes: Vec::new(),
            nt_rereduces: Vec::new(),
            all_nts: BTreeSet::new(),
        };

        // Scan all parser states for actions on terminal t.
        for s in 0..num_states {
            for action in table.actions(s, t) {
                match action {
                    Action::Shift(to) => {
                        tc.shifts.push((s, *to));
                    }
                    Action::Reduce(rule_idx) => {
                        let rule = &table.rules[*rule_idx as usize];
                        let pop_count = rule.rhs.len();
                        let nt = rule.lhs;
                        tc.reduces.push((s, pop_count, nt));
                        tc.all_nts.insert(nt);
                    }
                    Action::Accept => {
                        // Accept is like a shift to a virtual "accept" state.
                        // We handle this in the NWA by adding a final weight.
                    }
                }
            }
        }

        // Compute reduce cascades for each nonterminal.
        // After reducing to nonterminal `nt`, we're at goto(revealed, nt).
        // From there, check what happens with terminal t.
        let mut visited_nts: BTreeSet<NonterminalId> = BTreeSet::new();
        let mut nt_queue: VecDeque<NonterminalId> = tc.all_nts.iter().copied().collect();

        while let Some(nt) = nt_queue.pop_front() {
            if !visited_nts.insert(nt) {
                continue;
            }

            // For each possible revealed state, check goto(revealed, nt).
            for revealed in 0..num_states {
                if let Some(goto_state) = table.goto_target(revealed, nt) {
                    // From goto_state, check actions for terminal t.
                    for action in table.actions(goto_state, t) {
                        match action {
                            Action::Shift(shift_to) => {
                                tc.nt_escapes.push((nt, revealed, goto_state, *shift_to));
                            }
                            Action::Reduce(rule_idx) => {
                                let rule = &table.rules[*rule_idx as usize];
                                let pop2 = rule.rhs.len();
                                let nt2 = rule.lhs;
                                tc.nt_rereduces.push((nt, revealed, pop2, nt2));
                                tc.all_nts.insert(nt2);
                                if !visited_nts.contains(&nt2) {
                                    nt_queue.push_back(nt2);
                                }
                            }
                            Action::Accept => {}
                        }
                    }
                }
            }
        }

        if !tc.shifts.is_empty() || !tc.reduces.is_empty() {
            result.insert(t, tc);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// NWA construction from characterizations
// ---------------------------------------------------------------------------

/// Build the parser NWA from terminal characterizations and vocab preprocessing.
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
    vocab_pre: &VocabPreprocessing,
) -> Nwa {
    use std::time::Instant;

    let t0 = Instant::now();
    let characterizations = characterize_terminal(table, grammar);
    eprintln!("[glrmask::dwa]   characterize:  {:.3}s", t0.elapsed().as_secs_f64());

    // Pre-compute used terminals from characterizations to filter token iteration.
    let characterized_terminals: std::collections::BTreeSet<TerminalId> =
        characterizations.keys().copied().collect();

    let t0 = Instant::now();
    let terminal_dwa = build_terminal_dwa(tokenizer, vocab, vocab_pre, grammar, &characterized_terminals);
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
        vocab_pre.num_tsids,
        vocab_pre.max_token,
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
                    let label_str = if label == crate::compiler::labels::DEFAULT_LABEL {
                        "DEFAULT".to_string()
                    } else if crate::compiler::labels::is_negative_label(label) {
                        format!("neg({})", crate::compiler::labels::negative_to_positive_label(label))
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
                    let label_str = if label == crate::compiler::labels::DEFAULT_LABEL {
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

/// Check if a CompDwa is acyclic (no self-loops or back-edges) over *all* states.
///
/// Used to decide whether `minimize_acyclic` can be applied.
fn is_acyclic(dwa: &CompDwa) -> bool {
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
    fn dfs(u: usize, states: &[CompDwaState], color: &mut [u8]) -> bool {
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
fn find_cycle_in_non_accepting_states(dwa: &CompDwa) -> Option<Vec<usize>> {
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
        states: &[CompDwaState],
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
    vocab_pre: &VocabPreprocessing,
) -> CompDwa {
    let (dwa, _) = build_parser_dwa_impl(table, grammar, tokenizer, vocab, vocab_pre, false);
    dwa
}

/// Build the full parser DWA, returning an [`AutomataDebug`] bundle alongside.
#[allow(dead_code)]
pub fn build_parser_dwa_with_debug(
    table: &GlrTable,
    grammar: &GlrGrammar,
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
) -> (CompDwa, crate::compiler::debug::AutomataDebug) {
    let (dwa, dbg) = build_parser_dwa_impl(table, grammar, tokenizer, vocab, vocab_pre, true);
    (dwa, dbg.expect("debug=true must produce Some"))
}

fn build_parser_dwa_impl(
    table: &GlrTable,
    grammar: &GlrGrammar,
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    capture_debug: bool,
) -> (CompDwa, Option<crate::compiler::debug::AutomataDebug>) {
    use std::time::Instant;

    // --- Step 1: Characterize & build terminal DWA ---
    let t = Instant::now();
    let characterizations = characterize_terminal(table, grammar);
    let characterized_terminals: std::collections::BTreeSet<TerminalId> =
        characterizations.keys().copied().collect();
    eprintln!("[glrmask::dwa]   characterize:  {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let (terminal_dwa, terminal_debug) = if capture_debug {
        use crate::compiler::terminal_dwa::build_terminal_dwa_with_debug;
        let (dwa, dbg) = build_terminal_dwa_with_debug(tokenizer, vocab, vocab_pre, grammar, &characterized_terminals);
        (dwa, Some(dbg))
    } else {
        (build_terminal_dwa(tokenizer, vocab, vocab_pre, grammar, &characterized_terminals), None)
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
        vocab_pre.num_tsids,
        vocab_pre.max_token,
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
    let dwa = determinize(&nwa);
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
        let result = minimize_acyclic(&dwa);
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
            vocab_pre: vocab_pre.clone(),
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
    use crate::Vocab;
    use crate::compiler::glr::grammar::GlrGrammar;
    use crate::compiler::grammar_def::GrammarDef;
    use crate::compiler::grammar_def::tests::*;
    use crate::compiler::tokenizer_dfa::TokenizerDfa;
    use crate::automata::weighted::weight::TokenSet;

    fn make_vocab_and_preprocessing(
        gdef: &GrammarDef,
    ) -> (Vocab, TokenizerDfa, VocabPreprocessing) {
        let tok = TokenizerDfa::from_grammar_def(gdef);
        // Build vocab: one token per terminal pattern.
        let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
        for (i, td) in gdef.terminals.iter().enumerate() {
            entries.push((i as u32, td.name.as_bytes().to_vec()));
        }
        let vocab = Vocab::new(entries, None);
        let vp = VocabPreprocessing::compute(&tok, &vocab, None);
        (vocab, tok, vp)
    }

    #[test]
    fn test_characterize_simple_ab() {
        let gdef = simple_ab_grammar(); // S → a b
        let gg = GlrGrammar::from_grammar_def(&gdef);
        let table = GlrTable::build(&gg);
        let chars = characterize_terminal(&table, &gg);

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
        let mut nwa = Nwa::new(1, 2);
        let start = nwa.add_state();
        let mid = nwa.add_state();
        let end = nwa.add_state();
        nwa.start_states.push(start);

        let w = Weight::from_entries(vec![(0, 0, TokenSet::from_iter([2..=2]))]);
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
