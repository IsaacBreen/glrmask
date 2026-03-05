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
//! 2. **Build NWA** from characterizations (labels = parser state IDs, weights = token sets).
//! 3. **Determinize + minimize** → CompDwa → Dwa.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::{CompDwa, CompDwaState};
use crate::automata::weighted::minimize::minimize_acyclic;
use crate::automata::weighted::nwa::Nwa;
use crate::automata::weighted::weight::Weight;
use crate::compiler::glr::grammar::GlrGrammar;
use crate::compiler::glr::table::{Action, GlrTable};
use crate::compiler::grammar_def::{NonterminalId, TerminalId};
use crate::compiler::vocab_pre::VocabPreprocessing;
use crate::ds::rangeset::RangeSet;

/// A special label that matches any parser state ("wildcard" / "default").
/// No longer used in NWA construction (expanded to explicit transitions),
/// but kept for reference and potential use in runtime DEFAULT handling.
#[allow(dead_code)]
pub const DEFAULT_LABEL: i32 = i32::MAX - 1;

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
/// causes another reduction with `pop_count` and `nt_to`.
type NtRereduce = (NonterminalId, u32, usize, NonterminalId);

/// Stack pattern characterization for a single terminal.
#[derive(Debug, Clone)]
struct TerminalCharacterization {
    shifts: Vec<InitialShift>,
    reduces: Vec<InitialReduce>,
    nt_escapes: Vec<NtEscape>,
    nt_rereduces: Vec<NtRereduce>,
    /// All nonterminals involved in reduce cascades.
    all_nts: BTreeSet<NonterminalId>,
}

/// Characterize a single terminal: find all stack patterns that allow it.
fn characterize_terminal(
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

/// Compute all valid revealed states for a nonterminal, following rereduce chains.
///
/// A revealed state `r` is valid for nonterminal `nt` if, starting from `nt`,
/// there exists a chain of rereduces (with pop_count=1, which re-reveal the same
/// state) that eventually leads to an nt_escape at `r`.
fn compute_valid_revealed(start_nt: NonterminalId, tc: &TerminalCharacterization) -> BTreeSet<u32> {
    let mut valid = BTreeSet::new();
    let mut visited: BTreeSet<(NonterminalId, u32)> = BTreeSet::new();
    let mut queue: VecDeque<(NonterminalId, u32)> = VecDeque::new();

    // Seed: direct escapes from start_nt.
    for &(n, r, _, _) in &tc.nt_escapes {
        if n == start_nt {
            valid.insert(r);
        }
    }

    // Seed: rereduces with pop_count=1 from start_nt.
    for &(n, r, pop2, nt_to) in &tc.nt_rereduces {
        if n == start_nt && pop2 == 1 {
            queue.push_back((nt_to, r));
        }
    }

    while let Some((nt, r)) = queue.pop_front() {
        if !visited.insert((nt, r)) {
            continue;
        }

        // Check if (nt, r) has a direct escape.
        for &(n, er, _, _) in &tc.nt_escapes {
            if n == nt && er == r {
                valid.insert(r);
            }
        }

        // Follow rereduces with pop_count=1 at (nt, r).
        for &(n, rr, pop2, nt_to) in &tc.nt_rereduces {
            if n == nt && rr == r && pop2 == 1 {
                queue.push_back((nt_to, r));
            }
        }
    }

    valid
}

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
/// Instead of using DEFAULT_LABEL (wildcard), self-loop transitions are
/// expanded to explicit transitions for all parser states. This ensures
/// the standard determinizer handles them correctly.
pub fn build_parser_nwa(table: &GlrTable, grammar: &GlrGrammar, vocab: &VocabPreprocessing) -> Nwa {
    let characterizations = characterize_terminal(table, grammar);

    let num_tsids = vocab.num_tsids;
    let max_token = vocab.max_token;
    let num_parser_states = table.num_states;
    let mut nwa = Nwa::new(num_tsids, max_token);
    let max_pos = nwa.max_position();

    // Create the start state and register it.
    let start = nwa.add_state();
    nwa.start_states.push(start);

    let w_all = Weight::all(max_pos, num_tsids);

    // Helper: add a "skip any parser state" self-loop by expanding to all
    // concrete parser state labels. This replaces DEFAULT_LABEL self-loops
    // so the determinizer handles it correctly.
    let add_skip_self_loop = |nwa: &mut Nwa, q: u32, w: &Weight| {
        for s in 0..num_parser_states {
            nwa.add_transition(q, s as i32, q, w.clone());
        }
    };

    // For each characterized terminal, build NWA paths.
    for (&terminal, tc) in &characterizations {
        let token_weight = terminal_to_weight(terminal, vocab);
        if token_weight.is_empty() {
            continue; // No tokens match this terminal.
        }

        // --- Initial shifts ---
        for &(from_state, _to_state) in &tc.shifts {
            let q = nwa.add_state();
            let q_final = nwa.add_state();

            nwa.add_epsilon(start, q, w_all.clone());
            add_skip_self_loop(&mut nwa, q, &w_all);
            nwa.add_transition(q, from_state as i32, q_final, token_weight.clone());
            nwa.set_final_weight(q_final, token_weight.clone());
        }

        // --- Initial reduces ---
        // For each reduce (from_state, pop_count, nt), find all valid revealed
        // states (via nt_escapes and rereduce chains with pop_count=1) and build
        // explicit NWA paths.
        //
        // The NWA reads the stack bottom-to-top. For a reduce that pops `p` states
        // from `from_state`, revealing state `r`, the word pattern is:
        //   [self-loop*, r, any{p-1}, from_state]
        //
        // This means: match arbitrary prefix, then the revealed state, then p-1
        // intermediate states (any label), then from_state at the top.
        for &(from_state, pop_count, nt) in &tc.reduces {
            let valid_revealed = compute_valid_revealed(nt, tc);

            for r in valid_revealed {
                let q = nwa.add_state();
                nwa.add_epsilon(start, q, w_all.clone());
                add_skip_self_loop(&mut nwa, q, &w_all);

                // Read revealed state.
                let mut current = nwa.add_state();
                nwa.add_transition(q, r as i32, current, w_all.clone());

                // Skip pop_count - 1 intermediate states (any label).
                for _ in 0..pop_count.saturating_sub(1) {
                    let next = nwa.add_state();
                    for s in 0..num_parser_states {
                        nwa.add_transition(current, s as i32, next, w_all.clone());
                    }
                    current = next;
                }

                // Read from_state at the top → accept with token weight.
                let final_q = nwa.add_state();
                nwa.add_transition(current, from_state as i32, final_q, token_weight.clone());
                nwa.set_final_weight(final_q, token_weight.clone());
            }
        }
    }

    nwa
}

/// Check if a CompDwa is acyclic (no self-loops or back-edges).
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

/// Convert a terminal ID to a weight: the set of tokens that match this terminal
/// across all TSIDs.
fn terminal_to_weight(terminal: TerminalId, vocab: &VocabPreprocessing) -> Weight {
    let num_tsids = vocab.num_tsids;

    // Collect (tsid, token_set) pairs.
    let mut entries: Vec<(u32, u32, RangeSet)> = Vec::new();

    for tsid in 0..num_tsids {
        if let Some(rs) = vocab.possible_matches[tsid as usize].get(&terminal)
            && !rs.is_empty()
        {
            entries.push((tsid, tsid, rs.clone()));
        }
    }

    if entries.is_empty() {
        Weight::empty(num_tsids)
    } else {
        Weight::from_entries(entries, num_tsids)
    }
}

/// Build the full parser DWA by constructing the NWA, determinizing, and minimizing.
pub fn build_parser_dwa(
    table: &GlrTable,
    grammar: &GlrGrammar,
    vocab: &VocabPreprocessing,
) -> CompDwa {
    let nwa = build_parser_nwa(table, grammar, vocab);

    // Determinize: NWA → CompDwa (general, handles cycles).
    let dwa = determinize(&nwa);

    // Minimize only if the DWA is acyclic. The skip-all-labels self-loops
    // in the NWA produce cyclic DWAs, and minimize_acyclic breaks on cycles.
    if is_acyclic(&dwa) {
        minimize_acyclic(&dwa)
    } else {
        dwa
    }
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
        let vp = VocabPreprocessing::compute(&tok, &vocab);
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
        let (_vocab, _tok, vp) = make_vocab_and_preprocessing(&gdef);

        let nwa = build_parser_nwa(&table, &gg, &vp);
        assert!(nwa.num_states() > 1);
        assert!(nwa.num_transitions() > 0);
    }

    #[test]
    fn test_build_parser_dwa_simple() {
        let gdef = simple_ab_grammar();
        let gg = GlrGrammar::from_grammar_def(&gdef);
        let table = GlrTable::build(&gg);
        let (_vocab, _tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &vp);
        assert!(dwa.num_states() > 0);
    }

    #[test]
    fn test_build_parser_dwa_choice() {
        let gdef = choice_grammar(); // S → a | b
        let gg = GlrGrammar::from_grammar_def(&gdef);
        let table = GlrTable::build(&gg);
        let (_vocab, _tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &vp);
        assert!(dwa.num_states() > 0);
    }
}
