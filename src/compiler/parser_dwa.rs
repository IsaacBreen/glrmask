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

use crate::automata::weighted::dwa::CompDwa;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::minimize::minimize_acyclic;
use crate::automata::weighted::nwa::Nwa;
use crate::automata::weighted::weight::Weight;
use crate::compiler::glr::grammar::GlrGrammar;
use crate::compiler::glr::table::{Action, GlrTable};
use crate::compiler::grammar_def::{NonterminalId, TerminalId};
use crate::compiler::vocab_pre::VocabPreprocessing;
use crate::ds::rangeset::RangeSet;

/// A special label that matches any parser state ("wildcard" / "default").
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

/// Build the parser NWA from terminal characterizations and vocab preprocessing.
///
/// The NWA labels are parser state IDs (non-negative i32).
/// DEFAULT_LABEL is used as a wildcard that matches any parser state.
/// The weights encode which LLM tokens are valid, in TSID × token space.
pub fn build_parser_nwa(
    table: &GlrTable,
    grammar: &GlrGrammar,
    vocab: &VocabPreprocessing,
) -> Nwa {
    let characterizations = characterize_terminal(table, grammar);

    let num_tsids = vocab.num_tsids;
    let max_token = vocab.max_token;
    let mut nwa = Nwa::new(num_tsids, max_token);
    let max_pos = nwa.max_position();

    // Start state (already state 0 in Nwa::new).
    let start = 0;

    let w_all = Weight::all(max_pos, num_tsids);

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
            nwa.add_transition(q, DEFAULT_LABEL, q, w_all.clone());
            nwa.add_transition(q, from_state as i32, q_final, token_weight.clone());
            nwa.set_final_weight(q_final, token_weight.clone());
        }

        // --- Initial reduces + NT escapes/rereduces ---
        let mut nt_states: BTreeMap<NonterminalId, u32> = BTreeMap::new();

        for &nt in &tc.all_nts {
            let ns = nwa.add_state();
            nt_states.insert(nt, ns);
        }

        for &(from_state, _pop_count, nt) in &tc.reduces {
            let q = nwa.add_state();
            nwa.add_epsilon(start, q, w_all.clone());
            nwa.add_transition(q, DEFAULT_LABEL, q, w_all.clone());

            let after_read = nwa.add_state();
            nwa.add_transition(q, from_state as i32, after_read, w_all.clone());

            let nt_target = *nt_states.get(&nt).unwrap();
            nwa.add_epsilon(after_read, nt_target, w_all.clone());
        }

        // --- NT escapes ---
        for &(nt, revealed, _goto_state, _shift_to) in &tc.nt_escapes {
            if let Some(&nt_state) = nt_states.get(&nt) {
                let q_final = nwa.add_state();
                nwa.add_transition(nt_state, revealed as i32, q_final, token_weight.clone());
                nwa.set_final_weight(q_final, token_weight.clone());
            }
        }

        // --- NT rereduces ---
        for &(nt_from, revealed, _pop2, nt_to) in &tc.nt_rereduces {
            if let Some(&from_state) = nt_states.get(&nt_from) {
                if let Some(&to_state) = nt_states.get(&nt_to) {
                    nwa.add_transition(from_state, revealed as i32, to_state, w_all.clone());
                }
            }
        }
    }

    nwa
}

/// Convert a terminal ID to a weight: the set of tokens that match this terminal
/// across all TSIDs.
fn terminal_to_weight(terminal: TerminalId, vocab: &VocabPreprocessing) -> Weight {
    let num_tsids = vocab.num_tsids;

    // Collect (tsid, token_set) pairs.
    let mut entries: Vec<(u32, u32, RangeSet)> = Vec::new();

    for tsid in 0..num_tsids {
        if let Some(rs) = vocab.possible_matches[tsid as usize].get(&terminal) {
            if !rs.is_empty() {
                entries.push((tsid, tsid, rs.clone()));
            }
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

    // Minimize: CompDwa → CompDwa.
    let dwa = minimize_acyclic(&dwa);

    dwa
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::grammar::GlrGrammar;
    use crate::compiler::grammar_def::tests::*;
    use crate::compiler::grammar_def::GrammarDef;
    use crate::compiler::tokenizer_dfa::TokenizerDfa;
    use crate::Vocab;

    fn make_vocab_and_preprocessing(gdef: &GrammarDef) -> (Vocab, TokenizerDfa, VocabPreprocessing) {
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
