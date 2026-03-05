//! Constraint and ConstraintState — the main runtime types.
//!
//! `Constraint` holds all compiled artifacts needed at inference time.
//! `ConstraintState` tracks per-sequence state and computes token masks.

use std::collections::{BTreeMap, BTreeSet};

use crate::automata::dfa::DEAD;
use crate::automata::weighted::dwa::CompDwa;
use crate::compiler::glr::table::{Action, GlrTable};
use crate::compiler::grammar_def::TerminalId;
use crate::compiler::tokenizer_dfa::TokenizerDfa;
use crate::ds::bitset::BitSet;
use crate::ds::rangeset::RangeSet;
use crate::GlrMaskError;

/// A compiled grammar constraint, ready for inference.
///
/// Immutable after creation. Thread-safe (`Send + Sync`).
/// Create [`ConstraintState`] instances from this to track per-sequence state.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Constraint {
    /// The compiled parser DWA.
    /// Labels = parser state IDs (i32), weights = token bitvectors.
    pub(crate) parser_dwa: CompDwa,

    /// The GLR parse table.
    pub(crate) table: GlrTable,

    /// The byte-level tokenizer DFA.
    pub(crate) tokenizer: TokenizerDfa,

    /// Number of token-set IDs.
    pub(crate) num_tsids: u32,

    /// Tokenizer DFA state → TSID mapping.
    /// `state_to_tsid[dfa_state]` = compacted TSID (u32::MAX if unreachable).
    pub(crate) state_to_tsid: Vec<u32>,

    /// TSID → tokenizer DFA state mapping.
    pub(crate) tsid_to_state: Vec<u32>,

    /// Per-TSID: { terminal_id → token RangeSet }.
    /// `possible_matches[tsid][terminal] = set of allowed token IDs`.
    pub(crate) possible_matches: Vec<BTreeMap<TerminalId, RangeSet>>,

    /// Maximum token ID in the vocabulary.
    pub(crate) max_token: u32,

    /// EOS token ID, if any.
    pub(crate) eos_token_id: Option<u32>,

    /// Token ID → byte sequence mapping.
    pub(crate) token_bytes: BTreeMap<u32, Vec<u8>>,
}

impl Constraint {
    /// Number of DWA states.
    pub fn num_dwa_states(&self) -> u32 {
        self.parser_dwa.num_states()
    }

    /// Number of parser states (GLR table states).
    pub fn num_parser_states(&self) -> u32 {
        self.table.num_states as u32
    }

    /// Number of token-set IDs.
    pub fn num_tsids(&self) -> u32 {
        self.num_tsids
    }

    /// Create a new `ConstraintState` at the start position.
    pub fn start(&self) -> ConstraintState {
        // The initial parser state is 0.
        // The initial tokenizer state is 0 (initial DFA state).
        let initial_parser_state = 0u32;
        let initial_tok_state = self.tokenizer.initial_state();

        let mut state = BTreeMap::new();
        state.insert(initial_tok_state, vec![vec![initial_parser_state]]);

        ConstraintState { state }
    }
}

// ---------------------------------------------------------------------------
// ConstraintState
// ---------------------------------------------------------------------------

/// Per-sequence constraint state.
///
/// Tracks the current parse + tokenizer state. Computes token masks and
/// advances state when tokens are committed.
///
/// State is a map from tokenizer DFA state → list of parser stacks.
/// Each parser stack is a Vec<u32> of parser state IDs, bottom to top.
#[derive(Debug, Clone)]
pub struct ConstraintState {
    /// tokenizer DFA state → list of parser state stacks.
    /// Each stack is bottom-to-top.
    pub(crate) state: BTreeMap<u32, Vec<Vec<u32>>>,
}

impl ConstraintState {
    /// Whether the constraint is still active (has valid parse stacks).
    pub fn is_active(&self) -> bool {
        !self.state.is_empty()
    }

    /// Compute the allowed-token mask.
    ///
    /// Returns a BitSet where bit `i` is set iff token `i` is allowed.
    pub fn compute_mask(&self, constraint: &Constraint) -> BitSet {
        super::mask::compute_mask(
            &self.state,
            &constraint.parser_dwa,
            &constraint.state_to_tsid,
            constraint.max_token,
            constraint.num_tsids,
        )
    }

    /// Whether the current state is accepting (grammar allows end-of-input here).
    ///
    /// This checks if any of the current parser stacks can reach an Accept
    /// action by processing EOF (which may require reduce cascades first).
    pub fn is_accepting(&self, constraint: &Constraint) -> bool {
        let eof = crate::compiler::glr::grammar::EOF;
        for stacks in self.state.values() {
            for stack in stacks {
                if can_accept(&constraint.table, stack, eof) {
                    return true;
                }
            }
        }
        false
    }

    /// Commit a token: advance the constraint state.
    ///
    /// Processes the token's byte sequence through the tokenizer to find
    /// matched terminals, then steps the GLR parser accordingly.
    pub fn commit(
        &mut self,
        constraint: &Constraint,
        token_id: u32,
    ) -> std::result::Result<(), GlrMaskError> {
        let token_bytes = constraint
            .token_bytes
            .get(&token_id)
            .ok_or_else(|| {
                GlrMaskError::InvalidInput(format!("Token ID {} not in vocabulary", token_id))
            })?
            .clone();

        if token_bytes.is_empty() {
            return Ok(());
        }

        let mut new_state: BTreeMap<u32, Vec<Vec<u32>>> = BTreeMap::new();

        // Processing queue: (byte_offset, tokenizer_state, stacks)
        let mut queue: BTreeMap<usize, BTreeMap<u32, Vec<Vec<u32>>>> = BTreeMap::new();

        // Seed the queue with the current state at offset 0.
        queue.insert(0, self.state.clone());

        while let Some((offset, states_at_offset)) = queue.pop_first() {
            for (tok_state, stacks) in states_at_offset {
                let remaining = &token_bytes[offset..];
                if remaining.is_empty() {
                    // All bytes consumed — add to new state.
                    new_state
                        .entry(tok_state)
                        .or_default()
                        .extend(stacks.iter().cloned());
                    continue;
                }

                // Run tokenizer on remaining bytes.
                let result = constraint.tokenizer.execute_all_matches(remaining, tok_state);

                // Process each intermediate match.
                for (match_offset, matched_terminals) in &result.matches {
                    let abs_offset = offset + match_offset;

                    for &terminal_id in matched_terminals {
                        // Step GLR parser on this terminal for each stack.
                        let new_stacks = step_glr_all(
                            &constraint.table,
                            &stacks,
                            terminal_id,
                        );

                        if !new_stacks.is_empty() {
                            // After matching a terminal, reset tokenizer to initial state.
                            let initial_tok = constraint.tokenizer.initial_state();
                            if abs_offset == token_bytes.len() {
                                // All bytes consumed — add directly to new state.
                                new_state
                                    .entry(initial_tok)
                                    .or_default()
                                    .extend(new_stacks);
                            } else {
                                // More bytes to process.
                                queue
                                    .entry(abs_offset)
                                    .or_default()
                                    .entry(initial_tok)
                                    .or_default()
                                    .extend(new_stacks);
                            }
                        }
                    }
                }

                // Track the tokenizer end state (for partial matches).
                if result.end_state != DEAD {
                    new_state
                        .entry(result.end_state)
                        .or_default()
                        .extend(stacks);
                }
            }
        }

        // Deduplicate stacks within each tokenizer state.
        for stacks in new_state.values_mut() {
            let deduped: BTreeSet<Vec<u32>> = stacks.drain(..).collect();
            *stacks = deduped.into_iter().collect();
        }

        // Remove empty entries.
        new_state.retain(|_, stacks| !stacks.is_empty());

        self.state = new_state;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GLR parser stepping (runtime)
// ---------------------------------------------------------------------------

/// Step the GLR parser on a terminal for a list of stacks.
///
/// For each stack, applies shift/reduce actions and returns the resulting stacks.
/// Stacks that can't consume this terminal are discarded.
fn step_glr_all(
    table: &GlrTable,
    stacks: &[Vec<u32>],
    terminal: TerminalId,
) -> Vec<Vec<u32>> {
    let mut result_stacks = Vec::new();

    for stack in stacks {
        let new = step_glr_single(table, stack, terminal);
        result_stacks.extend(new);
    }

    result_stacks
}

/// Step a single GLR stack on a terminal.
///
/// Returns all resulting stacks after processing the terminal (may be 0, 1, or many
/// due to GLR nondeterminism from shift-reduce or reduce-reduce conflicts).
fn step_glr_single(
    table: &GlrTable,
    stack: &[u32],
    terminal: TerminalId,
) -> Vec<Vec<u32>> {
    if stack.is_empty() {
        return Vec::new();
    }

    let top = *stack.last().unwrap();
    let actions = table.actions(top, terminal);

    let mut results = Vec::new();

    for action in actions {
        match action {
            Action::Shift(to) => {
                let mut new_stack = stack.to_vec();
                new_stack.push(*to);
                results.push(new_stack);
            }
            Action::Reduce(rule_idx) => {
                let rule = &table.rules[*rule_idx as usize];
                let pop_count = rule.rhs.len();
                let nt = rule.lhs;

                if stack.len() < pop_count + 1 {
                    continue; // Stack too shallow.
                }

                let mut new_stack = stack[..stack.len() - pop_count].to_vec();
                let revealed = *new_stack.last().unwrap();

                if let Some(goto_state) = table.goto_target(revealed, nt) {
                    new_stack.push(goto_state);
                    // After reduce, try to consume the terminal from the new top.
                    let further = step_glr_single(table, &new_stack, terminal);
                    if further.is_empty() {
                        // Reduce succeeded but terminal can't be consumed yet.
                        // This happens in cascading reduces — keep going.
                    }
                    results.extend(further);
                }
            }
            Action::Accept => {
                // Accept — this stack is done.
                results.push(stack.to_vec());
            }
        }
    }

    results
}

/// Check if a stack can reach Accept via EOF (possibly after reduce cascades).
fn can_accept(table: &GlrTable, stack: &[u32], eof: TerminalId) -> bool {
    can_accept_inner(table, stack, eof, 0)
}

fn can_accept_inner(table: &GlrTable, stack: &[u32], eof: TerminalId, depth: usize) -> bool {
    if stack.is_empty() || depth > 100 {
        return false;
    }

    let top = *stack.last().unwrap();
    let actions = table.actions(top, eof);

    for action in actions {
        match action {
            Action::Accept => return true,
            Action::Reduce(rule_idx) => {
                let rule = &table.rules[*rule_idx as usize];
                let pop_count = rule.rhs.len();
                let nt = rule.lhs;

                if stack.len() < pop_count + 1 {
                    continue;
                }

                let mut new_stack = stack[..stack.len() - pop_count].to_vec();
                let revealed = *new_stack.last().unwrap();

                if let Some(goto_state) = table.goto_target(revealed, nt) {
                    new_stack.push(goto_state);
                    if can_accept_inner(table, &new_stack, eof, depth + 1) {
                        return true;
                    }
                }
            }
            Action::Shift(_) => {
                // Can't shift on EOF to accept.
            }
        }
    }
    false
}
