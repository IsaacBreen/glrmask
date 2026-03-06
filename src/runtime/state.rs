//! Constraint and ConstraintState — the main runtime types.
//!
//! `Constraint` holds all compiled artifacts needed at inference time.
//! `ConstraintState` tracks per-sequence state and computes token masks.

use std::collections::BTreeMap;

use crate::GlrMaskError;
use crate::automata::dfa::DEAD;
use crate::automata::weighted::dwa::CompDwa;
use crate::compiler::glr::table::{Action, GlrTable};
use crate::compiler::grammar_def::TerminalId;
use crate::compiler::tokenizer_dfa::TokenizerDfa;
use crate::ds::bitset::BitSet;
use crate::ds::rangeset::RangeSet;

use super::gss_acc::{TerminalsDisallowed, terminals_disallowed_fresh};
use super::leveled_gss::LeveledGSS;

/// A GSS (Graph-Structured Stack) for the GLR parser.
///
/// Stack items are `u32` parser state IDs.
/// Accumulator is `TerminalsDisallowed` (currently unused but reserved for future mask pruning).
pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

/// A compiled grammar constraint, ready for inference.
///
/// Immutable after creation. Thread-safe (`Send + Sync`).
/// Create [`ConstraintState`] instances from this to track per-sequence state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

    /// Per-TSID: tokens that reach a non-dead tokenizer state without
    /// completing any terminal match. These tokens advance the tokenizer
    /// without triggering parser actions.
    pub(crate) passthrough_tokens: Vec<RangeSet>,

    /// Maximum token ID in the vocabulary.
    pub(crate) max_token: u32,

    /// EOS token ID, if any.
    pub(crate) eos_token_id: Option<u32>,

    /// Token ID → byte sequence mapping.
    pub(crate) token_bytes: BTreeMap<u32, Vec<u8>>,
}

impl Constraint {
    /// Compile a constraint from an EBNF grammar string.
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        let gdef = crate::frontend::ebnf::parse_ebnf(ebnf)?;
        Ok(crate::compiler::pipeline::compile(&gdef, vocab))
    }

    /// Compile a constraint from a Lark grammar string.
    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        let gdef = crate::frontend::lark::parse_lark(lark)?;
        Ok(crate::compiler::pipeline::compile(&gdef, vocab))
    }

    /// Compile a constraint from a JSON Schema string.
    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        let gdef = crate::frontend::json_schema::json_schema_to_grammar(schema)?;
        Ok(crate::compiler::pipeline::compile(&gdef, vocab))
    }

    /// Serialize this constraint to a byte vector (bincode format).
    pub fn save(&self) -> crate::Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| crate::GlrMaskError::Serialization(format!("serialize: {e}")))
    }

    /// Deserialize a constraint from bytes (bincode format).
    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| crate::GlrMaskError::Serialization(format!("deserialize: {e}")))
    }

    /// Save to a file.
    pub fn save_to_file(&self, path: &std::path::Path) -> crate::Result<()> {
        let bytes = self.save()?;
        std::fs::write(path, bytes)
            .map_err(|e| crate::GlrMaskError::Serialization(format!("write: {e}")))?;
        Ok(())
    }

    /// Load from a file.
    pub fn load_from_file(path: &std::path::Path) -> crate::Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| crate::GlrMaskError::Serialization(format!("read: {e}")))?;
        Self::load(&bytes)
    }

    /// Number of DWA states.
    pub fn num_dwa_states(&self) -> u32 {
        self.parser_dwa.num_states()
    }

    /// Number of parser states (GLR table states).
    pub fn num_parser_states(&self) -> u32 {
        self.table.num_states
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
        let gss = LeveledGSS::from_stacks(&[(vec![initial_parser_state], terminals_disallowed_fresh())]);
        state.insert(initial_tok_state, gss);

        ConstraintState { state }
    }

    /// Debug dump of internal state for troubleshooting.
    pub fn debug_dump(&self) {
        eprintln!("--- Constraint Debug Dump ---");
        eprintln!("num_tsids: {}", self.num_tsids);
        eprintln!("max_token: {}", self.max_token);
        eprintln!("state_to_tsid: {:?}", self.state_to_tsid);
        eprintln!("tsid_to_state: {:?}", self.tsid_to_state);
        eprintln!("Tokenizer DFA states: {}", self.tokenizer.dfa.num_states());
        for s in 0..self.tokenizer.dfa.num_states() {
            let fin = self.tokenizer.matched_terminals(s as u32);
            if !fin.is_empty() {
                eprintln!("  tok DFA state {}: finalizers={:?}", s, fin);
            }
            // Show non-dead transitions for this state
            let mut trans = Vec::new();
            for b in 0u16..=255u16 {
                let next = self.tokenizer.dfa.get_transition(s as u32, b as u8);
                if next != crate::automata::dfa::DEAD {
                    trans.push((b as u8, next));
                }
            }
            if !trans.is_empty() && trans.len() <= 20 {
                eprintln!("  tok DFA state {}: transitions={:?}", s, trans);
            } else if !trans.is_empty() {
                eprintln!("  tok DFA state {}: {} transitions", s, trans.len());
            }
        }
        eprintln!("DWA max_token: {}", self.parser_dwa.max_token);
        eprintln!("DWA states: {}", self.parser_dwa.states.len());
        for (tsid, pm) in self.possible_matches.iter().enumerate() {
            for (term, rs) in pm {
                let vals: Vec<u32> = rs.iter_values().collect();
                eprintln!("possible_matches[tsid={}][term={}] = {:?}", tsid, term, vals);
            }
        }
        eprintln!("--- End Debug Dump ---");
    }

    /// Debug: trace tokenizer behavior for specific bytes from a given starting state.
    pub fn debug_tokenizer(&self, input: &[u8], start_state: u32) {
        let result = self.tokenizer.execute_all_matches(input, start_state);
        eprintln!(
            "[debug_tokenizer] input={:?} start={} -> end={} matches={:?}",
            input, start_state, result.end_state, result.matches
        );
        // Also trace byte by byte
        let mut state = start_state;
        for (i, &byte) in input.iter().enumerate() {
            let next = self.tokenizer.dfa.get_transition(state, byte);
            let is_dead = next == crate::automata::dfa::DEAD;
            let finals = if !is_dead {
                self.tokenizer.dfa.finalizers(next).iter().copied().collect::<Vec<_>>()
            } else {
                vec![]
            };
            eprintln!(
                "  byte[{}]=0x{:02X} state {}->{}{}{}",
                i, byte, state, next,
                if is_dead { " DEAD" } else { "" },
                if !finals.is_empty() { format!(" finalizers={:?}", finals) } else { String::new() }
            );
            state = next;
            if is_dead { break; }
        }
    }

    /// Get the tokenizer's initial state (for debugging).
    pub fn tokenizer_initial_state(&self) -> u32 {
        self.tokenizer.initial_state()
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
/// State is a map from tokenizer DFA state → GSS of parser stacks.
/// The GSS provides structural sharing for efficient GLR parsing.
#[derive(Debug, Clone)]
pub struct ConstraintState {
    /// tokenizer DFA state → GSS of parser state stacks.
    pub(crate) state: BTreeMap<u32, ParserGSS>,
}

impl ConstraintState {
    /// Whether the constraint is still active (has valid parse stacks).
    pub fn is_active(&self) -> bool {
        !self.state.is_empty()
    }

    /// Compute the allowed-token mask.
    ///
    /// Returns a BitSet where bit `i` is set iff token `i` is allowed.
    ///
    /// Two-phase approach:
    /// 1. DWA walk produces an overapproximation (fast, ~O(stacks × DWA states))
    /// 2. Post-filter: simulate `commit()` for each candidate to remove false positives
    pub fn compute_mask(&self, constraint: &Constraint) -> BitSet {
        // Phase 1: DWA overapproximation.
        let stacks_map: BTreeMap<u32, Vec<Vec<u32>>> = self.state.iter().map(|(&tok_state, gss)| {
            let stacks: Vec<Vec<u32>> = gss.to_stacks().into_iter().map(|(s, _acc)| s).collect();
            (tok_state, stacks)
        }).collect();

        let mut mask = super::mask::compute_mask(
            &stacks_map,
            &constraint.parser_dwa,
            &constraint.state_to_tsid,
            constraint.max_token,
            constraint.num_tsids,
        );

        // Phase 2: Post-filter by simulating commit for each candidate token.
        // This removes false positives from the DWA overapproximation.
        //
        // A token is valid iff after commit the resulting state:
        // - Has stacks at the initial tokenizer state (clean terminal boundary), OR
        // - Has passthrough stacks where the tokenizer can still reach terminals
        //   that the parser can consume (verified via reachable_terminals).
        let reachable = constraint.tokenizer.compute_reachable_terminals();
        let initial_tok = constraint.tokenizer.initial_state();
        let candidates: Vec<usize> = mask.iter_ones().collect();
        let dwa_count = candidates.len();
        let mut filtered = 0usize;
        for token_id in candidates {
            let mut trial = self.clone();
            if trial.commit(constraint, token_id as u32).is_err() {
                mask.clear(token_id);
                filtered += 1;
                continue;
            }
            let viable = has_viable_state(&trial.state, &constraint.table, &reachable, initial_tok);
            if !viable {
                mask.clear(token_id);
                filtered += 1;
            }
            // Debug: print info for specific suspicious tokens
            if token_id == 17405 || token_id == 16792 {
                let token_bytes = constraint.token_bytes.get(&(token_id as u32));
                eprintln!(
                    "[debug] token_id={} bytes={:?} trial_state_entries={} viable={} self_state_entries={}",
                    token_id,
                    token_bytes,
                    trial.state.len(),
                    viable,
                    self.state.len(),
                );
                for (&ts, gss) in &trial.state {
                    let stacks = gss.to_stacks();
                    eprintln!("  tok_state={} num_stacks={} stacks={:?}", ts, stacks.len(), stacks.iter().map(|(s,_)| s.clone()).collect::<Vec<_>>());
                }
            }
        }
        if filtered > 0 {
            eprintln!("[post-filter] DWA candidates: {}, filtered out: {}, remaining: {}", dwa_count, filtered, dwa_count - filtered);
        }

        mask
    }

    /// Whether the current state is accepting (grammar allows end-of-input here).
    ///
    /// This checks if any of the current parser stacks can reach an Accept
    /// action by processing EOF (which may require reduce cascades first).
    pub fn is_accepting(&self, constraint: &Constraint) -> bool {
        let eof = crate::compiler::glr::grammar::EOF;
        for gss in self.state.values() {
            for (stack, _acc) in gss.to_stacks() {
                if can_accept(&constraint.table, &stack, eof) {
                    return true;
                }
            }
        }
        false
    }

    /// Commit a token: advance the constraint state.
    ///
    /// Processes the token's byte sequence through the tokenizer to find
    /// matched terminals, then steps the GLR parser accordingly using the GSS.
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

        let mut new_state: BTreeMap<u32, ParserGSS> = BTreeMap::new();

        // Processing queue: (byte_offset, tokenizer_state, gss)
        let mut queue: BTreeMap<usize, BTreeMap<u32, ParserGSS>> = BTreeMap::new();

        // Seed the queue with the current state at offset 0.
        queue.insert(0, self.state.clone());

        while let Some((offset, states_at_offset)) = queue.pop_first() {
            for (tok_state, gss) in states_at_offset {
                let remaining = &token_bytes[offset..];
                if remaining.is_empty() {
                    // All bytes consumed — add to new state.
                    new_state
                        .entry(tok_state)
                        .and_modify(|existing| *existing = existing.merge(&gss))
                        .or_insert_with(|| gss.clone());
                    continue;
                }

                // Run tokenizer on remaining bytes.
                let result = constraint
                    .tokenizer
                    .execute_all_matches(remaining, tok_state);

                // Process each intermediate match.
                for (match_offset, matched_terminals) in &result.matches {
                    let abs_offset = offset + match_offset;

                    for &terminal_id in matched_terminals {
                        // Step GLR parser on this terminal using the GSS.
                        let new_gss = step_glr_gss(&constraint.table, &gss, terminal_id);

                        if !new_gss.is_empty() {
                            // After matching a terminal, reset tokenizer to initial state.
                            let initial_tok = constraint.tokenizer.initial_state();
                            if abs_offset == token_bytes.len() {
                                // All bytes consumed — add directly to new state.
                                new_state
                                    .entry(initial_tok)
                                    .and_modify(|existing| *existing = existing.merge(&new_gss))
                                    .or_insert(new_gss);
                            } else {
                                // More bytes to process.
                                queue
                                    .entry(abs_offset)
                                    .or_default()
                                    .entry(initial_tok)
                                    .and_modify(|existing| *existing = existing.merge(&new_gss))
                                    .or_insert(new_gss);
                            }
                        }
                    }
                }

                // Track the tokenizer end state (for partial matches).
                if result.end_state != DEAD {
                    new_state
                        .entry(result.end_state)
                        .and_modify(|existing| *existing = existing.merge(&gss))
                        .or_insert(gss);
                }
            }
        }

        // Remove empty entries.
        new_state.retain(|_, gss| !gss.is_empty());

        self.state = new_state;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GSS-based GLR parser stepping (runtime)
// ---------------------------------------------------------------------------

/// Step the GLR parser on a terminal using the GSS.
///
/// This is the core GLR stepping function. It:
/// 1. Groups stacks by top state via `peek()` + `isolate()`
/// 2. Looks up actions for each (state, terminal) pair
/// 3. Handles shifts with `push`, reduces with `popn` + goto + `push`
/// 4. Merges all results with balanced merge
///
/// This is equivalent to grammars2024's `process_token_gss`.
fn step_glr_gss(table: &GlrTable, gss: &ParserGSS, terminal: TerminalId) -> ParserGSS {
    if gss.is_empty() {
        return LeveledGSS::empty();
    }

    // Group stacks by top parser state.
    let mut heads_by_state: BTreeMap<u32, ParserGSS> = BTreeMap::new();
    for top_state in gss.peek() {
        let iso = gss.isolate(Some(top_state));
        heads_by_state
            .entry(top_state)
            .and_modify(|acc| *acc = acc.merge(&iso))
            .or_insert(iso);
    }

    let mut shifted: Vec<ParserGSS> = Vec::new();

    // Cache popn results keyed by (gss pointer, pop length)
    let mut popn_cache: std::collections::HashMap<(usize, isize), ParserGSS> =
        std::collections::HashMap::new();
    // Cache: for a given popped GSS, pre-computed edge map: state_id -> [(top_val, iso)]
    let mut popped_edge_map_cache: std::collections::HashMap<
        usize,
        BTreeMap<u32, Vec<(u32, ParserGSS)>>,
    > = std::collections::HashMap::new();
    // Keep consumed state_gss values alive to avoid ABA problem with pointer keys.
    let mut _gss_anchor: Vec<ParserGSS> = Vec::new();

    while let Some((state_id, state_gss)) = heads_by_state.pop_first() {
        _gss_anchor.push(state_gss.clone());
        let actions = table.actions(state_id, terminal);

        for action in actions {
            match action {
                Action::Shift(to) => {
                    shifted.push(state_gss.push(*to));
                }
                Action::Reduce(rule_idx) => {
                    let rule = &table.rules[*rule_idx as usize];
                    let pop_count = rule.rhs.len();
                    let nt = rule.lhs;

                    // Get or compute the popped GSS.
                    let pop_key = (state_gss.ptr_key(), pop_count as isize);
                    let popped = popn_cache
                        .entry(pop_key)
                        .or_insert_with(|| state_gss.popn(pop_count as isize))
                        .clone();
                    if popped.is_empty() {
                        continue;
                    }

                    // Get or compute the edge map for this popped GSS.
                    let popped_ptr = popped.ptr_key();
                    let edge_map = popped_edge_map_cache
                        .entry(popped_ptr)
                        .or_insert_with(|| {
                            let mut map: BTreeMap<u32, Vec<(u32, ParserGSS)>> = BTreeMap::new();
                            for top_val in popped.peek() {
                                let iso = popped.isolate(Some(top_val));
                                map.entry(top_val).or_default().push((top_val, iso));
                            }
                            map
                        });

                    // Group edges by goto target, then batch push.
                    let mut next_id_edges: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
                    for (from_id, _edge_isos) in edge_map.iter() {
                        if let Some(goto_state) = table.goto_target(*from_id, nt) {
                            next_id_edges.entry(goto_state).or_default().push(*from_id);
                        }
                    }

                    for (next_id, from_ids) in next_id_edges {
                        let batch_gss =
                            popped.isolate_many(from_ids.into_iter().map(Some));
                        let pushed = batch_gss.push(next_id);
                        heads_by_state
                            .entry(next_id)
                            .and_modify(|acc| *acc = acc.merge(&pushed))
                            .or_insert(pushed);
                    }
                }
                Action::Accept => {
                    // Accept — keep the stack as-is (shifted into results).
                    shifted.push(state_gss.clone());
                }
            }
        }
    }

    if shifted.is_empty() {
        return LeveledGSS::empty();
    }
    if shifted.len() == 1 {
        return shifted.into_iter().next().unwrap();
    }
    // Balanced merge: O(n log n) instead of O(n²).
    while shifted.len() > 1 {
        let mut next = Vec::with_capacity((shifted.len() + 1) / 2);
        let mut iter = shifted.into_iter();
        while let Some(a) = iter.next() {
            if let Some(b) = iter.next() {
                next.push(a.merge(&b));
            } else {
                next.push(a);
            }
        }
        shifted = next;
    }
    shifted.into_iter().next().unwrap()
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

/// Check if a state has viable continuations.
///
/// A state is viable if at least one (tok_state, gss) entry satisfies:
/// 1. tok_state is the initial tokenizer state (clean terminal boundary), OR
/// 2. At least one reachable terminal from tok_state has valid parser actions
///    for some top parser state in the GSS.
///
/// This filters out states where the tokenizer is mid-match but no reachable
/// terminal matches any parser action — such states are effectively dead.
fn has_viable_state(
    state: &BTreeMap<u32, ParserGSS>,
    table: &GlrTable,
    reachable: &[std::collections::BTreeSet<crate::compiler::grammar_def::TerminalId>],
    initial_tok_state: u32,
) -> bool {
    if state.is_empty() {
        return false;
    }
    let eof = crate::compiler::glr::grammar::EOF;

    for (&tok_state, gss) in state {
        if gss.is_empty() {
            continue;
        }

        if tok_state == initial_tok_state {
            // Clean terminal boundary — stacks here are definitely viable.
            // But also check that the parser can actually do something
            // (accept EOF or shift some terminal).
            return true;
        }

        // Non-initial tok state: check if any reachable terminal from this
        // tokenizer state has valid parser actions for any top state.
        let top_states = gss.peek();

        if let Some(reachable_terms) = reachable.get(tok_state as usize) {
            for &terminal in reachable_terms {
                for &top in &top_states {
                    let actions = table.actions(top, terminal);
                    if !actions.is_empty() {
                        return true;
                    }
                }
            }
        }

        // Also check: can any stack accept via EOF? (for passthroughs that
        // happen to be accepting — e.g., the last token leaves a valid state)
        for (stack, _) in gss.to_stacks() {
            if can_accept(table, &stack, eof) {
                // The stack can accept, but tokenizer is mid-match.
                // This is NOT viable — the tokenizer needs to flush.
                // Only accept if the tok state also has finalizers that
                // the parser can consume. Already checked above via reachable.
            }
        }
    }

    false
}
