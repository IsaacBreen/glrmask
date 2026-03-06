//! Constraint and ConstraintState — the main runtime types.
//!
//! `Constraint` holds all compiled artifacts needed at inference time.
//! `ConstraintState` tracks per-sequence state and computes token masks.

use std::collections::BTreeMap;

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
    #[doc(hidden)]
    pub fn save_to_file(&self, path: &std::path::Path) -> crate::Result<()> {
        let bytes = self.save()?;
        std::fs::write(path, bytes)
            .map_err(|e| crate::GlrMaskError::Serialization(format!("write: {e}")))?;
        Ok(())
    }

    /// Load from a file.
    #[doc(hidden)]
    pub fn load_from_file(path: &std::path::Path) -> crate::Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| crate::GlrMaskError::Serialization(format!("read: {e}")))?;
        Self::load(&bytes)
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
    pub fn tokenizer_initial_state(&self) -> u32 {
        self.tokenizer.initial_state()
    }

    /// Number of `u32` words required in a mask buffer for this vocabulary.
    ///
    /// Allocate the buffer with `vec![0u32; constraint.mask_len()]`.
    /// Token `i` is allowed iff `buf[i / 32] & (1u32 << (i % 32)) != 0`.
    pub fn mask_len(&self) -> usize {
        (self.max_token as usize / 32) + 1
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
    /// Compute the allowed-token mask.
    ///
    /// Returns a BitSet where bit `i` is set iff token `i` is allowed.
    ///
    /// Two-phase approach:
    /// 1. DWA walk produces an overapproximation (fast, ~O(stacks × DWA states))
    /// 2. Post-filter: simulate `commit()` for each candidate to remove false positives
    ///
    /// **Note**: prefer [`mask`] or [`fill_mask`] which return `u32` words matching the
    /// plan's public API. This method is retained for white-box tests only.
    #[doc(hidden)]
    pub fn compute_mask(&self, constraint: &Constraint) -> BitSet {
        // Phase 1: DWA overapproximation.
        let mut stacks_map: BTreeMap<u32, Vec<Vec<u32>>> = self.state.iter().map(|(&tok_state, gss)| {
            let stacks: Vec<Vec<u32>> = gss.to_stacks().into_iter().map(|(s, _acc)| s).collect();
            (tok_state, stacks)
        }).collect();

        // ε-reduce closure: extend stacks at initial tokenizer state.
        // After a commit, stacks may have un-reduced ε-productions at the top.
        // The DWA expects fully-reduced stacks, so apply all possible ε-reductions
        // (pop_count=0 rules) to produce additional stack variants.
        let initial_tok = constraint.tokenizer.initial_state();
        if let Some(stacks) = stacks_map.get_mut(&initial_tok) {
            let mut extra: Vec<Vec<u32>> = Vec::new();
            for stack in stacks.iter() {
                epsilon_reduce_stacks(&constraint.table, stack, &mut extra);
            }
            stacks.extend(extra);
        }

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
        let _dwa_count = candidates.len();
        let mut _filtered = 0usize;
        for token_id in candidates {
            // Skip EOS token in post-filter — handled separately below.
            if constraint.eos_token_id == Some(token_id as u32) {
                continue;
            }
            let mut trial = self.clone();
            trial.commit(constraint, token_id as u32);
            let viable = has_viable_state(&trial.state, &constraint.table, &reachable, initial_tok);
            if !viable {
                mask.clear(token_id);
                _filtered += 1;
            }

        }


        // Phase 3: Rescue — trial-commit tokens that the DWA might have missed.
        //
        // The DWA overapproximation can miss tokens when cascading reduces
        // (e.g., _star → _star "," JSON_INTEGER with pop_count > 0) create
        // stack patterns the determinized DWA doesn't represent.
        //
        // For each TSID in the current state, collect all token IDs from
        // possible_matches that are NOT already in the mask, then trial-commit
        // each to verify viability.
        {
            let mut rescue_candidates = BitSet::new(constraint.max_token as usize + 1);
            for &tok_state in self.state.keys() {
                let tsid = if (tok_state as usize) < constraint.state_to_tsid.len() {
                    constraint.state_to_tsid[tok_state as usize]
                } else {
                    continue;
                };
                if tsid == u32::MAX || (tsid as usize) >= constraint.possible_matches.len() {
                    continue;
                }
                for token_set in constraint.possible_matches[tsid as usize].values() {
                    for token_id in token_set.iter_values() {
                        if token_id <= constraint.max_token && !mask.get(token_id as usize) {
                            rescue_candidates.set(token_id as usize);
                        }
                    }
                }
            }

            let mut _rescued = 0usize;
            for token_id in rescue_candidates.iter_ones() {
                // Skip EOS — handled separately below.
                if constraint.eos_token_id == Some(token_id as u32) {
                    continue;
                }
                let mut trial = self.clone();
                trial.commit(constraint, token_id as u32);
                if has_viable_state(&trial.state, &constraint.table, &reachable, initial_tok) {
                    mask.set(token_id);
                    _rescued += 1;
                }
            }

        }

        // EOS token handling: EOS is not a regular byte-sequence token.
        // Remove it from the DWA/post-filter result and add it back only
        // when the current state is accepting (grammar allows end-of-input).
        if let Some(eos_id) = constraint.eos_token_id {
            mask.clear(eos_id as usize);
            if self.is_accepting(constraint) {
                mask.set(eos_id as usize);
            }
        }

        mask
    }

    /// Whether the current state is accepting (grammar allows end-of-input here).
    ///
    /// This checks if any of the current parser stacks can reach an Accept
    /// action by processing EOF (which may require reduce cascades first).
    ///
    /// Only checks stacks at the initial tokenizer state (clean terminal boundary).
    /// Stacks at non-initial tokenizer states are mid-match and cannot accept.
    ///
    /// **Note**: prefer [`is_finished`] which matches the plan's public API.
    /// This method is retained for white-box tests only.
    #[doc(hidden)]
    pub fn is_accepting(&self, constraint: &Constraint) -> bool {
        let eof = crate::compiler::glr::grammar::EOF;
        let initial_tok = constraint.tokenizer.initial_state();
        if let Some(gss) = self.state.get(&initial_tok) {
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
    /// Infallible. If `token_id` is not in the vocabulary, the method is a
    /// no-op and the parser state is left unchanged. The next call to
    /// [`mask`] / [`fill_mask`] will reflect whatever state the parser is in
    /// after any bytes that *were* successfully committed.
    pub fn commit(
        &mut self,
        constraint: &Constraint,
        token_id: u32,
    ) {
        if let Some(bytes) = constraint.token_bytes.get(&token_id) {
            let bytes = bytes.clone();
            self.process_bytes_raw(constraint, &bytes);
        }
        // Unknown token_id → no-op (caller should only commit tokens from the mask)
    }

    // -----------------------------------------------------------------------
    // Plan-conforming public API
    // -----------------------------------------------------------------------

    /// Compute the allowed-token mask as a `Vec<u32>`.
    ///
    /// Token `i` is allowed iff `result[i / 32] & (1u32 << (i % 32)) != 0`.
    /// Allocate the buffer with [`Constraint::mask_len`] words.
    pub fn mask(&self, constraint: &Constraint) -> Vec<u32> {
        let bitset = self.compute_mask(constraint);
        let mut buf = vec![0u32; constraint.mask_len()];
        bitset.fill_u32_mask(&mut buf);
        buf
    }

    /// Fill a pre-allocated mask buffer.
    ///
    /// `buf` must be at least `constraint.mask_len()` words long.
    /// Token `i` is allowed iff `buf[i / 32] & (1u32 << (i % 32)) != 0`.
    pub fn fill_mask(&self, constraint: &Constraint, buf: &mut [u32]) {
        let bitset = self.compute_mask(constraint);
        bitset.fill_u32_mask(buf);
    }

    /// Whether the grammar has been fully satisfied (EOS is valid at current position).
    pub fn is_finished(&self, constraint: &Constraint) -> bool {
        self.is_accepting(constraint)
    }

    /// Commit raw bytes, advancing tokenizer and parser state.
    ///
    /// Infallible. If the bytes produce no valid parse continuations the next
    /// mask will simply be empty.
    pub fn commit_bytes(&mut self, constraint: &Constraint, bytes: &[u8]) {
        self.process_bytes_raw(constraint, bytes);
    }

    /// Commit multiple tokens in sequence (batch convenience wrapper).
    ///
    /// Equivalent to calling [`commit`] for each token ID in order.
    pub fn commit_tokens(&mut self, constraint: &Constraint, tokens: &[u32]) {
        for &token in tokens {
            self.commit(constraint, token);
        }
    }

    /// Return the sequence of tokens forced by the current grammar state.
    ///
    /// A token is *forced* when it is the only non-EOS option in the mask.
    /// The method repeatedly computes the mask, collects any single forced
    /// token, simulates a commit, and continues until the state is no longer
    /// deterministic. Returns an empty `Vec` when no tokens are forced.
    ///
    /// The caller is responsible for committing the returned tokens via
    /// [`commit_tokens`].
    pub fn force(&self, constraint: &Constraint) -> Vec<u32> {
        let mut result = Vec::new();
        let mut trial = self.clone();
        loop {
            let bitset = trial.compute_mask(constraint);
            // Build a copy with the EOS bit cleared so we see only real tokens.
            let forced_token = if let Some(eos_id) = constraint.eos_token_id {
                let mut without_eos = bitset.clone();
                without_eos.clear(eos_id as usize);
                if without_eos.count_ones() == 1 {
                    without_eos.iter_ones().next().map(|i| i as u32)
                } else {
                    None
                }
            } else {
                if bitset.count_ones() == 1 {
                    bitset.iter_ones().next().map(|i| i as u32)
                } else {
                    None
                }
            };

            let Some(token) = forced_token else { break };
            result.push(token);
            trial.commit(constraint, token);
            if trial.state.is_empty() {
                break;
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Core byte-processing engine shared by `commit` and `commit_bytes`.
    fn process_bytes_raw(&mut self, constraint: &Constraint, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }

        let bytes_len = bytes.len();
        let mut new_state: BTreeMap<u32, ParserGSS> = BTreeMap::new();

        // Processing queue: (byte_offset, tokenizer_state → gss)
        let mut queue: BTreeMap<usize, BTreeMap<u32, ParserGSS>> = BTreeMap::new();
        queue.insert(0, self.state.clone());

        while let Some((offset, states_at_offset)) = queue.pop_first() {
            for (tok_state, gss) in states_at_offset {
                let remaining = &bytes[offset..];
                if remaining.is_empty() {
                    new_state
                        .entry(tok_state)
                        .and_modify(|existing| *existing = existing.merge(&gss))
                        .or_insert_with(|| gss.clone());
                    continue;
                }

                let result = constraint
                    .tokenizer
                    .execute_all_matches(remaining, tok_state);

                for (match_offset, matched_terminals) in &result.matches {
                    let abs_offset = offset + match_offset;

                    for &terminal_id in matched_terminals {
                        let new_gss = step_glr_gss(&constraint.table, &gss, terminal_id);

                        if !new_gss.is_empty() {
                            let initial_tok = constraint.tokenizer.initial_state();
                            if abs_offset == bytes_len {
                                new_state
                                    .entry(initial_tok)
                                    .and_modify(|existing| *existing = existing.merge(&new_gss))
                                    .or_insert(new_gss);
                            } else {
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

                if result.end_state != DEAD {
                    new_state
                        .entry(result.end_state)
                        .and_modify(|existing| *existing = existing.merge(&gss))
                        .or_insert(gss);
                }
            }
        }

        new_state.retain(|_, gss| !gss.is_empty());
        self.state = new_state;
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

/// Compute ε-reduce closure for a single stack.
///
/// For each ε-production (pop_count=0 rule) that can fire at the top state,
/// push the goto state to produce a new extended stack. This is applied
/// recursively until no more ε-reductions are possible.
///
/// The original stack is NOT included in `out` — only newly produced variants.
fn epsilon_reduce_stacks(table: &GlrTable, stack: &[u32], out: &mut Vec<Vec<u32>>) {
    let mut worklist: Vec<Vec<u32>> = vec![stack.to_vec()];
    let mut seen_tops: std::collections::BTreeSet<Vec<u32>> = std::collections::BTreeSet::new();
    seen_tops.insert(stack.to_vec());

    while let Some(current) = worklist.pop() {
        let top_state = match current.last() {
            Some(&s) => s,
            None => continue,
        };

        // Check all terminals for ε-reduce actions at the top state.
        for t in 0..table.num_terminals {
            for action in table.actions(top_state, t) {
                if let Action::Reduce(rule_idx) = action {
                    let rule = &table.rules[*rule_idx as usize];
                    if rule.rhs.len() == 0 {
                        // ε-production: pop 0, goto from top_state for the LHS nonterminal.
                        if let Some(goto_state) = table.goto_target(top_state, rule.lhs) {
                            let mut extended = current.clone();
                            extended.push(goto_state);
                            if seen_tops.insert(extended.clone()) {
                                out.push(extended.clone());
                                worklist.push(extended);
                            }
                        }
                    }
                }
            }
        }
    }
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
