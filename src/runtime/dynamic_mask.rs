//! Direct dynamic mask generation.
//!
//! This implementation intentionally does not consult the parser DWA. It walks
//! the vocabulary byte trie while advancing the lexer and GLR parser directly.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::automata::lexer::Lexer;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{
    advance_stacks, stack_may_advance_on, stack_may_advance_on_any, ParserGSS,
};
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::LeveledGSS;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::grammar::flat::TerminalID;

use super::artifact::{
    cache_dynamic_continuation_partition,
    lookup_dynamic_continuation_partition,
    Constraint,
    DynamicContinuationPartition,
    DynamicMaskVocab,
};
use super::state::ConstraintState;

type ExclusionMap = BTreeMap<u32, BTreeSet<TerminalID>>;
type Exclusions = Arc<ExclusionMap>;
type ParserStacks = LeveledGSS<u32, ()>;

/// Operation counts for one direct dynamic-mask computation.
///
/// This is deliberately driven through a compile-time observer. The normal
/// dynamic-mask path uses `NoopDynamicMaskObserver`, so the compiler removes
/// all profiling work from the hot path.
#[derive(Clone, Debug, Default)]
pub struct DynamicMaskProfile {
    pub total_ns: u64,
    pub work_items: u64,
    pub trie_edges: u64,
    pub traversal_pushes: u64,
    pub lexer_executions: u64,
    pub lexer_first_byte_rejects: u64,
    pub parser_candidate_first_byte_rejects: u64,
    pub exclusion_lexer_executions: u64,
    pub terminal_matches: u64,
    pub parser_child_attempts: u64,
    pub parser_child_accepts: u64,
    pub boundary_checks: u64,
    pub parser_admission_cache_hits: u64,
    pub parser_admission_cache_misses: u64,
    pub terminal_loop_checks: u64,
    pub terminal_loop_no_future: u64,
    pub terminal_loop_no_candidate: u64,
    pub terminal_loop_byte_rejects: u64,
    pub terminal_loop_boundary_rejects: u64,
    pub terminal_loop_exclusion_rejects: u64,
    pub terminal_loop_mark_all: u64,
    pub terminal_loop_mark_current_only: u64,
    pub terminal_loop_child_mark_all: u64,
    pub terminal_loop_post_edge_mark_all: u64,
    pub loop_partition_uses: u64,
    pub loop_partition_loop_bytes_total: u64,
    pub loop_partition_safe_tokens: u64,
    pub loop_partition_exception_tokens: u64,
    pub marked_subtree_tokens: u64,
}

trait DynamicMaskObserver {
    #[inline]
    fn work_item(&mut self) {}
    #[inline]
    fn trie_edge(&mut self) {}
    #[inline]
    fn traversal_push(&mut self) {}
    #[inline]
    fn lexer_execution(&mut self) {}
    #[inline]
    fn lexer_first_byte_reject(&mut self) {}
    #[inline]
    fn parser_candidate_first_byte_reject(&mut self) {}
    #[inline]
    fn exclusion_lexer_execution(&mut self) {}
    #[inline]
    fn terminal_match(&mut self) {}
    #[inline]
    fn parser_child_attempt(&mut self) {}
    #[inline]
    fn parser_child_accept(&mut self) {}
    #[inline]
    fn boundary_check(&mut self) {}
    #[inline]
    fn parser_admission_cache_hit(&mut self) {}
    #[inline]
    fn parser_admission_cache_miss(&mut self) {}
    #[inline]
    fn terminal_loop_check(&mut self) {}
    #[inline]
    fn terminal_loop_no_future(&mut self) {}
    #[inline]
    fn terminal_loop_no_candidate(&mut self) {}
    #[inline]
    fn terminal_loop_byte_reject(&mut self) {}
    #[inline]
    fn terminal_loop_boundary_reject(&mut self) {}
    #[inline]
    fn terminal_loop_exclusion_reject(&mut self) {}
    #[inline]
    fn terminal_loop_mark_all(&mut self) {}
    #[inline]
    fn terminal_loop_mark_current_only(&mut self) {}
    #[inline]
    fn terminal_loop_child_mark_all(&mut self) {}
    #[inline]
    fn terminal_loop_post_edge_mark_all(&mut self) {}
    #[inline]
    fn loop_partition(&mut self, _loop_bytes: usize, _safe_tokens: usize, _exception_tokens: usize) {}
    #[inline]
    fn marked_subtree_tokens(&mut self, _count: u64) {}
}

struct NoopDynamicMaskObserver;

impl DynamicMaskObserver for NoopDynamicMaskObserver {}

impl DynamicMaskObserver for DynamicMaskProfile {
    #[inline]
    fn work_item(&mut self) {
        self.work_items += 1;
    }
    #[inline]
    fn trie_edge(&mut self) {
        self.trie_edges += 1;
    }
    #[inline]
    fn traversal_push(&mut self) {
        self.traversal_pushes += 1;
    }
    #[inline]
    fn lexer_execution(&mut self) {
        self.lexer_executions += 1;
    }
    #[inline]
    fn lexer_first_byte_reject(&mut self) {
        self.lexer_first_byte_rejects += 1;
    }
    #[inline]
    fn parser_candidate_first_byte_reject(&mut self) {
        self.parser_candidate_first_byte_rejects += 1;
    }
    #[inline]
    fn exclusion_lexer_execution(&mut self) {
        self.exclusion_lexer_executions += 1;
    }
    #[inline]
    fn terminal_match(&mut self) {
        self.terminal_matches += 1;
    }
    #[inline]
    fn parser_child_attempt(&mut self) {
        self.parser_child_attempts += 1;
    }
    #[inline]
    fn parser_child_accept(&mut self) {
        self.parser_child_accepts += 1;
    }
    #[inline]
    fn boundary_check(&mut self) {
        self.boundary_checks += 1;
    }
    #[inline]
    fn parser_admission_cache_hit(&mut self) {
        self.parser_admission_cache_hits += 1;
    }
    #[inline]
    fn parser_admission_cache_miss(&mut self) {
        self.parser_admission_cache_misses += 1;
    }
    #[inline]
    fn terminal_loop_check(&mut self) {
        self.terminal_loop_checks += 1;
    }
    #[inline]
    fn terminal_loop_no_future(&mut self) {
        self.terminal_loop_no_future += 1;
    }
    #[inline]
    fn terminal_loop_no_candidate(&mut self) {
        self.terminal_loop_no_candidate += 1;
    }
    #[inline]
    fn terminal_loop_byte_reject(&mut self) {
        self.terminal_loop_byte_rejects += 1;
    }
    #[inline]
    fn terminal_loop_boundary_reject(&mut self) {
        self.terminal_loop_boundary_rejects += 1;
    }
    #[inline]
    fn terminal_loop_exclusion_reject(&mut self) {
        self.terminal_loop_exclusion_rejects += 1;
    }
    #[inline]
    fn terminal_loop_mark_all(&mut self) {
        self.terminal_loop_mark_all += 1;
    }
    #[inline]
    fn terminal_loop_mark_current_only(&mut self) {
        self.terminal_loop_mark_current_only += 1;
    }
    #[inline]
    fn terminal_loop_child_mark_all(&mut self) {
        self.terminal_loop_child_mark_all += 1;
    }
    #[inline]
    fn terminal_loop_post_edge_mark_all(&mut self) {
        self.terminal_loop_post_edge_mark_all += 1;
    }
    #[inline]
    fn loop_partition(&mut self, loop_bytes: usize, safe_tokens: usize, exception_tokens: usize) {
        self.loop_partition_uses += 1;
        self.loop_partition_loop_bytes_total += loop_bytes as u64;
        self.loop_partition_safe_tokens += safe_tokens as u64;
        self.loop_partition_exception_tokens += exception_tokens as u64;
    }
    #[inline]
    fn marked_subtree_tokens(&mut self, count: u64) {
        self.marked_subtree_tokens += count;
    }
}

#[derive(Clone)]
struct TraverseWork<'a> {
    node: &'a VocabPrefixTreeNode,
    tokenizer_state: u32,
    gss: ParserStacks,
    exclusions: Exclusions,
}

#[inline]
fn set_mask_bit(buf: &mut [u32], token_id: u32) {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    if let Some(slot) = buf.get_mut(word) {
        *slot |= 1u32 << bit;
    }
}

#[inline]
fn or_mask(buf: &mut [u32], mask: &[u32]) {
    for (target, source) in buf.iter_mut().zip(mask) {
        *target |= *source;
    }
}

fn update_eos_mask(state: &ConstraintState<'_>, buf: &mut [u32]) {
    let Some(token_id) = state.constraint.eos_token_id else {
        return;
    };
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    let Some(slot) = buf.get_mut(word) else {
        return;
    };
    *slot &= !(1u32 << bit);
    if state.is_complete() {
        *slot |= 1u32 << bit;
    }
}

/// Dynamic masking keeps terminal restrictions in `Exclusions`. The parser
/// table routines still use `ParserGSS`, so give their stack operations an
/// otherwise-unused empty accumulator.
fn with_empty_accumulators(stacks: &ParserStacks) -> ParserGSS {
    stacks.apply(|_| TerminalsDisallowed::new())
}

/// Parser-side information is invariant while a trie walk continues through a
/// lexer state without finalizing a terminal.  A single dynamic mask routinely
/// revisits that `(lexer state, parser GSS)` pair tens of thousands of times.
/// Keep a strong GSS reference in the value, not just its pointer key: that
/// makes the pointer an unambiguous identity for the whole traversal and avoids
/// accidental allocator-address reuse after a work item is dropped.
struct ParserAdmission {
    stacks: ParserStacks,
    has_future_terminal: bool,
    candidate_terminals: BitSet,
    any_terminal_admissible: bool,
    /// Bytes that preserve the residual language of every terminal that can
    /// matter from this lexer/parser pair.  It is the intersection of their
    /// terminal-sensitive quotient-loop sets.
    common_terminal_loop_bytes: U8Set,
    /// First bytes that can still reach or complete at least one parser-side
    /// candidate terminal.  This is a much stronger edge filter than merely
    /// testing whether the lexer has *some* transition on the byte.
    candidate_first_bytes: U8Set,
}

#[derive(Default)]
struct DynamicMaskTraversalCache {
    parser_admissions: FxHashMap<(u32, usize), ParserAdmission>,
}

impl DynamicMaskTraversalCache {
    fn parser_admission<O: DynamicMaskObserver>(
        &mut self,
        constraint: &Constraint,
        vocab: &DynamicMaskVocab,
        tokenizer_state: u32,
        stacks: &ParserStacks,
        observer: &mut O,
    ) -> &ParserAdmission {
        let key = (tokenizer_state, stacks.ptr_key());
        if self.parser_admissions.contains_key(&key) {
            let cached = self
                .parser_admissions
                .get(&key)
                .expect("parser admission cache entry disappeared");
            debug_assert!(cached.stacks.ptr_eq(stacks));
            observer.parser_admission_cache_hit();
            return cached;
        }

        observer.parser_admission_cache_miss();
        let future_terminals = constraint.tokenizer.tokens_accessible_from_state(tokenizer_state);
        let has_future_terminal = !future_terminals.is_empty();
        let top_states = stacks.peek_values();
        let mut candidate_terminals = BitSet::empty(future_terminals.len());
        for terminal in future_terminals.iter() {
            let terminal = terminal as TerminalID;
            if Some(terminal) == constraint.ignore_terminal
                || top_states
                    .iter()
                    .any(|&parser_state| constraint.table.advance_row_allows(parser_state, terminal))
            {
                candidate_terminals.set(terminal as usize);
            }
        }

        // `candidate_terminals` is a conservative superset of genuinely
        // applicable parser terminals: row support can include a guarded shift
        // whose lower-stack guard fails.  Validate the set once, exactly, and
        // cache the answer instead of reconstructing a ParserGSS for every trie
        // node that shares this lexer/parser state.
        let any_terminal_admissible = if candidate_terminals.is_empty() {
            false
        } else if constraint
            .ignore_terminal
            .is_some_and(|terminal| candidate_terminals.contains(terminal as usize))
        {
            true
        } else {
            let parser_gss = with_empty_accumulators(stacks);
            stack_may_advance_on_any(&constraint.table, &parser_gss, &candidate_terminals)
        };

        let mut common_terminal_loop_bytes = U8Set::all();
        let mut candidate_first_bytes = U8Set::empty();
        if !candidate_terminals.is_empty() {
            let loops_by_state = vocab.terminal_self_loop_bytes(&constraint.tokenizer);
            let loops = loops_by_state
                .get(tokenizer_state as usize)
                .expect("tokenizer state missing terminal loop row");
            for terminal in candidate_terminals.iter() {
                common_terminal_loop_bytes &= *loops
                    .get(terminal)
                    .expect("terminal missing from terminal loop row");
            }

            for byte in 0..=u8::MAX {
                let Some(next_state) = constraint.tokenizer.step(tokenizer_state, byte) else {
                    continue;
                };
                let reaches_candidate = !candidate_terminals.is_disjoint(
                    constraint.tokenizer.tokens_accessible_from_state(next_state),
                ) || !candidate_terminals.is_disjoint(
                    constraint.tokenizer.matched_terminal_bitset(next_state),
                );
                if reaches_candidate {
                    candidate_first_bytes.insert(byte);
                }
            }
        }

        self.parser_admissions.insert(
            key,
            ParserAdmission {
                stacks: stacks.clone(),
                has_future_terminal,
                candidate_terminals,
                any_terminal_admissible,
                common_terminal_loop_bytes,
                candidate_first_bytes,
            },
        );
        self.parser_admissions
            .get(&key)
            .expect("parser admission cache insertion disappeared")
    }
}

/// Advance every outstanding exclusion through one compressed vocabulary-trie
/// edge. If any excluded terminal matches anywhere on the edge, this traversal
/// branch would duplicate that terminal and is rejected. Otherwise each entry
/// follows its lexer state and keeps only terminals still accessible there.
fn advance_exclusions<O: DynamicMaskObserver>(
    constraint: &Constraint,
    segment: &[u8],
    exclusions: &Exclusions,
    observer: &mut O,
) -> Option<Exclusions> {
    if exclusions.is_empty() {
        return Some(exclusions.clone());
    }

    let mut advanced = ExclusionMap::new();
    for (&tokenizer_state, blocked) in exclusions.iter() {
        observer.exclusion_lexer_execution();
        let execution = constraint
            .tokenizer
            .execute_from_state_all_widths(segment, tokenizer_state);
        if execution
            .matches
            .iter()
            .any(|matched| blocked.contains(&matched.id))
        {
            return None;
        }

        let Some(end_state) = execution.end_state else {
            continue;
        };
        let accessible = constraint.tokenizer.tokens_accessible_from_state(end_state);
        let next_blocked = advanced.entry(end_state).or_default();
        next_blocked.extend(
            blocked
                .iter()
                .copied()
                .filter(|terminal| accessible.contains(*terminal as usize)),
        );
    }
    Some(Arc::new(advanced))
}

/// Record that a terminal committed at this token boundary cannot be matched
/// again by the parallel lexer continuation carried in `exclusions`.
fn with_excluded_terminal(
    exclusions: &Exclusions,
    tokenizer_state: u32,
    terminal: TerminalID,
) -> Exclusions {
    let mut next = (**exclusions).clone();
    next.entry(tokenizer_state).or_default().insert(terminal);
    Arc::new(next)
}

fn parser_child<O: DynamicMaskObserver>(
    constraint: &Constraint,
    stacks: &ParserStacks,
    terminal: TerminalID,
    observer: &mut O,
) -> Option<ParserStacks> {
    observer.parser_child_attempt();
    // Ignore terminals reset the lexer but deliberately leave the parser alone.
    if Some(terminal) == constraint.ignore_terminal {
        observer.parser_child_accept();
        return Some(stacks.clone());
    }
    let parser_gss = with_empty_accumulators(stacks);
    if !stack_may_advance_on(&constraint.table, &parser_gss, terminal) {
        return None;
    }
    let advanced = advance_stacks(&constraint.table, &parser_gss, terminal).apply(|_| ());
    if advanced.is_empty() {
        None
    } else {
        observer.parser_child_accept();
        Some(advanced)
    }
}

fn token_boundary_allowed<O: DynamicMaskObserver>(
    constraint: &Constraint,
    vocab: &DynamicMaskVocab,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicMaskTraversalCache,
    observer: &mut O,
) -> bool {
    observer.boundary_check();
    cache
        .parser_admission(constraint, vocab, tokenizer_state, stacks, observer)
        .any_terminal_admissible
}

fn mark_reachable_tokens<O: DynamicMaskObserver>(
    vocab: &DynamicMaskVocab,
    node: &VocabPrefixTreeNode,
    buf: &mut [u32],
    observer: &mut O,
) {
    observer.marked_subtree_tokens(node.reachable_token_ids().len() as u64);
    for canonical_token_id in node.reachable_token_ids().iter() {
        let token_ids = vocab
            .token_ids
            .get(&(canonical_token_id as u32))
            .expect("dynamic vocabulary trie node lacks token ids");
        for &token_id in token_ids.iter() {
            set_mask_bit(buf, token_id);
        }
    }
}

/// The terminal-wise loop map is a quotient loop, not necessarily a raw DFA
/// loop.  If every possible remaining byte is a loop for every terminal that
/// the parser can currently admit, then the no-finalization continuation has
/// exactly the same terminal language at every descendant vocabulary node.
/// Consequently every token below `node` is admissible through that
/// continuation and no lexer/parser walk is needed.
///
/// `Exclusions` carry parallel lexer continuations that reject a branch if one
/// of their blocked terminals matches.  They may only be skipped when each
/// blocked terminal has the same quotient loop property.  A currently accepting
/// blocked terminal can only reject descendants after one more byte, not the
/// token ending at the current node; a nonaccepting one cannot newly match
/// anywhere in the subtree.
enum TerminalLoopSubtree {
    CannotSkip,
    MarkAllTokens,
    MarkCurrentNodeOnly,
}

fn terminal_loop_bytes<O: DynamicMaskObserver>(
    constraint: &Constraint,
    vocab: &DynamicMaskVocab,
    remaining_bytes: U8Set,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    exclusions: &Exclusions,
    cache: &mut DynamicMaskTraversalCache,
    observer: &mut O,
) -> TerminalLoopSubtree {
    observer.terminal_loop_check();
    let admission = cache.parser_admission(constraint, vocab, tokenizer_state, stacks, observer);
    if !admission.has_future_terminal {
        observer.terminal_loop_no_future();
        return TerminalLoopSubtree::CannotSkip;
    }

    if admission.candidate_terminals.is_empty() {
        observer.terminal_loop_no_candidate();
        return TerminalLoopSubtree::CannotSkip;
    }
    if !remaining_bytes.is_subset(&admission.common_terminal_loop_bytes) {
        observer.terminal_loop_byte_reject();
        return TerminalLoopSubtree::CannotSkip;
    }
    if !admission.any_terminal_admissible {
        observer.terminal_loop_boundary_reject();
        return TerminalLoopSubtree::CannotSkip;
    }

    if !exclusions.is_empty() {
        let loops_by_state = vocab.terminal_self_loop_bytes(&constraint.tokenizer);
        for (&excluded_state, blocked_terminals) in exclusions.iter() {
            let Some(exclusion_loops) = loops_by_state.get(excluded_state as usize) else {
                return TerminalLoopSubtree::CannotSkip;
            };
            for &terminal in blocked_terminals {
                let Some(loop_bytes) = exclusion_loops.get(terminal as usize) else {
                    return TerminalLoopSubtree::CannotSkip;
                };
                if !remaining_bytes.is_subset(loop_bytes) {
                    observer.terminal_loop_exclusion_reject();
                    return TerminalLoopSubtree::CannotSkip;
                }
                if constraint
                    .tokenizer
                    .matched_terminal_bitset(excluded_state)
                    .contains(terminal as usize)
                {
                    observer.terminal_loop_mark_current_only();
                    return TerminalLoopSubtree::MarkCurrentNodeOnly;
                }
            }
        }
    }

    observer.terminal_loop_mark_all();
    TerminalLoopSubtree::MarkAllTokens
}

fn terminal_loop_subtree<O: DynamicMaskObserver>(
    constraint: &Constraint,
    vocab: &DynamicMaskVocab,
    node: &VocabPrefixTreeNode,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    exclusions: &Exclusions,
    cache: &mut DynamicMaskTraversalCache,
    observer: &mut O,
) -> TerminalLoopSubtree {
    terminal_loop_bytes(
        constraint,
        vocab,
        U8Set::from_words(*node.subtree_bytes()),
        tokenizer_state,
        stacks,
        exclusions,
        cache,
        observer,
    )
}

/// Recognize a raw lexer loop that is already reachable from the current
/// state on every byte in its alphabet. Tokens made entirely of that alphabet
/// can take the no-finalization path straight through the loop, so a global
/// vocabulary partition for the alphabet is exact even before the first byte.
fn raw_continuation_loop_partition<O: DynamicMaskObserver>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicMaskTraversalCache,
    observer: &mut O,
) -> Option<Arc<super::artifact::DynamicLoopPartition>> {
    // The shared partition includes an empty token if the vocabulary has one;
    // that token does not consume a byte to enter the loop state. Keep the
    // ordinary exact path for that unusual vocabulary shape.
    if vocab.trie.has_empty_string_token() {
        return None;
    }

    let admission = cache.parser_admission(
        state.constraint,
        vocab,
        tokenizer_state,
        stacks,
        observer,
    );
    if admission.candidate_terminals.is_empty() || !admission.any_terminal_admissible {
        return None;
    }

    let mut incoming_bytes = FxHashMap::<u32, U8Set>::default();
    for byte in admission.candidate_first_bytes.iter() {
        if let Some(next_state) = state.constraint.tokenizer.step(tokenizer_state, byte) {
            incoming_bytes.entry(next_state).or_default().insert(byte);
        }
    }

    let mut best: Option<Arc<super::artifact::DynamicLoopPartition>> = None;
    for (loop_state, covered_first_bytes) in incoming_bytes {
        let loop_bytes = state.constraint.tokenizer.self_loop_bytes(loop_state);
        if loop_bytes.is_empty() || !loop_bytes.is_subset(&covered_first_bytes) {
            continue;
        }
        if !token_boundary_allowed(
            state.constraint,
            vocab,
            loop_state,
            stacks,
            cache,
            observer,
        ) {
            continue;
        }

        let partition = vocab.loop_partition(loop_bytes);
        if partition.safe_token_count <= partition.exception_token_count {
            continue;
        }
        if best.as_ref().map_or(true, |current| {
            partition.safe_token_count > current.safe_token_count
        }) {
            best = Some(partition);
        }
    }
    best
}

/// Build (or reuse) an exact token partition for the no-finalization lexer
/// continuation. A token is safe when consuming all of its bytes reaches a
/// lexer state from which the unchanged parser stacks can still consume some
/// terminal. Such a token is admissible regardless of any terminal matches it
/// may have crossed, because the dynamic semantics retain that continuation as
/// a separate live path.
fn direct_continuation_partition<O: DynamicMaskObserver>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicMaskTraversalCache,
    observer: &mut O,
) -> Option<Arc<DynamicContinuationPartition>> {
    if let Some(partition) =
        lookup_dynamic_continuation_partition(vocab, tokenizer_state, stacks)
    {
        return Some(partition);
    }

    let (candidate_terminals, can_enter_loop) = {
        let admission = cache.parser_admission(
            state.constraint,
            vocab,
            tokenizer_state,
            stacks,
            observer,
        );
        if admission.candidate_terminals.is_empty() || !admission.any_terminal_admissible {
            return None;
        }

        // Building a vocabulary-wide partition is worthwhile only when this
        // state can enter a real lexer loop. Literal-only states instead stay
        // on the compact trie path below. This is structural, not a workload
        // knob.
        let can_enter_loop = admission.candidate_first_bytes.iter().any(|byte| {
        state
            .constraint
            .tokenizer
            .step(tokenizer_state, byte)
            .is_some_and(|next_state| {
                !state
                    .constraint
                    .tokenizer
                    .self_loop_bytes(next_state)
                    .is_empty()
            })
        });
        (admission.candidate_terminals.clone(), can_enter_loop)
    };
    if !can_enter_loop {
        return None;
    }

    let mut safe_mask = vec![0u32; vocab.output_mask_words];
    // `canonical_tokens` are already lexicographically sorted and bytewise
    // deduplicated by `build_dynamic_mask_vocab`, so the exception subset is
    // too. Keep borrowed byte slices and use the presorted trie builder below.
    let mut exception_entries = Vec::<(usize, &[u8])>::new();
    let mut safe_canonical_tokens = 0usize;
    let mut endpoint_allowed = FxHashMap::<u32, bool>::default();

    for (index, &(canonical_token_id, ref bytes)) in vocab.canonical_tokens.iter().enumerate() {
        let mut end_state = tokenizer_state;
        let mut reaches_end = true;
        let mut saw_candidate_match = false;
        for &byte in bytes.iter() {
            let Some(next_state) = state.constraint.tokenizer.step(end_state, byte) else {
                reaches_end = false;
                break;
            };
            end_state = next_state;
            saw_candidate_match |= !state
                .constraint
                .tokenizer
                .matched_terminal_bitset(end_state)
                .is_disjoint(&candidate_terminals);
        }

        let safe = if !reaches_end {
            false
        } else if let Some(&allowed) = endpoint_allowed.get(&end_state) {
            allowed
        } else {
            let allowed = token_boundary_allowed(
                state.constraint,
                vocab,
                end_state,
                stacks,
                cache,
                observer,
            );
            endpoint_allowed.insert(end_state, allowed);
            allowed
        };

        if safe {
            safe_canonical_tokens += 1;
            let token_ids = &vocab.canonical_aliases[index];
            for &token_id in token_ids.iter() {
                let word = token_id as usize / 32;
                debug_assert!(word < safe_mask.len());
                safe_mask[word] |= 1u32 << (token_id % 32);
            }
        } else if saw_candidate_match {
            exception_entries.push((canonical_token_id as usize, bytes.as_ref()));
        }
    }

    let exception_canonical_tokens = exception_entries.len();
    let partition = Arc::new(DynamicContinuationPartition {
        tokenizer_state,
        stacks: stacks.clone(),
        safe_mask: Arc::from(safe_mask.into_boxed_slice()),
        exception_trie: Arc::new(VocabPrefixTree::build_presorted(&exception_entries)),
        safe_canonical_tokens,
        exception_canonical_tokens,
    });
    Some(cache_dynamic_continuation_partition(vocab, partition))
}

fn walk_dynamic_from_seed<'a, O: DynamicMaskObserver>(
    state: &ConstraintState<'_>,
    vocab: &DynamicMaskVocab,
    traversal: &mut Vec<TraverseWork<'a>>,
    initial_tsid: u32,
    buf: &mut [u32],
    cache: &mut DynamicMaskTraversalCache,
    observer: &mut O,
) {
    while let Some(current) = traversal.pop() {
        observer.work_item();
        let subtree_action = terminal_loop_subtree(
            state.constraint,
            vocab,
            current.node,
            current.tokenizer_state,
            &current.gss,
            &current.exclusions,
            cache,
            observer,
        );
        if matches!(subtree_action, TerminalLoopSubtree::MarkAllTokens) {
            mark_reachable_tokens(vocab, current.node, buf, observer);
            continue;
        }

        if current.node.has_token()
            && (current.tokenizer_state == initial_tsid
                || token_boundary_allowed(
                    state.constraint,
                    vocab,
                    current.tokenizer_state,
                    &current.gss,
                    cache,
                    observer,
                ))
        {
            let canonical_token_id = current.node.token_id() as u32;
            let token_ids = vocab
                .token_ids
                .get(&canonical_token_id)
                .expect("dynamic vocabulary trie node lacks token ids");
            for &token_id in token_ids.iter() {
                set_mask_bit(buf, token_id);
            }
        }

        if matches!(subtree_action, TerminalLoopSubtree::MarkCurrentNodeOnly) {
            continue;
        }

        for (segment, child) in current.node.iter_children() {
            observer.trie_edge();

            debug_assert!(!segment.is_empty());
            let admission = cache.parser_admission(
                state.constraint,
                vocab,
                current.tokenizer_state,
                &current.gss,
                observer,
            );
            if !admission.candidate_first_bytes.contains(segment[0]) {
                observer.lexer_first_byte_reject();
                observer.parser_candidate_first_byte_reject();
                continue;
            }

            if current.exclusions.is_empty() {
                let child_bytes =
                    U8Set::from_bytes(segment) | U8Set::from_words(*child.subtree_bytes());
                if matches!(
                    terminal_loop_bytes(
                        state.constraint,
                        vocab,
                        child_bytes,
                        current.tokenizer_state,
                        &current.gss,
                        &current.exclusions,
                        cache,
                        observer,
                    ),
                    TerminalLoopSubtree::MarkAllTokens,
                ) {
                    observer.terminal_loop_child_mark_all();
                    mark_reachable_tokens(vocab, child, buf, observer);
                    continue;
                }
            }

            let Some(segment_exclusions) =
                advance_exclusions(state.constraint, segment, &current.exclusions, observer)
            else {
                continue;
            };

            let mut segment_queue = VecDeque::new();
            segment_queue.push_back((0usize, current.tokenizer_state, current.gss.clone()));

            while let Some((position, tokenizer_state, gss)) = segment_queue.pop_front() {
                observer.lexer_execution();
                let admission = cache.parser_admission(
                    state.constraint,
                    vocab,
                    tokenizer_state,
                    &gss,
                    observer,
                );
                if admission.candidate_terminals.is_empty()
                    || !admission.any_terminal_admissible
                {
                    continue;
                }

                let candidates = &admission.candidate_terminals;
                let mut end_state = tokenizer_state;
                let mut reached_edge_end = true;
                for (offset, &byte) in segment[position..].iter().enumerate() {
                    let Some(next_state) = state.constraint.tokenizer.step(end_state, byte) else {
                        reached_edge_end = false;
                        break;
                    };
                    end_state = next_state;
                    let next_position = position + offset + 1;

                    let finalizers = state.constraint.tokenizer.matched_terminal_bitset(end_state);
                    if !finalizers.is_disjoint(candidates) {
                        for terminal in finalizers.iter() {
                            let terminal = terminal as TerminalID;
                            if !candidates.contains(terminal as usize) {
                                continue;
                            }
                            observer.terminal_match();
                            let Some(advanced_parser) =
                                parser_child(state.constraint, &gss, terminal, observer)
                            else {
                                continue;
                            };

                            if next_position == segment.len() {
                                let exclusions = with_excluded_terminal(
                                    &segment_exclusions,
                                    end_state,
                                    terminal,
                                );
                                traversal.push(TraverseWork {
                                    node: child,
                                    tokenizer_state: initial_tsid,
                                    gss: advanced_parser,
                                    exclusions,
                                });
                                observer.traversal_push();
                            } else {
                                segment_queue.push_back((
                                    next_position,
                                    initial_tsid,
                                    advanced_parser,
                                ));
                            }
                        }
                    }

                    if state
                        .constraint
                        .tokenizer
                        .tokens_accessible_from_state(end_state)
                        .is_disjoint(candidates)
                    {
                        reached_edge_end = false;
                        break;
                    }
                }

                if reached_edge_end {
                    match terminal_loop_subtree(
                        state.constraint,
                        vocab,
                        child,
                        end_state,
                        &gss,
                        &segment_exclusions,
                        cache,
                        observer,
                    ) {
                        TerminalLoopSubtree::MarkAllTokens => {
                            observer.terminal_loop_post_edge_mark_all();
                            mark_reachable_tokens(vocab, child, buf, observer);
                        }
                        TerminalLoopSubtree::CannotSkip
                        | TerminalLoopSubtree::MarkCurrentNodeOnly => {
                            traversal.push(TraverseWork {
                                node: child,
                                tokenizer_state: end_state,
                                gss,
                                exclusions: segment_exclusions.clone(),
                            });
                            observer.traversal_push();
                        }
                    }
                }
            }
        }
    }
}

fn fill_mask_dynamic_inner<O: DynamicMaskObserver>(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
    observer: &mut O,
) {
    let vocab = &state.constraint.dynamic_mask_vocab;

    buf.fill(0);
    let initial_tsid = state.constraint.tokenizer.initial_state();
    let mut traversal = Vec::<TraverseWork<'_>>::new();
    let mut cache = DynamicMaskTraversalCache::default();

    for (&tokenizer_state, gss) in &state.state {
        for (stacks, exclusions) in gss.partition_by_accumulator() {
            traversal.push(TraverseWork {
                node: &vocab.trie.root,
                tokenizer_state,
                gss: stacks,
                exclusions: exclusions.0,
            });
            observer.traversal_push();
        }
    }

    for (&tokenizer_state, gss) in &state.state {
        for (stacks, exclusions) in gss.partition_by_accumulator() {
            if exclusions.0.is_empty() {
                if let Some(partition) = raw_continuation_loop_partition(
                    state,
                    vocab,
                    tokenizer_state,
                    &stacks,
                    &mut cache,
                    observer,
                ) {
                    or_mask(buf, &partition.safe_mask);
                    if partition.exception_token_count != 0 {
                        let mut traversal = vec![TraverseWork {
                            node: &partition.exception_trie.root,
                            tokenizer_state,
                            gss: stacks,
                            exclusions: exclusions.0,
                        }];
                        observer.traversal_push();
                        walk_dynamic_from_seed(
                            state,
                            vocab,
                            &mut traversal,
                            initial_tsid,
                            buf,
                            &mut cache,
                            observer,
                        );
                    }
                    continue;
                }

                if let Some(partition) = direct_continuation_partition(
                    state,
                    vocab,
                    tokenizer_state,
                    &stacks,
                    &mut cache,
                    observer,
                ) {
                    let represented_tokens = partition.safe_canonical_tokens
                        + partition.exception_canonical_tokens;
                    if represented_tokens < vocab.canonical_tokens.len()
                        || partition.safe_canonical_tokens > partition.exception_canonical_tokens
                    {
                        or_mask(buf, &partition.safe_mask);
                        if partition.exception_canonical_tokens != 0 {
                            let mut traversal = vec![TraverseWork {
                                node: &partition.exception_trie.root,
                                tokenizer_state,
                                gss: stacks,
                                exclusions: exclusions.0,
                            }];
                            observer.traversal_push();
                            walk_dynamic_from_seed(
                                state,
                                vocab,
                                &mut traversal,
                                initial_tsid,
                                buf,
                                &mut cache,
                                observer,
                            );
                        }
                        continue;
                    }
                }
            }

            let loop_bytes = if exclusions.0.is_empty() {
                let admission = cache.parser_admission(
                    state.constraint,
                    vocab,
                    tokenizer_state,
                    &stacks,
                    observer,
                );
                (admission.any_terminal_admissible
                    && !admission.candidate_terminals.is_empty()
                    && admission.common_terminal_loop_bytes.len() >= 32)
                    .then_some(admission.common_terminal_loop_bytes)
            } else {
                None
            };

            if let Some(loop_bytes) = loop_bytes {
                let partition = vocab.loop_partition(loop_bytes);
                if partition.safe_token_count >= 1024
                    && partition.exception_token_count.saturating_mul(2)
                        < vocab.canonical_token_bytes.len()
                {
                    observer.loop_partition(
                        loop_bytes.len(),
                        partition.safe_token_count,
                        partition.exception_token_count,
                    );
                    or_mask(buf, &partition.safe_mask);
                    if partition.exception_token_count != 0 {
                        let mut traversal = vec![TraverseWork {
                            node: &partition.exception_trie.root,
                            tokenizer_state,
                            gss: stacks,
                            exclusions: exclusions.0,
                        }];
                        observer.traversal_push();
                        walk_dynamic_from_seed(
                            state,
                            vocab,
                            &mut traversal,
                            initial_tsid,
                            buf,
                            &mut cache,
                            observer,
                        );
                    }
                    continue;
                }
            }

            let mut traversal = vec![TraverseWork {
                node: &vocab.trie.root,
                tokenizer_state,
                gss: stacks,
                exclusions: exclusions.0,
            }];
            observer.traversal_push();
            walk_dynamic_from_seed(
                state,
                vocab,
                &mut traversal,
                initial_tsid,
                buf,
                &mut cache,
                observer,
            );
        }
    }

    update_eos_mask(state, buf);
}

pub(crate) fn fill_mask_dynamic(state: &ConstraintState<'_>, buf: &mut [u32]) {
    fill_mask_dynamic_inner(state, buf, &mut NoopDynamicMaskObserver);
}

pub(crate) fn fill_mask_dynamic_profiled(
    state: &ConstraintState<'_>,
    buf: &mut [u32],
) -> DynamicMaskProfile {
    let started = Instant::now();
    let mut profile = DynamicMaskProfile::default();
    fill_mask_dynamic_inner(state, buf, &mut profile);
    profile.total_ns = started.elapsed().as_nanos() as u64;
    profile
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Constraint, Vocab};

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word).is_some_and(|word| word & (1u32 << bit) != 0)
    }

    fn direct_mask(state: &ConstraintState<'_>) -> Vec<u32> {
        let mut mask = vec![0u32; state.constraint.mask_len()];
        state.fill_mask_dynamic(&mut mask);
        mask
    }

    fn assert_dynamic_parity(state: &ConstraintState<'_>) {
        assert_eq!(state.mask(), direct_mask(state));
    }

    #[test]
    fn dynamic_mask_matches_normal_for_repeat_and_cross_terminal_tokens() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b"ab".to_vec()),
                (4, b"aab".to_vec()),
                (5, b"aaa".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B | A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));

        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 2));

        state.commit_token(2).unwrap();
        assert!(state.is_finished());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_trie_is_rebuilt_after_load() {
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec())],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert_dynamic_parity(&loaded.start());
    }

    #[test]
    fn dynamic_mask_keeps_duplicate_byte_token_aliases() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (7, b"a".to_vec()),
                (12, b"ab".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a';
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        let mask = direct_mask(&state);
        assert!(token_allowed(&mask, 0));
        assert!(token_allowed(&mask, 7));
        assert!(token_allowed(&mask, 12));

        state.commit_token(7).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 1));
    }

    #[test]
    fn dynamic_mask_matches_normal_across_an_ignore_terminal() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b" b".to_vec()),
                (4, b"  b".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+;
t A ::= 'a'+;
t B ::= 'b';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));

        state.commit_token(3).unwrap();
        assert!(state.is_finished());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_matches_normal_at_every_reachable_small_state() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"b".to_vec()),
                (3, b"bb".to_vec()),
                (4, b"c".to_vec()),
                (5, b"ab".to_vec()),
                (6, b"ba".to_vec()),
                (7, b"a c".to_vec()),
                (8, b"b c".to_vec()),
                (9, b" aa".to_vec()),
                (10, b" bb".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
ignore WS;
t WS ::= ' '+;
t A ::= 'a'+;
t B ::= 'b'+;
t C ::= 'c';
nt start ::= A B C | B A C | A C | B C;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        fn visit(state: ConstraintState<'_>, depth: usize) {
            assert_dynamic_parity(&state);
            if depth == 3 {
                return;
            }
            let mask = state.mask();
            for token_id in 0..11u32 {
                if !token_allowed(&mask, token_id) {
                    continue;
                }
                let mut next = state.clone();
                next.commit_token(token_id).unwrap();
                visit(next, depth + 1);
            }
        }

        visit(constraint.start(), 0);
    }

    #[test]
    fn dynamic_mask_matches_normal_when_one_repeated_terminal_crosses_tokens() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"aa".to_vec()),
                (2, b"aaa".to_vec()),
                (3, b"aaaa".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        fn visit(state: ConstraintState<'_>, depth: usize) {
            assert_dynamic_parity(&state);
            if depth == 3 {
                return;
            }
            let mask = state.mask();
            for token_id in 0..4u32 {
                if !token_allowed(&mask, token_id) {
                    continue;
                }
                let mut next = state.clone();
                next.commit_token(token_id).unwrap();
                visit(next, depth + 1);
            }
        }

        visit(constraint.start(), 0);
    }

    #[test]
    fn dynamic_mask_matches_normal_for_a_partial_json_string() {
        let vocab = Vocab::new(
            vec![
                (0, b"\"".to_vec()),
                (1, b"a".to_vec()),
                (2, b"b".to_vec()),
                (3, b"\\\"".to_vec()),
                (4, b"\"a".to_vec()),
                (5, b"a\"".to_vec()),
            ],
            None,
        );
        let constraint =
            Constraint::from_json_schema(r#"{"type":"string"}"#, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_handles_monolithic_json_number() {
        let vocab = Vocab::new(
            vec![
                (0, b"-".to_vec()),
                (1, b"0".to_vec()),
                (2, b"1".to_vec()),
                (3, b"2".to_vec()),
                (4, b"3".to_vec()),
                (5, b".".to_vec()),
                (6, b"e".to_vec()),
                (7, b"+".to_vec()),
            ],
            None,
        );
        let constraint =
            Constraint::from_json_schema(r#"{"type":"number"}"#, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        for bytes in [b"1".as_slice(), b".".as_slice(), b"2".as_slice(), b"e".as_slice(), b"-".as_slice(), b"3".as_slice()] {
            state.commit_bytes(bytes).unwrap();
            assert_dynamic_parity(&state);
        }
        assert!(state.is_complete());
    }

    #[test]
    fn dynamic_mask_keeps_other_gss_paths_when_one_path_excludes_a_terminal() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"ab".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'ab';
t B ::= 'a';
t C ::= 'c';
t D ::= 'b';
nt start ::= A C | B D;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_token(0).unwrap();

        let paths = state
            .state
            .values()
            .flat_map(|gss| gss.to_stacks())
            .collect::<Vec<_>>();
        assert!(paths.iter().any(|(_, exclusions)| exclusions.is_empty()));
        assert!(paths.iter().any(|(_, exclusions)| !exclusions.is_empty()));

        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 1));
        state.commit_token(1).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_handles_overlapping_live_terminal_paths() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"b".to_vec()),
                (3, b"bc".to_vec()),
                (4, b"c".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'ab';
t B ::= 'b' | 'bc';
nt start ::= A B;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));
        state.commit_token(3).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_mask_handles_a_live_cross_terminal_prefix() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"abc".to_vec()),
                (3, b"bc".to_vec()),
                (4, b"c".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'abc';
nt start ::= A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(0).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));
        assert!(!token_allowed(&state.mask(), 4));
        state.commit_token(3).unwrap();
        assert!(state.is_complete());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn terminal_loop_skip_handles_an_already_accepting_exclusion() {
        use std::collections::{BTreeMap, BTreeSet};

        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"aa".to_vec()), (2, b"aaa".to_vec())],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a'+;
nt start ::= A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let state = constraint.start();
        let initial = constraint.tokenizer.initial_state();
        let (stacks, _) = state
            .state
            .get(&initial)
            .unwrap()
            .partition_by_accumulator()
            .into_iter()
            .next()
            .unwrap();
        let continuation = constraint
            .tokenizer
            .execute_from_state_all_widths(b"a", initial)
            .end_state
            .unwrap();
        let (_, node) = constraint
            .dynamic_mask_vocab
            .trie
            .root
            .iter_children()
            .next()
            .unwrap();

        let empty = Arc::new(BTreeMap::new());
        let mut observer = NoopDynamicMaskObserver;
        let mut cache = DynamicMaskTraversalCache::default();
        assert!(matches!(
            terminal_loop_subtree(
                &constraint,
                &constraint.dynamic_mask_vocab,
                node,
                continuation,
                &stacks,
                &empty,
                &mut cache,
                &mut observer,
            ),
            TerminalLoopSubtree::MarkAllTokens,
        ));

        let exclusions = Arc::new(BTreeMap::from([(
            continuation,
            BTreeSet::from([0]),
        )]));
        assert!(matches!(
            terminal_loop_subtree(
                &constraint,
                &constraint.dynamic_mask_vocab,
                node,
                continuation,
                &stacks,
                &exclusions,
                &mut cache,
                &mut observer,
            ),
            TerminalLoopSubtree::MarkCurrentNodeOnly,
        ));
    }

}
