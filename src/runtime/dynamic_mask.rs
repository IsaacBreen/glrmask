//! Direct dynamic mask generation.
//!
//! This implementation intentionally does not consult the parser DWA. It walks
//! the vocabulary byte trie while advancing the lexer and GLR parser directly.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::automata::lexer::Lexer;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{
    advance_stacks, stack_may_advance_on, stack_may_advance_on_any, ParserGSS,
};
use crate::ds::leveled_gss::LeveledGSS;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

use super::artifact::{Constraint, DynamicMaskStateKey, DynamicMaskTrie};
use super::state::ConstraintState;

type ExclusionMap = BTreeMap<u32, BTreeSet<TerminalID>>;
type Exclusions = Arc<ExclusionMap>;
type ParserStacks = LeveledGSS<u32, ()>;

#[derive(Default)]
struct DynamicTraversalCache {
    boundary: FxHashMap<(u32, usize), (ParserStacks, bool)>,
    lexer_relevant: FxHashMap<(u32, usize), (ParserStacks, bool)>,
    parser_children: FxHashMap<(usize, TerminalID), (ParserStacks, Option<ParserStacks>)>,
}

#[derive(Clone)]
struct TraverseWork {
    trie_index: usize,
    node: u32,
    tokenizer_state: u32,
    gss: ParserStacks,
    exclusions: Exclusions,
    continuation_filter: Option<(usize, u64)>,
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

/// Advance outstanding tokenizer-state-correlated exclusions through one
/// compressed vocabulary-trie edge. A blocked match rejects the traversal only
/// when it belongs to the active lexer branch. The same match on a parallel
/// residual kills that residual alone. Surviving entries follow their own lexer
/// state and keep only terminals still accessible there.
fn advance_exclusions(
    constraint: &Constraint,
    segment: &[u8],
    active_tokenizer_state: u32,
    exclusions: &Exclusions,
) -> Option<Exclusions> {
    if exclusions.is_empty() {
        return Some(exclusions.clone());
    }

    let mut advanced = ExclusionMap::new();
    for (&tokenizer_state, blocked) in exclusions.iter() {
        let execution = constraint
            .tokenizer
            .execute_from_state_all_widths(segment, tokenizer_state);
        if execution
            .matches
            .iter()
            .any(|matched| blocked.contains(&matched.id))
        {
            if tokenizer_state == active_tokenizer_state {
                return None;
            }
            // Restrictions are correlated with their tokenizer-state branch.
            // A blocked match on a parallel residual kills that residual, not
            // the independently reset/active lexer branch being traversed.
            continue;
        }

        for &end_state in &execution.end_state {
            let accessible = constraint.tokenizer.tokens_accessible_from_state(end_state);
            let next_blocked = advanced.entry(end_state).or_default();
            next_blocked.extend(
                blocked
                    .iter()
                    .copied()
                    .filter(|terminal| accessible.contains(*terminal as usize)),
            );
        }
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

fn parser_child(
    constraint: &Constraint,
    stacks: &ParserStacks,
    terminal: TerminalID,
) -> Option<ParserStacks> {
    // Ignore terminals reset the lexer but deliberately leave the parser alone.
    if Some(terminal) == constraint.ignore_terminal {
        return Some(stacks.clone());
    }
    let parser_gss = with_empty_accumulators(stacks);
    if !stack_may_advance_on(&constraint.table, &parser_gss, terminal) {
        return None;
    }
    let advanced = advance_stacks(&constraint.table, &parser_gss, terminal).apply(|_| ());
    (!advanced.is_empty()).then_some(advanced)
}

fn parser_child_cached(
    constraint: &Constraint,
    stacks: &ParserStacks,
    terminal: TerminalID,
    cache: &mut DynamicTraversalCache,
) -> Option<ParserStacks> {
    let key = (stacks.ptr_key(), terminal);
    if let Some((cached_stacks, result)) = cache.parser_children.get(&key) {
        debug_assert!(cached_stacks.ptr_eq(stacks));
        return result.clone();
    }
    let result = parser_child(constraint, stacks, terminal);
    cache
        .parser_children
        .insert(key, (stacks.clone(), result.clone()));
    result
}

fn token_boundary_allowed(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
) -> bool {
    let parser_gss = with_empty_accumulators(stacks);
    constraint
        .tokenizer
        .tokens_accessible_from_state(tokenizer_state)
        .iter()
        .any(|terminal| {
            let terminal = terminal as TerminalID;
            Some(terminal) == constraint.ignore_terminal
                || stack_may_advance_on(&constraint.table, &parser_gss, terminal)
        })
}

fn token_boundary_allowed_cached(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicTraversalCache,
) -> bool {
    let key = (tokenizer_state, stacks.ptr_key());
    if let Some((cached_stacks, result)) = cache.boundary.get(&key) {
        debug_assert!(cached_stacks.ptr_eq(stacks));
        return *result;
    }
    let result = token_boundary_allowed(constraint, tokenizer_state, stacks);
    cache
        .boundary
        .insert(key, (stacks.clone(), result));
    result
}

fn lexer_state_relevant_cached(
    constraint: &Constraint,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    cache: &mut DynamicTraversalCache,
) -> bool {
    let key = (tokenizer_state, stacks.ptr_key());
    if let Some((cached_stacks, result)) = cache.lexer_relevant.get(&key) {
        debug_assert!(cached_stacks.ptr_eq(stacks));
        return *result;
    }

    let accessible = constraint
        .tokenizer
        .tokens_accessible_from_state(tokenizer_state);
    let matched = constraint.tokenizer.matched_terminal_bitset(tokenizer_state);
    let ignore_relevant = constraint.ignore_terminal.is_some_and(|terminal| {
        accessible.contains(terminal as usize) || matched.contains(terminal as usize)
    });
    let result = if ignore_relevant {
        true
    } else {
        let parser_gss = with_empty_accumulators(stacks);
        stack_may_advance_on_any(&constraint.table, &parser_gss, accessible)
            || stack_may_advance_on_any(&constraint.table, &parser_gss, matched)
    };
    cache
        .lexer_relevant
        .insert(key, (stacks.clone(), result));
    result
}

#[inline]
fn mark_subtree_tokens(
    constraint: &Constraint,
    trie: &DynamicMaskTrie,
    node: u32,
    buf: &mut [u32],
) {
    for &canonical_token_id in trie.subtree_tokens(node) {
        let token_ids = constraint
            .dynamic_mask_vocab
            .token_ids(canonical_token_id)
            .expect("dynamic vocabulary trie node lacks token ids");
        for &token_id in token_ids {
            set_mask_bit(buf, token_id);
        }
    }
}

enum RawSelfLoopSubtree {
    CannotSkip,
    MarkAllTokens,
    MarkCurrentNodeOnly,
}

#[inline]
fn cached_self_loop_bytes(
    constraint: &Constraint,
    tokenizer_state: u32,
    cache: &mut FxHashMap<u32, U8Set>,
) -> U8Set {
    *cache
        .entry(tokenizer_state)
        .or_insert_with(|| constraint.tokenizer.self_loop_bytes(tokenizer_state))
}

/// A raw tokenizer-state self-loop is a particularly strong residual-language
/// certificate: consuming the byte leaves every lexer possibility exactly
/// unchanged. If every remaining byte below a vocabulary-trie node has that
/// property, the no-finalization continuation witnesses every token in the
/// subtree without any per-token lexer or parser work.
///
/// Parallel exclusion continuations can use the same shortcut only while they
/// also remain in place. If a blocked terminal is already accepting there,
/// every non-empty descendant would immediately match it; the token ending at
/// the current node can still be retained.
fn raw_self_loop_subtree(
    constraint: &Constraint,
    trie: &DynamicMaskTrie,
    node: u32,
    tokenizer_state: u32,
    stacks: &ParserStacks,
    exclusions: &Exclusions,
    initial_tsid: u32,
    self_loop_cache: &mut FxHashMap<u32, U8Set>,
    traversal_cache: &mut DynamicTraversalCache,
) -> RawSelfLoopSubtree {
    // Work at the initial state may represent either an untouched lexer or a
    // lexer reset after an in-token terminal match. The current work item does
    // not distinguish those cases, so keep this optimization conservative.
    if tokenizer_state == initial_tsid {
        return RawSelfLoopSubtree::CannotSkip;
    }

    let subtree_bytes = U8Set::from_words(trie.subtree_bytes(node));
    let loop_bytes = cached_self_loop_bytes(constraint, tokenizer_state, self_loop_cache);
    if !subtree_bytes.is_subset(&loop_bytes)
        || !token_boundary_allowed_cached(constraint, tokenizer_state, stacks, traversal_cache)
    {
        return RawSelfLoopSubtree::CannotSkip;
    }

    for (&excluded_state, blocked_terminals) in exclusions.iter() {
        let exclusion_loops = cached_self_loop_bytes(constraint, excluded_state, self_loop_cache);
        if !subtree_bytes.is_subset(&exclusion_loops) {
            return RawSelfLoopSubtree::CannotSkip;
        }
        let matched = constraint.tokenizer.matched_terminal_bitset(excluded_state);
        if blocked_terminals
            .iter()
            .any(|&terminal| matched.contains(terminal as usize))
        {
            return RawSelfLoopSubtree::MarkCurrentNodeOnly;
        }
    }

    RawSelfLoopSubtree::MarkAllTokens
}

fn dynamic_mask_state_key(state: &ConstraintState<'_>) -> DynamicMaskStateKey {
    state
        .state
        .iter()
        .map(|(&tokenizer_state, gss)| {
            let mut paths = gss
                .to_stacks()
                .into_iter()
                .map(|(stack, exclusions)| {
                    let exclusion_entries = exclusions
                        .0
                        .iter()
                        .map(|(&excluded_state, terminals)| {
                            (excluded_state, terminals.iter().copied().collect::<Vec<_>>())
                        })
                        .collect::<Vec<_>>();
                    (stack, exclusion_entries)
                })
                .collect::<Vec<_>>();
            paths.sort_unstable();
            (tokenizer_state, paths)
        })
        .collect()
}

pub(crate) fn fill_mask_dynamic(state: &ConstraintState<'_>, buf: &mut [u32]) {
    let vocab = &state.constraint.dynamic_mask_vocab;
    let profile = std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some();
    let total_started_at = profile.then(std::time::Instant::now);
    let key_started_at = profile.then(std::time::Instant::now);
    let cache_key = dynamic_mask_state_key(state);
    let key_ms = key_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

    if vocab.copy_cached_mask(&cache_key, buf) {
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][dynamic_mask] generation={} cache_hit=true key_ms={:.3} total_ms={:.3}",
                state.generation,
                key_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return;
    }

    buf.fill(0);
    let initial_tsid = state.constraint.tokenizer.initial_state();
    let mut traversal = Vec::<TraverseWork>::with_capacity(4096);
    let mut segment_stack = Vec::<(usize, u32, ParserStacks)>::with_capacity(8);
    let mut self_loop_cache = FxHashMap::<u32, U8Set>::default();
    let mut traversal_cache = DynamicTraversalCache::default();
    let tries = [vocab.trie.clone()];
    let mut continuation_partitions = Vec::new();
    let mut work_items = 0usize;
    let mut trie_edges = 0usize;
    let mut lexer_executions = 0usize;
    let mut subtree_marks = 0usize;
    let mut subtree_mark_tokens = 0usize;
    let mut continuation_groups_admitted = 0usize;
    let mut continuation_groups_traversed = 0usize;
    if profile {
        eprintln!(
            "[glrmask/profile][dynamic_mask_config] tokenizer_states={} epsilon={} fast_transition_rows={}",
            state.constraint.tokenizer.num_states(),
            state.constraint.tokenizer.has_epsilon_transitions(),
            state.constraint.tokenizer_fast_transitions.len(),
        );
    }

    for (&tokenizer_state, gss) in &state.state {
        for (stacks, exclusions) in gss.partition_by_accumulator() {
            if profile {
                let loop_bytes = cached_self_loop_bytes(
                    state.constraint,
                    tokenizer_state,
                    &mut self_loop_cache,
                );
                eprintln!(
                    "[glrmask/profile][dynamic_seed] generation={} tokenizer_state={} initial={} stack_paths={} exclusions={} loop_bytes={} boundary_allowed={}",
                    state.generation,
                    tokenizer_state,
                    tokenizer_state == initial_tsid,
                    stacks.path_count_at_most(1_000_000),
                    exclusions.0.values().map(BTreeSet::len).sum::<usize>(),
                    loop_bytes.len(),
                    token_boundary_allowed_cached(
                        state.constraint,
                        tokenizer_state,
                        &stacks,
                        &mut traversal_cache,
                    ),
                );
            }
            if exclusions.0.is_empty()
                && let Some(partition) = vocab.cached_continuation_partition(tokenizer_state)
            {
                let mut admitted_groups = 0u64;
                for (group_id, group) in partition.groups.iter().enumerate() {
                    let admitted = group.end_states.iter().any(|&end_state| {
                        token_boundary_allowed_cached(
                            state.constraint,
                            end_state,
                            &stacks,
                            &mut traversal_cache,
                        )
                    });
                    if admitted {
                        or_mask(buf, &group.mask);
                        admitted_groups |= 1u64 << group_id;
                        continuation_groups_admitted += 1;
                    }
                }
                let all_groups = if partition.groups.len() == 64 {
                    u64::MAX
                } else {
                    (1u64 << partition.groups.len()) - 1
                };
                let required_groups = all_groups & !admitted_groups;
                if required_groups != 0 {
                    let partition_index = continuation_partitions.len();
                    continuation_partitions.push(partition);
                    traversal.push(TraverseWork {
                        trie_index: 0,
                        node: 0,
                        tokenizer_state,
                        gss: stacks.clone(),
                        exclusions: exclusions.0.clone(),
                        continuation_filter: Some((partition_index, required_groups)),
                    });
                    continuation_groups_traversed += required_groups.count_ones() as usize;
                }
                continue;
            }

            traversal.push(TraverseWork {
                trie_index: 0,
                node: 0,
                tokenizer_state,
                gss: stacks,
                exclusions: exclusions.0,
                continuation_filter: None,
            });
        }
    }

    while let Some(current) = traversal.pop() {
        work_items += 1;
        let trie = &tries[current.trie_index];
        let node = trie.node(current.node);
        if let Some((partition_index, required_groups)) = current.continuation_filter
            && continuation_partitions[partition_index].subtree_groups(current.node)
                & required_groups
                == 0
        {
            continue;
        }
        let subtree_action = raw_self_loop_subtree(
            state.constraint,
            trie,
            current.node,
            current.tokenizer_state,
            &current.gss,
            &current.exclusions,
            initial_tsid,
            &mut self_loop_cache,
            &mut traversal_cache,
        );
        if matches!(subtree_action, RawSelfLoopSubtree::MarkAllTokens) {
            subtree_marks += 1;
            subtree_mark_tokens += trie.subtree_tokens(current.node).len();
            mark_subtree_tokens(state.constraint, trie, current.node, buf);
            continue;
        }

        let token_is_required = current.continuation_filter.is_none_or(
            |(partition_index, required_groups)| {
                node.token_id
                    .and_then(|token_id| {
                        continuation_partitions[partition_index].token_group(token_id)
                    })
                    .is_some_and(|group| required_groups & (1u64 << group) != 0)
            },
        );
        if token_is_required
            && node.token_id.is_some()
            && (current.tokenizer_state == initial_tsid
                || token_boundary_allowed_cached(
                    state.constraint,
                    current.tokenizer_state,
                    &current.gss,
                    &mut traversal_cache,
                ))
        {
            let canonical_token_id = node.token_id.expect("token leaf checked");
            let token_ids = vocab
                .token_ids(canonical_token_id)
                .expect("dynamic vocabulary trie node lacks token ids");
            for &token_id in token_ids {
                set_mask_bit(buf, token_id);
            }
        }

        if matches!(subtree_action, RawSelfLoopSubtree::MarkCurrentNodeOnly) {
            continue;
        }

        for edge in trie.children(current.node) {
            if let Some((partition_index, required_groups)) = current.continuation_filter
                && continuation_partitions[partition_index].subtree_groups(edge.child)
                    & required_groups
                    == 0
            {
                continue;
            }
            trie_edges += 1;
            let segment = trie.edge_bytes(edge);
            let Some(segment_exclusions) =
                advance_exclusions(
                    state.constraint,
                    segment,
                    current.tokenizer_state,
                    &current.exclusions,
                )
            else {
                continue;
            };

            segment_stack.clear();
            segment_stack.push((0usize, current.tokenizer_state, current.gss.clone()));

            while let Some((position, tokenizer_state, gss)) = segment_stack.pop() {
                lexer_executions += 1;
                let execution = state
                    .constraint
                    .tokenizer
                    .execute_from_state_all_widths(&segment[position..], tokenizer_state);

                for matched in &execution.matches {
                    debug_assert!(matched.width > 0);
                    let Some(advanced_parser) = parser_child_cached(
                        state.constraint,
                        &gss,
                        matched.id,
                        &mut traversal_cache,
                    )
                    else {
                        continue;
                    };

                    let next_position = position + matched.width;
                    if next_position == segment.len() {
                        let mut exclusions = segment_exclusions.clone();
                        for &end_state in &execution.end_state {
                            exclusions = with_excluded_terminal(&exclusions, end_state, matched.id);
                        }
                        traversal.push(TraverseWork {
                            trie_index: current.trie_index,
                            node: edge.child,
                            tokenizer_state: initial_tsid,
                            gss: advanced_parser,
                            exclusions,
                            continuation_filter: current.continuation_filter,
                        });
                    } else {
                        segment_stack.push((next_position, initial_tsid, advanced_parser));
                    }
                }

                for &end_state in &execution.end_state {
                    if !lexer_state_relevant_cached(
                        state.constraint,
                        end_state,
                        &gss,
                        &mut traversal_cache,
                    ) {
                        continue;
                    }
                    traversal.push(TraverseWork {
                        trie_index: current.trie_index,
                        node: edge.child,
                        tokenizer_state: end_state,
                        gss: gss.clone(),
                        exclusions: segment_exclusions.clone(),
                        continuation_filter: current.continuation_filter,
                    });
                }
            }
        }
    }

    update_eos_mask(state, buf);
    vocab.cache_mask(cache_key, buf);
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][dynamic_mask] generation={} cache_hit=false key_ms={:.3} work_items={} trie_edges={} lexer_execs={} subtree_marks={} subtree_tokens={} continuation_admitted={} continuation_traversed={} boundary_cache={} relevant_cache={} child_cache={} total_ms={:.3}",
            state.generation,
            key_ms,
            work_items,
            trie_edges,
            lexer_executions,
            subtree_marks,
            subtree_mark_tokens,
            continuation_groups_admitted,
            continuation_groups_traversed,
            traversal_cache.boundary.len(),
            traversal_cache.lexer_relevant.len(),
            traversal_cache.parser_children.len(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
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
    fn dynamic_mask_preserves_repeated_terminal_after_ignore_reset_inside_token() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"aa".to_vec()),
                (4, b"bb".to_vec()),
                (5, b"cc".to_vec()),
                (6, b"ab".to_vec()),
                (7, b"ac".to_vec()),
                (8, b"ba".to_vec()),
                (9, b"bc".to_vec()),
                (10, b"abc".to_vec()),
                (11, b"aab".to_vec()),
                (12, b"abb".to_vec()),
                (13, b"acc".to_vec()),
                (14, b" ".to_vec()),
                (15, b"  ".to_vec()),
                (16, b" a".to_vec()),
                (17, b"a ".to_vec()),
                (18, b" a ".to_vec()),
                (19, b"ab c".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
ignore WS;
lexer group ws ::= WS;
lexer group a ::= A;
lexer group b ::= B;
lexer group c ::= C;
t WS ::= " "+;
t A ::= "a"+;
t B ::= "b";
t C ::= "c";
nt item ::= A | B | C;
nt start ::= item item? item?;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_token(0).unwrap();
        state.commit_token(16).unwrap();

        assert_dynamic_parity(&state);
        assert!(token_allowed(&direct_mask(&state), 0));
        assert!(token_allowed(&direct_mask(&state), 3));
        assert!(token_allowed(&direct_mask(&state), 17));
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

}
