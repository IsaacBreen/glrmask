//! Direct dynamic mask generation.
//!
//! This implementation intentionally does not consult the parser DWA. It walks
//! the vocabulary byte trie while advancing the lexer and GLR parser directly.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::OnceLock;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::automata::lexer::tokenizer::TokenizerMatch;
use crate::automata::lexer::Lexer;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{advance_stacks, stack_may_advance_on, ParserGSS};
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use crate::grammar::flat::TerminalID;

use super::artifact::Constraint;
use super::state::ConstraintState;

/// A node in the parser-state trie cache. Children are indexed by the lexer
/// state at which the terminal was matched as well as the terminal itself,
/// because disallowed-follow accumulators are keyed by lexer state.
struct ParserTrieNode {
    gss: ParserGSS,
    terminal_children: FxHashMap<(u32, TerminalID, Option<u32>), usize>,
    /// Lazily populated token-boundary answers, one for each active lexer
    /// state. The cache is state-sensitive for the same accumulator reason.
    token_boundary_allowed: FxHashMap<u32, bool>,
    admissible_terminals: FxHashMap<u32, BitSet>,
    /// For an unrestricted GSS, parser-admissible terminals are independent of
    /// lexer state. Keeping this once per parser node avoids recomputing the
    /// same GLR admission set for every lexer state reached in a wide string.
    unrestricted_admissible_terminals: Option<BitSet>,
}

impl ParserTrieNode {
    fn new(gss: ParserGSS) -> Self {
        Self {
            gss,
            terminal_children: FxHashMap::default(),
            token_boundary_allowed: FxHashMap::default(),
            admissible_terminals: FxHashMap::default(),
            unrestricted_admissible_terminals: None,
        }
    }
}

#[derive(Clone, Copy)]
struct TraverseWork<'a> {
    node: &'a VocabPrefixTreeNode,
    tokenizer_state: u32,
    parser_node: usize,
    exclusion: Option<(u32, TerminalID)>,
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
fn mask_bit_is_set(buf: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    buf.get(word).is_some_and(|slot| *slot & (1u32 << bit) != 0)
}

/// Set every token bit in an inclusive ID range. Dynamic vocab tries use
/// canonical token IDs, which are not byte-sorted, but their reachable-ID sets
/// retain compact numeric ranges from the original vocabulary.
fn set_mask_range(buf: &mut [u32], start: usize, end: usize) {
    let capacity = buf.len().saturating_mul(32);
    if start >= capacity {
        return;
    }
    let end = end.min(capacity - 1);
    if start > end {
        return;
    }

    let first_word = start / 32;
    let last_word = end / 32;
    for word in first_word..=last_word {
        let first_bit = if word == first_word { start % 32 } else { 0 };
        let last_bit = if word == last_word { end % 32 } else { 31 };
        let high = if last_bit == 31 {
            u32::MAX
        } else {
            (1u32 << (last_bit + 1)) - 1
        };
        let low = if first_bit == 0 {
            0
        } else {
            (1u32 << first_bit) - 1
        };
        buf[word] |= high & !low;
    }
}

fn set_reachable_token_ids(buf: &mut [u32], node: &VocabPrefixTreeNode) {
    for range in node.reachable_token_ids().ranges() {
        set_mask_range(buf, *range.start(), *range.end());
    }
}

fn expand_dynamic_token_aliases(constraint: &Constraint, buf: &mut [u32]) {
    for (&canonical, aliases) in constraint.dynamic_mask_token_aliases.iter() {
        if mask_bit_is_set(buf, canonical) {
            for &alias in aliases.iter() {
                set_mask_bit(buf, alias);
            }
        }
    }
}

fn subtree_bytes_self_loop(node: &VocabPrefixTreeNode, self_loop_bytes: U8Set) -> bool {
    U8Set::from_words(*node.subtree_bytes()).is_subset(&self_loop_bytes)
}

fn cached_self_loop_bytes(
    constraint: &Constraint,
    tokenizer_state: u32,
    cache: &mut FxHashMap<u32, U8Set>,
) -> U8Set {
    *cache
        .entry(tokenizer_state)
        .or_insert_with(|| constraint.tokenizer.self_loop_bytes(tokenizer_state))
}

/// A sufficient condition for every token endpoint in this vocabulary subtree
/// to be admissible. Every suffix byte keeps the lexer in the same live state,
/// and there are no branch-local terminal exclusions to preserve. A token may
/// therefore end anywhere in the subtree: either an already-matched terminal
/// can advance the parser now, or the unchanged lexer state is completable by a
/// later terminal. This is especially important inside unbounded string tokens.
fn whole_subtree_is_allowed(
    constraint: &Constraint,
    current: TraverseWork<'_>,
    nodes: &mut [ParserTrieNode],
    self_loop_cache: &mut FxHashMap<u32, U8Set>,
) -> bool {
    if current.exclusion.is_some()
        || !nodes[current.parser_node]
            .gss
            .all_accs_satisfy(|disallowed: &TerminalsDisallowed| disallowed.is_empty())
        || !subtree_bytes_self_loop(
            current.node,
            cached_self_loop_bytes(constraint, current.tokenizer_state, self_loop_cache),
        )
    {
        return false;
    }

    let terminal_can_advance_now = !constraint
        .tokenizer
        .matched_terminal_bitset(current.tokenizer_state)
        .is_disjoint(
            unrestricted_admissible_terminals(constraint, current.parser_node, nodes)
                .expect("whole-subtree shortcut requires an unrestricted parser node"),
        );
    terminal_can_advance_now
        || token_boundary_allowed(
            constraint,
            current.tokenizer_state,
            current.parser_node,
            nodes,
        )
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

/// Match a terminal at the current lexer state, dropping accumulator branches
/// which have explicitly forbidden that terminal. Once a terminal is selected,
/// its pending follow restrictions are consumed, just as in commit.
fn prune_for_terminal(
    gss: &ParserGSS,
    tokenizer_state: u32,
    terminal: TerminalID,
    execution_end_state: Option<u32>,
) -> ParserGSS {
    // This is the accumulator half of normal byte commit: reject branches that
    // have forbidden this terminal at the current lexer state, and carry a
    // surviving restriction forward to the lexer state at the end of the bytes
    // being tested. Restrictions for any other lexer state belong to other
    // tokenization branches and are intentionally not copied.
    if gss.all_accs_satisfy(|disallowed: &TerminalsDisallowed| disallowed.is_empty()) {
        return gss.clone();
    }
    gss.apply_and_prune_no_promote(|disallowed: &TerminalsDisallowed| {
        if disallowed
            .get(&tokenizer_state)
            .is_some_and(|set| set.contains(&terminal))
        {
            return None;
        }

        let mut remapped = BTreeMap::new();
        if let Some(end_state) = execution_end_state {
            if let Some(blocked) = disallowed.get(&tokenizer_state) {
                remapped.insert(end_state, blocked.clone());
            }
        }
        Some(TerminalsDisallowed(Arc::new(remapped)))
    })
}

/// Keep the branch that has not yet committed a terminal while the lexer
/// consumes the whole segment. This mirrors the accumulator remapping done by
/// the ordinary byte-commit path.
fn remap_continuation(
    constraint: &Constraint,
    gss: &ParserGSS,
    tokenizer_state: u32,
    end_state: u32,
    matched_terminals: &[TokenizerMatch],
) -> ParserGSS {
    // With no accumulated restrictions an uncommitted lexer continuation is
    // unchanged. In particular, there is no need to compute which just-matched
    // terminals were actionable merely to remap an empty map.
    if gss.all_accs_satisfy(|disallowed: &TerminalsDisallowed| disallowed.is_empty()) {
        return gss.clone();
    }

    let actionable: BTreeSet<TerminalID> = matched_terminals
        .iter()
        .filter_map(|matched| {
            let terminal = matched.id;
            (Some(terminal) != constraint.ignore_terminal
                && stack_may_advance_on(&constraint.table, gss, terminal))
                .then_some(terminal)
        })
        .collect();

    gss.apply_and_prune_no_promote(|disallowed: &TerminalsDisallowed| {
        if let Some(blocked) = disallowed.get(&tokenizer_state) {
            if !actionable.is_empty() && actionable.iter().all(|terminal| blocked.contains(terminal)) {
                return None;
            }
        }

        let mut remapped = BTreeMap::new();
        if let Some(blocked) = disallowed.get(&tokenizer_state) {
            remapped.insert(end_state, blocked.clone());
        }
        Some(TerminalsDisallowed(Arc::new(remapped)))
    })
}

fn parser_child(
    constraint: &Constraint,
    parser_node: usize,
    tokenizer_state: u32,
    terminal: TerminalID,
    execution_end_state: Option<u32>,
    nodes: &mut Vec<ParserTrieNode>,
) -> Option<usize> {
    // Ignore terminals reset the lexer but deliberately leave the parser alone.
    if Some(terminal) == constraint.ignore_terminal {
        return Some(parser_node);
    }

    // This is the same future-terminal restriction attached by the ordinary
    // commit path. It prevents a branch that matched `terminal` earlier in a
    // token from matching the same still-live terminal again at this segment's
    // end state.
    let future_disallow_state = execution_end_state.filter(|&end_state| {
        constraint
            .tokenizer
            .possible_future_terminals(end_state)
            .contains(terminal as usize)
    });
    // Existing accumulator restrictions are remapped to the actual lexer
    // end-state even when this terminal does not itself stay live there, so the
    // cache key must retain `execution_end_state`, not merely the new exclusion.
    let key = (tokenizer_state, terminal, execution_end_state);
    if let Some(&child) = nodes[parser_node].terminal_children.get(&key) {
        return Some(child);
    }

    let unrestricted = nodes[parser_node]
        .gss
        .all_accs_satisfy(|disallowed: &TerminalsDisallowed| disallowed.is_empty());
    let mut advanced = if unrestricted {
        let gss = &nodes[parser_node].gss;
        if !stack_may_advance_on(&constraint.table, gss, terminal) {
            return None;
        }
        advance_stacks(&constraint.table, gss, terminal)
    } else {
        let pruned = prune_for_terminal(
            &nodes[parser_node].gss,
            tokenizer_state,
            terminal,
            execution_end_state,
        );
        if pruned.is_empty() || !stack_may_advance_on(&constraint.table, &pruned, terminal) {
            return None;
        }
        advance_stacks(&constraint.table, &pruned, terminal)
    };
    if advanced.is_empty() {
        return None;
    }
    if let Some(end_state) = future_disallow_state {
        advanced = advanced.apply(|disallowed: &TerminalsDisallowed| {
            disallowed.with_insert(end_state, terminal)
        });
    }

    let child = nodes.len();
    nodes.push(ParserTrieNode::new(advanced));
    nodes[parser_node].terminal_children.insert(key, child);
    Some(child)
}

fn continuation_node(
    constraint: &Constraint,
    parser_node: usize,
    tokenizer_state: u32,
    end_state: u32,
    matched_terminals: &[TokenizerMatch],
    nodes: &mut Vec<ParserTrieNode>,
) -> Option<usize> {
    if nodes[parser_node]
        .gss
        .all_accs_satisfy(|disallowed: &TerminalsDisallowed| disallowed.is_empty())
    {
        return Some(parser_node);
    }
    let remapped = remap_continuation(
        constraint,
        &nodes[parser_node].gss,
        tokenizer_state,
        end_state,
        matched_terminals,
    );
    if remapped.is_empty() {
        return None;
    }
    let node = nodes.len();
    nodes.push(ParserTrieNode::new(remapped));
    Some(node)
}

fn unrestricted_admissible_terminals<'a>(
    constraint: &Constraint,
    parser_node: usize,
    nodes: &'a mut [ParserTrieNode],
) -> Option<&'a BitSet> {
    if !nodes[parser_node]
        .gss
        .all_accs_satisfy(|disallowed: &TerminalsDisallowed| disallowed.is_empty())
    {
        return None;
    }
    if nodes[parser_node].unrestricted_admissible_terminals.is_none() {
        let mut admissible = BitSet::new(constraint.tokenizer.num_terminals() as usize);
        for terminal_index in 0..constraint.tokenizer.num_terminals() as usize {
            let terminal = terminal_index as TerminalID;
            if Some(terminal) == constraint.ignore_terminal
                || stack_may_advance_on(&constraint.table, &nodes[parser_node].gss, terminal)
            {
                admissible.set(terminal_index);
            }
        }
        nodes[parser_node].unrestricted_admissible_terminals = Some(admissible);
    }
    nodes[parser_node].unrestricted_admissible_terminals.as_ref()
}

fn lexer_state_has_admissible_terminal(
    constraint: &Constraint,
    tokenizer_state: u32,
    parser_node: usize,
    nodes: &mut [ParserTrieNode],
) -> bool {
    if let Some(admissible) = unrestricted_admissible_terminals(constraint, parser_node, nodes) {
        return !constraint
            .tokenizer
            .possible_future_terminals(tokenizer_state)
            .is_disjoint(admissible)
            || !constraint
                .tokenizer
                .matched_terminal_bitset(tokenizer_state)
                .is_disjoint(admissible);
    }

    if !nodes[parser_node]
        .admissible_terminals
        .contains_key(&tokenizer_state)
    {
        let gss = &nodes[parser_node].gss;
        let mut admissible = BitSet::new(constraint.tokenizer.num_terminals() as usize);
        for terminal_index in 0..constraint.tokenizer.num_terminals() as usize {
            let terminal = terminal_index as TerminalID;
            if Some(terminal) == constraint.ignore_terminal {
                admissible.set(terminal_index);
                continue;
            }
            let pruned = prune_for_terminal(gss, tokenizer_state, terminal, None);
            if !pruned.is_empty() && stack_may_advance_on(&constraint.table, &pruned, terminal) {
                admissible.set(terminal_index);
            }
        }
        nodes[parser_node]
            .admissible_terminals
            .insert(tokenizer_state, admissible);
    }

    let admissible = nodes[parser_node]
        .admissible_terminals
        .get(&tokenizer_state)
        .expect("dynamic admissible-terminal cache was just populated");
    !constraint
        .tokenizer
        .possible_future_terminals(tokenizer_state)
        .is_disjoint(admissible)
        || !constraint
            .tokenizer
            .matched_terminal_bitset(tokenizer_state)
            .is_disjoint(admissible)
}

fn token_boundary_allowed(
    constraint: &Constraint,
    tokenizer_state: u32,
    parser_node: usize,
    nodes: &mut [ParserTrieNode],
) -> bool {
    if !nodes[parser_node]
        .token_boundary_allowed
        .contains_key(&tokenizer_state)
    {
        let allowed = if let Some(admissible) =
            unrestricted_admissible_terminals(constraint, parser_node, nodes)
        {
            !constraint
                .tokenizer
                .tokens_accessible_from_state(tokenizer_state)
                .is_disjoint(admissible)
        } else {
            let gss = &nodes[parser_node].gss;
            constraint
                .tokenizer
                .tokens_accessible_from_state(tokenizer_state)
                .iter()
                .any(|terminal| {
                    let terminal = terminal as TerminalID;
                    if Some(terminal) == constraint.ignore_terminal {
                        return true;
                    }
                    let pruned = prune_for_terminal(gss, tokenizer_state, terminal, None);
                    !pruned.is_empty() && stack_may_advance_on(&constraint.table, &pruned, terminal)
                })
        };
        nodes[parser_node]
            .token_boundary_allowed
            .insert(tokenizer_state, allowed);
    }

    *nodes[parser_node]
        .token_boundary_allowed
        .get(&tokenizer_state)
        .expect("dynamic token-boundary cache was just populated")
}

fn excluded_by_first_byte(
    constraint: &Constraint,
    segment: &[u8],
    exclusion: Option<(u32, TerminalID)>,
) -> bool {
    let Some((tokenizer_state, terminal)) = exclusion else {
        return false;
    };
    debug_assert!(!segment.is_empty());
    constraint
        .tokenizer
        .step(tokenizer_state, segment[0])
        .is_some_and(|next_state| {
            constraint
                .tokenizer
                .matched_terminal_bitset(next_state)
                .contains(terminal as usize)
        })
}

pub(crate) fn fill_mask_dynamic(state: &ConstraintState<'_>, buf: &mut [u32]) {
    assert!(
        state.constraint.dynamic_mask_available,
        "dynamic mask generation is unavailable: the lexer persistence property does not hold"
    );
    if state.try_fill_dynamic_mask_from_cache(buf) {
        return;
    }
    fill_mask_dynamic_uncached(state, buf);
    state.store_dynamic_mask_cache(buf);
}

fn fill_mask_dynamic_uncached(state: &ConstraintState<'_>, buf: &mut [u32]) {
    buf.fill(0);
    let initial_tsid = state.constraint.tokenizer.initial_state();
    let mut parser_nodes = Vec::<ParserTrieNode>::new();
    let mut traversal = Vec::<TraverseWork<'_>>::new();
    let mut self_loop_cache = FxHashMap::<u32, U8Set>::default();

    for (&tokenizer_state, gss) in &state.state {
        if gss.is_empty() {
            continue;
        }
        let parser_node = parser_nodes.len();
        parser_nodes.push(ParserTrieNode::new(gss.clone()));
        traversal.push(TraverseWork {
            node: &state.constraint.dynamic_mask_vocab_trie.root,
            tokenizer_state,
            parser_node,
            exclusion: None,
        });
    }

    while let Some(current) = traversal.pop() {
        if whole_subtree_is_allowed(
            state.constraint,
            current,
            &mut parser_nodes,
            &mut self_loop_cache,
        ) {
            set_reachable_token_ids(buf, current.node);
            continue;
        }

        if current.node.has_token()
            && (current.tokenizer_state == initial_tsid
                || token_boundary_allowed(
                    state.constraint,
                    current.tokenizer_state,
                    current.parser_node,
                    &mut parser_nodes,
                ))
        {
            set_mask_bit(buf, current.node.token_id() as u32);
        }

        for (segment, child) in current.node.iter_children() {
            if excluded_by_first_byte(state.constraint, segment, current.exclusion) {
                continue;
            }

            // Most trie branches die on their first byte. Avoid allocating a
            // match vector or scanning the rest of a compressed edge for them.
            let Some(first_state) = state
                .constraint
                .tokenizer
                .step(current.tokenizer_state, segment[0])
            else {
                continue;
            };
            if !lexer_state_has_admissible_terminal(
                state.constraint,
                first_state,
                current.parser_node,
                &mut parser_nodes,
            ) {
                continue;
            }

            let mut segment_stack = SmallVec::<[(usize, u32, usize); 4]>::new();
            segment_stack.push((0usize, current.tokenizer_state, current.parser_node));
            let mut matches = SmallVec::<[crate::automata::lexer::tokenizer::TokenizerMatch; 4]>::new();

            while let Some((position, tokenizer_state, parser_node)) = segment_stack.pop() {
                let first_state = if position == 0 {
                    first_state
                } else {
                    let Some(first_state) = state
                        .constraint
                        .tokenizer
                        .step(tokenizer_state, segment[position])
                    else {
                        continue;
                    };
                    first_state
                };
                if !lexer_state_has_admissible_terminal(
                    state.constraint,
                    first_state,
                    parser_node,
                    &mut parser_nodes,
                ) {
                    continue;
                }
                let end_state = state
                    .constraint
                    .tokenizer
                    .execute_from_state_all_widths_after_first_into(
                        &segment[position..],
                        first_state,
                        &mut matches,
                    );
                for matched in &matches {
                    debug_assert!(matched.width > 0);
                    let Some(advanced_parser) = parser_child(
                        state.constraint,
                        parser_node,
                        tokenizer_state,
                        matched.id,
                        end_state,
                        &mut parser_nodes,
                    ) else {
                        continue;
                    };

                    let next_position = position + matched.width;
                    if next_position == segment.len() {
                        traversal.push(TraverseWork {
                            node: child,
                            tokenizer_state: initial_tsid,
                            parser_node: advanced_parser,
                            exclusion: end_state.map(|end_state| (end_state, matched.id)),
                        });
                    } else {
                        segment_stack.push((next_position, initial_tsid, advanced_parser));
                    }
                }

                if let Some(end_state) = end_state {
                    if let Some(continuation) = continuation_node(
                        state.constraint,
                        parser_node,
                        tokenizer_state,
                        end_state,
                        &matches,
                        &mut parser_nodes,
                    ) {
                        traversal.push(TraverseWork {
                            node: child,
                            tokenizer_state: end_state,
                            parser_node: continuation,
                            exclusion: None,
                        });
                    }
                }
            }
        }
    }

    expand_dynamic_token_aliases(state.constraint, buf);
    update_eos_mask(state, buf);
}

pub(crate) fn assert_normal_mask_matches_dynamic(
    state: &ConstraintState<'_>,
    normal_mask: &[u32],
) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        if cfg!(debug_assertions) {
            return true;
        }
        std::env::var("GLRMASK_ASSERT_DYNAMIC_MASK_EQUIVALENCE")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    });
    if !enabled || !state.constraint.dynamic_mask_available {
        return;
    }

    let mut dynamic_mask = vec![0u32; state.constraint.mask_len()];
    fill_mask_dynamic(state, &mut dynamic_mask);
    assert_eq!(
        normal_mask,
        dynamic_mask.as_slice(),
        "normal parser-DWA mask disagrees with direct dynamic mask"
    );
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

    fn assert_dynamic_parity(state: &ConstraintState<'_>) {
        assert_eq!(state.mask(), state.dynamic_mask());
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
        assert!(constraint.dynamic_mask_available);

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 4));

        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 2));

        state.commit_token(2).unwrap();
        assert!(state.is_finished());
        assert_dynamic_parity(&state);
    }

    #[test]
    fn dynamic_availability_and_trie_are_rebuilt_after_load() {
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
        assert!(loaded.dynamic_mask_available);
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
        assert!(constraint.dynamic_mask_available);

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        let mask = state.dynamic_mask();
        assert!(token_allowed(&mask, 0));
        assert!(token_allowed(&mask, 7));
        assert!(token_allowed(&mask, 12));

        state.commit_token(7).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.dynamic_mask(), 1));
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
        assert!(constraint.dynamic_mask_available);

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        state.commit_token(1).unwrap();
        assert_dynamic_parity(&state);
        assert!(token_allowed(&state.mask(), 3));
        assert!(token_allowed(&state.mask(), 4));

        state.commit_token(4).unwrap();
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
        assert!(constraint.dynamic_mask_available);

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
        assert!(constraint.dynamic_mask_available);

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
        assert!(constraint.dynamic_mask_available);

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
    fn dynamic_mask_handles_split_json_number_phases() {
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
        assert!(constraint.dynamic_mask_available);

        let mut state = constraint.start();
        assert_dynamic_parity(&state);
        for bytes in [b"1".as_slice(), b".".as_slice(), b"2".as_slice(), b"e".as_slice(), b"-".as_slice(), b"3".as_slice()] {
            state.commit_bytes(bytes).unwrap();
            assert_dynamic_parity(&state);
        }
        assert!(state.is_complete());
    }

    #[test]
    fn dynamic_mask_is_unavailable_for_a_live_cross_terminal_prefix() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"ab".to_vec()),
                (2, b"abc".to_vec()),
            ],
            None,
        );
        let grammar = r#"
start start;
t A ::= 'a' | 'abc';
nt start ::= A;
"#;
        let constraint = Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(!constraint.dynamic_mask_available);

        let state = constraint.start();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = state.dynamic_mask();
        }))
        .is_err());
    }
}
