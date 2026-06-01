//! Sparse root CanMatch collection path.
//!
//! The dense grouped collector is the normal path.  For very small root
//! signatures, the sparse collector is easier to reason about and can avoid
//! building the full interval trie.  It remains compile-time only.

use super::prelude::*;
use super::collector::{IntervalCanMatchMap, TerminalRangeGroup, TrieClassBuildResult};
use super::terminal_sequences::CanMatchComputer;

pub(super) fn group_scan_relation_vocab_validation_enabled() -> bool {
    std::env::var("GLRMASK_VALIDATE_GROUP_SCAN_RELATION_VOCAB")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub(super) fn group_scan_relation_vocab_legacy_enabled() -> bool {
    std::env::var("GLRMASK_SCAN_RELATION_USE_LEGACY_VOCAB_SWEEP")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub(super) fn sparse_root_collect_enabled() -> bool {
    std::env::var("GLRMASK_SCAN_RELATION_SPARSE_ROOT_COLLECT")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

pub(super) fn sparse_root_state_limit() -> usize {
    std::env::var("GLRMASK_SCAN_RELATION_SPARSE_ROOT_MAX_STATES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128)
}

pub(super) fn sparse_root_terminal_limit() -> usize {
    std::env::var("GLRMASK_SCAN_RELATION_SPARSE_ROOT_MAX_TERMINALS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16)
}

pub(super) fn root_terminal_union_count(tokenizer: &Tokenizer, states: &[u32]) -> usize {
    let mut seen = vec![false; tokenizer.num_terminals as usize];
    let mut count = 0usize;
    for &state in states {
        for terminal in tokenizer
            .matched_terminals_iter(state)
            .chain(tokenizer.possible_future_terminals_iter(state))
        {
            let slot = terminal as usize;
            if slot < seen.len() && !seen[slot] {
                seen[slot] = true;
                count += 1;
            }
        }
    }
    count
}

fn interval_map_from_sparse_matches(
    matches: &FxHashMap<TerminalID, RangeSetBlaze<u32>>,
) -> IntervalCanMatchMap {
    let mut by_ranges = BTreeMap::<Vec<(u32, u32)>, Vec<TerminalID>>::new();
    for (&terminal, token_ids) in matches {
        let ranges: Vec<(u32, u32)> = token_ids
            .ranges()
            .map(|range| (*range.start(), *range.end()))
            .collect();
        if !ranges.is_empty() {
            by_ranges.entry(ranges).or_default().push(terminal);
        }
    }

    let mut map = Vec::with_capacity(by_ranges.len());
    for (ranges, mut terminals) in by_ranges {
        terminals.sort_unstable();
        terminals.dedup();
        if !terminals.is_empty() {
            map.push(TerminalRangeGroup {
                terminals: terminals.into_boxed_slice(),
                ranges,
            });
        }
    }
    map.sort_unstable_by(|left, right| {
        left.terminals
            .as_ref()
            .cmp(right.terminals.as_ref())
            .then_with(|| left.ranges.cmp(&right.ranges))
    });
    map
}

pub(super) fn collect_sparse_root_can_match(
    tokenizer: &Tokenizer,
    root: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
    entries: &[u32],
    canonical_state: Option<&[u32]>,
) -> TrieClassBuildResult {
    let mut computer = CanMatchComputer::new_with_canonical_state(tokenizer, canonical_state);
    let mut state_classes = vec![u32::MAX; tokenizer.num_states() as usize];
    let mut class_maps = Vec::<Arc<IntervalCanMatchMap>>::new();
    let mut map_to_class = FxHashMap::<IntervalCanMatchMap, u32>::default();

    for &state in entries {
        let sparse_matches = computer.can_match_for_node(root, state);
        let interval_map = interval_map_from_sparse_matches(sparse_matches.as_ref());
        let class_id = if let Some(&class_id) = map_to_class.get(&interval_map) {
            class_id
        } else {
            let class_id = class_maps.len() as u32;
            map_to_class.insert(interval_map.clone(), class_id);
            class_maps.push(Arc::new(interval_map));
            class_id
        };

        if let Some(slot) = state_classes.get_mut(state as usize) {
            *slot = class_id;
        }
    }

    TrieClassBuildResult {
        state_classes,
        class_maps,
    }
}
