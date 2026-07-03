//! Shared types used across the terminal DWA build pipeline.

use crate::automata::weighted::dwa::DWA;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::grammar::flat::TerminalID;

/// Color identifier (index into graph-coloring partition).
pub(crate) type ColorId = u32;

/// Terminal coloring: maps each terminal to a color based on GLR table row
/// adjacency. Terminals with the same color never appear in the same action
/// row, so they can share NWA transitions.
#[derive(Debug, Clone)]
pub(crate) struct TerminalColoring {
    pub(crate) terminal_to_color: Vec<ColorId>,
    pub(crate) num_colors: usize,
}

impl TerminalColoring {
    pub(crate) fn identity(num_terminals: usize) -> Self {
        Self {
            terminal_to_color: (0..num_terminals as ColorId).collect(),
            num_colors: num_terminals,
        }
    }

    #[inline]
    pub(crate) fn color_for(&self, terminal_id: TerminalID) -> ColorId {
        self.terminal_to_color
            .get(terminal_id as usize)
            .copied()
            .unwrap_or(terminal_id)
    }
}

/// Per-partition build profile counters.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TerminalDwaBuildProfile {
    pub(crate) future_terminal_additions: u64,
    pub(crate) match_transition_additions: u64,
    /// TI-only: time spent interning scanner-mode subsets while constructing
    /// transport contexts. This is profile-gated and remains zero otherwise.
    pub(crate) transport_mode_set_intern_ms: f64,
    /// TI-only: time spent grouping transport modes by scanner destination.
    pub(crate) transport_context_plan_ms: f64,
    /// TI-only: time spent constructing and looking up transported output labels.
    pub(crate) transport_output_remap_ms: f64,
    pub(crate) transport_output_remap_cache_hits: u64,
    pub(crate) transport_output_remap_cache_misses: u64,
    pub(crate) transport_root_count: u64,
    pub(crate) transport_context_count: u64,
    pub(crate) transport_root_seed_ms: f64,
    pub(crate) transport_trie_walk_ms: f64,
    pub(crate) transport_flush_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TerminalDwaPhaseProfile {
    pub(crate) id_map_ms: f64,
    pub(crate) terminal_dwa_ms: f64,
    pub(crate) compact_ms: f64,
    pub(crate) split_terminal_dwa_total_ms: f64,
    pub(crate) global_merge_ms: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct LocalIdMapTerminalDwa {
    pub(crate) id_map: InternalIdMap,
    pub(crate) dwa: DWA,
    pub(crate) profile: TerminalDwaPhaseProfile,
}

impl TerminalDwaPhaseProfile {
    pub(crate) fn total_ms(self) -> f64 {
        self.id_map_ms + self.terminal_dwa_ms + self.compact_ms
    }

    pub(crate) fn add_assign(&mut self, other: Self) {
        self.id_map_ms += other.id_map_ms;
        self.terminal_dwa_ms += other.terminal_dwa_ms;
        self.compact_ms += other.compact_ms;
        self.split_terminal_dwa_total_ms += other.split_terminal_dwa_total_ms;
        self.global_merge_ms += other.global_merge_ms;
    }
}

/// Terminal path length classification for L1/L2+ split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalPathLength {
    /// Terminal's first-byte bitset is disjoint from vocab byte bitset – ignorable.
    Zero,
    /// Single-step paths only – fast special case for id_map/DWA.
    One,
    /// Multi-terminal token paths possible – full treatment required.
    TwoPlus,
}

pub(crate) fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

/// Nested Rayon joins can execute sibling outer tasks while an inner task is
/// pending. With one worker that makes a partition wall timer include unrelated
/// partitions. Use a serial outer schedule only for this profiling case.
pub(crate) fn compile_profile_uses_serial_partition_schedule() -> bool {
    compile_profile_enabled() && rayon::current_num_threads() == 1
}

/// Preserve the normal Rayon join except in a one-worker compile profile.
/// In that case an inner join can run sibling outer partition work, making the
/// caller's inclusive timer non-compositional.
pub(crate) fn compile_profile_join<A, B, Left, Right>(left: Left, right: Right) -> (A, B)
where
    A: Send,
    B: Send,
    Left: FnOnce() -> A + Send,
    Right: FnOnce() -> B + Send,
{
    if compile_profile_uses_serial_partition_schedule() {
        (left(), right())
    } else {
        rayon::join(left, right)
    }
}
