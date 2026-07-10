//! Shared types used across the terminal DWA build pipeline.

use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::stages::equiv_types::{InternalIdMap, MappedArtifact};
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

/// The independently-built terminal-DWA pieces produced by one vocabulary
/// partition.  The third slot is the cheap L1 construction over tokens split
/// away from an L2P terminal set because they never cross an L2P boundary.
#[derive(Debug, Default)]
pub(crate) struct PartitionTerminalDwas {
    pub(crate) l1: Option<LocalIdMapTerminalDwa>,
    pub(crate) l2p: Option<LocalIdMapTerminalDwa>,
    pub(crate) l2p_single_l1: Option<LocalIdMapTerminalDwa>,
    pub(crate) profile: TerminalDwaPhaseProfile,
}

impl PartitionTerminalDwas {
    pub(crate) fn is_empty(&self) -> bool {
        self.l1.is_none() && self.l2p.is_none() && self.l2p_single_l1.is_none()
    }
}

/// Globally merged terminal-DWA families.  L1 includes both ordinary L1
/// terminals and the cheap L1 construction over the vocabulary subset split
/// away from L2P.  Keeping the families separate lets parser construction run
/// independently before the parser DWAs are unioned.
#[derive(Debug)]
pub(crate) struct TerminalDwaFamilies {
    pub(crate) l1: Option<MappedArtifact<TerminalAutomaton>>,
    pub(crate) l2p: Option<MappedArtifact<TerminalAutomaton>>,
}

impl TerminalDwaFamilies {
    pub(crate) fn len(&self) -> usize {
        usize::from(self.l1.is_some()) + usize::from(self.l2p.is_some())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.l1.is_none() && self.l2p.is_none()
    }

    pub(crate) fn into_vec(self) -> Vec<MappedArtifact<TerminalAutomaton>> {
        self.l1.into_iter().chain(self.l2p).collect()
    }
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
