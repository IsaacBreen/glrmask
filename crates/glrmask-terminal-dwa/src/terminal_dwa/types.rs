//! Shared types used across the terminal DWA build pipeline.

use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::stages::equiv_types::MappedArtifact;
use crate::grammar::flat::TerminalID;

/// Color identifier (index into graph-coloring partition).
pub type ColorId = u32;

/// Terminal coloring: maps each terminal to a color based on GLR table row
/// adjacency. Terminals with the same color never appear in the same action
/// row, so they can share NWA transitions.
#[derive(Debug, Clone)]
pub struct TerminalColoring {
    pub terminal_to_color: Vec<ColorId>,
    pub num_colors: usize,
}

impl TerminalColoring {
    pub fn identity(num_terminals: usize) -> Self {
        Self {
            terminal_to_color: (0..num_terminals as ColorId).collect(),
            num_colors: num_terminals,
        }
    }

    #[inline]
    pub fn color_for(&self, terminal_id: TerminalID) -> ColorId {
        self.terminal_to_color
            .get(terminal_id as usize)
            .copied()
            .unwrap_or(terminal_id)
    }

    /// Give externally protected lexer residuals unique colors so terminal-DWA
    /// transition sharing cannot erase their identities before late state-map
    /// expansion.
    pub fn isolate_terminals(
        &mut self,
        terminals: impl IntoIterator<Item = TerminalID>,
    ) {
        for terminal in terminals {
            let terminal = terminal as usize;
            if terminal >= self.terminal_to_color.len() {
                continue;
            }
            self.terminal_to_color[terminal] = self.num_colors as ColorId;
            self.num_colors += 1;
        }
    }
}

/// Per-partition build profile counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct TerminalDwaBuildProfile {
    pub future_terminal_additions: u64,
    pub match_transition_additions: u64,
    pub trie_walk_ms: f64,
    pub flush_ms: f64,
    pub flush_leaf_ms: f64,
    pub flush_future_ms: f64,
    pub flush_weight_ms: f64,
    pub trie_self_loop_ms: f64,
    pub trie_execute_ms: f64,
    pub trie_match_filter_ms: f64,
    pub trie_end_state_ms: f64,
    pub trie_match_process_ms: f64,
    pub trie_continuation_weight_ms: f64,
    pub trie_execute_calls: u64,
    pub trie_execute_input_bytes: u64,
    pub trie_matches: u64,
    pub trie_end_states: u64,
    pub trie_self_loop_checks: u64,
    pub trie_self_loop_skips: u64,
    pub trie_self_loop_source_nodes: u64,
    pub trie_self_loop_skipped_source_nodes: u64,
    pub trie_self_loop_cache_misses: u64,
}

pub use glrmask_dwa_merge::__private::{LocalIdMapTerminalDwa, TerminalDwaPhaseProfile};

/// The independently-built terminal-DWA pieces produced by one vocabulary
/// partition.  The third slot is the cheap L1 construction over tokens split
/// away from an L2P terminal set because they never cross an L2P boundary.
#[derive(Debug, Default)]
pub struct PartitionTerminalDwas {
    pub l1: Option<LocalIdMapTerminalDwa>,
    pub l2p: Option<LocalIdMapTerminalDwa>,
    pub l2p_single_l1: Option<LocalIdMapTerminalDwa>,
    pub profile: TerminalDwaPhaseProfile,
}

impl PartitionTerminalDwas {
    pub fn is_empty(&self) -> bool {
        self.l1.is_none() && self.l2p.is_none() && self.l2p_single_l1.is_none()
    }
}

/// Globally merged terminal-DWA families.  L1 includes both ordinary L1
/// terminals and the cheap L1 construction over the vocabulary subset split
/// away from L2P.  Keeping the families separate lets parser construction run
/// independently before the parser DWAs are unioned.
#[derive(Debug)]
pub struct TerminalDwaFamilies {
    pub l1: Option<MappedArtifact<TerminalAutomaton>>,
    pub l2p: Option<MappedArtifact<TerminalAutomaton>>,
    pub special: Option<MappedArtifact<TerminalAutomaton>>,
}

impl TerminalDwaFamilies {
    pub fn len(&self) -> usize {
        usize::from(self.l1.is_some())
            + usize::from(self.l2p.is_some())
            + usize::from(self.special.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.l1.is_none() && self.l2p.is_none() && self.special.is_none()
    }

    pub fn into_vec(self) -> Vec<MappedArtifact<TerminalAutomaton>> {
        self.l1
            .into_iter()
            .chain(self.l2p)
            .chain(self.special)
            .collect()
    }

    pub fn max_original_token_id(&self) -> Option<u32> {
        [&self.l1, &self.l2p, &self.special]
            .into_iter()
            .filter_map(|family| family.as_ref())
            .filter_map(|family| {
                family
                    .id_map()
                    .vocab_tokens
                    .original_to_internal
                    .iter()
                    .rposition(|&internal| internal != u32::MAX)
                    .map(|token_id| token_id as u32)
            })
            .max()
    }
}

/// Terminal path length classification for L1/L2+ split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalPathLength {
    /// Terminal's first-byte bitset is disjoint from vocab byte bitset – ignorable.
    Zero,
    /// Single-step paths only – fast special case for id_map/DWA.
    One,
    /// Multi-terminal token paths possible – full treatment required.
    TwoPlus,
}

pub fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

/// Disable only coarse-grained compiler fan-out while leaving Rayon available
/// to the algorithms inside each coarse unit of work.
///
/// This is intentionally different from setting `RAYON_NUM_THREADS=1`: callers
/// use it to profile one partition/branch/template family at a time while still
/// allowing that individual operation to use the full worker pool.
pub fn macro_parallelism_disabled() -> bool {
    std::env::var("GLRMASK_DISABLE_MACRO_PARALLELISM")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

pub fn report_macro_item_timings(label: &str, timings_ms: &[f64]) {
    let profile_enabled = compile_profile_enabled()
        || std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some();
    if !macro_parallelism_disabled() || !profile_enabled || timings_ms.is_empty() {
        return;
    }
    let mut sorted = timings_ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    let total_work_ms = sorted.iter().sum::<f64>();
    let max_item_ms = sorted[count - 1];
    let mean_ms = total_work_ms / count as f64;
    let p50_ms = if count % 2 == 0 {
        (sorted[count / 2 - 1] + sorted[count / 2]) * 0.5
    } else {
        sorted[count / 2]
    };
    let p90_ms = sorted[((count as f64 * 0.90).ceil() as usize).clamp(1, count) - 1];
    let ideal_parallelism = if max_item_ms > 0.0 {
        total_work_ms / max_item_ms
    } else {
        0.0
    };
    eprintln!(
        "[glrmask/profile][macro_fanout] label={label} count={count} total_work_ms={total_work_ms:.3} max_item_ms={max_item_ms:.3} mean_ms={mean_ms:.3} p50_ms={p50_ms:.3} p90_ms={p90_ms:.3} ideal_parallelism={ideal_parallelism:.2}",
    );
}

/// Nested Rayon joins can execute sibling outer tasks while an inner task is
/// pending. With one worker that makes a partition wall timer include unrelated
/// partitions. Use a serial outer schedule only for this profiling case.
pub fn compile_profile_uses_serial_partition_schedule() -> bool {
    macro_parallelism_disabled()
        || (compile_profile_enabled() && rayon::current_num_threads() == 1)
}

/// Preserve the normal Rayon join except in a one-worker compile profile.
/// In that case an inner join can run sibling outer partition work, making the
/// caller's inclusive timer non-compositional.
pub fn compile_profile_join<A, B, Left, Right>(
    label: &'static str,
    left: Left,
    right: Right,
) -> (A, B)
where
    A: Send,
    B: Send,
    Left: FnOnce() -> A + Send,
    Right: FnOnce() -> B + Send,
{
    if compile_profile_uses_serial_partition_schedule() {
        let left_started = std::time::Instant::now();
        let left = left();
        let left_ms = left_started.elapsed().as_secs_f64() * 1000.0;
        let right_started = std::time::Instant::now();
        let right = right();
        let right_ms = right_started.elapsed().as_secs_f64() * 1000.0;
        report_macro_item_timings(label, &[left_ms, right_ms]);
        (left, right)
    } else {
        rayon::join(left, right)
    }
}
