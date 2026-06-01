//! Terminal-DWA build policy.
//!
//! This module is intentionally separated from the builder.  The mathematical
//! object being built does not depend on environment variables; only the route
//! taken to construct it does.  Keeping this file small and explicit makes the
//! publication story cleaner: options select algorithms, while the surrounding
//! modules define the denotation.

use std::sync::OnceLock;

use crate::compile::terminal_dwa::classify::{PairPartitionCostFn, PairPartitionObjective};

/// How the caller vocabulary is split before constructing local Terminal DWAs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VocabPartitionScheme {
    /// Partition by coarse byte/character class.
    CharType,
    /// Partition by estimated multi-step pair-partition cost.
    PairPartitionCost,
    /// Compare coarse character partitioning against cost partitioning and pick
    /// the safer cheaper strategy for this grammar/vocab pair.
    AutoPairPartitionCost,
}

impl VocabPartitionScheme {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::CharType => "char_type",
            Self::PairPartitionCost => "pair_partition_cost",
            Self::AutoPairPartitionCost => "auto_pair_partition_cost",
        }
    }
}

fn parse_truthy(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
}

pub(crate) fn vocab_partition_scheme_from_env() -> VocabPartitionScheme {
    match std::env::var("GLRMASK_PARTITION_SCHEME").as_deref() {
        Ok("char_type") | Err(_) => VocabPartitionScheme::CharType,
        Ok("pair_partition_cost") => VocabPartitionScheme::PairPartitionCost,
        Ok("auto_pair_partition_cost") => VocabPartitionScheme::AutoPairPartitionCost,
        Ok(other) => panic!(
            "Invalid GLRMASK_PARTITION_SCHEME={other}; expected one of: char_type, pair_partition_cost, auto_pair_partition_cost"
        ),
    }
}

pub(crate) fn pair_partition_cost_fn_from_env() -> PairPartitionCostFn {
    match std::env::var("GLRMASK_PAIR_PARTITION_COST_FN").as_deref() {
        Ok("size") | Err(_) => PairPartitionCostFn::Size,
        Ok("size_log") => PairPartitionCostFn::SizeLog,
        Ok("log_log") => PairPartitionCostFn::LogLog,
        Ok("union_size") => PairPartitionCostFn::UnionSize,
        Ok(other) => panic!(
            "Invalid GLRMASK_PAIR_PARTITION_COST_FN={other}; expected one of: size, size_log, log_log, union_size"
        ),
    }
}

pub(crate) fn pair_partition_objective_from_env() -> PairPartitionObjective {
    match std::env::var("GLRMASK_PAIR_PARTITION_COST_OBJECTIVE").as_deref() {
        Ok("max") | Err(_) => PairPartitionObjective::Max,
        Ok("sum") => PairPartitionObjective::Sum,
        Ok(other) => panic!(
            "Invalid GLRMASK_PAIR_PARTITION_COST_OBJECTIVE={other}; expected one of: max, sum"
        ),
    }
}

pub(crate) fn pair_partition_count_from_env() -> usize {
    std::env::var("GLRMASK_PAIR_PARTITION_COST_PARTITIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(10)
}

pub(crate) fn pair_partition_auto_second_largest_limit_from_env() -> usize {
    std::env::var("GLRMASK_PAIR_PARTITION_AUTO_SECOND_LARGEST_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(12_000)
}

pub(crate) fn pair_partition_auto_max_estimated_pair_partition_terminals_from_env() -> usize {
    std::env::var("GLRMASK_PAIR_PARTITION_AUTO_MAX_ESTIMATED_TERMINALS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(7)
}

pub(crate) fn pair_partition_auto_min_estimated_pair_partition_terminals_from_env() -> usize {
    std::env::var("GLRMASK_PAIR_PARTITION_AUTO_MIN_ESTIMATED_TERMINALS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(6)
}

pub(crate) fn pair_partition_auto_min_grammar_terminals_from_env() -> usize {
    std::env::var("GLRMASK_PAIR_PARTITION_AUTO_MIN_GRAMMAR_TERMINALS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(12)
}

pub(crate) fn global_max_length_env_override() -> Option<bool> {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| std::env::var("GLRMASK_USE_GLOBAL_MAX_LENGTH").ok().map(|value| parse_truthy(&value)))
}
