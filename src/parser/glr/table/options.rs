//! Configuration for GLR table construction and optimization.
//!
//! The GLR table has two meanings that should stay separate:
//!
//! 1. a mathematical transition system over LR items and parser stacks; and
//! 2. an engineered execution representation optimized for masking/commit speed.
//!
//! This module owns the engineering policy knobs so table construction,
//! optimization, and parser advance do not read process environment variables in
//! their algorithmic bodies.

/// Policy for constructing and optimizing GLR tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GLRTableOptions {
    pub(crate) default_action_rows: bool,
    pub(crate) stack_shift_predecessor_canonicalization: bool,
    pub(crate) recognizer_suffix_quotient: bool,
    pub(crate) recognizer_suffix_quotient_max_states: usize,
    pub(crate) recognizer_suffix_quotient_max_alts: usize,
    pub(crate) recognizer_suffix_quotient_max_width: usize,
    pub(crate) max_guarded_stack_effects: Option<usize>,
    pub(crate) unit_reduction_inlining: bool,
    pub(crate) profile_table_build: bool,
}

impl GLRTableOptions {
    pub(crate) const fn deterministic_default() -> Self {
        Self {
            default_action_rows: true,
            stack_shift_predecessor_canonicalization: true,
            recognizer_suffix_quotient: true,
            recognizer_suffix_quotient_max_states: 4096,
            recognizer_suffix_quotient_max_alts: 16,
            recognizer_suffix_quotient_max_width: 8,
            max_guarded_stack_effects: None,
            unit_reduction_inlining: true,
            profile_table_build: false,
        }
    }

    pub(crate) fn from_env() -> Self {
        Self {
            default_action_rows: !env_flag_enabled("GLRMASK_DISABLE_DEFAULT_ACTION_ROWS", false),
            stack_shift_predecessor_canonicalization: !env_flag_enabled(
                "GLRMASK_DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION",
                false,
            ),
            recognizer_suffix_quotient: !env_flag_enabled(
                "GLRMASK_DISABLE_RECOGNIZER_SUFFIX_QUOTIENT",
                false,
            ),
            recognizer_suffix_quotient_max_states: env_usize(
                "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_STATES",
                4096,
            ),
            recognizer_suffix_quotient_max_alts: env_usize(
                "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_ALTS",
                16,
            ),
            recognizer_suffix_quotient_max_width: env_usize(
                "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_WIDTH",
                8,
            ),
            max_guarded_stack_effects: env_optional_usize("GLRMASK_MAX_GUARDED_STACK_EFFECTS"),
            unit_reduction_inlining: !env_flag_enabled("GLRMASK_DISABLE_UNIT_REDUCTION_INLINING", false),
            profile_table_build: std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
                || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some(),
        }
    }
}

pub(crate) fn table_options_from_env() -> GLRTableOptions {
    GLRTableOptions::from_env()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn env_optional_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
}

fn env_flag_enabled(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(default)
}
