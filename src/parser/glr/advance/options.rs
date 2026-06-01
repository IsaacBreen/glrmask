//! Parser stack-advance policy.
//!
//! Runtime parser advancement has a small number of engineering knobs: whether
//! expensive safety fallbacks are allowed and whether per-wave traces should be
//! recorded.  Historically those knobs were read directly inside the advance
//! algorithm.  This module collects them into one typed policy object so the
//! mathematical algorithm can be read separately from environment-variable
//! compatibility.

use std::sync::OnceLock;

/// Execution policy for GLR stack advancement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParserAdvanceOptions {
    /// Disable materializing concrete stacks when guarded stack-effect execution
    /// cannot remain virtual.
    pub(crate) disable_guarded_stack_to_stacks_fallback: bool,
    /// Disable materializing concrete stacks when ordinary stack-effect
    /// execution cannot remain virtual.
    pub(crate) disable_stack_effect_to_stacks_fallback: bool,
    /// Capture the detailed parser-advance wave trace used by profiling and
    /// debugging tools.
    pub(crate) trace_enabled: bool,
}

impl ParserAdvanceOptions {
    pub(crate) const fn deterministic_default() -> Self {
        Self {
            disable_guarded_stack_to_stacks_fallback: false,
            disable_stack_effect_to_stacks_fallback: false,
            trace_enabled: false,
        }
    }

    pub(crate) fn from_env() -> Self {
        Self {
            disable_guarded_stack_to_stacks_fallback: env_flag_enabled(
                "GLRMASK_DISABLE_GUARDED_STACK_TO_STACKS_FALLBACK",
            ),
            disable_stack_effect_to_stacks_fallback: env_flag_enabled(
                "GLRMASK_DISABLE_STACK_EFFECT_TO_STACKS_FALLBACK",
            ),
            trace_enabled: env_flag_enabled("GLRMASK_PROFILE_ADVANCE_TRACE"),
        }
    }

    pub(crate) fn global() -> &'static Self {
        static OPTIONS: OnceLock<ParserAdvanceOptions> = OnceLock::new();
        OPTIONS.get_or_init(ParserAdvanceOptions::from_env)
    }
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}
