//! Parser-DWA construction policy.
//!
//! The mathematical denotation of the Parser DWA is independent of these
//! switches.  Options may choose a faster or smaller equivalent construction,
//! but they must not change which `(lexer_state, token)` pairs appear in
//! `[[PDWA]](rho)`.

#[derive(Debug, Clone, Copy)]
pub(crate) struct ParserDwaOptions {
    /// Whether to run weighted-DWA minimization after fallback/default
    /// determinization.  The current production default skips minimization
    /// because continuation sharing and default normalization make the DWA
    /// small enough for runtime use while avoiding an expensive compile-time
    /// tail.
    pub(crate) skip_minimization: bool,
}

impl ParserDwaOptions {
    pub(crate) fn from_environment(
        pre_minimize_states: usize,
        pre_minimize_transitions: usize,
    ) -> Self {
        Self {
            skip_minimization: should_skip_parser_dwa_minimization(
                pre_minimize_states,
                pre_minimize_transitions,
            ),
        }
    }
}

fn skip_parser_dwa_minimization_env_override() -> Option<bool> {
    std::env::var("GLRMASK_SKIP_PARSER_DWA_MINIMIZE")
        .ok()
        .map(|value| {
            let trimmed = value.trim();
            !(trimmed.is_empty()
                || trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false"))
        })
}

#[inline]
pub(crate) fn should_skip_parser_dwa_minimization(
    _pre_minimize_states: usize,
    _pre_minimize_transitions: usize,
) -> bool {
    // Parser-DWA minimization is behavior-preserving but comparatively expensive
    // on the large-schema tail path.  The preceding construction already shares
    // continuation subgraphs and applies default/fallback normalization, so the
    // unminimized DWA is small enough for the runtime fast-transition cache.
    // Keep an escape hatch for size-sensitive experiments.
    skip_parser_dwa_minimization_env_override().unwrap_or(true)
}

