pub mod compile;
pub(crate) mod constraint_possible_matches;
pub mod glr;
pub mod grammar;
pub(crate) mod pipeline;
pub(crate) mod terminal_run_collapse;
pub(crate) mod pm_profile;
pub(crate) mod possible_matches;
pub mod stages;

pub(crate) use compile::compile_owned;

/// Exact bounded-terminal synthesis is enabled by default. Runtime always keeps
/// the full exact lexer, while terminal/parser DWA construction may use a
/// certified smaller representative lexer. Retain an explicit opt-out only for
/// diagnostics and performance comparisons; it must not change schema
/// semantics.
pub(crate) fn synthetic_bounded_terminals_enabled() -> bool {
    match std::env::var("GLRMASK_SYNTHETIC_BOUNDED_TERMINALS") {
        Err(_) => true,
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => panic!(
                "invalid GLRMASK_SYNTHETIC_BOUNDED_TERMINALS={other:?}; expected one of 1/0, true/false, yes/no, or on/off"
            ),
        },
    }
}
