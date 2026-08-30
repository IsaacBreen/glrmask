fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|value| {
        let normalized = value.trim().to_ascii_lowercase();
        !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
    })
}

pub(crate) fn macro_parallelism_disabled() -> bool {
    env_flag("GLRMASK_DISABLE_MACRO_PARALLELISM").unwrap_or(false)
}

/// Commit-template automata remain an explicit experimental opt-in. They must
/// not change ordinary static compilation unless the caller requests them.
/// The disable flag wins so an operator always has an unambiguous rollback
/// switch when testing the feature.
pub fn commit_template_dfas_enabled() -> bool {
    if env_flag("GLRMASK_DISABLE_COMMIT_TEMPLATE_DFAS") == Some(true) {
        return false;
    }
    env_flag("GLRMASK_ENABLE_COMMIT_TEMPLATE_DFAS").unwrap_or(false)
}

pub(crate) mod characterize;
pub(crate) mod compile_bundle;
pub(crate) mod compile_dfa;

pub use compile_dfa::Templates;
