fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|value| {
        let normalized = value.trim().to_ascii_lowercase();
        !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
    })
}

/// Commit-template automata and their construction proofs are production
/// defaults. The explicit disable flag wins over the legacy enable flag so an
/// operator always has an unambiguous rollback switch.
pub(crate) fn commit_template_dfas_enabled() -> bool {
    if env_flag("GLRMASK_DISABLE_COMMIT_TEMPLATE_DFAS") == Some(true) {
        return false;
    }
    env_flag("GLRMASK_ENABLE_COMMIT_TEMPLATE_DFAS").unwrap_or(true)
}

pub mod characterize;
pub mod compile_bundle;
pub mod compile_dfa;

pub use compile_dfa::Templates;
