//! Environment-controlled Commit options.
//!
//! Commit has exactly two template-DFA switches: one enabling the optimized
//! parser-stack recognizer path and one asking the runtime to validate that
//! path against the table-based GLR advance.  Keeping these reads here makes
//! the transition relation independent of process environment.

use std::sync::OnceLock;

static TEMPLATE_ADVANCE_ENABLED: OnceLock<bool> = OnceLock::new();
static VALIDATE_TEMPLATE_ADVANCE_ENABLED: OnceLock<bool> = OnceLock::new();

pub(super) fn template_advance_enabled() -> bool {
    *TEMPLATE_ADVANCE_ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_DISABLE_TEMPLATE_DFA_ADVANCE").is_none())
}

pub(super) fn validate_template_advance_enabled() -> bool {
    *VALIDATE_TEMPLATE_ADVANCE_ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_VALIDATE_TEMPLATE_DFA_ADVANCE").is_some())
}

