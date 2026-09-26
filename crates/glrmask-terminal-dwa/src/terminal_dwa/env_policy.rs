//! Rollback overrides for validated terminal-boundary optimizations.

fn value_enabled(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no" | "off"
    )
}

pub(crate) fn enabled(name: &str) -> bool {
    std::env::var_os(name)
        .map(|value| value_enabled(&value.to_string_lossy()))
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_boolean_override_values_are_unambiguous() {
        for value in ["", "0", "false", "FALSE", " no ", "Off"] {
            assert!(!value_enabled(value), "{value:?}");
        }
        for value in ["1", "true", "yes", "on", "garbage"] {
            assert!(value_enabled(value), "{value:?}");
        }
    }
}
