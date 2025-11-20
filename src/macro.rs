/// Small, focused helper macros and debug utilities.
///
/// The debug macros are intentionally simple and always print a single
/// line to stderr when enabled:
///
///   [D<level> file.rs:123] message
///
/// The effective debug level is read once from the `MACRO_DEBUG_LEVEL`
/// environment variable (default: 0 = disabled).

use once_cell::sync::Lazy;
use std::env;

/// Returns the current debug level, read once from `MACRO_DEBUG_LEVEL`.
pub fn get_macro_debug_level() -> usize {
    static LEVEL: Lazy<usize> = Lazy::new(|| {
        env::var("MACRO_DEBUG_LEVEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    });
    *LEVEL
}

/// Checks if a given debug level is enabled.
#[inline]
pub fn is_debug_level_enabled(level: usize) -> bool {
    level <= get_macro_debug_level()
}

/// Macro for creating a sequence of parsers (regex-level).
#[macro_export]
macro_rules! seq_fast {
    ($($x:expr),* $(,)?) => {
        $crate::interface::seq_fast(vec![$($x),*])
    };
}

/// Macro for creating a choice of parsers (regex-level).
#[macro_export]
macro_rules! choice_fast {
    ($($x:expr),* $(,)?) => {
        $crate::interface::choice_fast(vec![$($x),*])
    };
}

/// Main debug macro: one-line, file/line annotated.
#[macro_export]
macro_rules! debug {
    ($level:expr, $($arg:tt)*) => {
        if $crate::r#macro::is_debug_level_enabled($level) {
            eprintln!(
                "[D{} {}:{}] {}",
                $level,
                file!(),
                line!(),
                format_args!($($arg)*)
            );
        }
    };
}

/// Legacy alias to `debug!`.
#[macro_export]
macro_rules! debug_line {
    ($level:expr, $($arg:tt)*) => {
        $crate::debug!($level, $($arg)*);
    };
}

/// Starts a debug message and returns a small token.
/// The message is printed immediately; `debug_end!` can append more.
#[macro_export]
macro_rules! debug_start {
    ($level:expr, $($arg:tt)*) => {{
        if $crate::r#macro::is_debug_level_enabled($level) {
            eprint!(
                "[D{} {}:{}] {}",
                $level,
                file!(),
                line!(),
                format_args!($($arg)*)
            );
            Some(())
        } else {
            None::<()>
        }
    }};
}

/// Completes a debug message started with `debug_start!`.
#[macro_export]
macro_rules! debug_end {
    ($token:expr, $($arg:tt)*) => {
        if $token.is_some() {
            eprintln!("{}", format_args!($($arg)*));
        }
    };
}

/// Starts a simple debug timer. Returns an opaque token for `debug_timer_end!`.
#[macro_export]
macro_rules! debug_timer_start {
    ($level:expr, $($arg:tt)*) => {{
        if $crate::r#macro::is_debug_level_enabled($level) {
            let start = std::time::Instant::now();
            let msg = format!($($arg)*);
            eprintln!(
                "[D{} {}:{}] START {}",
                $level,
                file!(),
                line!(),
                msg
            );
            Some((start, msg))
        } else {
            None::<(std::time::Instant, String)>
        }
    }};
}

/// Ends a debug timer started with `debug_timer_start!`.
/// The optional `thresh = ...` parameter is accepted but ignored for simplicity.
#[macro_export]
macro_rules! debug_timer_end {
    // Form with explicit threshold – we ignore the threshold and reuse the plain form.
    ($token:expr, thresh = $thresh:expr, $($arg:tt)*) => {
        $crate::debug_timer_end!($token, $($arg)*);
    };
    // Plain form.
    ($token:expr, $($arg:tt)*) => {{
        if let Some((start, _msg)) = $token {
            let elapsed = start.elapsed();
            eprintln!(
                "[DT {}:{}] {} ({:?})",
                file!(),
                line!(),
                format_args!($($arg)*),
                elapsed
            );
        }
    }};
}
