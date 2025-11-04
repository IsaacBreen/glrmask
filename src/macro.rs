/// Macro for creating a sequence of parsers
#[macro_export]
macro_rules! seq_fast {
    ($($x:expr),* $(,)?) => {
        $crate::interface::seq_fast(vec![$($x),*])
    };
}

/// Macro for creating a choice of parsers
#[macro_export]
macro_rules! choice_fast {
    ($($x:expr),* $(,)?) => {
        $crate::interface::choice_fast(vec![$($x),*])
    };
}

use once_cell::sync::Lazy;
use std::env;

// Import the Local timezone functionality

/// Returns the current debug level, read from the `MACRO_DEBUG_LEVEL` environment variable.
///
/// This function reads the environment variable once and caches the result for subsequent calls.
/// If `MACRO_DEBUG_LEVEL` is not set or contains an invalid value, it defaults to `5`.
pub fn get_macro_debug_level() -> usize {
    static MACRO_DEBUG_LEVEL: Lazy<usize> =
        Lazy::new(|| env::var("MACRO_DEBUG_LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(5));
    *MACRO_DEBUG_LEVEL
}

/// A list of filenames (not full paths) to allow debug messages from.
/// If this list is empty, all files are allowed (respecting `MACRO_DEBUG_LEVEL`).
/// Example: `&["parser.rs", "constraint.rs"]`
pub const ALLOWED_FILES: &[&str] = &[
    // "parser.rs", // Example: Uncomment to allow messages from parser.rs
    // "constraint.rs", // Example: Uncomment to allow messages from constraint.rs
    // "interface.rs",
    // Add more filenames here as needed
];

/// Checks if a given debug level is enabled based on `MACRO_DEBUG_LEVEL`.
pub fn is_debug_level_enabled(level: usize) -> bool {
    level <= get_macro_debug_level()
}

/// Internal implementation detail for the `debug!` macro.
/// This macro contains the shared logic and configuration to avoid duplication.
/// It should not be used directly.
#[doc(hidden)]
#[macro_export]
macro_rules! __debug_impl {
    ($level:expr, $user_fmt:expr, $($user_args:tt)*) => {{
        // Runtime check against the message's level and file path
        if $level <= $crate::r#macro::get_macro_debug_level() {
            let current_file_path = std::path::Path::new(file!());
            // Extract the filename, default to empty string if extraction fails
            let current_filename = current_file_path.file_name()
                .map_or("", |os_str| os_str.to_str().unwrap_or(""));

            // Allow if ALLOWED_FILES is empty (no filter) or if the current file is in the list
            if $crate::r#macro::ALLOWED_FILES.is_empty() || $crate::r#macro::ALLOWED_FILES.contains(&current_filename) {
                // Optional: Keep this if you want compile-time stripping based on a feature flag
                // #[cfg(feature = "debug")]
                { // Use a block to scope the 'now' variable and the import
                    // Make chrono, file! and line! available inside the macro expansion
                    // use chrono::Local;
                    // let now = Local::now(); // Timestamp removed for brevity, uncomment if needed
                    println!(
                        // The complete format string is constructed here
                        concat!("[DEBUG] {}] {}:{}: ", $user_fmt),
                        // concat!("[DEBUG {} {}] {}:{}: ", $user_fmt), // For timestamp
                        // now.format("%Y-%m-%d %H:%M:%S%.3f"), // Uncomment for timestamp
                        // now.format("%H:%M:%S%.3f"), // Uncomment for timestamp
                        $level,           // Argument for the first {} in the prefix
                        file!(),          // Argument for the second {} in the prefix
                        line!(),          // Argument for the third {} in the prefix
                        $($user_args)*    // Arguments for the user-provided format part
                    );
                }
            }
        }
    }};
}

#[macro_export]
macro_rules! debug {
    // Arm for format literals, e.g., debug!(1, "value is {}", 42)
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {
        // Delegate to the internal implementation, passing the user's format string and arguments.
        $crate::__debug_impl!($level, $fmt, $($($arg)*)?);
    };

    // Arm for single expressions, e.g., debug!(1, my_variable)
    ($level:expr, $msg:expr) => {
        // Delegate to the internal implementation, providing a debug format specifier "{:?}"
        // and the user's expression as the argument.
        $crate::__debug_impl!($level, "{:?}", $msg);
    };
}