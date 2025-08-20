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

use chrono::Local; // Import the Local timezone functionality

/// Internal implementation detail for the `debug!` macro.
/// This macro contains the shared logic and configuration to avoid duplication.
/// It should not be used directly.
#[doc(hidden)]
#[macro_export]
macro_rules! __debug_impl {
    ($level:expr, $user_fmt:expr, $($user_args:tt)*) => {{
        // --- Configuration (Defined Once) ---
        // Define the compile-time debug level (adjust as needed)
        const MACRO_DEBUG_LEVEL: usize = 7;
        // List of filenames (not full paths) to allow debug messages from.
        // If empty, all files are allowed (respecting MACRO_DEBUG_LEVEL).
        // Example: &["parser.rs", "constraint.rs"]
        const ALLOWED_FILES: &[&str] = &[
            // "parser.rs", // Example: Uncomment to allow messages from parser.rs
            // "constraint.rs", // Example: Uncomment to allow messages from constraint.rs
            // "interface.rs",
            // Add more filenames here as needed
        ];
        // --- End Configuration ---

        // Runtime check against the message's level and file path
        if $level <= MACRO_DEBUG_LEVEL {
            let current_file_path = std::path::Path::new(file!());
            // Extract the filename, default to empty string if extraction fails
            let current_filename = current_file_path.file_name()
                .map_or("", |os_str| os_str.to_str().unwrap_or(""));

            // Allow if ALLOWED_FILES is empty (no filter) or if the current file is in the list
            if ALLOWED_FILES.is_empty() || ALLOWED_FILES.contains(&current_filename) {
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