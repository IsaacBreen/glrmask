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

#[macro_export]
macro_rules! debug {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        // --- Configuration ---
        // Define the compile-time debug level (adjust as needed)
        const MACRO_DEBUG_LEVEL: usize = 2;
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
            if ALLOWED_FILES.is_empty() || ALLOWED_FILES.contains(¤t_filename) {
                // Optional: Keep this if you want compile-time stripping based on a feature flag
                // #[cfg(feature = "debug")]
                { // Use a block to scope the 'now' variable and the import
                    // Make chrono, file! and line! available inside the macro expansion
                    use chrono::Local;
                    let now = Local::now(); // Timestamp removed for brevity, uncomment if needed
                    println!(
                        // Add file, line placeholders. Add timestamp placeholder if needed.
                        // concat!("[DEBUG] {}] {}:{}: ", $fmt), // The complete format string
                        concat!("[DEBUG {} {}] {}:{}: ", $fmt), // The complete format string
                        // now.format("%Y-%m-%d %H:%M:%S%.3f"), // Uncomment for timestamp
                        now.format("%H:%M:%S%.3f"), // Uncomment for timestamp
                        $level,           // Argument for the first {} in the prefix
                        file!(),          // Argument for the second {} in the prefix
                        line!(),          // Argument for the third {} in the prefix
                        $($($arg)*)?      // Arguments for the placeholders in the original $fmt
                    );
                }
            }
        }
    }};

    ($level:expr, $msg:expr) => {{
        // --- Configuration ---
        // Define the compile-time debug level (adjust as needed)
        const MACRO_DEBUG_LEVEL: usize = 2;
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
            if ALLOWED_FILES.is_empty() || ALLOWED_FILES.contains(¤t_filename) {
                // Optional: Keep this if you want compile-time stripping based on a feature flag
                // #[cfg(feature = "debug")]
                { // Use a block to scope the 'now' variable and the import
                    // Make chrono, file! and line! available inside the macro expansion
                    use chrono::Local;
                    let now = Local::now(); // Timestamp removed for brevity, uncomment if needed
                    println!(
                        // Add file, line placeholders. Add timestamp placeholder if needed.
                        // "[DEBUG {}] {}:{}: {:?}", // Format string for the expression variant
                        "[DEBUG {} {}] {}:{}: {:?}", // Format string for the expression variant
                        // now.format("%Y-%m-%d %H:%M:%S%.3f"), // Uncomment for timestamp
                        now.format("%H:%M:%S%.3f"), // Uncomment for timestamp
                        $level,           // Argument for the first {} in the prefix
                        file!(),          // Argument for the second {} in the prefix
                        line!(),          // Argument for the third {} in the prefix
                        $msg // Forward the original message expression
                    );
                }
            }
        }
    }};
}