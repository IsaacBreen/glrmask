/// Macro for creating a sequence of parsers
#[macro_export]
macro_rules! seq_fast {
    ($($x:expr),* $(,)?) => {
        $crate::interface::tokenizer_combinators::seq_fast(vec![$($x),*])
    };
}

/// Macro for creating a choice of parsers
#[macro_export]
macro_rules! choice_fast {
    ($($x:expr),* $(,)?) => {
        $crate::interface::tokenizer_combinators::choice_fast(vec![$($x),*])
    };
}

use chrono::Local; // Import the Local timezone functionality

#[macro_export]
macro_rules! debug {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        // --- Configuration ---
        const MACRO_DEBUG_LEVEL: usize = 5;
        // List of filenames (not full paths) to allow debug messages from.
        // Example: &["parser.rs", "constraint.rs"]
        const ALLOWED_FILES: &[&str] = &[
            // "parser.rs", // Example: Uncomment to allow messages from parser.rs
            // "constraint.rs", // Example: Uncomment to allow messages from constraint.rs
            // Add more filenames here as needed
        ];
        // --- End Configuration ---

            // #[cfg(feature = "debug")] // Keep this if you want compile-time stripping
            { // Use a block to scope the 'now' variable and the import
                // Make chrono, file! and line! available inside the macro expansion
                use chrono::Local;
                let now = Local::now();
                println!(
                    concat!("[DEBUG {}] {}:{}: ", $fmt), // Add timestamp, file, line placeholders
                    file!(), line!(), // Add file and line number
                    $level
                    $(, $($arg)*)? // Forward the original arguments
                );
            }
        }

        // Runtime check against the message's level and file path
        if $level <= MACRO_DEBUG_LEVEL {
            let current_file_path = std::path::Path::new(file!());
            let current_filename = current_file_path.file_name().map_or("", |os_str| os_str.to_str().unwrap_or(""));

            // Allow if ALLOWED_FILES is empty (no filter) or if the current file is in the list
            if ALLOWED_FILES.is_empty() || ALLOWED_FILES.contains(&current_filename) {
                // #[cfg(feature = "debug")] // Keep this if you want compile-time stripping
                { // Use a block to scope the 'now' variable and the import
                    // Make chrono, file! and line! available inside the macro expansion
                    use chrono::Local;
                    let now = Local::now();
                    println!(
                        concat!("[DEBUG {}] {}:{}: ", $fmt), // Add timestamp, file, line placeholders
                        file!(), line!(), // Add file and line number
                        $level
                        $(, $($arg)*)? // Forward the original arguments
                    );
                }
            }
        }
    }};

    ($level:expr, $msg:expr) => {{
        // Define the compile-time debug level (adjust as needed)
        const MACRO_DEBUG_LEVEL: usize = 5;
        // List of filenames (not full paths) to allow debug messages from.
        // Example: &["parser.rs", "constraint.rs"]
        const ALLOWED_FILES: &[&str] = &[
            // "parser.rs", // Example: Uncomment to allow messages from parser.rs
            // "constraint.rs", // Example: Uncomment to allow messages from constraint.rs
            // Add more filenames here as needed
        ];

        // Runtime check against the message's level and file path
        if $level <= MACRO_DEBUG_LEVEL {
            let current_file_path = std::path::Path::new(file!());
            let current_filename = current_file_path.file_name().map_or("", |os_str| os_str.to_str().unwrap_or(""));

            // Allow if ALLOWED_FILES is empty (no filter) or if the current file is in the list
            if ALLOWED_FILES.is_empty() || ALLOWED_FILES.contains(&current_filename) {
            // #[cfg(feature = "debug")] // Keep this if you want compile-time stripping
            { // Use a block to scope the 'now' variable and the import
                // Make chrono, file! and line! available inside the macro expansion
                use chrono::Local;
                let now = Local::now();
                println!(
                    "[DEBUG {}] {}:{}: {:?}", // Add timestamp, file, line placeholders
                    file!(), line!(), // Add file and line number
                    $level,
                    $msg // Forward the original message expression
                );
            }
        }
    }};
    }
}

