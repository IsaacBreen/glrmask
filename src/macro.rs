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
        // Define the compile-time debug level (adjust as needed)
        const MACRO_DEBUG_LEVEL: usize = 2;

        // Runtime check against the message's level
        if $level <= MACRO_DEBUG_LEVEL {
            // #[cfg(feature = "debug")] // Keep this if you want compile-time stripping
            { // Use a block to scope the 'now' variable and the import
                // Make chrono, file! and line! available inside the macro expansion
                use chrono::Local;
                let now = Local::now();
                println!(
                    concat!("[DEBUG {}] {}:{}: ", $fmt), // Add timestamp, file, line placeholders
                    // now.format("%Y-%m-%d %H:%M:%S%.3f"), // Format the timestamp (YYYY-MM-DD HH:MM:SS.ms)
                    file!(), line!(), // Add file and line number
                    $level
                    $(, $($arg)*)? // Forward the original arguments
                );
            }
        }
    }};

    ($level:expr, $msg:expr) => {{
        // Define the compile-time debug level (adjust as needed)
        const MACRO_DEBUG_LEVEL: usize = 2;

        // Runtime check against the message's level
        if $level <= MACRO_DEBUG_LEVEL {
            // #[cfg(feature = "debug")] // Keep this if you want compile-time stripping
            { // Use a block to scope the 'now' variable and the import
                // Make chrono, file! and line! available inside the macro expansion
                use chrono::Local;
                let now = Local::now();
                println!(
                    "[DEBUG {}] {}:{}: {:?}", // Add timestamp, file, line placeholders
                    // now.format("%Y-%m-%d %H:%M:%S%.3f"), // Format the timestamp
                    file!(), line!(), // Add file and line number
                    $level,
                    $msg // Forward the original message expression
                );
            }
        }
    }};
}

