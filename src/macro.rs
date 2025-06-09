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
            if ALLOWED_FILES.is_empty() || ALLOWED_FILES.contains(&current_filename) {
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
            if ALLOWED_FILES.is_empty() || ALLOWED_FILES.contains(&current_filename) {
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

/// A module for detailed debug tracking, such as counting line hits.
///
/// This module provides a global hit counter that can be used to track how many
/// times specific lines of code are executed. It is designed to be used from
/// anywhere in the code without needing to pass around a logger object.
///
/// The functionality is enabled by the `debug_hits` feature flag.
pub mod debug_hits {
    use std::collections::HashMap;
    use std::mem::MaybeUninit;
    use std::sync::{Mutex, Once};

    /// A struct to hold hit counts for various code locations.
    /// The `counts` field is public to allow direct access to the data.
    pub struct HitCounter {
        pub counts: HashMap<String, u64>,
    }

    impl HitCounter {
        fn new() -> Self {
            HitCounter {
                counts: HashMap::new(),
            }
        }

        /// Increments the hit count for a given key.
        fn hit(&mut self, key: &str) {
            *self.counts.entry(key.to_string()).or_insert(0) += 1;
        }

        /// Resets all hit counts to zero.
        fn reset(&mut self) {
            self.counts.clear();
        }

        /// Prints the hit counts to the console in a sorted, readable format.
        pub fn print_summary(&self) {
            if self.counts.is_empty() {
                println!("--- Hit Counts (no hits recorded) ---");
                return;
            }

            let mut sorted_counts: Vec<_> = self.counts.iter().collect();
            // Sort by count descending, then by key ascending
            sorted_counts.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));

            println!("--- Hit Summary ---");
            for (key, count) in sorted_counts {
                println!("{:>8} : {}", count, key);
            }
            println!("-------------------");
        }
    }

    static mut GLOBAL_HIT_COUNTER: MaybeUninit<Mutex<HitCounter>> = MaybeUninit::uninit();
    static INIT: Once = Once::new();

    fn get_instance() -> &'static Mutex<HitCounter> {
        INIT.call_once(|| unsafe {
            GLOBAL_HIT_COUNTER
                .as_mut_ptr()
                .write(Mutex::new(HitCounter::new()));
        });
        unsafe { &*GLOBAL_HIT_COUNTER.as_ptr() }
    }

    /// Increments the hit count for a given string key.
    /// This is the backing function for the `hit!` macro.
    pub fn hit(key: &str) {
        get_instance().lock().unwrap().hit(key);
    }

    /// Resets all hit counts.
    pub fn reset_hits() {
        get_instance().lock().unwrap().reset();
    }

    /// Prints a summary of all hit counts in a readable format.
    pub fn print_summary() {
        get_instance().lock().unwrap().print_summary();
    }

    /// Provides mutable access to the raw hit counter data.
    /// The lock is held while the closure `f` is executed.
    ///
    /// # Panics
    /// Panics if the mutex is poisoned.
    pub fn with_hits_mut<F, R>(f: F) -> R
    where
        F: FnOnce(&mut HitCounter) -> R,
    {
        let mut guard = get_instance().lock().unwrap();
        f(&mut *guard)
    }

    /// Provides immutable access to the raw hit counter data.
    /// The lock is held while the closure `f` is executed.
    ///
    /// # Panics
    /// Panics if the mutex is poisoned.
    pub fn with_hits<F, R>(f: F) -> R
    where
        F: FnOnce(&HitCounter) -> R,
    {
        let guard = get_instance().lock().unwrap();
        f(&*guard)
    }
}

/// Macro to record a "hit" for a line of code, optionally with a descriptive string.
/// This functionality is only enabled when the `debug_hits` feature is active in your Cargo.toml.
///
/// The macro assumes this file (`macro.rs`) is included in your crate root (`lib.rs` or `main.rs`)
/// as `mod macro;`. If you use `mod macros;`, you will need to change `$crate::macro` to `$crate::macros`.
#[macro_export]
macro_rules! hit {
    ($s:expr) => {
        #[cfg(feature = "debug_hits")]
        {
            $crate::macro::debug_hits::hit(&format!("{}:{}: {}", file!(), line!(), $s));
        }
    };
    () => {
        #[cfg(feature = "debug_hits")]
        {
            $crate::macro::debug_hits::hit(&format!("{}:{}", file!(), line!()));
        }
    };
}
