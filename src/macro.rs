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
use std::sync::Mutex;

/// Returns the current debug level, read from the `MACRO_DEBUG_LEVEL` environment variable.
pub fn get_macro_debug_level() -> usize {
    static MACRO_DEBUG_LEVEL: Lazy<usize> =
        Lazy::new(|| env::var("MACRO_DEBUG_LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(5));
    *MACRO_DEBUG_LEVEL
}

/// Checks if a given debug level is enabled based on `MACRO_DEBUG_LEVEL`.
pub fn is_debug_level_enabled(level: usize) -> bool {
    level <= get_macro_debug_level()
}

/// Tracks the last filename printed by the debug macro to avoid repetition.
pub static LAST_DEBUG_FILE: Lazy<Mutex<String>> = Lazy::new(|| Mutex::new(String::new()));

/// Tracks the last time a debug message was printed.
pub static LAST_DEBUG_TIME: Lazy<Mutex<Option<std::time::Instant>>> = Lazy::new(|| Mutex::new(None));

/// Tracks if the last debug message was a 'start' message that didn't print a newline.
pub static PENDING_INCOMPLETE_LINE: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

/// A list of filenames (not full paths) to allow debug messages from.
pub const ALLOWED_FILES: &[&str] = &[
    // "parser.rs",
    // "constraint.rs",
];

/// Internal implementation for the new grouped format (debug!).
/// Uses ANSI colors: Bold Cyan for files, Dark Gray for line numbers.
#[doc(hidden)]
#[macro_export]
macro_rules! __debug_grouped_impl {
    ($level:expr, $user_fmt:expr, $($user_args:tt)*) => {{
        if $level <= $crate::r#macro::get_macro_debug_level() {
            let current_file_path = std::path::Path::new(file!());
            let current_filename = current_file_path.file_name()
                .map_or("", |os_str| os_str.to_str().unwrap_or(""));

            if $crate::r#macro::ALLOWED_FILES.is_empty() || $crate::r#macro::ALLOWED_FILES.contains(&current_filename) {
                let mut last_file_guard = $crate::r#macro::LAST_DEBUG_FILE.lock().unwrap();
                let mut last_time_guard = $crate::r#macro::LAST_DEBUG_TIME.lock().unwrap();
                let mut pending_guard = $crate::r#macro::PENDING_INCOMPLETE_LINE.lock().unwrap();

                if *pending_guard {
                    println!();
                    *pending_guard = false;
                }
                let now = std::time::Instant::now();

                let elapsed_str = if let Some(last_time) = *last_time_guard {
                    let diff = now.duration_since(last_time);
                    if diff.as_millis() > 1 {
                        format!("\x1b[35m+{:?}\x1b[0m ", diff)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                *last_time_guard = Some(now);

                let current_file_str = file!();

                // If filename changed, print it in Bold Cyan
                if *last_file_guard != current_file_str {
                    // \x1b[1;36m = Bold Cyan, \x1b[0m = Reset
                    println!("\x1b[1;36m{}\x1b[0m", current_file_str);
                    *last_file_guard = current_file_str.to_string();
                }

                // Print line number in Dark Gray, then the message
                // \x1b[90m = Dark Gray (Bright Black)
                println!(
                    concat!("{}\x1b[90m  {:>4}\x1b[0m  ", $user_fmt),
                    elapsed_str,
                    line!(),
                    $($user_args)*
                );
            }
        }
    }};
}

/// Internal implementation for the start of a debug span (debug_start!).
/// Prints the message without a newline and sets the pending flag.
#[doc(hidden)]
#[macro_export]
macro_rules! __debug_start_impl {
    ($level:expr, $user_fmt:expr, $($user_args:tt)*) => {{
        if $level <= $crate::r#macro::get_macro_debug_level() {
            let current_file_path = std::path::Path::new(file!());
            let current_filename = current_file_path.file_name()
                .map_or("", |os_str| os_str.to_str().unwrap_or(""));

            if $crate::r#macro::ALLOWED_FILES.is_empty() || $crate::r#macro::ALLOWED_FILES.contains(&current_filename) {
                let mut last_file_guard = $crate::r#macro::LAST_DEBUG_FILE.lock().unwrap();
                let mut last_time_guard = $crate::r#macro::LAST_DEBUG_TIME.lock().unwrap();
                let mut pending_guard = $crate::r#macro::PENDING_INCOMPLETE_LINE.lock().unwrap();

                if *pending_guard {
                    println!();
                    *pending_guard = false;
                }
                let now = std::time::Instant::now();

                let elapsed_str = if let Some(last_time) = *last_time_guard {
                    let diff = now.duration_since(last_time);
                    if diff.as_millis() > 1 {
                        format!("\x1b[35m+{:?}\x1b[0m ", diff)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                *last_time_guard = Some(now);

                let current_file_str = file!();

                if *last_file_guard != current_file_str {
                    println!("\x1b[1;36m{}\x1b[0m", current_file_str);
                    *last_file_guard = current_file_str.to_string();
                }

                print!(
                    concat!("{}\x1b[90m  {:>4}\x1b[0m  ", $user_fmt),
                    elapsed_str,
                    line!(),
                    $($user_args)*
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
                *pending_guard = true;
                Some(())
            } else { None }
        } else { None }
    }};
}

/// Internal implementation for the old format (debug_line!).
/// Uses ANSI colors: Bold Yellow for the tag.
#[doc(hidden)]
#[macro_export]
macro_rules! __debug_line_impl {
    ($level:expr, $user_fmt:expr, $($user_args:tt)*) => {{
        if $level <= $crate::r#macro::get_macro_debug_level() {
            let current_file_path = std::path::Path::new(file!());
            let current_filename = current_file_path.file_name()
                .map_or("", |os_str| os_str.to_str().unwrap_or(""));

            if $crate::r#macro::ALLOWED_FILES.is_empty() || $crate::r#macro::ALLOWED_FILES.contains(&current_filename) {
                let mut pending_guard = $crate::r#macro::PENDING_INCOMPLETE_LINE.lock().unwrap();
                if *pending_guard {
                    println!();
                    *pending_guard = false;
                }

                // \x1b[1;33m = Bold Yellow
                println!(
                    concat!("\x1b[1;33m[DEBUG] {}]\x1b[0m {}:{}: ", $user_fmt),
                    $level,
                    file!(),
                    line!(),
                    $($user_args)*
                );
            }
        }
    }};
}

/// The main debug macro.
/// Prints filename (Bold Cyan) only when it changes.
/// Prints line numbers (Dark Gray) indented.
#[macro_export]
macro_rules! debug {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {
        $crate::__debug_grouped_impl!($level, $fmt, $($($arg)*)?);
    };
    ($level:expr, $msg:expr) => {
        $crate::__debug_grouped_impl!($level, "{:?}", $msg);
    };
}

/// The legacy debug macro.
/// Prints [DEBUG] (Yellow) level] file:line: msg.
#[macro_export]
macro_rules! debug_line {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {
        $crate::__debug_line_impl!($level, $fmt, $($($arg)*)?);
    };
    ($level:expr, $msg:expr) => {
        $crate::__debug_line_impl!($level, "{:?}", $msg);
    };
}

/// Starts a debug message that will be completed later.
/// Returns a token that must be passed to `debug_end!`.
#[macro_export]
macro_rules! debug_start {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {
        $crate::__debug_start_impl!($level, $fmt, $($($arg)*)?)
    };
    ($level:expr, $msg:expr) => {
        $crate::__debug_start_impl!($level, "{:?}", $msg)
    };
}

/// Completes a debug message started with `debug_start!`.
/// If no other debug messages were printed in between, it appends to the same line.
/// Otherwise, it prints the message on a new line with a continuation marker.
#[macro_export]
macro_rules! debug_end {
    ($token:expr, $fmt:literal $(, $($arg:tt)*)?) => {
        if $token.is_some() {
            let mut pending_guard = $crate::r#macro::PENDING_INCOMPLETE_LINE.lock().unwrap();
            if *pending_guard {
                println!($fmt $(, $($arg)*)?);
                *pending_guard = false;
            } else {
                println!(concat!("... ", $fmt) $(, $($arg)*)?);
            }
        }
    };
    ($token:expr, $msg:expr) => {
        $crate::debug_end!($token, "{:?}", $msg);
    };
}
