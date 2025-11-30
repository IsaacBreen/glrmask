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

// =============================================================================
// Debug Level System
// =============================================================================
//
// Control via MACRO_DEBUG_LEVEL environment variable (default: 1)
//
// Level 0: Silent
//   - Errors only (via eprintln!)
//   - Use for: CI/CD, production, scripting
//
// Level 1: Milestones (default)
//   - Major completion checkmarks (✓)
//   - Key timing information
//   - High-level warnings (summary counts)
//   - Use for: Normal usage, quick runs
//
// Level 2: Summary Stats
//   - Same as level 1, plus:
//   - Aggregate stats (counts, sizes, compression ratios)
//   - Use for: Understanding build characteristics
//
// Level 3: Pipeline Stages
//   - Named pipeline phases (▸ arrow prefix)
//   - Stage timing with deltas
//   - Detailed warning messages
//   - Use for: Debugging pipeline, understanding flow
//
// Level 4: Substeps
//   - Operations within stages (• bullet prefix)
//   - Progress bars enabled
//   - Finer-grained timing
//   - Use for: Debugging specific stages
//
// Level 5: Algorithm Details
//   - Internal algorithm stats (dim text)
//   - Data structure sizes
//   - Intermediate results
//   - Use for: Deep debugging, performance analysis
//
// Level 6+: Verbose Traces
//   - File:line prefixes for every message
//   - Full data structure dumps
//   - Use for: Development, tracing execution
//
// =============================================================================

// ANSI Color Codes
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    
    // Regular colors
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";
    pub const GRAY: &str = "\x1b[90m";
    
    // Bold colors
    pub const BOLD_GREEN: &str = "\x1b[1;32m";
    pub const BOLD_CYAN: &str = "\x1b[1;36m";
    pub const BOLD_YELLOW: &str = "\x1b[1;33m";
    pub const BOLD_RED: &str = "\x1b[1;31m";
    pub const BOLD_WHITE: &str = "\x1b[1;37m";
    pub const BOLD_BLUE: &str = "\x1b[1;34m";
    pub const BOLD_MAGENTA: &str = "\x1b[1;35m";
    
    // Symbols
    pub const CHECK: &str = "✓";
    pub const ARROW: &str = "→";
    pub const BULLET: &str = "•";
    pub const WARN: &str = "⚠";
    pub const PLAY: &str = "▸";
    pub const BOX: &str = "■";
    pub const LINE: &str = "─";
    pub const CORNER: &str = "└";
    pub const PIPE: &str = "│";
}

/// Returns the current debug level from `MACRO_DEBUG_LEVEL` env var.
pub fn get_macro_debug_level() -> usize {
    static MACRO_DEBUG_LEVEL: Lazy<usize> =
        Lazy::new(|| env::var("MACRO_DEBUG_LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(1));
    *MACRO_DEBUG_LEVEL
}

/// Checks if a given debug level is enabled.
pub fn is_debug_level_enabled(level: usize) -> bool {
    level <= get_macro_debug_level()
}

/// Returns true if progress bars should be shown (level 4+).
pub fn should_show_progress_bars() -> bool {
    get_macro_debug_level() >= 4
}

/// Tracks the last filename printed (for level 6+ file headers).
pub static LAST_DEBUG_FILE: Lazy<Mutex<String>> = Lazy::new(|| Mutex::new(String::new()));

/// Tracks the last time a debug message was printed.
pub static LAST_DEBUG_TIME: Lazy<Mutex<Option<std::time::Instant>>> = Lazy::new(|| Mutex::new(None));

/// A list of filenames to allow debug messages from. Empty = all allowed.
pub const ALLOWED_FILES: &[&str] = &[];

/// Formats a duration in a human-readable way.
pub fn format_duration(d: std::time::Duration) -> String {
    let micros = d.as_micros();
    if micros < 1000 {
        format!("{}µs", micros)
    } else {
        let millis = d.as_millis();
        if millis < 1000 {
            format!("{}ms", millis)
        } else {
            let secs = d.as_secs_f64();
            if secs < 60.0 {
                format!("{:.2}s", secs)
            } else {
                let mins = secs / 60.0;
                format!("{:.1}m", mins)
            }
        }
    }
}

/// Formats a byte count in human-readable form.
pub fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Returns elapsed time since last debug message, if significant.
pub fn get_elapsed_suffix(now: std::time::Instant, threshold_ms: u64) -> String {
    let mut last_time_guard = LAST_DEBUG_TIME.lock().unwrap();
    let suffix = if let Some(last_time) = *last_time_guard {
        let diff = now.duration_since(last_time);
        if diff.as_millis() >= threshold_ms as u128 {
            format!(" \x1b[35m+{}\x1b[0m", format_duration(diff))
        } else {
            String::new()
        }
    } else {
        String::new()
    };
    *last_time_guard = Some(now);
    suffix
}

// =============================================================================
// Core Debug Macro - Level-based formatting
// =============================================================================

#[doc(hidden)]
#[macro_export]
macro_rules! __debug_impl {
    ($level:expr, $user_fmt:expr, $($user_args:tt)*) => {{
        if $level <= $crate::r#macro::get_macro_debug_level() {
            use $crate::r#macro::colors::*;
            
            let msg = format!($user_fmt, $($user_args)*);
            let now = std::time::Instant::now();
            
            // Level 6+: Show file:line info with verbose output
            if $crate::r#macro::get_macro_debug_level() >= 6 {
                let current_file_str = file!();
                let mut last_file_guard = $crate::r#macro::LAST_DEBUG_FILE.lock().unwrap();
                
                // Print file header if changed
                if *last_file_guard != current_file_str {
                    println!("{GRAY}─── {}{RESET}", current_file_str);
                    *last_file_guard = current_file_str.to_string();
                }
                
                let elapsed = $crate::r#macro::get_elapsed_suffix(now, 10);
                println!("{GRAY}{:>4}{RESET}  {}{}", line!(), msg, elapsed);
            } else {
                // Levels 1-5: Clean output with level-based formatting
                let elapsed = $crate::r#macro::get_elapsed_suffix(now, 50);
                
                // Apply visual hierarchy based on the message's level
                // Indentation: Level 3 = 3 spaces, Level 4 = 5 spaces, Level 5 = 7 spaces
                match $level {
                    1 | 2 => {
                        // High-level info: no prefix
                        println!("{}{}", msg, elapsed);
                    }
                    3 => {
                        // Pipeline stage: arrow prefix (3-space indent)
                        println!("   {CYAN}{PLAY}{RESET} {}{}", msg, elapsed);
                    }
                    4 => {
                        // Substep: bullet prefix (5-space indent)
                        println!("     {DIM}{BULLET}{RESET} {}{}", msg, elapsed);
                    }
                    5 => {
                        // Detail: dim text (7-space indent)
                        println!("       {DIM}{}{}{RESET}", msg, elapsed);
                    }
                    _ => {
                        // Level 6+ without verbose mode (shouldn't happen)
                        println!("{}{}", msg, elapsed);
                    }
                }
            }
        }
    }};
}

/// Generic debug macro. Use semantic helpers when possible.
#[macro_export]
macro_rules! debug {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {
        $crate::__debug_impl!($level, $fmt, $($($arg)*)?);
    };
    ($level:expr, $msg:expr) => {
        $crate::__debug_impl!($level, "{}", $msg);
    };
}

// =============================================================================
// Semantic Output Helpers - Use these instead of raw debug!
// =============================================================================

/// Level 1: Major milestone with checkmark (always visible at level 1+)
/// Usage: log_milestone!("Loaded grammar", "20 productions");
#[macro_export]
macro_rules! log_milestone {
    ($name:expr, $detail:expr) => {
        if $crate::r#macro::is_debug_level_enabled(1) {
            use $crate::r#macro::colors::*;
            println!("  {BOLD_GREEN}{CHECK}{RESET}  {} {DIM}({}){RESET}", $name, $detail);
        }
    };
    ($name:expr) => {
        if $crate::r#macro::is_debug_level_enabled(1) {
            use $crate::r#macro::colors::*;
            println!("  {BOLD_GREEN}{CHECK}{RESET}  {}", $name);
        }
    };
}

/// Level 2: Summary statistic (key metrics)
/// Usage: log_stat!("States", 50);
#[macro_export]
macro_rules! log_stat {
    ($name:expr, $value:expr) => {
        if $crate::r#macro::is_debug_level_enabled(2) {
            use $crate::r#macro::colors::*;
            println!("   {DIM}{}{RESET} {:<25} {CYAN}{}{RESET}", $crate::r#macro::colors::PIPE, $name, $value);
        }
    };
    ($name:expr, $value:expr, $unit:expr) => {
        if $crate::r#macro::is_debug_level_enabled(2) {
            use $crate::r#macro::colors::*;
            println!("   {DIM}{}{RESET} {:<25} {CYAN}{}{RESET} {}", $crate::r#macro::colors::PIPE, $name, $value, $unit);
        }
    };
}

/// Level 3: Pipeline stage (major phase in the compilation)
/// Usage: log_stage!("Building tokenizer");
#[macro_export]
macro_rules! log_stage {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        if $crate::r#macro::is_debug_level_enabled(3) {
            use $crate::r#macro::colors::*;
            let msg = format!($fmt $(, $($arg)*)?);
            let now = std::time::Instant::now();
            let elapsed = $crate::r#macro::get_elapsed_suffix(now, 10);
            println!("   {BOLD_BLUE}{}{RESET} {}{}", $crate::r#macro::colors::PLAY, msg, elapsed);
        }
    };
}

/// Level 3: Pipeline stage completion with timing
#[macro_export]
macro_rules! log_stage_done {
    ($name:expr, $start:expr) => {
        if $crate::r#macro::is_debug_level_enabled(3) {
            use $crate::r#macro::colors::*;
            let elapsed = $start.elapsed();
            println!("   {BOLD_BLUE}{}{RESET} {} {MAGENTA}({}){RESET}", 
                $crate::r#macro::colors::CHECK, $name, $crate::r#macro::format_duration(elapsed));
        }
    };
    ($name:expr, $start:expr, $detail:expr) => {
        if $crate::r#macro::is_debug_level_enabled(3) {
            use $crate::r#macro::colors::*;
            let elapsed = $start.elapsed();
            println!("   {BOLD_BLUE}{}{RESET} {} {DIM}[{}]{RESET} {MAGENTA}({}){RESET}", 
                $crate::r#macro::colors::CHECK, $name, $detail, $crate::r#macro::format_duration(elapsed));
        }
    };
}

/// Level 4: Substep within a stage (operation)
/// Usage: log_substep!("Computing first sets");
#[macro_export]
macro_rules! log_substep {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        if $crate::r#macro::is_debug_level_enabled(4) {
            use $crate::r#macro::colors::*;
            let msg = format!($fmt $(, $($arg)*)?);
            println!("     {CYAN}{}{RESET} {}", $crate::r#macro::colors::BULLET, msg);
        }
    };
}

/// Level 4: Substep completion with timing
#[macro_export]
macro_rules! log_substep_done {
    ($name:expr, $start:expr) => {
        if $crate::r#macro::is_debug_level_enabled(4) {
            use $crate::r#macro::colors::*;
            let elapsed = $start.elapsed();
            if elapsed.as_millis() >= 10 {
                println!("     {CYAN}{}{RESET} {} {MAGENTA}({}){RESET}", 
                    $crate::r#macro::colors::BULLET, $name, $crate::r#macro::format_duration(elapsed));
            }
        }
    };
    ($name:expr, $start:expr, $detail:expr) => {
        if $crate::r#macro::is_debug_level_enabled(4) {
            use $crate::r#macro::colors::*;
            let elapsed = $start.elapsed();
            println!("     {CYAN}{}{RESET} {} {DIM}[{}]{RESET} {MAGENTA}({}){RESET}", 
                $crate::r#macro::colors::BULLET, $name, $detail, $crate::r#macro::format_duration(elapsed));
        }
    };
}

/// Level 5: Detail/algorithm info (dim, indented)
/// Usage: log_detail!("DFA states: {}", 50);
#[macro_export]
macro_rules! log_detail {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        if $crate::r#macro::is_debug_level_enabled(5) {
            use $crate::r#macro::colors::*;
            let msg = format!($fmt $(, $($arg)*)?);
            println!("       {DIM}{}{RESET}", msg);
        }
    };
}

/// Level 1: Warning message (always visible at level 1+)
#[macro_export]
macro_rules! log_warn {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        if $crate::r#macro::is_debug_level_enabled(1) {
            use $crate::r#macro::colors::*;
            let msg = format!($fmt $(, $($arg)*)?);
            println!("  {BOLD_YELLOW}{}{RESET} {}", $crate::r#macro::colors::WARN, msg);
        }
    };
}

/// Level 0: Error message (always shown)
#[macro_export]
macro_rules! log_error {
    ($fmt:literal $(, $($arg:tt)*)?) => {{
        use $crate::r#macro::colors::*;
        let msg = format!($fmt $(, $($arg)*)?);
        eprintln!("{BOLD_RED}Error:{RESET} {}", msg);
    }};
}

/// Level 1: Success message with checkmark
#[macro_export]
macro_rules! log_success {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        if $crate::r#macro::is_debug_level_enabled(1) {
            use $crate::r#macro::colors::*;
            let msg = format!($fmt $(, $($arg)*)?);
            println!("{BOLD_GREEN}{} {}{RESET}", $crate::r#macro::colors::CHECK, msg);
        }
    };
}

/// Level 1: Section Header
/// Usage: log_header!("Stage 1: Parsing");
#[macro_export]
macro_rules! log_header {
    ($title:expr) => {
        if $crate::r#macro::is_debug_level_enabled(1) {
            use $crate::r#macro::colors::*;
            println!("\n{BOLD_MAGENTA}═══ {} ═══{RESET}", $title);
        }
    };
}

/// Level 1: Separator Line
#[macro_export]
macro_rules! log_separator {
    () => {
        if $crate::r#macro::is_debug_level_enabled(1) {
            use $crate::r#macro::colors::*;
            println!("{DIM}────────────────────────────────────────{RESET}");
        }
    };
}

/// Level 2: Key-Value Table Row
/// Usage: log_kv!("States", 50);
#[macro_export]
macro_rules! log_kv {
    ($key:expr, $value:expr) => {
        if $crate::r#macro::is_debug_level_enabled(2) {
            use $crate::r#macro::colors::*;
            println!("   {DIM}{}{RESET} {:<25} {CYAN}{}{RESET}", $crate::r#macro::colors::PIPE, $key, $value);
        }
    };
}


// =============================================================================
// Timer Helpers for measuring operations
// =============================================================================

/// Start a timer (returns Instant). Used with log_stage_done!, log_substep_done!
#[macro_export]
macro_rules! timer_start {
    () => {
        std::time::Instant::now()
    };
}

// Backwards compatibility aliases
#[macro_export]
macro_rules! debug_line {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {
        $crate::debug!($level, $fmt $(, $($arg)*)?);
    };
    ($level:expr, $msg:expr) => {
        $crate::debug!($level, "{}", $msg);
    };
}
