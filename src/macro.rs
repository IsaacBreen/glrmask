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
// Hierarchical Debug Level System
// =============================================================================
//
// Control via environment variables:
//   MACRO_DEBUG_LEVEL - verbosity level (default: 1)
//   MACRO_LINE_LEVELS - comma-separated levels that draw │ lines (default: "")
//
// Level 0: Errors only
// Level 1: Major milestones (headings)
// Level 2: Section headers (subheadings)
// Level 3: Pipeline stages
// Level 4: Substeps (• prefix)
// Level 5: Algorithm details (dim)
// Level 6+: Verbose traces with file:line
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
    
    // Symbols
    pub const CHECK: &str = "✓";
    pub const ARROW: &str = "→";
    pub const BULLET: &str = "•";
    pub const WARN: &str = "⚠";
    pub const LINE: &str = "│";
    pub const LINE_END: &str = "└─";
    pub const BOX: &str = "■";
}

// =============================================================================
// Configuration for Each Debug Level
// =============================================================================

/// Configuration for a single debug level's output formatting.
#[derive(Clone, Copy)]
pub struct DebugLevelConfig {
    /// Symbol at the start of messages at this level (e.g., "• ")
    pub heading_symbol: &'static str,
    /// Symbol that propagates to all HIGHER levels when this level draws a line (e.g., "│ ")
    pub line_prefix: &'static str,
    /// ANSI style codes for this level
    pub style_start: &'static str,
    pub style_end: &'static str,
    /// Minimum milliseconds before showing +Xms timing suffix
    pub timing_threshold_ms: u64,
}

/// Configuration for section-closing "alt" messages (e.g., "└─ Done in 1.5s")
#[derive(Clone, Copy)]
pub struct DebugAltConfig {
    /// Symbol at the start (e.g., "└─ ")
    pub heading_symbol: &'static str,
    /// ANSI style codes
    pub style_start: &'static str,
    pub style_end: &'static str,
}

/// Default configurations for each level (indices 0-6)
pub static LEVEL_CONFIGS: [DebugLevelConfig; 7] = [
    // Level 0: Errors - no symbol, no line (handled separately)
    DebugLevelConfig {
        heading_symbol: "",
        line_prefix: "",
        style_start: "",
        style_end: "",
        timing_threshold_ms: 0,
    },
    // Level 1: Major milestones / outer sections
    DebugLevelConfig {
        heading_symbol: "",
        line_prefix: "│ ",
        style_start: "\x1b[1m", // Bold
        style_end: "\x1b[0m",
        timing_threshold_ms: 100,
    },
    // Level 2: Section headers / subsections  
    DebugLevelConfig {
        heading_symbol: "",
        line_prefix: "│ ",
        style_start: "\x1b[1m", // Bold
        style_end: "\x1b[0m",
        timing_threshold_ms: 50,
    },
    // Level 3: Pipeline stages
    DebugLevelConfig {
        heading_symbol: "",
        line_prefix: "  ",
        style_start: "",
        style_end: "",
        timing_threshold_ms: 50,
    },
    // Level 4: Substeps with bullet
    DebugLevelConfig {
        heading_symbol: "• ",
        line_prefix: "  ",
        style_start: "",
        style_end: "",
        timing_threshold_ms: 50,
    },
    // Level 5: Algorithm details - dim
    DebugLevelConfig {
        heading_symbol: "  ",
        line_prefix: "  ",
        style_start: "\x1b[2m", // Dim
        style_end: "\x1b[0m",
        timing_threshold_ms: 0,
    },
    // Level 6+: Verbose traces
    DebugLevelConfig {
        heading_symbol: "",
        line_prefix: "",
        style_start: "\x1b[90m", // Gray
        style_end: "\x1b[0m",
        timing_threshold_ms: 10,
    },
];

/// Alt (section-closing) configurations for each level
pub static ALT_CONFIGS: [DebugAltConfig; 7] = [
    // Level 0
    DebugAltConfig { heading_symbol: "", style_start: "", style_end: "" },
    // Level 1
    DebugAltConfig { heading_symbol: "└─ ", style_start: "\x1b[1m", style_end: "\x1b[0m" },
    // Level 2
    DebugAltConfig { heading_symbol: "└─ ", style_start: "\x1b[1m", style_end: "\x1b[0m" },
    // Level 3
    DebugAltConfig { heading_symbol: "└─ ", style_start: "", style_end: "" },
    // Level 4
    DebugAltConfig { heading_symbol: "└─ ", style_start: "", style_end: "" },
    // Level 5
    DebugAltConfig { heading_symbol: "", style_start: "\x1b[2m", style_end: "\x1b[0m" },
    // Level 6+
    DebugAltConfig { heading_symbol: "", style_start: "\x1b[90m", style_end: "\x1b[0m" },
];

// =============================================================================
// Runtime State
// =============================================================================

/// Returns the current debug level from `MACRO_DEBUG_LEVEL` env var.
pub fn get_macro_debug_level() -> usize {
    static MACRO_DEBUG_LEVEL: Lazy<usize> =
        Lazy::new(|| env::var("MACRO_DEBUG_LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(1));
    *MACRO_DEBUG_LEVEL
}

/// Returns which levels should draw lines, from `MACRO_LINE_LEVELS` env var.
/// Format: comma-separated level numbers, e.g., "1,2" means levels 1 and 2 draw lines.
/// If not set, defaults to levels 1-4 drawing lines for hierarchical output.
pub fn get_line_levels() -> &'static [bool; 7] {
    static LINE_LEVELS: Lazy<[bool; 7]> = Lazy::new(|| {
        let mut levels = [true, true, true, true, true, false, false]; // default: 1-4 draw lines
        if let Ok(val) = env::var("MACRO_LINE_LEVELS") {
            if val.is_empty() {
                // Explicit empty = no lines at all
                return [false; 7];
            }
            levels = [false; 7];
            for part in val.split(',') {
                if let Ok(n) = part.trim().parse::<usize>() {
                    if n < 7 {
                        levels[n] = true;
                    }
                }
            }
        }
        levels
    });
    &LINE_LEVELS
}

/// Returns true if a given level should draw a line (│).
pub fn level_draws_line(level: usize) -> bool {
    let levels = get_line_levels();
    level < 7 && levels[level]
}

/// Checks if a given debug level is enabled.
pub fn is_debug_level_enabled(level: usize) -> bool {
    level <= get_macro_debug_level()
}

/// Tracks the last filename printed (for level 6+ file headers).
pub static LAST_DEBUG_FILE: Lazy<Mutex<String>> = Lazy::new(|| Mutex::new(String::new()));

/// Tracks the last time a debug message was printed.
pub static LAST_DEBUG_TIME: Lazy<Mutex<Option<std::time::Instant>>> = Lazy::new(|| Mutex::new(None));

// Stack of active section levels (for tracking nested sections)
pub static SECTION_STACK: Lazy<Mutex<Vec<usize>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// A list of filenames to allow debug messages from. Empty = all allowed.
pub const ALLOWED_FILES: &[&str] = &[];

// =============================================================================
// Formatting Helpers
// =============================================================================

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

/// Build the line prefix string based on message level.
/// Level 1-2: no indentation (top-level messages)
/// Level 3+: "│ " (single level of indentation)
pub fn build_line_prefix(msg_level: usize) -> String {
    match msg_level {
        0..=2 => String::new(),
        _ => if level_draws_line(3) { "│ ".to_string() } else { "  ".to_string() },
    }
}

/// Build the line prefix for an alt (closing) message.
pub fn build_alt_line_prefix(msg_level: usize) -> String {
    // For closing messages, use same prefix as regular messages at that level
    build_line_prefix(msg_level)
}

// =============================================================================
// Core Debug Implementation
// =============================================================================

/// Print a debug message at the given level.
pub fn print_debug(level: usize, message: &str, file: &str, line: u32) {
    if level > get_macro_debug_level() {
        return;
    }
    
    let now = std::time::Instant::now();
    let cfg = &LEVEL_CONFIGS[level.min(6)];
    
    // Level 6+: Show file:line info
    if get_macro_debug_level() >= 6 && level >= 6 {
        let mut last_file_guard = LAST_DEBUG_FILE.lock().unwrap();
        if *last_file_guard != file {
            println!("\x1b[90m─── {}\x1b[0m", file);
            *last_file_guard = file.to_string();
        }
        let elapsed = get_elapsed_suffix(now, cfg.timing_threshold_ms);
        println!("\x1b[90m{:>4}\x1b[0m  {}{}", line, message, elapsed);
        return;
    }
    
    // Build the line prefix from lower levels
    let line_prefix = build_line_prefix(level);
    
    // Get timing suffix
    let elapsed = get_elapsed_suffix(now, cfg.timing_threshold_ms);
    
    // Print with styling
    println!("{}{}{}{}{}{}", 
        line_prefix,
        cfg.style_start,
        cfg.heading_symbol,
        message,
        cfg.style_end,
        elapsed
    );
}

/// Print an "alt" (section-closing) message at the given level.
/// Only prints if the level has line drawing enabled.
pub fn print_debug_alt(level: usize, message: &str, _file: &str, _line: u32) {
    if level > get_macro_debug_level() {
        return;
    }
    
    // Alt messages only show if this level draws lines
    if !level_draws_line(level) {
        return;
    }
    
    let now = std::time::Instant::now();
    let alt_cfg = &ALT_CONFIGS[level.min(6)];
    let level_cfg = &LEVEL_CONFIGS[level.min(6)];
    
    // Build prefix from levels BELOW this one
    let line_prefix = build_alt_line_prefix(level);
    
    // Get timing suffix
    let elapsed = get_elapsed_suffix(now, level_cfg.timing_threshold_ms);
    
    // Print with alt styling
    println!("{}{}{}{}{}{}", 
        line_prefix,
        alt_cfg.style_start,
        alt_cfg.heading_symbol,
        message,
        alt_cfg.style_end,
        elapsed
    );
}

// =============================================================================
// Core Debug Macros
// =============================================================================

/// Internal implementation macro for debug printing.
#[doc(hidden)]
#[macro_export]
macro_rules! __debug_impl {
    ($level:expr, $user_fmt:expr, $($user_args:tt)*) => {{
        if $level <= $crate::r#macro::get_macro_debug_level() {
            let msg = format!($user_fmt, $($user_args)*);
            $crate::r#macro::print_debug($level, &msg, file!(), line!());
        }
    }};
}

/// Internal implementation macro for alt (section-closing) debug printing.
#[doc(hidden)]
#[macro_export]
macro_rules! __debug_alt_impl {
    ($level:expr, $user_fmt:expr, $($user_args:tt)*) => {{
        if $level <= $crate::r#macro::get_macro_debug_level() {
            let msg = format!($user_fmt, $($user_args)*);
            $crate::r#macro::print_debug_alt($level, &msg, file!(), line!());
        }
    }};
}

/// Generic debug macro. Use for standard log messages.
/// Usage: debug!(level, "message") or debug!(level, "format {}", args)
#[macro_export]
macro_rules! debug {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {
        $crate::__debug_impl!($level, $fmt, $($($arg)*)?);
    };
    ($level:expr, $msg:expr) => {
        $crate::__debug_impl!($level, "{}", $msg);
    };
}

/// Alt debug macro for section-closing messages (e.g., "└─ Done in 1.5s").
/// Only prints if the level has line drawing enabled.
/// Usage: debug_alt!(level, "message") or debug_alt!(level, "format {}", args)
#[macro_export]
macro_rules! debug_alt {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {
        $crate::__debug_alt_impl!($level, $fmt, $($($arg)*)?);
    };
    ($level:expr, $msg:expr) => {
        $crate::__debug_alt_impl!($level, "{}", $msg);
    };
}

// =============================================================================
// Semantic Output Helpers (Backwards Compatibility)
// =============================================================================

/// Level 1: Major milestone with checkmark
#[macro_export]
macro_rules! log_milestone {
    ($name:expr, $detail:expr) => {
        if $crate::r#macro::is_debug_level_enabled(1) {
            use $crate::r#macro::colors::*;
            let prefix = $crate::r#macro::build_line_prefix(1);
            println!("{}  {BOLD_GREEN}{CHECK}{RESET}  {} {DIM}({}){RESET}", prefix, $name, $detail);
        }
    };
    ($name:expr) => {
        if $crate::r#macro::is_debug_level_enabled(1) {
            use $crate::r#macro::colors::*;
            let prefix = $crate::r#macro::build_line_prefix(1);
            println!("{}  {BOLD_GREEN}{CHECK}{RESET}  {}", prefix, $name);
        }
    };
}

/// Level 2: Summary statistic
#[macro_export]
macro_rules! log_stat {
    ($name:expr, $value:expr) => {
        if $crate::r#macro::is_debug_level_enabled(2) {
            use $crate::r#macro::colors::*;
            let prefix = $crate::r#macro::build_line_prefix(2);
            println!("{}     {DIM}└─{RESET} {}: {CYAN}{}{RESET}", prefix, $name, $value);
        }
    };
    ($name:expr, $value:expr, $unit:expr) => {
        if $crate::r#macro::is_debug_level_enabled(2) {
            use $crate::r#macro::colors::*;
            let prefix = $crate::r#macro::build_line_prefix(2);
            println!("{}     {DIM}└─{RESET} {}: {CYAN}{}{RESET} {}", prefix, $name, $value, $unit);
        }
    };
}

/// Level 3: Pipeline stage
#[macro_export]
macro_rules! log_stage {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        $crate::debug!(3, $fmt $(, $($arg)*)?);
    };
}

/// Level 3: Pipeline stage completion with timing
#[macro_export]
macro_rules! log_stage_done {
    ($name:expr, $start:expr) => {
        if $crate::r#macro::is_debug_level_enabled(3) {
            use $crate::r#macro::colors::*;
            let elapsed = $start.elapsed();
            let prefix = $crate::r#macro::build_line_prefix(3);
            if $crate::r#macro::level_draws_line(3) {
                println!("{}└─ {} {MAGENTA}({}){RESET}", prefix, $name, $crate::r#macro::format_duration(elapsed));
            }
        }
    };
    ($name:expr, $start:expr, $detail:expr) => {
        if $crate::r#macro::is_debug_level_enabled(3) {
            use $crate::r#macro::colors::*;
            let elapsed = $start.elapsed();
            let prefix = $crate::r#macro::build_line_prefix(3);
            if $crate::r#macro::level_draws_line(3) {
                println!("{}└─ {} {DIM}[{}]{RESET} {MAGENTA}({}){RESET}", 
                    prefix, $name, $detail, $crate::r#macro::format_duration(elapsed));
            }
        }
    };
}

/// Level 4: Substep
#[macro_export]
macro_rules! log_substep {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        $crate::debug!(4, $fmt $(, $($arg)*)?);
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
                let prefix = $crate::r#macro::build_line_prefix(4);
                println!("{}  {CYAN}{}{RESET} {} {MAGENTA}({}){RESET}", 
                    prefix, $crate::r#macro::colors::BULLET, $name, $crate::r#macro::format_duration(elapsed));
            }
        }
    };
    ($name:expr, $start:expr, $detail:expr) => {
        if $crate::r#macro::is_debug_level_enabled(4) {
            use $crate::r#macro::colors::*;
            let elapsed = $start.elapsed();
            let prefix = $crate::r#macro::build_line_prefix(4);
            println!("{}  {CYAN}{}{RESET} {} {DIM}[{}]{RESET} {MAGENTA}({}){RESET}", 
                prefix, $crate::r#macro::colors::BULLET, $name, $detail, $crate::r#macro::format_duration(elapsed));
        }
    };
}

/// Level 5: Detail/algorithm info
#[macro_export]
macro_rules! log_detail {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        $crate::debug!(5, $fmt $(, $($arg)*)?);
    };
}

/// Level 1: Warning message
#[macro_export]
macro_rules! log_warn {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        if $crate::r#macro::is_debug_level_enabled(1) {
            use $crate::r#macro::colors::*;
            let msg = format!($fmt $(, $($arg)*)?);
            let prefix = $crate::r#macro::build_line_prefix(1);
            println!("{}  {BOLD_YELLOW}{}{RESET} {}", prefix, $crate::r#macro::colors::WARN, msg);
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
            let prefix = $crate::r#macro::build_line_prefix(1);
            println!("{}{BOLD_GREEN}{} {}{RESET}", prefix, $crate::r#macro::colors::CHECK, msg);
        }
    };
}

// =============================================================================
// Timer and Section Helpers
// =============================================================================

/// Start a timer (returns Instant). Used with log_stage_done!, log_substep_done!
#[macro_export]
macro_rules! timer_start {
    () => {
        std::time::Instant::now()
    };
}

/// Backwards compatibility alias
#[macro_export]
macro_rules! debug_line {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {
        $crate::debug!($level, $fmt $(, $($arg)*)?);
    };
    ($level:expr, $msg:expr) => {
        $crate::debug!($level, "{}", $msg);
    };
}
