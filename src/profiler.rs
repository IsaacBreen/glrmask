use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Holds all profiling data.
#[derive(Default)]
pub struct ProfilerData {
    hits: HashMap<String, u64>,
    timings: HashMap<String, Duration>,
    timing_stack: Vec<(String, Instant)>,
}

// Global, thread-safe profiler data instance.
static PROFILER: OnceLock<Mutex<ProfilerData>> = OnceLock::new();

fn profiler() -> &'static Mutex<ProfilerData> {
    PROFILER.get_or_init(|| Mutex::new(ProfilerData::default()))
}

/// Records a single hit for a named event.
/// This is also called automatically by the `time!` macro.
pub fn hit(name: &str) {
    let mut data = profiler().lock().unwrap();
    *data.hits.entry(name.to_string()).or_insert(0) += 1;
}

/// Resets all profiling data (hits and timings).
pub fn reset() {
    let mut data = profiler().lock().unwrap();
    data.hits.clear();
    data.timings.clear();
    data.timing_stack.clear();
}

/// Prints a summary of the collected profiling data to stdout.
/// The summary is sorted alphabetically by event name.
pub fn print_summary() {
    let data = profiler().lock().unwrap();
    println!("--- Profiler Summary ---");

    if data.hits.is_empty() && data.timings.is_empty() {
        println!("No data collected.");
        println!("--- End Profiler Summary ---");
        return;
    }

    if !data.hits.is_empty() {
        println!("\n[Hits]");
        let mut sorted_hits: Vec<_> = data.hits.iter().collect();
        sorted_hits.sort_by_key(|k| k.0);
        for (name, count) in sorted_hits {
            println!("  {:>10}x: {}", count, name);
        }
    }

    if !data.timings.is_empty() {
        println!("\n[Own Time]");
        let mut sorted_timings: Vec<_> = data.timings.iter().collect();
        sorted_timings.sort_by_key(|k| k.0);
        for (name, duration) in sorted_timings {
            println!("  {:>12.3}ms: {}", duration.as_secs_f64() * 1000.0, name);
        }
    }

    println!("\n--- End Profiler Summary ---");
}

/// Returns a clone of the hits data.
pub fn get_hits() -> HashMap<String, u64> {
    profiler().lock().unwrap().hits.clone()
}

/// Returns a clone of the timings data.
pub fn get_timings() -> HashMap<String, Duration> {
    profiler().lock().unwrap().timings.clone()
}


// Internal functions for timing blocks
fn time_block_start(name: String) {
    let mut data = profiler().lock().unwrap();
    let now = Instant::now();

    // Pause the parent timer
    if let Some((parent_name, parent_start_time)) = data.timing_stack.last() {
        let duration = now.duration_since(*parent_start_time);
        *data.timings.entry(parent_name.clone()).or_default() += duration;
    }

    // Push the new timer onto the stack
    data.timing_stack.push((name.clone(), now));

    // Also record a hit
    *data.hits.entry(name).or_insert(0) += 1;
}

fn time_block_end() {
    let mut data = profiler().lock().unwrap();
    let now = Instant::now();

    // End the current timer
    if let Some((name, start_time)) = data.timing_stack.pop() {
        let duration = now.duration_since(start_time);
        *data.timings.entry(name).or_default() += duration;
    }

    // Resume the parent timer by updating its start time
    if let Some(parent) = data.timing_stack.last_mut() {
        parent.1 = now;
    }
}

/// A guard object for timing a block of code using RAII.
/// Its creation marks the start of the block, and its destruction (when it goes out of scope)
/// marks the end. This is intended for use by the `time!` macro.
#[must_use]
pub struct TimedBlockGuard;

impl TimedBlockGuard {
    /// Creates a new guard, starting a timer for the given name.
    pub fn new(name: String) -> Self {
        time_block_start(name);
        TimedBlockGuard
    }
}

impl Drop for TimedBlockGuard {
    fn drop(&mut self) {
        time_block_end();
    }
}

/// Macro to time a block of code or an expression.
///
/// It measures the "own time" of the block, excluding time spent in nested
/// `time!` blocks. Timing a block also increments its "hit" count.
///
/// # Examples
///
/// ```ignore
/// use sep1::profiler::{time, print_summary, reset};
///
/// fn my_function() {
///     std::thread::sleep(std::time::Duration::from_millis(50));
/// }
///
/// time!("total", {
///     let result = time!("calculation", 1 + 1);
///     time!("my_function_call", my_function());
/// });
///
/// print_summary();
/// reset();
/// ```
#[macro_export]
macro_rules! time {
    ($name:expr, $block:expr) => {{
        let _guard = $crate::profiler::TimedBlockGuard::new(String::from($name));
        $block
    }};
}

/// Macro to record a single hit for a named event.
///
/// # Example
///
/// ```ignore
/// use sep1::profiler::{hit, print_summary};
///
/// if true {
///     hit!("some_condition was true");
/// }
///
/// print_summary();
/// ```
#[macro_export]
macro_rules! hit {
    ($name:expr) => {
        $crate::profiler::hit($name)
    };
}
