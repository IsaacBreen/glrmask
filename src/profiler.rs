use ordered_hash_map::OrderedHashMap;
use std::cell::RefCell;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// --- Configuration ---
// Set to false to disable all profiling macros at compile time,
// resulting in zero overhead for release builds.
pub const PROFILING_ENABLED: bool = true;
// --- End Configuration ---

/// Holds statistics for a timed block.
#[derive(Debug, Clone, Copy)]
pub struct TimingStats {
    pub total_time: Duration,
    pub count: u64,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl TimingStats {
    /// Creates new stats from the first measurement.
    fn new(duration: Duration) -> Self {
        Self {
            total_time: duration,
            count: 1,
            min_time: duration,
            max_time: duration,
        }
    }

    /// Updates stats with a new measurement.
    fn update(&mut self, duration: Duration) {
        self.total_time += duration;
        self.count += 1;
        if duration < self.min_time {
            self.min_time = duration;
        }
        if duration > self.max_time {
            self.max_time = duration;
        }
    }
}

/// The main struct holding all profiling data.
#[derive(Debug, Clone)]
pub struct ProfilerData {
    pub hit_counts: OrderedHashMap<String, u64>,
    pub timings: OrderedHashMap<String, TimingStats>,
}

impl ProfilerData {
    fn new() -> Self {
        Self {
            hit_counts: OrderedHashMap::new(),
            timings: OrderedHashMap::new(),
        }
    }
}

/// The global, thread-safe profiler data instance.
static PROFILER_DATA: OnceLock<Mutex<ProfilerData>> = OnceLock::new();

/// Returns a reference to the global profiler data, initializing it if necessary.
fn profiler_data() -> &'static Mutex<ProfilerData> {
    PROFILER_DATA.get_or_init(|| Mutex::new(ProfilerData::new()))
}

/// Resets all profiler data (hit counts and timings).
pub fn reset_profiler_data() {
    if PROFILING_ENABLED {
        let mut data = profiler_data().lock().unwrap();
        *data = ProfilerData::new();
    }
}

/// Prints a formatted summary of all collected profiling data to the console.
///
/// The summary is sorted to show the most significant results first (highest hit
/// counts, longest total times).
pub fn print_profiler_summary() {
    if !PROFILING_ENABLED {
        println!("Profiling is disabled.");
        return;
    }

    let data = profiler_data().lock().unwrap();

    println!("\n╔═════════════════════════════════╗");
    println!("║        Profiler Summary         ║");
    println!("╚═════════════════════════════════╝");

    if data.hit_counts.is_empty() && data.timings.is_empty() {
        println!("\nNo profiling data collected.");
    }

    if !data.hit_counts.is_empty() {
        println!("\n--- Hit Counts ---");
        let mut sorted_hits: Vec<_> = data.hit_counts.iter().collect();
        sorted_hits.sort_by(|a, b| b.1.cmp(a.1));

        for (id, count) in sorted_hits {
            println!("{:>10}x: {}", count, id);
        }
    }

    if !data.timings.is_empty() {
        println!("\n--- Timings ---");
        let mut sorted_timings: Vec<_> = data.timings.iter().collect();
        sorted_timings.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        for (id, stats) in sorted_timings {
            let avg_time = if stats.count > 0 {
                stats.total_time.div_f64(stats.count as f64)
            } else {
                Duration::from_secs(0)
            };
            println!(
                "'{id}':\n  total: {:?}, calls: {}, avg: {:?}, min: {:?}, max: {:?}",
                stats.total_time, stats.count, avg_time, stats.min_time, stats.max_time
            );
        }
    }

    println!("\n--- End Profiler Summary ---\n");
}

/// Provides safe, read-only access to the profiler data for custom analysis.
pub fn with_profiler_data<F, R>(f: F) -> R
where
    F: FnOnce(&ProfilerData) -> R,
{
    let data = profiler_data().lock().unwrap();
    f(&*data)
}

/// Provides safe, mutable access to the profiler data for custom manipulation.
pub fn with_profiler_data_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut ProfilerData) -> R,
{
    let mut data = profiler_data().lock().unwrap();
    f(&mut *data)
}

/// Macro to record a "hit" for a given string identifier.
///
/// This is useful for counting how many times a piece of code is executed.
/// If `PROFILING_ENABLED` is false, this macro compiles to nothing.
#[macro_export]
macro_rules! hit {
    ($id:expr) => {
        if $crate::profiler::PROFILING_ENABLED {
            let mut data = $crate::profiler::profiler_data().lock().unwrap();
            *data.hit_counts.entry($id.to_string()).or_insert(0) += 1;
        }
    };
}

/// Macro to time the execution of a block of code or an expression.
///
/// It takes an identifier string and the expression to time. It returns the
/// result of the expression, so it can be used with `let` bindings.
/// If `PROFILING_ENABLED` is false, this macro compiles to just the expression
/// itself, with no timing overhead.
#[macro_export]
macro_rules! time_block {
    ($id:expr, $block:expr) => {{
        if $crate::profiler::PROFILING_ENABLED {
            // Before starting, capture the time accumulated by any sibling blocks
            // that have already run, and reset the accumulator for our own children.
            let accumulated_from_siblings = $crate::profiler::NESTED_BLOCK_TIME.with(|cell| {
                cell.replace(::std::time::Duration::ZERO)
            });

            let start = std::time::Instant::now();
            let result = $block;
            let total_elapsed = start.elapsed();

            // After running, get the time accumulated by our direct children.
            let children_time = $crate::profiler::NESTED_BLOCK_TIME.with(|cell| {
                *cell.borrow()
            });

            // Calculate our own time by subtracting the time our children took.
            let own_time = total_elapsed.saturating_sub(children_time);

            // Update the global profiler data with our own time.
            let mut data = $crate::profiler::profiler_data().lock().unwrap();
            data.timings
                .entry($id.to_string())
                .and_modify(|stats| stats.update(own_time))
                .or_insert_with(|| $crate::profiler::TimingStats::new(own_time));

            // Now, update the thread-local accumulator for our parent.
            // We add our *total* elapsed time to the time that was already
            // accumulated by our prior siblings.
            $crate::profiler::NESTED_BLOCK_TIME.with(|cell| {
                *cell.borrow_mut() = accumulated_from_siblings + total_elapsed;
            });

            result
        } else {
            $block
        }
    }};
}

// This is public so the macro can access it, but it's not meant for direct use.
#[doc(hidden)]
thread_local! {
    /// Thread-local storage to track the total time spent in nested timed blocks.
    /// When a timed block ends, its total elapsed time is added to this value.
    /// The parent block then subtracts this from its own total time to get its "own time".
    pub static NESTED_BLOCK_TIME: RefCell<Duration> = RefCell::new(Duration::ZERO);
}
