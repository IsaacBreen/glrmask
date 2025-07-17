use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The measured overhead of a single timing block, used for correction.
const PROFILER_CORRECTION: Duration = Duration::from_nanos(1300);

/// Set this to `false` to completely disable profiling at runtime.
pub const PROFILING_ENABLED: bool = false;

/// A node in the profiler's call tree.
#[derive(Default, Clone)]
pub struct ProfileNode {
    /// Number of times this block was entered.
    pub hits: u64,
    /// Time spent in this block, excluding children.
    pub own_time: Duration,
    /// Time spent in this block, including children.
    pub total_time: Duration,
    /// Children nodes in the call tree.
    pub children: HashMap<String, ProfileNode>,
}

/// Holds all profiling data.
#[derive(Default)]
pub struct ProfilerData {
    /// The root of the call tree. Its children are the top-level timed blocks.
    call_tree: ProfileNode,
    /// A stack to keep track of the current path in the call tree.
    /// Each element is (name, start_time_for_total, start_time_for_own, sub_calls_count)
    timing_stack: Vec<(String, Instant, Instant, u64)>,
    /// Separate hits counter for the `hit!` macro.
    hits: HashMap<String, u64>,
}

// Global, thread-safe profiler data instance.
static PROFILER: OnceLock<Mutex<ProfilerData>> = OnceLock::new();

fn profiler() -> &'static Mutex<ProfilerData> {
    PROFILER.get_or_init(|| Mutex::new(ProfilerData::default()))
}

/// Records a single hit for a named event (for use with `hit!` macro).
pub fn hit(name: &str) {
    if PROFILING_ENABLED {
        let mut data = profiler().lock().unwrap();
        *data.hits.entry(name.to_string()).or_insert(0) += 1;
    }
}

/// Resets all profiling data (hits and timings).
pub fn reset() {
    let mut data = profiler().lock().unwrap();
    data.call_tree = ProfileNode::default();
    data.timing_stack.clear();
    data.hits.clear();
}

/// Formats a `Duration` into a human-readable string with appropriate units (s, ms, µs, ns).
fn format_duration(duration: Duration) -> String {
    let total_nanos = duration.as_nanos();
    if total_nanos >= 1_000_000_000 {
        format!("{:.3}s", duration.as_secs_f64())
    } else if total_nanos >= 1_000_000 {
        format!("{:.3}ms", duration.as_micros() as f64 / 1000.0)
    } else if total_nanos >= 1_000 {
        format!("{:.3}µs", duration.as_nanos() as f64 / 1000.0)
    } else {
        format!("{}ns", total_nanos)
    }
}

fn print_node_recursive(
    node: &ProfileNode,
    name: &str,
    indent_level: usize,
    parent_total_time: Duration,
) {
    let indent = "  ".repeat(indent_level);
    let name_with_indent = format!("{}{}", indent, name);

    let (total_per_hit, own_per_hit) = if node.hits > 0 {
        (
            node.total_time.mul_f64(1.0 / node.hits as f64),
            node.own_time.mul_f64(1.0 / node.hits as f64),
        )
    } else {
        (Duration::from_secs(0), Duration::from_secs(0))
    };

    let percentage_of_parent = if !parent_total_time.is_zero() {
        (node.total_time.as_secs_f64() / parent_total_time.as_secs_f64()) * 100.0
    } else {
        0.0
    };

    let percentage_own = if !node.total_time.is_zero() {
        (node.own_time.as_secs_f64() / node.total_time.as_secs_f64()) * 100.0
    } else {
        0.0
    };

    let total_str = format_duration(node.total_time);
    let own_str = format_duration(node.own_time);
    let total_per_hit_str = format_duration(total_per_hit);
    let own_per_hit_str = format_duration(own_per_hit);
    let percentage_of_parent_str = format!("{:.1}%", percentage_of_parent);
    let percentage_own_str = format!("{:.1}%", percentage_own);

    println!(
        "{:>10} {:>15} {:>15} {:>15} {:>15} {:>10} {:>15}  {}",
        node.hits,
        total_str,
        total_per_hit_str,
        own_str,
        own_per_hit_str,
        percentage_own_str,
        percentage_of_parent_str,
        name_with_indent
    );

    let mut sorted_children: Vec<_> = node.children.iter().collect();
    sorted_children.sort_by_key(|(name, _)| *name);

    for (child_name, child_node) in sorted_children {
        print_node_recursive(child_node, child_name, indent_level + 1, node.total_time);
    }
}

/// Prints a summary of the collected profiling data to stdout.
pub fn print_summary() {
    let data = profiler().lock().unwrap();
    println!("--- Profiler Summary ---");

    let no_timing_data = data.call_tree.children.is_empty();
    let no_hit_data = data.hits.is_empty();

    if no_timing_data && no_hit_data {
        println!("No data collected.");
        println!("--- End Profiler Summary ---");
        return;
    }

    if !no_timing_data {
        println!("\n[Hierarchical Timings]");
        println!(
            "{:>10} {:>15} {:>15} {:>15} {:>15} {:>10} {:>15}  {}",
            "Hits",
            "Total Time",
            "Total/Hit",
            "Own Time",
            "Own/Hit",
            "% Own",
            "% of Parent",
            "Name"
        );

        let root_total_time: Duration = data
            .call_tree
            .children
            .values()
            .map(|node| node.total_time)
            .sum();

        let mut sorted_children: Vec<_> = data.call_tree.children.iter().collect();
        sorted_children.sort_by_key(|(name, _)| *name);

        for (name, node) in sorted_children {
            print_node_recursive(node, name, 0, root_total_time);
        }
    }

    if !no_hit_data {
        println!("\n[Hits]");
        let mut sorted_hits: Vec<_> = data.hits.iter().collect();
        sorted_hits.sort_by_key(|k| k.0);
        for (name, count) in sorted_hits {
            println!("  {:>10}x: {}", count, name);
        }
    }

    println!("\n--- End Profiler Summary ---");
}

fn flatten_tree_recursive(
    nodes: &HashMap<String, ProfileNode>,
    flat_map: &mut HashMap<String, ProfileNode>,
) {
    for (name, node) in nodes {
        let entry = flat_map.entry(name.clone()).or_default();
        entry.hits += node.hits;
        entry.own_time += node.own_time;
        entry.total_time += node.total_time;

        if !node.children.is_empty() {
            flatten_tree_recursive(&node.children, flat_map);
        }
    }
}

/// Prints a summary of the collected profiling data as a flat list, merging all calls to the same function.
pub fn print_summary_flat() {
    let data = profiler().lock().unwrap();
    println!("--- Profiler Summary (Flat) ---");

    let no_timing_data = data.call_tree.children.is_empty();
    let no_hit_data = data.hits.is_empty();

    if no_timing_data && no_hit_data {
        println!("No data collected.");
        println!("--- End Profiler Summary (Flat) ---");
        return;
    }

    if !no_timing_data {
        let mut flat_map: HashMap<String, ProfileNode> = HashMap::new();
        flatten_tree_recursive(&data.call_tree.children, &mut flat_map);

        println!("\n[Flat Timings]");
        println!(
            "{:>10} {:>15} {:>15} {:>15} {:>15} {:>10}  {}",
            "Hits",
            "Total Time",
            "Total/Hit",
            "Own Time",
            "Own/Hit",
            "% Own",
            "Name"
        );

        let mut sorted_list: Vec<_> = flat_map.iter().collect();
        sorted_list.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        for (name, node) in sorted_list {
            let (total_per_hit, own_per_hit) = if node.hits > 0 {
                (
                    node.total_time.mul_f64(1.0 / node.hits as f64),
                    node.own_time.mul_f64(1.0 / node.hits as f64),
                )
            } else {
                (Duration::from_secs(0), Duration::from_secs(0))
            };

            let percentage_own = if !node.total_time.is_zero() {
                (node.own_time.as_secs_f64() / node.total_time.as_secs_f64()) * 100.0
            } else {
                0.0
            };

            let total_str = format_duration(node.total_time);
            let own_str = format_duration(node.own_time);
            let total_per_hit_str = format_duration(total_per_hit);
            let own_per_hit_str = format_duration(own_per_hit);
            let percentage_own_str = format!("{:.1}%", percentage_own);

            println!(
                "{:>10} {:>15} {:>15} {:>15} {:>15} {:>10}  {}",
                node.hits, total_str, total_per_hit_str, own_str, own_per_hit_str,
                percentage_own_str, name
            );
        }
    }

    if !no_hit_data {
        println!("\n[Hits]");
        let mut sorted_hits: Vec<_> = data.hits.iter().collect();
        sorted_hits.sort_by_key(|k| k.0);
        for (name, count) in sorted_hits {
            println!("  {:>10}x: {}", count, name);
        }
    }

    println!("\n--- End Profiler Summary (Flat) ---");
}

/// Returns a clone of the hits data from `hit!` macro.
pub fn get_hits() -> HashMap<String, u64> {
    profiler().lock().unwrap().hits.clone()
}

/// Returns a clone of the call tree.
pub fn get_call_tree() -> ProfileNode {
    profiler().lock().unwrap().call_tree.clone()
}

// Internal functions for timing blocks
fn time_block_start(name: String) {
    let mut data = profiler().lock().unwrap();
    let now = Instant::now();

    // Increment sub-call counter for the parent.
    if let Some(parent) = data.timing_stack.last_mut() {
        parent.3 += 1;
    }

    // Get parent's own time start if it exists. This immutable borrow ends immediately.
    let parent_own_time_start_opt = data.timing_stack.last().map(|(_, _, t, _)| *t);

    // Collect the path to the parent node to avoid conflicting borrows.
    let path: Vec<String> = data.timing_stack.iter().map(|(s, _, _, _)| s.clone()).collect();

    // Now, get a mutable reference to the parent node and traverse.
    let mut current_node = &mut data.call_tree;
    for node_name in &path {
        current_node = current_node.children.get_mut(node_name).unwrap();
    }

    // If there is a parent, pause its own-time clock.
    if let Some(parent_own_time_start) = parent_own_time_start_opt {
        let own_time_lapsed = now.duration_since(parent_own_time_start);
        current_node.own_time += own_time_lapsed;
    }

    // `current_node` is now the parent. Get or insert the new node and increment its hit count.
    let new_node = current_node.children.entry(name.clone()).or_default();
    new_node.hits += 1;

    // Push the new timer onto the stack.
    data.timing_stack.push((name, now, now, 0));
}

fn time_block_end() {
    let mut data = profiler().lock().unwrap();
    let now = Instant::now();

    // Pop the current timer from the stack.
    if let Some((name, total_start_time, own_start_time, sub_calls)) = data.timing_stack.pop() {
        let total_duration = now.duration_since(total_start_time);
        let own_duration = now.duration_since(own_start_time);

        // Total time correction: 1 for this block, +1 for each direct and indirect sub-call.
        let total_correction = PROFILER_CORRECTION.saturating_mul((sub_calls + 1) as u32);

        let corrected_total_duration = total_duration.saturating_sub(total_correction);
        // Own time is only corrected for its own measurement overhead.
        let corrected_own_duration = own_duration.saturating_sub(PROFILER_CORRECTION);

        // The `timing_stack` has been popped, so it now represents the parent path.
        // Collect the path to avoid conflicting borrows.
        let parent_path: Vec<String> =
            data.timing_stack.iter().map(|(s, _, _, _)| s.clone()).collect();

        // Get a mutable reference to the parent of the node that just ended.
        let mut parent_node = &mut data.call_tree;
        for node_name in &parent_path {
            parent_node = parent_node.children.get_mut(node_name).unwrap();
        }

        // Get the node that ended and update its timings.
        let ended_node = parent_node.children.get_mut(&name).unwrap();
        ended_node.total_time += corrected_total_duration;
        ended_node.own_time += corrected_own_duration;
    }

    // Resume the new parent timer by updating its own-time start time.
    if let Some(parent) = data.timing_stack.last_mut() {
        parent.2 = now; // parent.2 is own_time_start
    }
}

/// A guard object for timing a block of code using RAII.
/// Its creation marks the start of the block, and its destruction (when it goes out of scope)
/// marks the end. This is intended for use by the `time!` macro.
/// When profiling is disabled, this is a no-op.
#[must_use]
pub struct TimedBlockGuard {
    enabled: bool,
}

impl TimedBlockGuard {
    /// Creates a new guard, starting a timer for the given name.
    /// If profiling is disabled, returns a no-op guard.
    pub fn new(name: String) -> Self {
        if PROFILING_ENABLED {
            time_block_start(name);
            TimedBlockGuard { enabled: true }
        } else {
            TimedBlockGuard { enabled: false }
        }
    }
}

impl Drop for TimedBlockGuard {
    fn drop(&mut self) {
        if self.enabled {
            time_block_end();
        }
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
        let __profiler_name = ($name).into();
        let _guard = $crate::profiler::TimedBlockGuard::new(__profiler_name);
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
