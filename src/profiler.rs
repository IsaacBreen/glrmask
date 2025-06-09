use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

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
    /// Each element is (name, start_time_for_total, start_time_for_own)
    timing_stack: Vec<(String, Instant, Instant)>,
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
    let mut data = profiler().lock().unwrap();
    *data.hits.entry(name.to_string()).or_insert(0) += 1;
}

/// Resets all profiling data (hits and timings).
pub fn reset() {
    let mut data = profiler().lock().unwrap();
    data.call_tree = ProfileNode::default();
    data.timing_stack.clear();
    data.hits.clear();
}

fn print_node_recursive(
    node: &ProfileNode,
    name: &str,
    indent_level: usize,
    root_total_time: Duration,
    min_proportion_of_root: f64,
) {
    let indent = "  ".repeat(indent_level);
    let name_with_indent = format!("{}{}", indent, name);

    let total_ms = node.total_time.as_secs_f64() * 1000.0;
    let own_ms = node.own_time.as_secs_f64() * 1000.0;

    let proportion_of_root = if !root_total_time.is_zero() {
        node.total_time.as_secs_f64() / root_total_time.as_secs_f64()
    } else {
        0.0
    };
    let percentage_of_root = proportion_of_root * 100.0;

    println!(
        "{:<50} {:>10} {:>12.3}ms {:>12.3}ms {:>7.1}%",
        name_with_indent, node.hits, total_ms, own_ms, percentage_of_root
    );

    // Collapse children if this node's contribution is too small and it has children
    if proportion_of_root < min_proportion_of_root && !node.children.is_empty() {
        let collapsed_indent = "  ".repeat(indent_level + 1);
        println!("{}[... children collapsed ...]", collapsed_indent);
        return;
    }

    let mut sorted_children: Vec<_> = node.children.iter().collect();
    sorted_children.sort_by_key(|(name, _)| *name);

    for (child_name, child_node) in sorted_children {
        print_node_recursive(
            child_node,
            child_name,
            indent_level + 1,
            root_total_time,
            min_proportion_of_root,
        );
    }
}

/// Prints a summary of the collected profiling data to stdout.
///
/// Nodes whose total time is less than `min_proportion_of_root` of the total
/// profiled time will be collapsed.
pub fn print_summary(min_proportion_of_root: f64) {
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
            "{:<50} {:>10} {:>12} {:>12} {:>8}",
            "Name", "Hits", "Total Time", "Own Time", "% of Root"
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
            print_node_recursive(
                node,
                name,
                0,
                root_total_time,
                min_proportion_of_root,
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

    println!("\n--- End Profiler Summary ---");
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

    // Get parent's own time start if it exists. This immutable borrow ends immediately.
    let parent_own_time_start_opt = data.timing_stack.last().map(|(_, _, t)| *t);

    // Collect the path to the parent node to avoid conflicting borrows.
    let path: Vec<String> = data.timing_stack.iter().map(|(s, _, _)| s.clone()).collect();

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
    data.timing_stack.push((name, now, now));
}

fn time_block_end() {
    let mut data = profiler().lock().unwrap();
    let now = Instant::now();

    // Pop the current timer from the stack.
    if let Some((name, total_start_time, own_start_time)) = data.timing_stack.pop() {
        let total_duration = now.duration_since(total_start_time);
        let own_duration = now.duration_since(own_start_time);

        // The `timing_stack` has been popped, so it now represents the parent path.
        // Collect the path to avoid conflicting borrows.
        let parent_path: Vec<String> = data.timing_stack.iter().map(|(s, _, _)| s.clone()).collect();

        // Get a mutable reference to the parent of the node that just ended.
        let mut parent_node = &mut data.call_tree;
        for node_name in &parent_path {
            parent_node = parent_node.children.get_mut(node_name).unwrap();
        }

        // Get the node that ended and update its timings.
        let ended_node = parent_node.children.get_mut(&name).unwrap();
        ended_node.total_time += total_duration;
        ended_node.own_time += own_duration;
    }

    // Resume the new parent timer by updating its own-time start time.
    if let Some(parent) = data.timing_stack.last_mut() {
        parent.2 = now; // parent.2 is own_time_start
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
///     time!("my_function_call", my_function());
/// });
///
/// // Print full summary (collapse threshold is 0.0)
/// print_summary(0.0);
///
/// // Print summary, collapsing nodes that are less than 1% (0.01) of total time
/// print_summary(0.01);
///
/// reset();
/// ```
#[macro_export]
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
/// print_summary(0.0); // Show all nodes
/// ```
#[macro_export]
macro_rules! hit {
    ($name:expr) => {
        $crate::profiler::hit($name)
    };
}
