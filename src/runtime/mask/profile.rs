use std::fs::OpenOptions;
use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;

use crate::runtime::constraint::{DeltaReplayProfileStats, DenseToBufProfileStats};

pub(super) fn bool_env(name: &str) -> bool {
	std::env::var(name)
		.map(|value| {
			let normalized = value.trim().to_ascii_lowercase();
			matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
		})
		.unwrap_or(false)
}

pub(super) fn mask_queue_debug_enabled() -> bool {
	static ENABLED: OnceLock<bool> = OnceLock::new();
	*ENABLED.get_or_init(|| bool_env("GLRMASK_DEBUG_MASK_QUEUE"))
}

pub(super) fn mask_inner_profile_enabled() -> bool {
	static ENABLED: OnceLock<bool> = OnceLock::new();
	*ENABLED.get_or_init(|| bool_env("GLRMASK_PROFILE_MASK_INNER"))
}

pub(super) fn mask_delta_profile_enabled() -> bool {
	static ENABLED: OnceLock<bool> = OnceLock::new();
	*ENABLED.get_or_init(|| bool_env("GLRMASK_PROFILE_MASK_DELTA"))
}

pub(super) fn mask_queue_merge_profile_enabled() -> bool {
	static ENABLED: OnceLock<bool> = OnceLock::new();
	*ENABLED.get_or_init(|| bool_env("GLRMASK_PROFILE_MASK_QUEUE_MERGE"))
}

pub(super) fn mask_acc_merge_profile_enabled() -> bool {
	static ENABLED: OnceLock<bool> = OnceLock::new();
	*ENABLED.get_or_init(|| bool_env("GLRMASK_PROFILE_MASK_ACC_MERGE"))
}

pub(super) fn mask_fast_conversion_profile_enabled() -> bool {
	static ENABLED: OnceLock<bool> = OnceLock::new();
	*ENABLED.get_or_init(|| bool_env("GLRMASK_PROFILE_MASK_FAST_CONVERSION"))
}

pub(super) fn mask_single_path_to_stacks_fallback_disabled() -> bool {
	static DISABLED: OnceLock<bool> = OnceLock::new();
	*DISABLED.get_or_init(|| bool_env("GLRMASK_DISABLE_MASK_SINGLE_PATH_TO_STACKS_FALLBACK"))
}

fn emit_line_with_optional_file(line: &str, file_env_var: &str) {
	println!("{line}");

	let Ok(path) = std::env::var(file_env_var) else {
		return;
	};

	let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
		return;
	};
	let _ = writeln!(file, "{line}");
}

pub(super) fn emit_mask_queue_debug_line(line: &str) {
	emit_line_with_optional_file(line, "GLRMASK_DEBUG_MASK_QUEUE_FILE");
}

pub(super) fn emit_mask_inner_profile_line(line: &str) {
	emit_line_with_optional_file(line, "GLRMASK_PROFILE_MASK_INNER_FILE");
}

pub(super) fn emit_mask_queue_merge_profile_line(line: &str) {
	emit_line_with_optional_file(line, "GLRMASK_PROFILE_MASK_QUEUE_MERGE_FILE");
}

pub(super) fn emit_mask_acc_merge_profile_line(line: &str) {
	emit_line_with_optional_file(line, "GLRMASK_PROFILE_MASK_ACC_MERGE_FILE");
}

pub(super) fn emit_mask_fast_conversion_profile_line(line: &str) {
	let Ok(path) = std::env::var("GLRMASK_PROFILE_MASK_FAST_CONVERSION_FILE") else {
		return;
	};

	let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
		return;
	};
	let _ = writeln!(file, "{line}");
}

pub(super) fn elapsed_ns(start: Instant) -> u64 {
	start.elapsed().as_nanos() as u64
}

#[derive(Default)]
pub(super) struct MaskQueueDebugStats {
	pub enqueue_calls: u64,
	pub merge_hit_count: u64,
	pub insert_without_merge_count: u64,
	pub fuse_calls: u64,
	pub fuse_changed_depth: u64,
	pub stale_schedule_skips: u64,
	pub popped_items: u64,
	pub seed_decompose_callbacks: u64,
	pub loop_decompose_callbacks: u64,
	pub parser_dwa_transitions_enqueued: u64,
	pub enqueue_total_ns: u64,
	pub pop_total_ns: u64,
	pub fuse_total_ns: u64,
	pub lookup_total_ns: u64,
	pub merge_total_ns: u64,
	pub insert_total_ns: u64,
}

#[derive(Default)]
pub(super) struct MaskInnerProfileStats {
	pub total_ns: u64,
	pub seed_decompose_ns: u64,
	pub queue_pop_ns: u64,
	pub loop_decompose_total_ns: u64,
	pub loop_decompose_callback_ns: u64,
	pub transition_lookup_ns: u64,
	pub transition_apply_ns: u64,
	pub transition_apply_intersect_ns: u64,
	pub transition_apply_gss_ns: u64,
	pub token_accumulation_ns: u64,
	pub finalize_ns: u64,
	pub finalize_zero_ns: u64,
	pub finalize_dense_to_buf_ns: u64,
	pub finalize_eos_ns: u64,
	pub finalize_cache_ns: u64,
	pub delta_prev_available: u64,
	pub delta_added_bits: u64,
	pub delta_removed_bits: u64,
	pub delta_unchanged_words: u64,
	pub delta_unchanged_bits: u64,
	pub delta_added_cost: u64,
	pub delta_removed_cost: u64,
	pub delta_copy_cost_words: u64,
	pub delta_scratch_estimated_cost: u64,
	pub delta_estimated_cost: u64,
	pub delta_estimated_savings: u64,
	pub delta_used_seed: u64,
	pub delta_replay: DeltaReplayProfileStats,
	pub finalize_equal_dense_copy_seed: u64,
	pub finalize_delta_replay: u64,
	pub finalize_scratch_rebuild: u64,
	pub dense_to_buf: DenseToBufProfileStats,
}
