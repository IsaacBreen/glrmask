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

#[derive(Default, Clone, Copy)]
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

#[derive(Default, Clone, Copy)]
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

#[derive(Default, Clone, Copy)]
pub struct MaskProfile {
	pub total_ns: u64,
	pub cache_hit: u64,
	pub single_path_direct: u64,
	pub seed_decompose_ns: u64,
	pub queue_pop_ns: u64,
	pub loop_decompose_ns: u64,
	pub loop_decompose_callback_ns: u64,
	pub transition_lookup_ns: u64,
	pub transition_apply_ns: u64,
	pub transition_apply_intersect_ns: u64,
	pub transition_apply_gss_ns: u64,
	pub token_accumulation_ns: u64,
	pub enqueue_merge_ns: u64,
	pub queue_lookup_ns: u64,
	pub queue_merge_ns: u64,
	pub queue_insert_ns: u64,
	pub queue_fuse_ns: u64,
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
	pub delta_added_word_group_hits: u64,
	pub delta_added_word_group_entries: u64,
	pub delta_removed_word_group_hits: u64,
	pub delta_removed_word_group_entries: u64,
	pub delta_added_byte_group_hits: u64,
	pub delta_added_byte_group_entries: u64,
	pub delta_removed_byte_group_hits: u64,
	pub delta_removed_byte_group_entries: u64,
	pub delta_added_token_iterations: u64,
	pub delta_added_token_entries: u64,
	pub delta_removed_token_iterations: u64,
	pub delta_removed_token_entries: u64,
	pub finalize_equal_dense_copy_seed: u64,
	pub finalize_delta_replay: u64,
	pub finalize_scratch_rebuild: u64,
	pub dense_words_visited: u64,
	pub dense_complement_path_used: u64,
	pub dense_normal_full_word_hits: u64,
	pub dense_normal_group_complement_hits: u64,
	pub dense_complement_full_word_hits: u64,
	pub dense_complement_full_byte_groups: u64,
	pub dense_complement_full_nibble_groups: u64,
	pub dense_complement_remaining_bits: u64,
	pub dense_normal_token_iterations: u64,
	pub dense_complement_token_iterations: u64,
	pub dense_normal_sparse_entries: u64,
	pub dense_normal_group_complement_sparse_entries: u64,
	pub dense_complement_sparse_entries: u64,
	pub dense_complement_heavy_dense_clears: u64,
	pub dense_complement_max_sparse_span: u64,
	pub dense_group_or_sparse_entries: u64,
	pub dense_group_andnot_sparse_entries: u64,
	pub enqueue_calls: u64,
	pub merge_hits: u64,
	pub insert_without_merge_count: u64,
	pub fuse_calls: u64,
	pub fuse_changed_depth: u64,
	pub stale_schedule_skips: u64,
	pub popped_items: u64,
	pub seed_decompose_callbacks: u64,
	pub loop_decompose_callbacks: u64,
	pub parser_dwa_transitions_enqueued: u64,
	pub other_ns: u64,
}

impl MaskProfile {
	pub(super) fn from_parts(
		inner: MaskInnerProfileStats,
		queue: MaskQueueDebugStats,
		cache_hit: bool,
		single_path_direct: bool,
	) -> Self {
		let loop_decompose_ns = inner
			.loop_decompose_total_ns
			.saturating_sub(inner.loop_decompose_callback_ns);
		let enqueue_merge_ns = queue.enqueue_total_ns.saturating_sub(queue.fuse_total_ns);
		let accounted_ns = inner.seed_decompose_ns
			+ inner.queue_pop_ns
			+ loop_decompose_ns
			+ inner.transition_lookup_ns
			+ inner.transition_apply_ns
			+ inner.token_accumulation_ns
			+ enqueue_merge_ns
			+ queue.fuse_total_ns
			+ inner.finalize_ns;
		let other_ns = inner.total_ns.saturating_sub(accounted_ns);

		Self {
			total_ns: inner.total_ns,
			cache_hit: u64::from(cache_hit),
			single_path_direct: u64::from(single_path_direct),
			seed_decompose_ns: inner.seed_decompose_ns,
			queue_pop_ns: inner.queue_pop_ns,
			loop_decompose_ns,
			loop_decompose_callback_ns: inner.loop_decompose_callback_ns,
			transition_lookup_ns: inner.transition_lookup_ns,
			transition_apply_ns: inner.transition_apply_ns,
			transition_apply_intersect_ns: inner.transition_apply_intersect_ns,
			transition_apply_gss_ns: inner.transition_apply_gss_ns,
			token_accumulation_ns: inner.token_accumulation_ns,
			enqueue_merge_ns,
			queue_lookup_ns: queue.lookup_total_ns,
			queue_merge_ns: queue.merge_total_ns,
			queue_insert_ns: queue.insert_total_ns,
			queue_fuse_ns: queue.fuse_total_ns,
			finalize_ns: inner.finalize_ns,
			finalize_zero_ns: inner.finalize_zero_ns,
			finalize_dense_to_buf_ns: inner.finalize_dense_to_buf_ns,
			finalize_eos_ns: inner.finalize_eos_ns,
			finalize_cache_ns: inner.finalize_cache_ns,
			delta_prev_available: inner.delta_prev_available,
			delta_added_bits: inner.delta_added_bits,
			delta_removed_bits: inner.delta_removed_bits,
			delta_unchanged_words: inner.delta_unchanged_words,
			delta_unchanged_bits: inner.delta_unchanged_bits,
			delta_added_cost: inner.delta_added_cost,
			delta_removed_cost: inner.delta_removed_cost,
			delta_copy_cost_words: inner.delta_copy_cost_words,
			delta_scratch_estimated_cost: inner.delta_scratch_estimated_cost,
			delta_estimated_cost: inner.delta_estimated_cost,
			delta_estimated_savings: inner.delta_estimated_savings,
			delta_used_seed: inner.delta_used_seed,
			delta_added_word_group_hits: inner.delta_replay.added_word_group_hits,
			delta_added_word_group_entries: inner.delta_replay.added_word_group_entries,
			delta_removed_word_group_hits: inner.delta_replay.removed_word_group_hits,
			delta_removed_word_group_entries: inner.delta_replay.removed_word_group_entries,
			delta_added_byte_group_hits: inner.delta_replay.added_byte_group_hits,
			delta_added_byte_group_entries: inner.delta_replay.added_byte_group_entries,
			delta_removed_byte_group_hits: inner.delta_replay.removed_byte_group_hits,
			delta_removed_byte_group_entries: inner.delta_replay.removed_byte_group_entries,
			delta_added_token_iterations: inner.delta_replay.added_token_iterations,
			delta_added_token_entries: inner.delta_replay.added_token_entries,
			delta_removed_token_iterations: inner.delta_replay.removed_token_iterations,
			delta_removed_token_entries: inner.delta_replay.removed_token_entries,
			finalize_equal_dense_copy_seed: inner.finalize_equal_dense_copy_seed,
			finalize_delta_replay: inner.finalize_delta_replay,
			finalize_scratch_rebuild: inner.finalize_scratch_rebuild,
			dense_words_visited: inner.dense_to_buf.dense_words_visited,
			dense_complement_path_used: u64::from(inner.dense_to_buf.complement_path_used),
			dense_normal_full_word_hits: inner.dense_to_buf.normal_full_word_hits,
			dense_normal_group_complement_hits: inner.dense_to_buf.normal_group_complement_hits,
			dense_complement_full_word_hits: inner.dense_to_buf.complement_full_word_hits,
			dense_complement_full_byte_groups: inner.dense_to_buf.complement_full_byte_groups,
			dense_complement_full_nibble_groups: inner.dense_to_buf.complement_full_nibble_groups,
			dense_complement_remaining_bits: inner.dense_to_buf.complement_remaining_bits,
			dense_normal_token_iterations: inner.dense_to_buf.normal_token_iterations,
			dense_complement_token_iterations: inner.dense_to_buf.complement_token_iterations,
			dense_normal_sparse_entries: inner.dense_to_buf.normal_sparse_entries,
			dense_normal_group_complement_sparse_entries: inner.dense_to_buf.normal_group_complement_sparse_entries,
			dense_complement_sparse_entries: inner.dense_to_buf.complement_sparse_entries,
			dense_complement_heavy_dense_clears: inner.dense_to_buf.complement_heavy_dense_clears,
			dense_complement_max_sparse_span: inner.dense_to_buf.complement_max_sparse_span,
			dense_group_or_sparse_entries: inner.dense_to_buf.group_or_sparse_entries,
			dense_group_andnot_sparse_entries: inner.dense_to_buf.group_andnot_sparse_entries,
			enqueue_calls: queue.enqueue_calls,
			merge_hits: queue.merge_hit_count,
			insert_without_merge_count: queue.insert_without_merge_count,
			fuse_calls: queue.fuse_calls,
			fuse_changed_depth: queue.fuse_changed_depth,
			stale_schedule_skips: queue.stale_schedule_skips,
			popped_items: queue.popped_items,
			seed_decompose_callbacks: queue.seed_decompose_callbacks,
			loop_decompose_callbacks: queue.loop_decompose_callbacks,
			parser_dwa_transitions_enqueued: queue.parser_dwa_transitions_enqueued,
			other_ns,
		}
	}
}
