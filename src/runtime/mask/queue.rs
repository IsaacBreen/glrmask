use std::collections::BTreeMap;
use std::time::Instant;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::profile::{
	elapsed_ns,
	emit_mask_queue_merge_profile_line,
	mask_inner_profile_enabled,
	mask_queue_merge_profile_enabled,
	MaskQueueDebugStats,
};
use super::DenseMaskGSS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MaskQueueMode {
	Target,
	Depth,
	DepthMerge,
	DepthBatch4,
}

// A depth-first queue preserves the parser-DWA traversal order and avoids the
// potentially large cross-depth GSS unions performed by `Target`.  Keeping
// every same-target item separate, however, can replay exponentially many
// equivalent work items when a compact GSS represents a highly ambiguous
// parser state.  Batch four is deliberately conservative: ordinary one-, two-,
// and three-item buckets pay no merge cost, while a multiplying bucket is
// periodically collapsed back into one compact GSS.
const DEPTH_BATCH_MERGE_THRESHOLD: usize = 4;

pub(super) fn mask_queue_mode() -> MaskQueueMode {
	static MODE: std::sync::OnceLock<MaskQueueMode> = std::sync::OnceLock::new();
	*MODE.get_or_init(|| match std::env::var("GLRMASK_MASK_QUEUE_MODE") {
		Ok(value) if value.trim().eq_ignore_ascii_case("target") => MaskQueueMode::Target,
		Ok(value) if value.trim().eq_ignore_ascii_case("depth") => MaskQueueMode::Depth,
		Ok(value) if value.trim().eq_ignore_ascii_case("depth-merge") => MaskQueueMode::DepthMerge,
		_ => MaskQueueMode::DepthBatch4,
	})
}

pub(super) enum MaskQueueInner {
	Target {
		by_target: FxHashMap<u32, DenseMaskGSS>,
		ready_by_depth: BTreeMap<u32, SmallVec<[u32; 4]>>,
	},
	Depth {
		by_depth: BTreeMap<u32, FxHashMap<u32, SmallVec<[DenseMaskGSS; 2]>>>,
	},
}

pub(super) struct MaskQueue {
	mode: MaskQueueMode,
	inner: MaskQueueInner,
	debug: MaskQueueDebugStats,
}

impl Default for MaskQueue {
	fn default() -> Self {
		Self::new_with_mode(mask_queue_mode())
	}
}

impl MaskQueue {
	fn new_with_mode(mode: MaskQueueMode) -> Self {
		let inner = match mode {
			MaskQueueMode::Target => MaskQueueInner::Target {
				by_target: FxHashMap::default(),
				ready_by_depth: BTreeMap::new(),
			},
			MaskQueueMode::Depth
			| MaskQueueMode::DepthMerge
			| MaskQueueMode::DepthBatch4 => MaskQueueInner::Depth {
				by_depth: BTreeMap::new(),
			},
		};

		Self {
			mode,
			inner,
			debug: MaskQueueDebugStats::default(),
		}
	}

	pub(super) fn new() -> Self {
		Self::default()
	}

	pub(super) fn enqueue(&mut self, target: u32, gss: DenseMaskGSS) {
		if gss.is_empty() {
			return;
		}

		let inner_profile_enabled = mask_inner_profile_enabled();
		let merge_profile_enabled = mask_queue_merge_profile_enabled();
		let enqueue_start = if inner_profile_enabled { Some(Instant::now()) } else { None };
		self.debug.enqueue_calls += 1;

		match &mut self.inner {
			MaskQueueInner::Target {
				by_target,
				ready_by_depth,
			} => {
				let lookup_start = if inner_profile_enabled { Some(Instant::now()) } else { None };
				let existing = by_target.remove(&target);
				if let Some(start) = lookup_start {
					self.debug.lookup_total_ns += elapsed_ns(start);
				}

				let merged = match existing {
					Some(existing) => {
						self.debug.merge_hit_count += 1;
						let existing_depth = existing.max_depth();
						let incoming_depth = gss.max_depth();
						let merge_start = if inner_profile_enabled || merge_profile_enabled {
							Some(Instant::now())
						} else {
							None
						};
						let merged = existing.merge(&gss);
						let merge_ns = merge_start.map(elapsed_ns).unwrap_or(0);
						if inner_profile_enabled {
							self.debug.merge_total_ns += merge_ns;
						}
						let before_depth = merged.max_depth();
						self.debug.fuse_calls += 1;
						let fuse_start = if inner_profile_enabled { Some(Instant::now()) } else { None };
						let fused = merged.fuse(Some(1));
						if let Some(start) = fuse_start {
							self.debug.fuse_total_ns += elapsed_ns(start);
						}
						if fused.max_depth() != before_depth {
							self.debug.fuse_changed_depth += 1;
						}
						if merge_profile_enabled {
							let existing_summary = existing.summary();
							let incoming_summary = gss.summary();
							let line = format!(
								"[glrmask/debug][mask_queue_merge] mode=Target target={} existing_depth={} incoming_depth={} merged_depth={} merge_ns={} existing_top_values={} incoming_top_values={} existing_nodes={} incoming_nodes={} existing_edges={} incoming_edges={} existing_accs={} incoming_accs={}",
								target,
								existing_depth,
								incoming_depth,
								fused.max_depth(),
								merge_ns,
								existing_summary.top_values_count,
								incoming_summary.top_values_count,
								existing_summary.total_unique_nodes,
								incoming_summary.total_unique_nodes,
								existing_summary.total_edges,
								incoming_summary.total_edges,
								existing_summary.accumulator_instances,
								incoming_summary.accumulator_instances,
							);
							emit_mask_queue_merge_profile_line(&line);
						}
						fused
					}
					None => {
						self.debug.insert_without_merge_count += 1;
						gss
					}
				};

				let depth = merged.max_depth();
				let insert_start = if inner_profile_enabled { Some(Instant::now()) } else { None };
				by_target.insert(target, merged);
				ready_by_depth.entry(depth).or_default().push(target);
				if let Some(start) = insert_start {
					self.debug.insert_total_ns += elapsed_ns(start);
				}
			}
			MaskQueueInner::Depth { by_depth } => {
				let depth = gss.max_depth();
				let lookup_start = if inner_profile_enabled { Some(Instant::now()) } else { None };
				let target_items = by_depth
					.entry(depth)
					.or_default()
					.entry(target)
					.or_default();
				let existing = match self.mode {
					MaskQueueMode::DepthMerge => target_items.pop(),
					MaskQueueMode::DepthBatch4
						if target_items.len() + 1 >= DEPTH_BATCH_MERGE_THRESHOLD =>
					{
						let mut existing = target_items.pop().unwrap();
						while let Some(other) = target_items.pop() {
							existing = existing.merge(&other);
							self.debug.merge_hit_count += 1;
						}
						Some(existing)
					}
					MaskQueueMode::Target
					| MaskQueueMode::Depth
					| MaskQueueMode::DepthBatch4 => None,
				};
				if let Some(start) = lookup_start {
					self.debug.lookup_total_ns += elapsed_ns(start);
				}

				let merged = match existing {
					Some(existing) => {
						self.debug.merge_hit_count += 1;
						let existing_depth = existing.max_depth();
						let incoming_depth = gss.max_depth();
						let merge_start = if inner_profile_enabled || merge_profile_enabled {
							Some(Instant::now())
						} else {
							None
						};
						let merged = existing.merge(&gss);
						let merge_ns = merge_start.map(elapsed_ns).unwrap_or(0);
						if inner_profile_enabled {
							self.debug.merge_total_ns += merge_ns;
						}
						if merge_profile_enabled {
							let existing_summary = existing.summary();
							let incoming_summary = gss.summary();
							let line = format!(
								"[glrmask/debug][mask_queue_merge] mode=Depth target={} existing_depth={} incoming_depth={} merged_depth={} merge_ns={} existing_top_values={} incoming_top_values={} existing_nodes={} incoming_nodes={} existing_edges={} incoming_edges={} existing_accs={} incoming_accs={}",
								target,
								existing_depth,
								incoming_depth,
								merged.max_depth(),
								merge_ns,
								existing_summary.top_values_count,
								incoming_summary.top_values_count,
								existing_summary.total_unique_nodes,
								incoming_summary.total_unique_nodes,
								existing_summary.total_edges,
								incoming_summary.total_edges,
								existing_summary.accumulator_instances,
								incoming_summary.accumulator_instances,
							);
							emit_mask_queue_merge_profile_line(&line);
						}
						merged
					}
					None => {
						self.debug.insert_without_merge_count += 1;
						gss
					}
				};

				let insert_start = if inner_profile_enabled { Some(Instant::now()) } else { None };
				target_items.push(merged);
				if let Some(start) = insert_start {
					self.debug.insert_total_ns += elapsed_ns(start);
				}
			}
		}

		if let Some(start) = enqueue_start {
			self.debug.enqueue_total_ns += elapsed_ns(start);
		}
	}

	pub(super) fn pop_next(&mut self) -> Option<(u32, DenseMaskGSS)> {
		let pop_start = if mask_inner_profile_enabled() { Some(Instant::now()) } else { None };
		match &mut self.inner {
			MaskQueueInner::Target {
				by_target,
				ready_by_depth,
			} => loop {
				let mut depth_entry = ready_by_depth.last_entry()?;
				let depth = *depth_entry.key();
				let target = match depth_entry.get_mut().pop() {
					Some(target) => target,
					None => {
						depth_entry.remove();
						continue;
					}
				};

				if depth_entry.get().is_empty() {
					depth_entry.remove();
				}

				let Some(current) = by_target.get(&target) else {
					self.debug.stale_schedule_skips += 1;
					continue;
				};

				if current.max_depth() != depth {
					self.debug.stale_schedule_skips += 1;
					continue;
				}

				let gss = by_target
					.remove(&target)
					.expect("target must exist after stale-check");
				self.debug.popped_items += 1;
				if let Some(start) = pop_start {
					self.debug.pop_total_ns += elapsed_ns(start);
				}
				return Some((target, gss));
			},
			MaskQueueInner::Depth { by_depth } => loop {
				let mut depth_entry = by_depth.last_entry()?;
				let target = match depth_entry.get().keys().next().copied() {
					Some(target) => target,
					None => {
						depth_entry.remove();
						continue;
					}
				};
				let items = depth_entry
					.get_mut()
					.get_mut(&target)
					.expect("target must exist in depth bucket");
				let gss = items.pop().expect("target list must be non-empty");
				if items.is_empty() {
					depth_entry.get_mut().remove(&target);
				}
				if depth_entry.get().is_empty() {
					depth_entry.remove();
				}
				self.debug.popped_items += 1;
				if let Some(start) = pop_start {
					self.debug.pop_total_ns += elapsed_ns(start);
				}
				return Some((target, gss));
			},
		}
	}

	pub(super) fn debug_stats(&self) -> &MaskQueueDebugStats {
		&self.debug
	}

	pub(super) fn record_seed_decompose_callback(&mut self) {
		self.debug.seed_decompose_callbacks += 1;
	}

	pub(super) fn record_loop_decompose_callback(&mut self) {
		self.debug.loop_decompose_callbacks += 1;
	}

	pub(super) fn record_parser_dwa_transition_enqueue(&mut self) {
		self.debug.parser_dwa_transitions_enqueued += 1;
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::ds::leveled_gss::LeveledGSS;

	fn gss(stack: &[u32], bit: u64) -> DenseMaskGSS {
		let acc = super::super::DenseMaskAcc::from_dense(0, vec![bit]).unwrap();
		LeveledGSS::from_stacks(&[(stack.to_vec(), acc)])
	}

	#[test]
	fn depth_merge_coalesces_same_target_at_same_depth() {
		let left = gss(&[0, 1], 1);
		let right = gss(&[0, 2], 2);
		let expected = left.merge(&right);
		let mut queue = MaskQueue::new_with_mode(MaskQueueMode::DepthMerge);

		queue.enqueue(7, left);
		queue.enqueue(7, right);

		assert_eq!(queue.debug.merge_hit_count, 1);
		let (target, actual) = queue.pop_next().unwrap();
		assert_eq!(target, 7);
		assert_eq!(actual.semantically_eq(&expected, 16), Some(true));
		assert!(queue.pop_next().is_none());
	}

	#[test]
	fn depth_merge_keeps_different_depths_separate() {
		let mut queue = MaskQueue::new_with_mode(MaskQueueMode::DepthMerge);
		queue.enqueue(7, gss(&[0, 1], 1));
		queue.enqueue(7, gss(&[0, 1, 2], 2));

		assert_eq!(queue.debug.merge_hit_count, 0);
		assert!(queue.pop_next().is_some());
		assert!(queue.pop_next().is_some());
		assert!(queue.pop_next().is_none());
	}

	#[test]
	fn legacy_depth_mode_keeps_same_bucket_items_separate() {
		let mut queue = MaskQueue::new_with_mode(MaskQueueMode::Depth);
		queue.enqueue(7, gss(&[0, 1], 1));
		queue.enqueue(7, gss(&[0, 2], 2));

		assert_eq!(queue.debug.merge_hit_count, 0);
		assert!(queue.pop_next().is_some());
		assert!(queue.pop_next().is_some());
		assert!(queue.pop_next().is_none());
	}

	#[test]
	fn depth_batch_four_merges_only_after_fourth_same_bucket_item() {
		let stacks = [
			gss(&[0, 1], 1),
			gss(&[0, 2], 2),
			gss(&[0, 3], 4),
			gss(&[0, 4], 8),
		];
		let expected = stacks.iter().skip(1).fold(stacks[0].clone(), |acc, gss| acc.merge(gss));
		let mut queue = MaskQueue::new_with_mode(MaskQueueMode::DepthBatch4);

		for gss in stacks.iter().take(3) {
			queue.enqueue(7, gss.clone());
		}
		assert_eq!(queue.debug.merge_hit_count, 0);

		queue.enqueue(7, stacks[3].clone());
		assert_eq!(queue.debug.merge_hit_count, 3);
		let (target, actual) = queue.pop_next().unwrap();
		assert_eq!(target, 7);
		assert_eq!(actual.semantically_eq(&expected, 16), Some(true));
		assert!(queue.pop_next().is_none());
	}
}
