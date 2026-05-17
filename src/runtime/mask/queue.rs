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
}

pub(super) fn mask_queue_mode() -> MaskQueueMode {
	static MODE: std::sync::OnceLock<MaskQueueMode> = std::sync::OnceLock::new();
	*MODE.get_or_init(|| match std::env::var("GLRMASK_MASK_QUEUE_MODE") {
		Ok(value) if value.trim().eq_ignore_ascii_case("target") => MaskQueueMode::Target,
		_ => MaskQueueMode::Depth,
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
	inner: MaskQueueInner,
	debug: MaskQueueDebugStats,
}

impl Default for MaskQueue {
	fn default() -> Self {
		let inner = match mask_queue_mode() {
			MaskQueueMode::Target => MaskQueueInner::Target {
				by_target: FxHashMap::default(),
				ready_by_depth: BTreeMap::new(),
			},
			MaskQueueMode::Depth => MaskQueueInner::Depth {
				by_depth: BTreeMap::new(),
			},
		};

		Self {
			inner,
			debug: MaskQueueDebugStats::default(),
		}
	}
}

impl MaskQueue {
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
				let existing: Option<DenseMaskGSS> = None;
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
				by_depth.entry(depth).or_default().entry(target).or_default().push(merged);
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
