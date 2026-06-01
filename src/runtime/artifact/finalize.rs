//! Runtime artifact finalization entry points.
//!
//! Finalization is the operation
//!
//! ```text
//! semantic compiled artifact  ↦  semantic artifact + derived runtime caches
//! ```
//!
//! It is intentionally separate from compilation and deserialization.  Both
//! paths call the same operation after they have a `Constraint` value.

use super::Constraint;

impl Constraint {
	/// Rebuild every derived `#[serde(skip)]` runtime cache.
	pub(crate) fn rebuild_runtime_caches(&mut self) {
		self.rebuild_runtime_caches_impl();
	}
}
