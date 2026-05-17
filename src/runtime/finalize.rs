use super::artifact::Constraint;

impl Constraint {
	pub(crate) fn rebuild_runtime_caches(&mut self) {
		self.rebuild_runtime_caches_impl();
	}
}
