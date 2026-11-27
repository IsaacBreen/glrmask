use crate::{ConstraintState, Result, Vocab};
use crate::runtime::Constraint as RuntimeConstraint;


/// A compiled grammar, ready to start independent sequences.
/// It is Send + Sync, so it can be shared across threads using an Arc.
#[derive(Clone, Debug)]
pub struct Constraint {
    inner: RuntimeConstraint,
}

impl Constraint {
    pub(crate) fn from_runtime(inner: RuntimeConstraint) -> Self { Self { inner } }

    /// Starts a new sequence.
    pub fn start(&self) -> ConstraintState { self.inner.start() }

    /// The mask's length in bits: the largest token ID in the constraint (including special and end tokens) plus one, or zero if there are no token IDs.
    pub fn mask_bit_len(&self) -> usize { self.inner.mask_bit_len() }

    pub fn save(&self) -> Vec<u8> { self.inner.save() }

    /// May fail if saved by a different GLRMask version. Fails if `vocab` differs from the one it was compiled with.
    pub fn load(bytes: &[u8], vocab: &Vocab) -> Result<Self> {
        crate::artifact_load::guard(|| RuntimeConstraint::load(bytes, vocab).map(Self::from_runtime))
    }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn __inner(&self) -> &RuntimeConstraint { &self.inner }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn __inner_mut(&mut self) -> &mut RuntimeConstraint { &mut self.inner }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn __into_inner(self) -> RuntimeConstraint { self.inner }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn __from_inner(inner: RuntimeConstraint) -> Self { Self::from_runtime(inner) }
}
