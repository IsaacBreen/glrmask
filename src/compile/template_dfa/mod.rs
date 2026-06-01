//! Compatibility shim for the old compile::template_dfa path.
//! Publication code should use `compile::template_dfa`.

#[doc(hidden)]
pub(crate) use crate::compile::template_dfa::*;

#[doc(hidden)] pub(crate) mod characterize { pub(crate) use crate::compile::template_dfa::characterize::*; }
#[doc(hidden)] pub(crate) mod compile_bundle { pub(crate) use crate::compile::template_dfa::compile_bundle::*; }
#[doc(hidden)] pub(crate) mod compile_dfa { pub(crate) use crate::compile::template_dfa::compile_dfa::*; }
