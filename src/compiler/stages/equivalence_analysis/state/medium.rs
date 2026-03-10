#![allow(dead_code)]
//! State equivalence analysis — medium tier.
//!
//! Not yet implemented. This would be an intermediate fidelity approach,
//! potentially using a flat DFA walk without the batched hashing optimisation
//! of the fast implementation.
//!
//! When implemented, this should serve as a validation tier between slow
//! (reference) and fast (production).
