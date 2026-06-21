#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

#[path = "../../src/automata/mod.rs"]
pub(crate) mod automata;
#[path = "../../src/ds/mod.rs"]
pub(crate) mod ds;
mod compiler;
#[path = "../../src/error.rs"]
mod error;
mod grammar;
#[path = "../../src/runtime/mod.rs"]
pub(crate) mod runtime;

pub use error::{Error, GlrMaskError, Result};
pub use runtime::{Constraint, ConstraintState};

/// Versioned envelope for browser and provider runtime artifacts.
/// The payload is the explicit v1 execution-state contract, not `Constraint`'s
/// incidental serde layout.
pub const ARTIFACT_MAGIC: [u8; 8] = *b"GLRMASK\0";
pub const ARTIFACT_VERSION: u16 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeArtifact {
    bytes: Vec<u8>,
}

impl RuntimeArtifact {
    pub fn from_compiled_constraint(constraint: &Constraint) -> Self {
        Self::from_runtime_payload_v1(constraint.save_runtime_payload_v1())
    }

    /// Wrap bytes produced by `Constraint::save_runtime_payload_v1`.
    ///
    /// This accepts bytes instead of a `Constraint` so a native compiler and a
    /// separately linked runtime crate can share the same artifact contract.
    pub fn from_runtime_payload_v1(payload: Vec<u8>) -> Self {
        let mut bytes = Vec::with_capacity(ARTIFACT_MAGIC.len() + 2 + 8 + payload.len());
        bytes.extend_from_slice(&ARTIFACT_MAGIC);
        bytes.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8] { &self.bytes }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        const HEADER: usize = 8 + 2 + 8;
        if bytes.len() < HEADER || bytes[..8] != ARTIFACT_MAGIC {
            return Err(GlrMaskError::Serialization("invalid glrmask runtime artifact magic".to_owned()));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if version != ARTIFACT_VERSION {
            return Err(GlrMaskError::Serialization(format!("unsupported glrmask runtime artifact version {version}")));
        }
        let length = u64::from_le_bytes(bytes[10..18].try_into().expect("fixed artifact header")) as usize;
        if bytes.len() != HEADER.saturating_add(length) {
            return Err(GlrMaskError::Serialization("invalid glrmask runtime artifact payload length".to_owned()));
        }
        Ok(Self { bytes })
    }

    pub fn into_constraint(self) -> Result<Constraint> {
        Constraint::load_runtime_payload_v1(&self.bytes[18..])
    }
}

pub struct Session {
    // This field must remain before `constraint`: Rust drops fields in declaration
    // order, so the borrow-carrying state is dropped while its stable Box owner is
    // still alive. Moving Session moves only the Box pointer, never the allocation.
    state: ConstraintState<'static>,
    constraint: Box<Constraint>,
}

impl Session {
    pub fn from_artifact(artifact: RuntimeArtifact) -> Result<Self> {
        let constraint = Box::new(artifact.into_constraint()?);
        let constraint_ref = Self::stable_constraint_ref(&constraint);
        let state = constraint_ref.start();
        Ok(Self { state, constraint })
    }

    pub fn mask_words(&self) -> Vec<u32> { self.state.mask() }

    /// Fill a caller-owned packed original-vocabulary mask without allocating.
    /// This is the hot CFA/runtime path and matches the main executor's mask API.
    pub fn fill_mask(&self, words: &mut [u32]) {
        self.state.fill_mask(words);
    }

    /// As `fill_mask`, with timing measured entirely inside the runtime crate.
    pub fn fill_mask_timed_ns(&self, words: &mut [u32]) -> u64 {
        self.state.fill_mask_timed_ns(words)
    }

    pub fn mask_len(&self) -> usize {
        self.constraint.mask_len()
    }

    pub fn commit_token(&mut self, token_id: u32) -> std::result::Result<(), String> {
        self.state.commit_token(token_id)
    }

    /// Commit a sampled BPE token with timing measured inside the runtime crate.
    pub fn commit_token_timed_ns(&mut self, token_id: u32) -> std::result::Result<u64, String> {
        self.state.commit_token_timed_ns(token_id)
    }

    pub fn eos_allowed(&self) -> bool { self.state.is_complete() }

    pub fn is_finished(&self) -> bool { self.state.is_finished() }

    pub fn reset(&mut self) {
        let constraint_ref = Self::stable_constraint_ref(&self.constraint);
        self.state = constraint_ref.start();
    }

    fn stable_constraint_ref(constraint: &Box<Constraint>) -> &'static Constraint {
        unsafe {
            // The allocation is pinned by the Box for the full Session lifetime.
            // The only `'static` reference stays inside Session and `state` drops
            // before `constraint`; it cannot escape through this API.
            &*(constraint.as_ref() as *const Constraint)
        }
    }
}
