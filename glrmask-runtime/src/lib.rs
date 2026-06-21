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
/// The payload remains the current exact compiled-constraint encoding.
pub const ARTIFACT_MAGIC: [u8; 8] = *b"GLRMASK\0";
pub const ARTIFACT_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeArtifact {
    bytes: Vec<u8>,
}

impl RuntimeArtifact {
    pub fn from_compiled_constraint(constraint: &Constraint) -> Self {
        Self::from_compiled_payload(constraint.save())
    }

    pub fn from_compiled_payload(payload: Vec<u8>) -> Self {
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
        Constraint::load(&self.bytes[18..])
    }
}

pub struct Session {
    // `state` borrows the stable heap allocation owned by `constraint`. Moving a
    // Session moves only the Box pointer, never the Constraint allocation itself.
    constraint: Box<Constraint>,
    state: ConstraintState<'static>,
}

impl Session {
    pub fn from_artifact(artifact: RuntimeArtifact) -> Result<Self> {
        let constraint = Box::new(artifact.into_constraint()?);
        let constraint_ref: &'static Constraint = unsafe {
            // The Box allocation remains stable for the entire Session lifetime and
            // `state` is dropped before `constraint` because fields drop in order.
            &*(constraint.as_ref() as *const Constraint)
        };
        let state = constraint_ref.start();
        Ok(Self { constraint, state })
    }

    pub fn mask_words(&self) -> Vec<u32> { self.state.mask() }

    pub fn commit_token(&mut self, token_id: u32) -> std::result::Result<(), String> {
        self.state.commit_token(token_id)
    }

    pub fn eos_allowed(&self) -> bool { self.state.is_complete() }

    pub fn reset(&mut self) {
        let constraint_ref: &'static Constraint = unsafe {
            &*(self.constraint.as_ref() as *const Constraint)
        };
        self.state = constraint_ref.start();
    }
}
