//! Serialization format for compiled runtime artifacts.
//!
//! Public `save` now writes a small versioned envelope around the compiled
//! artifact.  `load` first tries the envelope and then falls back to the legacy
//! direct-bincode encoding used before the publication cleanup.  This gives the
//! crate a documented compatibility boundary without making old cache files
//! immediately unreadable.

use serde::{Deserialize, Serialize};

use super::Constraint;

const SERIALIZATION_FORMAT_VERSION: u32 = 1;
const SERIALIZATION_MAGIC: &str = "glrmask.constraint";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedArtifactEnvelope {
    magic: String,
    format_version: u32,
    features: SerializedArtifactFeatures,
    constraint: Constraint,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SerializedArtifactFeatures {
    /// Whether the artifact stores a final original-token -> internal-token map.
    final_internal_token_map: bool,
    /// Whether the artifact stores final internal-token byte representatives.
    internal_token_bytes: bool,
    /// Whether the artifact stores terminal display names for diagnostics.
    terminal_display_names: bool,
}

impl SerializedArtifactFeatures {
    fn from_constraint(constraint: &Constraint) -> Self {
        Self {
            final_internal_token_map: !constraint.original_token_to_internal.is_empty()
                && !constraint.internal_token_to_tokens.is_empty(),
            internal_token_bytes: !constraint.internal_token_bytes.is_empty(),
            terminal_display_names: !constraint.terminal_display_names.is_empty(),
        }
    }
}

impl Constraint {
    /// Serialize this compiled constraint.
    ///
    /// Derived runtime caches are not serialized; they are marked
    /// `#[serde(skip)]` and rebuilt by [`Constraint::load`].
    pub fn save(&self) -> Vec<u8> {
        let envelope = SerializedArtifactEnvelope {
            magic: SERIALIZATION_MAGIC.to_owned(),
            format_version: SERIALIZATION_FORMAT_VERSION,
            features: SerializedArtifactFeatures::from_constraint(self),
            constraint: self.clone(),
        };
        bincode::serialize(&envelope).expect("Constraint serialization should succeed")
    }

    /// Load a compiled constraint and rebuild its derived runtime caches.
    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        if let Ok(envelope) = bincode::deserialize::<SerializedArtifactEnvelope>(bytes) {
            if envelope.magic != SERIALIZATION_MAGIC {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "unexpected glrmask artifact magic: {}",
                    envelope.magic
                )));
            }
            if envelope.format_version != SERIALIZATION_FORMAT_VERSION {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "unsupported glrmask artifact version: {}",
                    envelope.format_version
                )));
            }
            let mut constraint = envelope.constraint;
            constraint.rebuild_runtime_caches();
            return Ok(constraint);
        }

        // Legacy fallback: pre-envelope artifacts were direct bincode encodings
        // of `Constraint`.
        let mut constraint: Self = bincode::deserialize(bytes)
            .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
        constraint.rebuild_runtime_caches();
        Ok(constraint)
    }
}
